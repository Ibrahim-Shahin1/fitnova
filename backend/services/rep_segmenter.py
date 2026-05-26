"""RepSegmenter — no-pose rep segmentation for upload + live form-analysis paths.

Fitness-AQA is a raw-pixel dataset.  The paper's premise is that pose is unreliable
in-the-wild; MediaPipe was removed from the serving path in Phase 5 (D-06).  Rep
segmentation therefore operates on motion energy (frame-difference signal) — no pose,
no optical flow.

Upload mode (ships first):
  segment_reps_by_motion_energy(video_path) → list[(start, end)]
  Low-res 64×64 grayscale scan → per-frame energy → smooth → threshold → merge.
  FALLBACK (guaranteed): if no boundary detected → return [(0, total_frames-1)].
  The Fitness-AQA clips are single-rep ~3 s @ 30 fps — the single-rep fallback
  is the statistically dominant case and is exactly correct for training-distribution
  clips (D-03, D-04).

Live mode (per D-03: sliding-window trigger):
  LiveWindowTrigger (per-connection helper) — enforces:
    buffer_len >= LIVE_REP_WINDOW_FRAMES  AND
    frames_since_last_trigger >= LIVE_MIN_GAP_FRAMES
  before returning True.  Keeps trigger arithmetic unit-testable independent of the
  WebSocket / JPEG decode layer (Plan 03 wires this into SquatLiveSession).

Why frame-difference over optical flow (D-04):
  cv2.calcOpticalFlowPyrLK requires extra OpenCV modules and is slower.
  Frame-difference is faster, zero additional dependencies, and good enough for ~3 s
  single-rep clips where the subject is the dominant motion source.
  [ASSUMED — see RESEARCH §Rep Segmentation §Assumptions A2/A3]

See: .planning/phases/05-backend-inference-integration-squat/05-RESEARCH.md §2.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module constants (SCREAMING_SNAKE — mirrors form_session.py convention)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_REP_WINDOW_FRAMES: int = 32     # clip window sent to the classifier per trigger
LIVE_MIN_GAP_FRAMES: int = 45        # (legacy LiveWindowTrigger) min gap between triggers
LIVE_BUFFER_MAX_FRAMES: int = 90     # rolling deque depth (~3 s of frames)

# Live rep-end detection (motion-energy state machine — replaces the fixed clock so
# feedback fires right after the rep, on the rep's frames, not on a timer).
LIVE_REP_ACTIVE_FACTOR: float = 1.8   # "moving" when energy > factor x running baseline
LIVE_REP_MIN_ENERGY: float = 4.0      # abs floor (0-255 gray) so sensor noise never fires
LIVE_REP_MIN_ACTIVE_FRAMES: int = 12  # min motion span to count as a rep (~0.4-0.5 s)
LIVE_REP_SETTLE_FRAMES: int = 5       # frames settled below threshold => rep ended
LIVE_REP_COOLDOWN_FRAMES: int = 8     # ignore frames right after firing (no double-fire)
LIVE_REP_MAX_WINDOW_FRAMES: int = 64  # cap the classified rep window
LIVE_BASELINE_DECAY: float = 0.9      # EMA decay for the resting-energy baseline

DEFAULT_MIN_REP_FRAMES: int = 60     # minimum frame span to count as a valid rep region
ENERGY_THRESHOLD_FACTOR: float = 0.5 # threshold = factor × mean(smoothed_energy)
SMOOTH_WINDOW: int = 5               # boxcar smoothing window width (frames)


# ─────────────────────────────────────────────────────────────────────────────
# Upload segmentation
# ─────────────────────────────────────────────────────────────────────────────


def segment_reps_by_motion_energy(
    video_path: str,
    min_rep_frames: int = DEFAULT_MIN_REP_FRAMES,
    energy_threshold_factor: float = ENERGY_THRESHOLD_FACTOR,
) -> list[tuple[int, int]]:
    """Segment a clip into rep intervals by frame-difference motion energy.

    Strategy:
      1. Decode all frames at low resolution (64×64 grayscale) for speed.
         Full resolution is not needed for motion energy computation.
      2. Per-frame motion energy = mean(abs(frame[i] - frame[i-1])).
      3. Smooth the energy signal with a boxcar of width SMOOTH_WINDOW.
      4. threshold = energy_threshold_factor × mean(smoothed); find contiguous
         regions where smoothed > threshold.
      5. Merge regions whose inter-region gap < min_rep_frames.
      6. Each merged region is one rep (start_frame, end_frame), 0-indexed into
         the full clip frame sequence.

    FALLBACK (ALWAYS returns >= 1 interval):
      If no region is detected (uniform or static clip), treat the entire clip as
      one rep: return [(0, total_frames-1)].  This is the correct behaviour for
      Fitness-AQA single-rep training clips (D-03).

    Args:
        video_path:              absolute path to the .mp4 clip.
        min_rep_frames:          minimum frame span to count as a valid active region.
                                 Shorter regions are merged into adjacent ones or dropped.
        energy_threshold_factor: active if smoothed_energy > factor × mean_energy.
                                 Lower = more sensitive (more reps detected).

    Returns:
        Non-empty list of (start_frame, end_frame) tuples, 0-indexed into the decoded
        frame sequence.  Guaranteed to contain at least one element.
    """
    cap = cv2.VideoCapture(video_path)
    frames_gray: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames_gray.append(gray)

    cap.release()

    total_frames = len(frames_gray)

    # Degenerate case: cannot compute frame-difference with fewer than 2 frames.
    if total_frames < 2:
        logger.warning(
            "rep_segmenter: decoded only %d frame(s) from %s — returning single-rep fallback.",
            total_frames,
            video_path,
        )
        return [(0, max(total_frames - 1, 0))]

    # Step 2: per-frame motion energy (frame-difference magnitude).
    energy = np.array(
        [np.mean(np.abs(frames_gray[i] - frames_gray[i - 1])) for i in range(1, total_frames)],
        dtype=np.float32,
    )
    # Prepend a zero so energy[i] aligns with frames_gray[i].
    energy = np.concatenate([[0.0], energy])  # shape: [total_frames]

    # Step 3: smooth with a boxcar window.
    kernel = np.ones(SMOOTH_WINDOW, dtype=np.float32) / SMOOTH_WINDOW
    smoothed = np.convolve(energy, kernel, mode="same")

    energy_mean = float(smoothed.mean())
    energy_peak = float(smoothed.max())
    threshold = energy_threshold_factor * energy_mean

    logger.info(
        "rep_segmenter: energy mean=%.3f peak=%.3f threshold=%.3f frames=%d path=%s",
        energy_mean,
        energy_peak,
        threshold,
        total_frames,
        video_path,
    )

    # Step 4: find contiguous above-threshold regions.
    active = smoothed > threshold
    rep_intervals: list[tuple[int, int]] = []
    in_region = False
    region_start = 0

    for i, is_active in enumerate(active):
        if is_active and not in_region:
            region_start = i
            in_region = True
        elif not is_active and in_region:
            if i - region_start >= min_rep_frames:
                rep_intervals.append((region_start, i - 1))
            in_region = False

    # Close any open region at the end.
    if in_region and total_frames - region_start >= min_rep_frames:
        rep_intervals.append((region_start, total_frames - 1))

    # Step 5: merge adjacent regions whose gap < min_rep_frames.
    if len(rep_intervals) > 1:
        merged: list[tuple[int, int]] = [rep_intervals[0]]
        for start, end in rep_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_rep_frames:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        rep_intervals = merged

    # Step 6: FALLBACK — always return at least one interval.
    # A single-rep clip with no detectable boundaries is the expected Fitness-AQA case.
    if not rep_intervals:
        logger.info(
            "rep_segmenter: no boundaries detected in %s — treating as single rep.",
            video_path,
        )
        rep_intervals = [(0, total_frames - 1)]

    logger.info(
        "rep_segmenter: detected %d interval(s) in %s.",
        len(rep_intervals),
        video_path,
    )
    return rep_intervals


# ─────────────────────────────────────────────────────────────────────────────
# Live window trigger
# ─────────────────────────────────────────────────────────────────────────────


class LiveWindowTrigger:
    """Per-connection inference trigger for the live WebSocket path (D-03).

    Enforces two independent conditions before returning True:
      1. buffer_len >= LIVE_REP_WINDOW_FRAMES  (enough frames for a full clip window)
      2. frames_since_last_trigger >= LIVE_MIN_GAP_FRAMES  (minimum inter-inference gap)

    Keeping this logic here (not in the WS handler) makes it unit-testable without
    a running WebSocket connection.  Plan 03's SquatLiveSession calls should_fire()
    on every incoming frame.

    Usage:
        trigger = LiveWindowTrigger()  # one instance per WebSocket connection
        # In add_frame():
        if trigger.should_fire(len(frame_buffer)):
            # extract window, classify, emit rep_result

    Thread safety: NOT thread-safe.  Each WebSocket connection must use its own
    LiveWindowTrigger instance (one per SquatLiveSession).
    """

    def __init__(self) -> None:
        self._frames_since_last: int = 0

    def should_fire(self, buffer_len: int) -> bool:
        """Increment internal counter; return True if both gate conditions are met.

        Resets the counter to 0 on fire, enforcing the minimum gap for the next trigger.

        Args:
            buffer_len: current number of frames in the rolling buffer (e.g. len(deque)).

        Returns:
            True if inference should run now; False otherwise.
        """
        self._frames_since_last += 1
        if (
            buffer_len >= LIVE_REP_WINDOW_FRAMES
            and self._frames_since_last >= LIVE_MIN_GAP_FRAMES
        ):
            self._frames_since_last = 0
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Live rep-end detector (motion-energy state machine) — the rep-aware trigger
# ─────────────────────────────────────────────────────────────────────────────


class LiveRepDetector:
    """Streaming motion-energy rep-end detector for the live WS path (no pose).

    Replaces the fixed-clock LiveWindowTrigger. The clock fired every N frames and
    classified whatever was in the rolling window at that tick — so pausing after a
    rep made it analyze the standing pose (lag + wrong-window bug). This detector
    fires right when a rep *ends*, returning the rep's frame span so the caller
    classifies exactly the rep's frames.

    State machine over per-frame frame-difference energy (downscaled grayscale):
      IDLE   --energy rises above adaptive threshold-->  MOVING
      MOVING --energy settles below threshold for SETTLE frames-->  fire (if the
              active span >= MIN_ACTIVE), enter COOLDOWN, back to IDLE

    Adaptive threshold = max(ACTIVE_FACTOR x running baseline, MIN_ENERGY). The
    baseline is an EMA of resting energy (updated while idle / in cooldown), so it
    adapts to lighting/camera without firing on sensor noise.

    NOTE: the thresholds are reasonable defaults but are genuinely device-dependent
    (camera, distance, lighting). They need calibration on the user's real squats —
    flagged as a human-verify item. Best results when the user starts the session
    standing still (establishes a low baseline) then squats and pauses between reps.

    Thread safety: NOT thread-safe — one instance per SquatLiveSession.
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._baseline: float | None = None
        self._moving: bool = False
        self._active_len: int = 0
        self._below: int = 0
        self._cooldown: int = 0

    def push(self, frame_rgb: np.ndarray) -> int | None:
        """Feed one RGB frame; return the rep's frame span on rep-end, else None.

        Args:
            frame_rgb: HxWx3 uint8 RGB frame.

        Returns:
            The rep window length (number of frames, capped at LIVE_REP_MAX_WINDOW_FRAMES)
            when a rep just ended; None otherwise.
        """
        small = cv2.resize(frame_rgb, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        energy = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray

        if self._baseline is None:
            self._baseline = energy
        threshold = max(LIVE_REP_ACTIVE_FACTOR * self._baseline, LIVE_REP_MIN_ENERGY)

        # Cooldown: ignore motion right after a fire; keep the baseline fresh.
        if self._cooldown > 0:
            self._cooldown -= 1
            self._baseline = LIVE_BASELINE_DECAY * self._baseline + (1 - LIVE_BASELINE_DECAY) * energy
            return None

        if not self._moving:
            if energy > threshold:
                self._moving = True
                self._active_len = 1
                self._below = 0
            else:
                # Resting — adapt the baseline toward the quiet energy level.
                self._baseline = LIVE_BASELINE_DECAY * self._baseline + (1 - LIVE_BASELINE_DECAY) * energy
            return None

        # Moving: accumulate the active span; watch for a settle.
        self._active_len += 1
        if energy <= threshold:
            self._below += 1
            if self._below >= LIVE_REP_SETTLE_FRAMES:
                span = self._active_len
                self._moving = False
                self._below = 0
                self._cooldown = LIVE_REP_COOLDOWN_FRAMES
                if span >= LIVE_REP_MIN_ACTIVE_FRAMES:
                    return min(span, LIVE_REP_MAX_WINDOW_FRAMES)
                return None  # too short to be a rep — discard
        else:
            self._below = 0
        return None
