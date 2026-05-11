"""
FormSession — manages one real-time exercise analysis session.

Each WebSocket connection gets its own FormSession instance.
The session:
  - Accepts JPEG frames, extracts poses and angular features
  - Maintains a 64-frame sliding window
  - Runs MT-TCN inference every INFERENCE_INTERVAL frames
  - Detects rep boundaries from the boundary probability signal
  - Accumulates per-rep quality scores and joint errors
  - On end_session(): compiles summary + calls GPT-4o-mini for coaching feedback
"""

import logging
import os
import time
from collections import deque
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .form_analyzer import FormAnalyzer
from .exercise_sanity import ExerciseMismatchDetector
# form_geometry removed from runtime (Day 0.5, 2026-05-11) — pure ML signal
# only. The module remains in the repo for history/ablation reuse but is
# not wired into any session. See plans/handoff-2026-05-11-md-this-handoff-is-sleepy-sparrow.md §6.
from backend.config.exercises import (
    get_metadata as _get_exercise_meta,
    relevant_joint_indices as _relevant_joint_indices,
)

logger = logging.getLogger(__name__)

WINDOW_SIZE         = 64    # frames
INFERENCE_INTERVAL  = 10    # run model every N frames (~1s at 10fps)
BOUNDARY_THRESHOLD  = 0.55  # boundary probability to count as a rep boundary
MIN_REP_GAP_FRAMES  = 20    # minimum frames between two rep boundaries
EXERCISE_CONF_THRESHOLD = 0.40  # min exercise confidence to count a boundary

# v5.2 boundary head saturates near 1.0 on real video AND quality output is
# compressed to [~0.35, ~0.55], so neither rising-edge boundary detection
# nor quality-cycle detection produces a reliable rep count.
#
# Two-layer fallback:
#   1. Quality-cycle detection (PRIMARY). Each rep → one dip+recovery in
#      the smoothed quality stream. Thresholds are loose given v5.2's
#      compressed range. If the future model has wider quality dynamic
#      range, these can be tightened.
#   2. Time-based fallback (SECONDARY). When session is active (exercise
#      stable + confident) but no quality cycle has fired for
#      TIME_BASED_REP_INTERVAL frames, force-increment one rep. Catches
#      the case where the quality stream has no detectable cycles.
QUALITY_REP_PEAK_DELTA       = 0.04    # peak − trough must reach this
QUALITY_TROUGH_FLOOR         = 0.85    # trough below this counts
TIME_BASED_REP_INTERVAL      = 30      # ≈ 3 sec at 10fps inference cadence

# v5.2-specific: disable the quality-cycle detector entirely. The model's
# quality output is too compressed and flat to produce reliable cycles, and
# letting it fire alongside the time fallback caused over-counting (8 reps
# for a 5-rep video). When QEVD-trained v6 lands with a wider quality range,
# this can be set back to True.
ENABLE_QUALITY_CYCLE_DETECTOR = False

# EMA smoothing factors (R5). Each new per-frame prediction replaces
# `alpha` of the previous smoothed value:
#     smoothed = alpha * new + (1 - alpha) * smoothed
# Lower alpha = smoother UI, more latency.
# Higher alpha = snappier UI, more flicker.
# Quality and joint_errors get the same alpha; exercise label has no EMA
# (it is categorical — we argmax the softmax of the latest window).
EMA_ALPHA_QUALITY     = 0.30
EMA_ALPHA_JOINT_ERR   = 0.30

JOINT_GROUP_NAMES   = [
    "Left Elbow", "Right Elbow",
    "Left Shoulder", "Right Shoulder",
    "Left Knee", "Right Knee",
    "Left Hip", "Right Hip",
    "Trunk / Spine", "Neck",
]

# v5.2 model knows only these 15 exercises. UI can show all 27 (per
# exercises.json), but if a user picks one outside this list the model's
# exercise-embedding input falls back to index 0 (squat). The other heads
# (quality / joint_err / boundary) still produce SOMETHING but the
# interpretation is unreliable for unsupported exercises.
V5_2_EXERCISE_TO_IDX = {
    "squat": 0, "deadlift": 1, "pushup": 2, "diamond_pushup": 3,
    "barbell_row": 4, "barbell_dead_row": 5,
    "dumbbell_overhead_shoulder_press": 6,
    "dumbbell_biceps_curls": 7, "dumbbell_hammer_curls": 8,
    "side_lateral_raise": 9, "dumbbell_reverse_lunge": 10,
    "burpees": 11, "clean_and_press": 12, "mule_kick": 13,
    "standing_ab_twists": 14,
}

# v6 (QEVD-trained) uses different class names — pluralised, lowercase, often
# with spaces. Map FitNova UI exercise IDs to QEVD class names. Anything not
# in this table falls through to the model's '__other__' bucket. The QEVD
# class set comes from qevd_exercise_map.json loaded by FormAnalyzer.
#
# Coverage: only the UI exercises that have a clean equivalent in QEVD's
# top-24. The rest fall back to '__other__' until v7 trains a wider taxonomy.
V6_FITNOVA_TO_QEVD = {
    "squat":            "squats",
    "pushup":           "pushups",
    "diamond_pushup":   "yoga pushup",     # closest QEVD class for narrow grip
    "burpees":          "inchworm",        # rough match — burpees not in QEVD top-24
    # Other FitNova exercises (deadlift, dumbbell curls, lateral raises, etc.)
    # have no QEVD equivalent; they map to '__other__'.
}


class FormSession:
    """
    Stateful per-connection session.
    Thread-safety: each WebSocket has its own instance; no shared state.
    """

    def __init__(
        self,
        analyzer: FormAnalyzer,
        exercise_hint: Optional[str] = None,
        selected_exercise: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
    ):
        self.analyzer       = analyzer
        self.selected_exercise = selected_exercise or exercise_hint
        self.exercise_hint  = self.selected_exercise  # legacy alias
        # Lazy OpenAI: if no client supplied AND no key in env, run without it.
        # `_generate_llm_feedback` already handles the None case by returning
        # the deterministic `_fallback_feedback`.
        if openai_client is not None:
            self.openai_client = openai_client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=api_key) if api_key else None

        # Reset MediaPipe VIDEO-mode state so each session starts with ts=0
        # and no forward-fill cache from a previous session.
        self.analyzer.reset_video_state(fps=30.0)

        # Sliding window buffers
        self._angle_buf: deque = deque(maxlen=WINDOW_SIZE)
        self._joint_buf: deque = deque(maxlen=WINDOW_SIZE)
        # v5.2 also needs (15, 4) canonical pose with visibility
        self._pose_buf: deque = deque(maxlen=WINDOW_SIZE)

        # v5.2 exercise-id mapping: name → integer for the model's
        # exercise-embedding input. Falls back to 0 (squat) if the user-
        # selected exercise isn't in v5.2's 15-class vocabulary.
        self._exercise_idx = self._resolve_exercise_idx(
            self.selected_exercise,
            model_version=getattr(analyzer, "_model_version", "v5.2"),
            qevd_exercise_map=getattr(analyzer, "_exercise_labels", None),
        )

        # Geometric form validator instantiation removed (Day 0.5).
        # See plans/handoff-2026-05-11-md-this-handoff-is-sleepy-sparrow.md §6.

        # Frame counters
        self._frame_count    = 0
        self._last_inference = 0

        # Rep tracking
        self._rep_count        = 0
        self._last_boundary_at = -MIN_REP_GAP_FRAMES
        self._prev_boundary_prob = 0.0
        self._recent_exercises: list[str] = []  # rolling window (size 3) for exercise stability check

        # v5.2 fallback rep-counter: tracks the smoothed-quality stream as a
        # state machine (climbing → trough → climbing → ...). Each completed
        # trough between two peaks counts as one rep. Robust to the v5.2
        # saturated-boundary-head failure mode.
        self._q_state              = "climbing"   # "climbing" | "descending"
        self._q_running_peak       = 0.0          # most recent peak value
        self._q_running_trough     = 1.0          # most recent trough value
        self._last_q_rep_at        = -MIN_REP_GAP_FRAMES

        # Per-rep accumulators (reset when new rep detected)
        self._rep_quality_buf:    List[float]       = []
        self._rep_joint_err_buf:  List[List[float]] = []

        # Session history
        self._all_reps:   List[dict] = []  # {rep_idx, quality, joint_errors}
        self._start_time  = time.time()

        # Latest inference result (returned on every frame even between inferences).
        # Values stored here are already EMA-smoothed (see `_update_inference_state`).
        self._latest_inference: dict = {
            "exercise":            self.selected_exercise or "detecting…",
            "exercise_confidence": 0.0,
            "quality":             0.5,
            "rep_count":           0,
            "joint_errors_frame":  [0.0] * 10,
        }
        # EMA state — raw floats held separately so we can seed them on the first
        # inference (instead of pulling the UI toward the initial 0.5 default).
        self._ema_quality:      Optional[float]       = None
        self._ema_joint_errors: Optional[List[float]] = None

        # Mismatch detection
        self._mismatch_detector = ExerciseMismatchDetector(
            selected_exercise=self.selected_exercise,
            model_dir=analyzer.model_dir,
        )
        self._pending_warning: Optional[dict] = None

        # Resolve the display label + relevant joint indices for the selection.
        # When the user has pre-selected an exercise, these drive the payload
        # and summary; the classifier head is treated as an internal signal for
        # the sanity detector only — it is never shown to the user.
        self._selected_display: Optional[str] = None
        self._selected_relevant_indices: Optional[List[int]] = None
        if self.selected_exercise:
            try:
                meta = _get_exercise_meta(self.selected_exercise)
                self._selected_display = meta.get("display_name", self.selected_exercise)
                self._selected_relevant_indices = _relevant_joint_indices(self.selected_exercise)
                # Seed _latest_inference so the very first frame emitted to the
                # client shows the user's selection — not "detecting…".
                self._latest_inference["exercise"]            = self._selected_display
                self._latest_inference["exercise_confidence"] = 1.0
            except KeyError:
                logger.warning(
                    "selected_exercise=%r is not in exercises.json; "
                    "falling back to classifier label.", self.selected_exercise
                )

    # ── Frame processing ──────────────────────────────────────────────────────

    def add_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> dict:
        """
        Process one camera frame.

        Returns a dict to send back to Flutter:
        {
          "type":               "frame_result",
          "timestamp_ms":       int,
          "landmarks":          [[x,y,z] × 15] or null,
          "joint_errors":       [float × 10],   current frame error probabilities
          "quality_score":      float,
          "rep_count":          int,
          "exercise_detected":  str,
          "confidence":         float,
          "status":             "ok" | "no_pose" | ...
        }
        """
        result = self.analyzer.process_frame(jpeg_bytes)

        if result["status"] not in ("ok", "pose_interpolated"):
            out = {
                "type":              "frame_result",
                "timestamp_ms":      timestamp_ms,
                "landmarks":         None,
                "joint_errors":      [0.0] * 10,
                "quality_score":     self._latest_inference["quality"],
                "rep_count":         self._rep_count,
                "exercise_detected": self._latest_inference["exercise"],
                "confidence":        0.0,
                "status":            result["status"],
            }
            if self._pending_warning is not None:
                out["mismatch_warning"] = self._pending_warning
                self._pending_warning = None
            return out

        # Buffer the normalised features
        angles_norm = np.array(result["angles_norm"], dtype=np.float32)  # (22,)
        self._angle_buf.append(angles_norm)
        # v4 mode uses flattened (T, 45) joints; v5.2 uses (T, 15, 4) pose with
        # visibility. We buffer both so the analyzer dispatch can pick whichever
        # is needed at inference time.
        jnt_key = "joints_norm" if "joints_norm" in result else "landmarks"
        lm = np.array(result[jnt_key], dtype=np.float32).flatten()       # (45,)
        self._joint_buf.append(lm)
        if "pose_canon" in result:
            self._pose_buf.append(np.array(result["pose_canon"], dtype=np.float32))  # (15, 4)

        # Geometric per-frame update removed (Day 0.5). Pure ML signal only.

        self._frame_count += 1

        # Run inference when window is full and interval reached
        if (len(self._angle_buf) == WINDOW_SIZE and
                self._frame_count - self._last_inference >= INFERENCE_INTERVAL):

            ang_win = np.stack(list(self._angle_buf), axis=0)   # (64, 22)
            jnt_win = np.stack(list(self._joint_buf), axis=0)   # (64, 45)
            pose_win = (
                np.stack(list(self._pose_buf), axis=0)          # (64, 15, 4)
                if len(self._pose_buf) == WINDOW_SIZE else None
            )
            inf = self.analyzer.predict_window(
                angles_window=ang_win,
                joints_window=jnt_win,
                pose_window=pose_win,
                exercise_idx=self._exercise_idx,
            )
            self._last_inference = self._frame_count
            self._update_inference_state(inf)

        # ── Pure ML signal — geometric layer fully removed (Day 0.5) ───────
        # Per user direction 2026-05-11 end of Day 2: rip out every
        # rule-based contribution from the runtime so v6.1 can be tested
        # purely. The geometric rule evaluators misfire both ways on real
        # video, AND the rep state machine is no longer trusted as a
        # fallback. If v6.1 ships and a rep counter is needed, it ships
        # as a separately-trained module — not a rule.
        # See plans/handoff-2026-05-11-md-this-handoff-is-sleepy-sparrow.md §6.
        ml_joint_errors = self._latest_inference["joint_errors_frame"]
        ml_quality      = float(self._latest_inference["quality"])

        # Pure ML signal payload (Day 0.5). Removed from the wire format vs
        # the rule-era schema:
        #   active_flags, quality_label, phase, geometric_ready, rep_count.
        # Flutter's FormFrameResult.fromJson defaults all of those to empty/
        # null/false, so the existing client tolerates the missing keys.
        # quality_score is still emitted (it's the v6 ML head's output) but
        # the v6.1 UI rework on Days 8-10 will drop it from the rendered UI.
        out = {
            "type":              "frame_result",
            "timestamp_ms":      timestamp_ms,
            "landmarks":         result["landmarks"],
            "joint_errors":      list(ml_joint_errors),
            "quality_score":     round(ml_quality, 3),
            "exercise_detected": self._latest_inference["exercise"],
            "confidence":        round(self._latest_inference["exercise_confidence"], 3),
            "status":            "ok",
        }
        if self._pending_warning is not None:
            out["mismatch_warning"] = self._pending_warning
            self._pending_warning = None
        return out

    def _update_inference_state(self, inf: dict) -> None:
        """Update latest inference results and detect new rep boundaries.

        Per-frame `quality` and `joint_errors` are EMA-smoothed before being
        stored (and hence before they are forwarded to Flutter on the next
        frame). This prevents the UI from flickering as the sliding window
        crosses rep boundaries.
        """
        # Exercise gating for rep-boundary detection.
        # When the user has pre-selected an exercise, the classifier is NOT
        # consulted for the gate — the selection is authoritative. Otherwise
        # we require the classifier mode-label over the last 5 windows to
        # agree and confidence to clear EXERCISE_CONF_THRESHOLD.
        if self._selected_display is not None:
            exercise_stable    = True
            exercise_confident = True
        else:
            self._recent_exercises.append(inf["exercise"])
            if len(self._recent_exercises) > 5:
                self._recent_exercises.pop(0)
            if len(self._recent_exercises) == 5:
                mode_label = max(set(self._recent_exercises), key=self._recent_exercises.count)
                mode_count = self._recent_exercises.count(mode_label)
                exercise_stable = (mode_count >= 4 and mode_label == inf["exercise"])
            else:
                exercise_stable = False
            exercise_confident = (
                float(inf.get("exercise_confidence", 1.0)) >= EXERCISE_CONF_THRESHOLD
            )

        # NOTE: the legacy boundary-rising-edge rep counter has been removed.
        # It only fires once per session on v5.2 because the boundary head
        # saturates near 1.0 — the rising edge happens at the FIRST inference
        # and never resets. The replacement (quality cycle + time-based
        # fallback) lives below, AFTER the EMA update so it operates on
        # smoothed values.
        boundary_probs = inf["boundary"]
        max_prob = float(max(boundary_probs))
        self._prev_boundary_prob = max_prob

        # ── EMA smoothing of quality + joint errors ──────────────────────────
        raw_quality = float(inf["quality"])
        raw_joint   = list(inf["joint_errors"][WINDOW_SIZE // 2])  # 10-vector

        if self._ema_quality is None:
            self._ema_quality = raw_quality
        else:
            self._ema_quality = (
                EMA_ALPHA_QUALITY * raw_quality
                + (1.0 - EMA_ALPHA_QUALITY) * self._ema_quality
            )

        if self._ema_joint_errors is None:
            self._ema_joint_errors = list(raw_joint)
        else:
            self._ema_joint_errors = [
                EMA_ALPHA_JOINT_ERR * r + (1.0 - EMA_ALPHA_JOINT_ERR) * s
                for r, s in zip(raw_joint, self._ema_joint_errors)
            ]

        # Accumulate per-rep stats on the SMOOTHED signal — keeps summary
        # figures consistent with what the user saw in real time.
        # IMPORTANT: must be appended BEFORE the rep-counter logic below
        # (so the snapshot it flushes contains the current frame too).
        self._rep_quality_buf.append(self._ema_quality)
        self._rep_joint_err_buf.append(list(self._ema_joint_errors))

        # ── v5.2 rep counter ────────────────────────────────────────────────
        # For v5.2 we disable the quality-cycle detector (ENABLE_QUALITY_CYCLE_DETECTOR=False)
        # because its compressed quality output produces noisy false-positive
        # cycles that double-count when stacked with the time fallback.
        # Pure time-based counting at TIME_BASED_REP_INTERVAL is the most
        # predictable behaviour for the model's current failure mode.
        if exercise_stable and exercise_confident:
            rep_fired = False

            # PRIMARY: quality-cycle detection (disabled for v5.2)
            if ENABLE_QUALITY_CYCLE_DETECTOR:
                q = self._ema_quality
                if self._q_state == "climbing":
                    if q >= self._q_running_peak:
                        self._q_running_peak = q
                    elif (self._q_running_peak - q) >= QUALITY_REP_PEAK_DELTA:
                        self._q_state = "descending"
                        self._q_running_trough = q
                else:  # descending
                    if q <= self._q_running_trough:
                        self._q_running_trough = q
                    elif (q - self._q_running_trough) >= QUALITY_REP_PEAK_DELTA:
                        if (self._q_running_trough <= QUALITY_TROUGH_FLOOR
                                and self._frame_count - self._last_q_rep_at > MIN_REP_GAP_FRAMES):
                            rep_fired = True
                        self._q_state = "climbing"
                        self._q_running_peak = q

            # SECONDARY: time-based fallback
            if (not rep_fired
                    and self._frame_count - self._last_q_rep_at >= TIME_BASED_REP_INTERVAL):
                rep_fired = True

            if rep_fired:
                self._rep_count += 1
                self._last_q_rep_at = self._frame_count
                # Save completed-rep snapshot
                if self._rep_quality_buf:
                    rep_quality = float(np.mean(self._rep_quality_buf))
                    rep_joint   = (np.array(self._rep_joint_err_buf).mean(axis=0).tolist()
                                   if self._rep_joint_err_buf else [0.0] * 10)
                    self._all_reps.append({
                        "rep_idx":     self._rep_count,
                        "quality":     rep_quality,
                        "joint_errors": rep_joint,
                    })
                    self._rep_quality_buf.clear()
                    self._rep_joint_err_buf.clear()

        # Update latest state. The user-visible `exercise` is the selected
        # display name when a selection exists (classifier is silent), with
        # confidence pinned to 1.0 because the user is authoritative. The raw
        # classifier output is kept under `predicted_exercise_raw` for the
        # sanity detector and the session summary.
        if self._selected_display is not None:
            self._latest_inference.update({
                "exercise":               self._selected_display,
                "exercise_confidence":    1.0,
                "predicted_exercise_raw": inf["exercise"],
                "quality":                self._ema_quality,
                "joint_errors_frame":     list(self._ema_joint_errors),
            })
        else:
            self._latest_inference.update({
                "exercise":               inf["exercise"],
                "exercise_confidence":    inf["exercise_confidence"],
                "predicted_exercise_raw": inf["exercise"],
                "quality":                self._ema_quality,
                "joint_errors_frame":     list(self._ema_joint_errors),
            })

        # Check for mismatch warning (always uses the RAW classifier label —
        # that is exactly the signal the detector is built around).
        warning = self._mismatch_detector.update(
            predicted_name=inf["exercise"],
            confidence=float(inf.get("exercise_confidence", 0.0)),
        )
        if warning is not None:
            self._pending_warning = warning

    # ── Session summary ───────────────────────────────────────────────────────

    def end_session(self) -> dict:
        """
        Finalise the session, compute aggregate stats, call GPT-4o-mini.

        Returns the full session summary dict.
        """
        duration_s = time.time() - self._start_time

        # Flush last partial rep
        if self._rep_quality_buf:
            rep_quality = float(np.mean(self._rep_quality_buf))
            rep_joint   = np.array(self._rep_joint_err_buf).mean(axis=0).tolist() \
                          if self._rep_joint_err_buf else [0.0] * 10
            self._rep_count += 1
            self._all_reps.append({
                "rep_idx":     self._rep_count,
                "quality":     rep_quality,
                "joint_errors": rep_joint,
            })

        total_reps     = max(self._rep_count, len(self._all_reps))
        per_rep_scores = [round(r["quality"], 3) for r in self._all_reps]
        avg_quality    = float(np.mean(per_rep_scores)) if per_rep_scores else 0.5

        # Find most common errors across all reps
        common_errors = self._compute_common_errors()

        # Quality trend
        trend = "stable"
        if len(per_rep_scores) >= 3:
            slope = np.polyfit(range(len(per_rep_scores)), per_rep_scores, 1)[0]
            trend = "improving" if slope > 0.02 else "declining" if slope < -0.02 else "stable"

        # LLM feedback
        # User's pre-selection is authoritative for aggregation + coaching;
        # the raw classifier label is kept for diagnostics only.
        exercise = (
            self._selected_display
            or self.selected_exercise
            or self._latest_inference.get("exercise", "Unknown exercise")
        )
        predicted_exercise = self._latest_inference.get(
            "predicted_exercise_raw",
            self._latest_inference.get("exercise", "unknown"),
        )
        llm_feedback = self._generate_llm_feedback(
            exercise, total_reps, per_rep_scores, common_errors, trend
        )

        # Per-rep detailed timeline (always included so the UI can show a
        # breakdown). Each entry carries the rep's mean quality, dominant
        # joint-error groups (max value across frames), and a coarse
        # frame-range so the Flutter UI can scrub the user's video.
        per_rep_details = []
        for rep in self._all_reps:
            errs = rep["joint_errors"]
            # Top-3 joint groups by magnitude (with their values)
            named = [
                (JOINT_GROUP_NAMES[i], float(errs[i]))
                for i in range(min(len(JOINT_GROUP_NAMES), len(errs)))
            ]
            named.sort(key=lambda x: -x[1])
            top_errors = [
                {"name": n, "value": round(v, 3)}
                for n, v in named[:3] if v > 0.30
            ]
            per_rep_details.append({
                "rep_idx":   rep["rep_idx"],
                "quality":   round(float(rep["quality"]), 3),
                "joint_errors": [round(float(e), 3) for e in errs],
                "top_errors": top_errors,
            })

        return {
            "type":            "session_summary",
            "exercise":        exercise,
            "predicted_exercise": predicted_exercise,
            "exercise_mismatch": (
                bool(self.selected_exercise)
                and predicted_exercise != "unknown"
                and predicted_exercise != self.selected_exercise
            ),
            "total_reps":      total_reps,
            "duration_seconds": round(duration_s),
            "average_quality": round(avg_quality, 3),
            "per_rep_scores":  per_rep_scores,
            "per_rep_details": per_rep_details,    # NEW: detailed per-rep timeline
            "common_errors":   common_errors,
            "quality_trend":   trend,
            "llm_feedback":    llm_feedback,
            "model_version":   getattr(self.analyzer, "_model_version", "v4"),
        }

    def _compute_common_errors(self) -> dict:
        """
        Aggregate per-joint errors across all reps.

        When the user has pre-selected an exercise, we restrict the output
        to the joints that matter for that exercise (``key_errors_detected``
        in exercises.json). For squats that means knees / hips / back — the
        classifier head and unrelated joints are never surfaced.

        The threshold is relaxed to 0.35 from the original 0.40 because the
        joint-error head is softly calibrated on MoCap and tends to
        under-fire on real phone video.

        Returns ``{joint_name: n_reps_with_error}`` sorted descending.
        """
        if not self._all_reps:
            return {}

        totals = np.zeros(10, dtype=float)
        for rep in self._all_reps:
            errs = np.array(rep["joint_errors"])
            totals += (errs > 0.35).astype(float)

        if self._selected_relevant_indices is not None:
            keep = set(self._selected_relevant_indices)
        else:
            keep = set(range(10))

        result = {}
        for i, name in enumerate(JOINT_GROUP_NAMES):
            if i not in keep:
                continue
            count = int(totals[i])
            if count > 0:
                result[name] = count

        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def _generate_llm_feedback(
        self,
        exercise: str,
        total_reps: int,
        per_rep_scores: list,
        common_errors: dict,
        trend: str,
    ) -> str:
        """Call GPT-4o-mini for personalised coaching feedback."""
        if not self.openai_client:
            return self._fallback_feedback(exercise, per_rep_scores, common_errors)

        avg = round(float(np.mean(per_rep_scores)) if per_rep_scores else 0.5, 2)
        errors_str = (
            ", ".join(f"{k} ({v} reps)" for k, v in list(common_errors.items())[:3])
            if common_errors else "no significant form errors detected"
        )

        prompt = (
            f"You are FitNova's certified strength and conditioning coach.\n\n"
            f"Exercise session summary:\n"
            f"- Exercise: {exercise.replace('_', ' ').title()}\n"
            f"- Total reps completed: {total_reps}\n"
            f"- Average form quality: {avg*100:.0f}%\n"
            f"- Form trend: {trend}\n"
            f"- Most common form issues: {errors_str}\n\n"
            f"Write a 3-4 sentence coaching response that:\n"
            f"1. Acknowledges what was done well\n"
            f"2. Clearly identifies the top 1-2 form issues with specific corrections\n"
            f"3. Gives one single actionable cue to focus on next set\n"
            f"Be encouraging, specific, and use plain language. "
            f"No bullet points — write in natural flowing sentences."
        )

        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are FitNova's form coaching assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM feedback failed: {e}")
            return self._fallback_feedback(exercise, per_rep_scores, common_errors)

    @staticmethod
    def _resolve_exercise_idx(
        name: Optional[str],
        model_version: str = "v5.2",
        qevd_exercise_map: Optional[dict] = None,
    ) -> int:
        """Map user-selected exercise name to the model's exercise-id input.

        Routing depends on the loaded model version:

        * **v6** (QEVD): map FitNova UI name → QEVD class name (via
          ``V6_FITNOVA_TO_QEVD``) → idx (via ``qevd_exercise_map``). Falls
          back to QEVD's '__other__' index when there's no equivalent. The
          per-class action head still produces something, but the
          conditioning then doesn't bias the joint_err head toward the
          relevant body parts.
        * **v5.2** (Fit3D): direct lookup in ``V5_2_EXERCISE_TO_IDX``.
          Falls back to 0 (squat).
        * **v4** / unknown: returns 0.

        Logs a clear warning whenever the fallback fires.
        """
        if not name:
            return 0

        # ── v6 routing ───────────────────────────────────────────────────────
        if model_version == "v6":
            if not qevd_exercise_map:
                logger.warning(
                    "v6 _resolve_exercise_idx: qevd_exercise_map is empty — "
                    "falling back to 0",
                )
                return 0
            qevd_name = V6_FITNOVA_TO_QEVD.get(name)
            if qevd_name and qevd_name in qevd_exercise_map:
                return int(qevd_exercise_map[qevd_name])
            other_idx = qevd_exercise_map.get("__other__", 0)
            logger.warning(
                "v6: selected_exercise=%r has no QEVD equivalent "
                "(FitNova→QEVD table covers: %s); falling back to "
                "'__other__' idx %d. Form-quality on this exercise is "
                "unreliable.",
                name, sorted(V6_FITNOVA_TO_QEVD.keys()), other_idx,
            )
            return int(other_idx)

        # ── v5.2 routing (legacy) ────────────────────────────────────────────
        if model_version == "v5.2":
            if name in V5_2_EXERCISE_TO_IDX:
                return V5_2_EXERCISE_TO_IDX[name]
            logger.warning(
                "v5.2: selected_exercise=%r is not in the vocabulary "
                "(supported: %s); falling back to exercise_idx=0 (squat). "
                "Form-quality scoring on this exercise is unreliable.",
                name, sorted(V5_2_EXERCISE_TO_IDX.keys()),
            )
            return 0

        # ── v4 / unknown ─────────────────────────────────────────────────────
        return 0

    @staticmethod
    def _fallback_feedback(
        exercise: str,
        per_rep_scores: list,
        common_errors: dict,
    ) -> str:
        """Deterministic fallback if LLM is unavailable."""
        avg = round(float(np.mean(per_rep_scores)) * 100 if per_rep_scores else 50)
        name = exercise.replace("_", " ").title()
        top_error = next(iter(common_errors), None)
        if top_error:
            return (
                f"Good effort on the {name}! Your average form score was {avg}%. "
                f"Pay attention to your {top_error.lower()} — focus on controlled movement "
                f"and maintaining proper alignment throughout each rep."
            )
        return (
            f"Great work on the {name}! Your average form score was {avg}%. "
            f"Keep focusing on controlled movement and full range of motion."
        )
