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
from backend.config.exercises import (
    get_metadata as _get_exercise_meta,
    relevant_joint_indices as _relevant_joint_indices,
)

logger = logging.getLogger(__name__)

WINDOW_SIZE         = 64    # frames
INFERENCE_INTERVAL  = 10    # run MT-TCN every N frames (~1s at 10fps)
BOUNDARY_THRESHOLD  = 0.55  # boundary probability to count as a rep boundary
MIN_REP_GAP_FRAMES  = 20    # minimum frames between two rep boundaries
EXERCISE_CONF_THRESHOLD = 0.40  # min exercise confidence to count a boundary

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
        self.openai_client  = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Reset MediaPipe VIDEO-mode state so each session starts with ts=0
        # and no forward-fill cache from a previous session.
        self.analyzer.reset_video_state(fps=30.0)

        # Sliding window buffers
        self._angle_buf: deque = deque(maxlen=WINDOW_SIZE)
        self._joint_buf: deque = deque(maxlen=WINDOW_SIZE)

        # Frame counters
        self._frame_count    = 0
        self._last_inference = 0

        # Rep tracking
        self._rep_count        = 0
        self._last_boundary_at = -MIN_REP_GAP_FRAMES
        self._prev_boundary_prob = 0.0
        self._recent_exercises: list[str] = []  # rolling window (size 3) for exercise stability check

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
        # Use skeleton-normalised world coords for joints input (matches training).
        # Fall back to image landmarks for legacy callers that don't have joints_norm.
        jnt_key = "joints_norm" if "joints_norm" in result else "landmarks"
        lm = np.array(result[jnt_key], dtype=np.float32).flatten()       # (45,)
        self._joint_buf.append(lm)

        self._frame_count += 1

        # Run MT-TCN when window is full and interval reached
        if (len(self._angle_buf) == WINDOW_SIZE and
                self._frame_count - self._last_inference >= INFERENCE_INTERVAL):

            ang_win = np.stack(list(self._angle_buf), axis=0)   # (64, 22)
            jnt_win = np.stack(list(self._joint_buf), axis=0)   # (64, 45)
            inf = self.analyzer.predict_window(ang_win, jnt_win)
            self._last_inference = self._frame_count
            self._update_inference_state(inf)

        # Current-frame joint errors from latest inference
        joint_errors_frame = self._latest_inference["joint_errors_frame"]

        out = {
            "type":              "frame_result",
            "timestamp_ms":      timestamp_ms,
            "landmarks":         result["landmarks"],
            "joint_errors":      joint_errors_frame,
            "quality_score":     round(self._latest_inference["quality"], 3),
            "rep_count":         self._rep_count,
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

        # Detect rep boundary: rising edge above threshold with minimum gap
        boundary_probs = inf["boundary"]
        max_prob = float(max(boundary_probs))  # peak anywhere in window
        max_idx = int(np.argmax(np.asarray(boundary_probs)))

        if (max_prob > BOUNDARY_THRESHOLD and
                self._prev_boundary_prob <= BOUNDARY_THRESHOLD and
                self._frame_count - self._last_boundary_at > MIN_REP_GAP_FRAMES and
                exercise_stable and exercise_confident):

            self._rep_count += 1
            # Anchor the boundary timestamp to where the peak actually occurred in absolute frames
            self._last_boundary_at = self._frame_count - (WINDOW_SIZE - 1 - max_idx)

            # Save completed rep data
            if self._rep_quality_buf:
                rep_quality = float(np.mean(self._rep_quality_buf))
                rep_joint   = np.array(self._rep_joint_err_buf).mean(axis=0).tolist() \
                              if self._rep_joint_err_buf else [0.0] * 10
                self._all_reps.append({
                    "rep_idx":     self._rep_count,
                    "quality":     rep_quality,
                    "joint_errors": rep_joint,
                })
                self._rep_quality_buf.clear()
                self._rep_joint_err_buf.clear()

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
        self._rep_quality_buf.append(self._ema_quality)
        self._rep_joint_err_buf.append(list(self._ema_joint_errors))

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
            "common_errors":   common_errors,
            "quality_trend":   trend,
            "llm_feedback":    llm_feedback,
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
