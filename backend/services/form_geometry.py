"""Rule-based geometric form validator.

Runs ALONGSIDE the trained ML model (form_analyzer.predict_window). For each
frame's MediaPipe landmarks, computes:

  rep_count_geometric   — int, advanced via state machine on hip-Y trajectory
  phase                 — "INIT" | "STANDING" | "DESCENDING" | "ASCENDING"
  active_flags          — list[str], names of currently-violated form rules
  joint_errors_geometric — (10,) float, per-joint-group severity in [0, 1]
  quality_geometric     — float in [0, 1], 1.0 = perfect form

The state machine (per master plan §II.4.4 + user spec):
  STANDING  ── hip_y > baseline + descent_threshold ──→ DESCENDING
  DESCENDING ── hip_y starts increasing again       ──→ ASCENDING
  ASCENDING  ── hip_y < baseline + return_threshold ──→ STANDING + (rep+1 IF
                                                          bottom_displacement
                                                          >= min_rep_disp)

Trigger policy: completion-only. A rep counts only when ASCENDING returns to
STANDING after a real BOTTOM was reached (filters out half-reps, bobbing).

Coordinate system: MediaPipe image-normalised (x in [0,1] left→right,
y in [0,1] top→bottom). DESCENDING in real life = y INCREASING in image.

This module has NO ML dependency. Pure NumPy.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from backend.config.exercise_rules import get_rules

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical 15-joint indices (kept here too so this file is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

L_SHOULDER, R_SHOULDER = 0, 1
L_HIP, R_HIP           = 6, 7
L_KNEE, R_KNEE         = 8, 9
L_ANKLE, R_ANKLE       = 10, 11
PELVIS, NECK           = 12, 13

N_JOINT_GROUPS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Phase enum (str-typed for JSON-friendly Flutter payload)
# ─────────────────────────────────────────────────────────────────────────────

PHASE_INIT       = "INIT"
PHASE_STANDING   = "STANDING"
PHASE_DESCENDING = "DESCENDING"
PHASE_ASCENDING  = "ASCENDING"


# ─────────────────────────────────────────────────────────────────────────────
# Compute primitives — pure functions over (15, 3) landmark arrays
# ─────────────────────────────────────────────────────────────────────────────


def trunk_angle_from_vertical(lms: np.ndarray) -> float:
    """Angle (degrees) between the trunk vector (pelvis→neck) and vertical UP.

    0° = perfectly upright torso.  90° = trunk parallel to ground.

    In image coords, vertical UP is (0, -1) (smaller y = higher in space).
    """
    pelvis = lms[PELVIS, :2]
    neck   = lms[NECK,   :2]
    trunk_vec = neck - pelvis           # points from pelvis up to neck
    norm = float(np.linalg.norm(trunk_vec))
    if norm < 1e-6:
        return 0.0
    # Cosine with vertical-UP. trunk_vec.y is negative when neck is above
    # pelvis (the typical case), and dot with (0, -1) is -trunk_vec.y, positive.
    cos_theta = float(-trunk_vec[1] / norm)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def knee_to_ankle_width_ratio(lms: np.ndarray) -> float:
    """Ratio of knee-knee horizontal spread to ankle-ankle horizontal spread.

    Used to detect knee valgus (knees caving in). Healthy ≈ 1.0; knees
    caving = ratio drops below 0.7.

    Returns NaN if either width is degenerate (< 1% of frame).
    """
    knee_dist  = float(abs(lms[L_KNEE, 0]  - lms[R_KNEE, 0]))
    ankle_dist = float(abs(lms[L_ANKLE, 0] - lms[R_ANKLE, 0]))
    if ankle_dist < 0.01:
        return float("nan")
    return knee_dist / ankle_dist


def hip_minus_knee_y(lms: np.ndarray) -> float:
    """How far below knee level the pelvis sits, in image-y units.

    Positive = pelvis is BELOW knees (deep squat, ass-to-grass).
    Zero     = pelvis at knee level (parallel squat).
    Negative = pelvis ABOVE knees (shallow / quarter squat).
    """
    pelvis_y = float(lms[PELVIS, 1])
    knee_y   = float((lms[L_KNEE, 1] + lms[R_KNEE, 1]) * 0.5)
    return pelvis_y - knee_y


# ─────────────────────────────────────────────────────────────────────────────
# Rep state machine
# ─────────────────────────────────────────────────────────────────────────────


class RepStateMachine:
    """Tracks STANDING ↔ DESCENDING ↔ ASCENDING phases over a single joint's
    Y trajectory. Counts a rep only when ASCENDING returns to STANDING after
    a real BOTTOM was reached (completion-trigger policy).

    All thresholds taken from the rep_tracker config block of the exercise's
    rule file.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.phase = PHASE_INIT
        self.count = 0
        self._calib_buf: deque = deque(maxlen=int(cfg["calibration_frames"]))
        self._standing_baseline_y: Optional[float] = None
        self._max_y_in_descent: Optional[float] = None
        self._bottom_y: Optional[float] = None
        # Per-rep flag accumulator — populated during DESCENDING/BOTTOM and
        # frozen at completion. Used to attach rule violations to specific reps.
        self._rep_in_progress_flags: list[str] = []

    @property
    def standing_baseline(self) -> Optional[float]:
        return self._standing_baseline_y

    @property
    def at_bottom(self) -> bool:
        """True when the current phase is the BOTTOM moment of a descent
        (just transitioned to ASCENDING). Used to gate 'fire_only_at: BOTTOM'
        rules."""
        return (self.phase == PHASE_ASCENDING
                and self._bottom_y is not None
                and self._max_y_in_descent is not None
                and abs(self._max_y_in_descent - self._bottom_y) < 1e-6)

    def update(self, lms: np.ndarray) -> None:
        cur_y = float(lms[self.cfg["key_joint_idx"], 1])

        # ── INIT: collect calibration frames ────────────────────────────────
        if self.phase == PHASE_INIT:
            self._calib_buf.append(cur_y)
            if len(self._calib_buf) >= self.cfg["calibration_frames"]:
                # Use the SMALLEST observed y as standing baseline (person at
                # their tallest = highest in space = smallest image y).
                self._standing_baseline_y = float(min(self._calib_buf))
                self.phase = PHASE_STANDING
                logger.debug(
                    "RepStateMachine: calibrated standing baseline = %.3f",
                    self._standing_baseline_y,
                )
            return

        # ── STANDING → DESCENDING ───────────────────────────────────────────
        if self.phase == PHASE_STANDING:
            # Allow baseline to drift down if person stands taller (rare but
            # happens after warmup).
            if cur_y < self._standing_baseline_y:
                self._standing_baseline_y = cur_y
            if cur_y > self._standing_baseline_y + self.cfg["descent_threshold"]:
                self.phase = PHASE_DESCENDING
                self._max_y_in_descent = cur_y
                self._bottom_y = None
                self._rep_in_progress_flags = []
            return

        # ── DESCENDING: track running max, watch for reversal ───────────────
        if self.phase == PHASE_DESCENDING:
            if cur_y > self._max_y_in_descent:
                self._max_y_in_descent = cur_y
            else:
                # cur_y has started decreasing — possibly the bottom.
                if (self._max_y_in_descent - cur_y) > self.cfg["velocity_reversal_threshold"]:
                    self.phase = PHASE_ASCENDING
                    self._bottom_y = self._max_y_in_descent
            return

        # ── ASCENDING → STANDING (rep completion check) ─────────────────────
        if self.phase == PHASE_ASCENDING:
            if cur_y < self._standing_baseline_y + self.cfg["return_threshold"]:
                # Returned to standing height. Was this a real rep?
                bottom_displacement = (self._bottom_y or 0.0) - self._standing_baseline_y
                if bottom_displacement >= self.cfg["min_rep_displacement"]:
                    self.count += 1
                    logger.debug(
                        "RepStateMachine: rep %d completed (displacement=%.3f)",
                        self.count, bottom_displacement,
                    )
                self.phase = PHASE_STANDING
                self._bottom_y = None
                self._max_y_in_descent = None
            return


# ─────────────────────────────────────────────────────────────────────────────
# Form-rule evaluator
# ─────────────────────────────────────────────────────────────────────────────


COMPUTE_FUNCTIONS = {
    "trunk_angle_from_vertical":   trunk_angle_from_vertical,
    "knee_to_ankle_width_ratio":   knee_to_ankle_width_ratio,
    "hip_above_knee_at_bottom":    hip_minus_knee_y,
}


# ─────────────────────────────────────────────────────────────────────────────
# Label conversion helpers (used by snapshot)
# ─────────────────────────────────────────────────────────────────────────────


def _severity_to_label(rule: dict, severity: float) -> str:
    """Map a 0-1 severity to a human-readable label using the rule's
    severity_tiers + severity_labels.

    Example for knees_caving with tiers=[0.30, 0.60, 0.85]:
        severity 0.40 -> "Small Knee Caving"   (in [0.30, 0.60))
        severity 0.70 -> "Notable Knee Caving" (in [0.60, 0.85))
        severity 0.95 -> "Dangerous Knee Caving"

    Falls back to the rule's display_name if tiers/labels are missing.
    """
    tiers  = rule.get("severity_tiers")
    labels = rule.get("severity_labels")
    if not tiers or not labels:
        return rule.get("display_name", rule["name"])

    # Pick the highest-tier label whose threshold is <= severity.
    label_idx = 0
    for i, t in enumerate(tiers):
        if severity >= t:
            label_idx = i
    label_idx = min(label_idx, len(labels) - 1)
    return labels[label_idx]


def _quality_to_label(rules: dict, quality: float) -> str:
    """Map a 0-1 quality to a categorical label using rules['quality_labels'].

    rules['quality_labels'] is a list of (threshold, label) tuples sorted
    in descending threshold order. We pick the first label whose threshold
    is <= quality.
    """
    table = rules.get("quality_labels", [(0.0, "Form")])
    for threshold, label in table:
        if quality >= threshold:
            return label
    return table[-1][1] if table else "Form"


def evaluate_rule(rule: dict, lms: np.ndarray, at_bottom: bool) -> tuple[bool, float]:
    """Evaluate one rule against current frame.

    Returns:
        (is_active, severity)  — severity in [0, 1].

    Rules with ``fire_only_at: "BOTTOM"`` evaluate to (False, 0.0) unless
    we're at the bottom of a rep.
    """
    if rule.get("fire_only_at") == "BOTTOM" and not at_bottom:
        return False, 0.0

    fn = COMPUTE_FUNCTIONS.get(rule["compute"])
    if fn is None:
        logger.warning("unknown compute function: %s", rule["compute"])
        return False, 0.0

    val = fn(lms)
    if math.isnan(val):
        return False, 0.0

    th = rule["thresholds"]
    name = rule["name"]

    # Each rule has its own threshold semantics — we hand-code these
    # rather than trying to generalise. Keeps the logic transparent and
    # auditable for the dissertation.
    if name == "back_excessive_lean":
        # val is degrees from vertical
        safe = th["safe_max_deg"]
        red  = th["red_flag_deg"]
        if val < safe:
            return False, 0.0
        # Linear ramp from safe → red, clamped to [0, 1]
        severity = (val - safe) / max(1e-3, red - safe)
        return True, max(0.0, min(1.0, severity))

    if name == "knees_caving":
        # val is knee/ankle ratio. < safe_min means caving.
        safe = th["safe_min"]
        red  = th["red_flag_min"]
        if val >= safe:
            return False, 0.0
        # Side-view filter: require minimum knee-knee separation
        # (already checked? No, val is already a ratio. We need raw separation.)
        # Caller will need to pass that — for now skip the side-view filter
        # at the rule level. Done in side-view detector below.
        severity = (safe - val) / max(1e-3, safe - red)
        return True, max(0.0, min(1.0, severity))

    if name == "insufficient_depth":
        # val is hip_y - knee_y. Negative = hip ABOVE knee = shallow.
        margin = th["shallow_margin"]   # negative number, e.g. -0.04
        if val >= margin:
            return False, 0.0
        # Scale: hip 4% above knee = severity 1.0; at margin = severity 0.
        severity = (margin - val) / max(1e-3, abs(margin))
        return True, max(0.0, min(1.0, severity))

    return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GeometricFormValidator — top-level class consumed by FormSession
# ─────────────────────────────────────────────────────────────────────────────


class GeometricFormValidator:
    """Stateful per-session geometric validator.

    Construct one per FormSession. Call ``update(landmarks)`` every frame the
    backend has a valid pose. Read ``snapshot()`` for the current state.

    Usage (FormSession-side)::

        self.geometry = GeometricFormValidator(selected_exercise="squat")
        ...
        # Each frame:
        if pose_data["status"] == "ok":
            self.geometry.update(pose_data["landmarks"])
        snap = self.geometry.snapshot()
        # snap["rep_count_geometric"], snap["quality_geometric"], etc.
    """

    def __init__(self, selected_exercise: Optional[str]):
        self.selected_exercise = selected_exercise
        self.rules = get_rules(selected_exercise)
        self.rep_machine: Optional[RepStateMachine] = None
        self._last_active_flags: list[dict] = []   # [{name, label, severity}, ...]
        self._last_joint_errors: np.ndarray = np.zeros(N_JOINT_GROUPS, dtype=np.float32)
        self._last_quality: float = 1.0
        self._last_quality_label: str = "Excellent Form"
        # Per-rep history (populated at completion)
        self._rep_history: list[dict] = []
        self._current_rep_active_flags: set[str] = set()
        self._last_rep_count: int = 0

        if self.rules is not None:
            self.rep_machine = RepStateMachine(self.rules["rep_tracker"])
            logger.info(
                "GeometricFormValidator: loaded %s rules (%d form_checks)",
                selected_exercise, len(self.rules.get("form_checks", [])),
            )
        else:
            logger.info(
                "GeometricFormValidator: no rules for exercise=%r — geometric "
                "layer disabled for this session",
                selected_exercise,
            )

    @property
    def enabled(self) -> bool:
        return self.rules is not None and self.rep_machine is not None

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, landmarks) -> None:
        """Process one frame's worth of canonical-15 landmarks.

        Args:
            landmarks: list of 15 [x, y, z] floats, OR np.ndarray (15, 3).
                       MediaPipe image-normalised coordinates.
        """
        if not self.enabled:
            return
        if landmarks is None:
            return

        lms = np.asarray(landmarks, dtype=np.float32)
        if lms.ndim != 2 or lms.shape[0] < 15 or lms.shape[1] < 2:
            return

        # 1. Update rep state machine
        prev_count = self.rep_machine.count
        self.rep_machine.update(lms)
        rep_just_completed = self.rep_machine.count > prev_count

        # 2. Evaluate form rules
        active_flags: list[dict] = []          # rich objects with display info
        active_flag_names: list[str] = []
        joint_errors = np.zeros(N_JOINT_GROUPS, dtype=np.float32)

        for rule in self.rules.get("form_checks", []):
            is_active, severity = evaluate_rule(
                rule, lms, at_bottom=self.rep_machine.at_bottom,
            )
            # Side-view filter for knees_caving: in a side view both ANKLES
            # overlap horizontally (the ratio becomes meaningless). We use
            # ankle separation (not knee separation) as the side-view
            # detector — knees may have collapsed inward in valgus, but if
            # ankles are wide apart we know we're in a front-ish view.
            if rule["name"] == "knees_caving" and is_active:
                ankle_sep = abs(float(lms[L_ANKLE, 0] - lms[R_ANKLE, 0]))
                min_sep = rule.get("min_horizontal_separation", 0.0)
                if ankle_sep < min_sep:
                    is_active = False
                    severity = 0.0

            if is_active and severity > 0.0:
                label = _severity_to_label(rule, severity)
                active_flags.append({
                    "name":     rule["name"],
                    "label":    label,
                    "severity": float(severity),
                })
                active_flag_names.append(rule["name"])
                self._current_rep_active_flags.add(rule["name"])
                for g in rule["joint_groups"]:
                    if 0 <= g < N_JOINT_GROUPS:
                        joint_errors[g] = max(joint_errors[g], severity)

        # 3. Quality
        quality = 1.0
        for flag_name in active_flag_names:
            for rule in self.rules.get("form_checks", []):
                if rule["name"] == flag_name:
                    quality -= rule.get("penalty", 0.1)
                    break
        floor = self.rules.get("quality_floor", 0.0)
        ceil  = self.rules.get("quality_ceil", 1.0)
        quality = max(floor, min(ceil, quality))
        quality_label = _quality_to_label(self.rules, quality)

        # 4. Snapshot persisted state
        self._last_active_flags = active_flags
        self._last_joint_errors = joint_errors
        self._last_quality      = float(quality)
        self._last_quality_label = quality_label

        # 5. On rep completion, archive the rep's accumulated flags
        if rep_just_completed:
            self._rep_history.append({
                "rep_idx":        self.rep_machine.count,
                "active_flags":   sorted(self._current_rep_active_flags),
            })
            self._current_rep_active_flags.clear()
            self._last_rep_count = self.rep_machine.count

    def snapshot(self) -> dict:
        """Return the current geometric state for the session/UI."""
        if not self.enabled:
            return {
                "enabled":                 False,
                "rep_count_geometric":     0,
                "phase":                   PHASE_INIT,
                "active_flags":            [],
                "joint_errors_geometric":  [0.0] * N_JOINT_GROUPS,
                "quality_geometric":       1.0,
                "quality_label":           "Excellent Form",
                "relevant_joint_groups":   list(range(N_JOINT_GROUPS)),  # show all
                "calibrated":              False,
            }

        return {
            "enabled":                 True,
            "rep_count_geometric":     int(self.rep_machine.count),
            "phase":                   self.rep_machine.phase,
            # active_flags is now a list of dicts: [{name, label, severity}]
            "active_flags":            list(self._last_active_flags),
            "joint_errors_geometric":  self._last_joint_errors.tolist(),
            "quality_geometric":       self._last_quality,
            "quality_label":           self._last_quality_label,
            "relevant_joint_groups":   list(self.rules.get(
                "relevant_joint_groups", range(N_JOINT_GROUPS),
            )),
            "calibrated":              self.rep_machine.standing_baseline is not None,
        }

    @property
    def rep_history(self) -> list[dict]:
        """Per-rep flag accumulator (populated at each rep completion).

        Used by FormSession.end_session() to attach rep-specific feedback.
        """
        return list(self._rep_history)
