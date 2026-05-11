"""Unit tests for the geometric form validator.

Covers:
  - Rep state machine: counts completion-only, rejects half-reps,
    no double-count on full cycles, calibration logic.
  - Form rule evaluator: trunk lean, knee valgus, depth.
  - GeometricFormValidator integration: full lifecycle on synthetic squat
    trajectories.

Run from repo root:
    pytest backend/services/test_form_geometry.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.services.form_geometry import (
    GeometricFormValidator, PHASE_ASCENDING, PHASE_DESCENDING, PHASE_INIT,
    PHASE_STANDING, RepStateMachine, evaluate_rule, hip_minus_knee_y,
    knee_to_ankle_width_ratio, trunk_angle_from_vertical,
)
from backend.config.exercise_rules.squat import SQUAT_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build synthetic 15-joint landmark frames
# ─────────────────────────────────────────────────────────────────────────────


def make_standing_lms(hip_y: float = 0.50,
                       trunk_lean_deg: float = 0.0,
                       knee_x_offset: float = 0.0,
                       hip_minus_knee: float = -0.20) -> np.ndarray:
    """Return a (15, 3) array representing a person standing.

    Args:
        hip_y: pelvis Y (image coords, 0=top, 1=bottom). Default 0.50.
        trunk_lean_deg: forward lean of the trunk vector from vertical.
        knee_x_offset: 0 = neutral knee width = ankle width.
                       negative = knees caving. positive = knees out.
        hip_minus_knee: default -0.20 = pelvis 20% above knee level (typical
                       for standing). Set to +0.05 for "below parallel" squat.
    """
    lms = np.zeros((15, 3), dtype=np.float32)
    # Default body layout — symmetric about x=0.5
    # Pelvis at (0.5, hip_y)
    pelvis_x = 0.5
    pelvis_y = hip_y

    # Place hips offset symmetrically so pelvis is their midpoint
    hip_offset = 0.07
    lms[6] = [pelvis_x - hip_offset, pelvis_y, 0]   # L_hip
    lms[7] = [pelvis_x + hip_offset, pelvis_y, 0]   # R_hip
    lms[12] = [pelvis_x, pelvis_y, 0]                # pelvis

    # Knees: pelvis_y - hip_minus_knee = knee_y (so larger hip_minus_knee
    # means deeper squat, knees relatively higher in image — wait that's
    # wrong. hip_minus_knee = pelvis_y - knee_y. So knee_y = pelvis_y - hpk.
    # If hip_minus_knee = +0.05, knee_y = pelvis_y - 0.05 → knees are 5% above
    # pelvis in image = pelvis 5% below knees = below parallel squat.
    # If hip_minus_knee = -0.20, knee_y = pelvis_y + 0.20 → knees 20% below
    # pelvis = standing.
    knee_y = pelvis_y - hip_minus_knee
    lms[8]  = [pelvis_x - hip_offset + knee_x_offset, knee_y, 0]  # L_knee
    lms[9]  = [pelvis_x + hip_offset - knee_x_offset, knee_y, 0]  # R_knee

    # Ankles: 20% below knees in image
    ankle_y = knee_y + 0.20
    lms[10] = [pelvis_x - hip_offset, ankle_y, 0]   # L_ankle
    lms[11] = [pelvis_x + hip_offset, ankle_y, 0]   # R_ankle

    # Trunk: place NECK directly along the pelvis→up vector rotated by lean.
    # This guarantees pelvis→neck has exactly the requested angle from
    # vertical (which is what trunk_angle_from_vertical computes).
    trunk_len = 0.23   # pelvis→neck distance
    angle_rad = math.radians(trunk_lean_deg)
    neck_x = pelvis_x + math.sin(angle_rad) * trunk_len
    neck_y = pelvis_y + -math.cos(angle_rad) * trunk_len
    lms[13] = [neck_x, neck_y, 0]

    # Shoulders: place at 80% of the way from pelvis to neck, offset
    # symmetrically. (Geometry of these doesn't affect the rules we test.)
    shoulder_mid_x = pelvis_x + 0.8 * (neck_x - pelvis_x)
    shoulder_mid_y = pelvis_y + 0.8 * (neck_y - pelvis_y)
    shoulder_offset = 0.10
    lms[0] = [shoulder_mid_x - shoulder_offset, shoulder_mid_y, 0]   # L_shoulder
    lms[1] = [shoulder_mid_x + shoulder_offset, shoulder_mid_y, 0]   # R_shoulder

    # Spine_mid: midpoint(pelvis, neck)
    lms[14] = [(pelvis_x + lms[13, 0]) * 0.5,
               (pelvis_y + lms[13, 1]) * 0.5, 0]

    return lms


# ─────────────────────────────────────────────────────────────────────────────
# Compute primitives
# ─────────────────────────────────────────────────────────────────────────────


def test_trunk_angle_perfectly_upright():
    lms = make_standing_lms(trunk_lean_deg=0.0)
    angle = trunk_angle_from_vertical(lms)
    assert angle < 1.0   # < 1° tolerance for synthetic data


def test_trunk_angle_30_degree_lean():
    lms = make_standing_lms(trunk_lean_deg=30.0)
    angle = trunk_angle_from_vertical(lms)
    assert math.isclose(angle, 30.0, abs_tol=1.0)


def test_trunk_angle_70_degree_lean():
    lms = make_standing_lms(trunk_lean_deg=70.0)
    angle = trunk_angle_from_vertical(lms)
    assert math.isclose(angle, 70.0, abs_tol=1.0)


def test_knee_to_ankle_ratio_neutral():
    lms = make_standing_lms(knee_x_offset=0.0)
    # Knee width = hip width = 0.14, ankle width = hip width = 0.14
    # Ratio should be exactly 1.0
    r = knee_to_ankle_width_ratio(lms)
    assert math.isclose(r, 1.0, abs_tol=0.01)


def test_knee_to_ankle_ratio_caving():
    """knee_x_offset > 0 brings knees inward → lower ratio."""
    lms = make_standing_lms(knee_x_offset=0.04)
    r = knee_to_ankle_width_ratio(lms)
    # knee_dist = 0.14 - 2*0.04 = 0.06, ankle_dist = 0.14 → ratio = 0.43
    assert r < 0.5


def test_knee_to_ankle_ratio_degenerate_returns_nan():
    lms = make_standing_lms()
    # Force ankle_dist near zero (side view simulation)
    lms[10, 0] = 0.50
    lms[11, 0] = 0.502
    r = knee_to_ankle_width_ratio(lms)
    assert math.isnan(r)


def test_hip_minus_knee_standing_negative():
    lms = make_standing_lms(hip_minus_knee=-0.20)
    assert math.isclose(hip_minus_knee_y(lms), -0.20, abs_tol=0.01)


def test_hip_minus_knee_below_parallel_positive():
    lms = make_standing_lms(hip_minus_knee=+0.05)
    assert math.isclose(hip_minus_knee_y(lms), +0.05, abs_tol=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Rule evaluator
# ─────────────────────────────────────────────────────────────────────────────


def test_back_lean_below_safe_threshold_inactive():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "back_excessive_lean")
    lms = make_standing_lms(trunk_lean_deg=30.0)   # well below 60° safe_max
    is_active, severity = evaluate_rule(rule, lms, at_bottom=False)
    assert not is_active
    assert severity == 0.0


def test_back_lean_above_red_flag_capped_at_one():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "back_excessive_lean")
    lms = make_standing_lms(trunk_lean_deg=90.0)   # past 85° red_flag
    is_active, severity = evaluate_rule(rule, lms, at_bottom=False)
    assert is_active
    assert severity == 1.0


def test_back_lean_between_thresholds_partial():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "back_excessive_lean")
    lms = make_standing_lms(trunk_lean_deg=72.0)   # between 60 and 85
    is_active, severity = evaluate_rule(rule, lms, at_bottom=False)
    assert is_active
    assert 0.0 < severity < 1.0


def test_knees_caving_inactive_when_neutral():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "knees_caving")
    lms = make_standing_lms(knee_x_offset=0.0)   # ratio = 1.0
    is_active, _ = evaluate_rule(rule, lms, at_bottom=False)
    assert not is_active


def test_knees_caving_inactive_at_mild_inward_track():
    """User feedback 2026-05-11: mild inward knee tracking is fine — a coach
    only flags when knees clearly cross inside the foot line. With the loosened
    thresholds (safe_min=0.55), this test verifies mild caving is silent."""
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "knees_caving")
    # knee_x_offset=0.03 -> knee_dist=0.08, ratio=0.57 (just above safe_min=0.55)
    lms = make_standing_lms(knee_x_offset=0.03)
    is_active, _ = evaluate_rule(rule, lms, at_bottom=False)
    assert not is_active


def test_knees_caving_fires_when_severely_caved():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "knees_caving")
    # knee_x_offset=0.06 -> knee_dist=0.02, ratio=0.14 (well below red_flag 0.25)
    lms = make_standing_lms(knee_x_offset=0.06)
    is_active, severity = evaluate_rule(rule, lms, at_bottom=False)
    assert is_active
    assert severity > 0.5


def test_insufficient_depth_only_fires_at_bottom():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "insufficient_depth")
    # Even with shallow geometry, must NOT fire if not at bottom
    lms = make_standing_lms(hip_minus_knee=-0.15)   # hip 15% above knees
    is_active, _ = evaluate_rule(rule, lms, at_bottom=False)
    assert not is_active

    # At bottom: SHOULD fire because hip is 15% above knees (well past the
    # -0.06 shallow_margin threshold)
    is_active, severity = evaluate_rule(rule, lms, at_bottom=True)
    assert is_active
    assert severity > 0.5


def test_deep_squat_passes_depth_check():
    rule = next(r for r in SQUAT_RULES["form_checks"]
                if r["name"] == "insufficient_depth")
    lms = make_standing_lms(hip_minus_knee=+0.05)   # below parallel
    is_active, _ = evaluate_rule(rule, lms, at_bottom=True)
    assert not is_active


# ─────────────────────────────────────────────────────────────────────────────
# Rep state machine
# ─────────────────────────────────────────────────────────────────────────────


def _simulate_squat_trajectory(start_y: float = 0.50,
                                bottom_y: float = 0.65,
                                n_descent: int = 30,
                                n_ascent: int = 30) -> list[float]:
    """Generate a smooth y-trajectory: start_y → bottom_y → start_y."""
    descent = np.linspace(start_y, bottom_y, n_descent).tolist()
    ascent  = np.linspace(bottom_y, start_y, n_ascent).tolist()
    return descent + ascent


def test_rep_state_machine_starts_in_init():
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    assert sm.phase == PHASE_INIT
    assert sm.count == 0


def test_rep_state_machine_calibrates_then_standing():
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    for _ in range(20):
        lms = make_standing_lms(hip_y=0.50)
        sm.update(lms)
    assert sm.phase == PHASE_STANDING
    assert sm.standing_baseline == pytest.approx(0.50, abs=1e-3)


def test_rep_state_machine_counts_one_full_squat():
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    # Calibrate at standing
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))
    assert sm.count == 0

    # One squat: 0.40 → 0.60 → 0.40
    trajectory = _simulate_squat_trajectory(0.40, 0.60, 30, 30)
    for y in trajectory:
        sm.update(make_standing_lms(hip_y=y))
    assert sm.count == 1, f"expected 1 rep, got {sm.count}"
    assert sm.phase == PHASE_STANDING


def test_rep_state_machine_counts_three_full_squats_no_double():
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))

    for _ in range(3):
        for y in _simulate_squat_trajectory(0.40, 0.60, 25, 25):
            sm.update(make_standing_lms(hip_y=y))

    assert sm.count == 3
    assert sm.phase == PHASE_STANDING


def test_rep_state_machine_rejects_shallow_bobbing():
    """Person bobs up and down by tiny amounts but never squats — no reps."""
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))

    # Tiny oscillation: 0.40 → 0.43 → 0.40 (only 0.03 displacement)
    # descent_threshold is 0.07, so this never even enters DESCENDING
    for _ in range(5):
        for y in _simulate_squat_trajectory(0.40, 0.43, 10, 10):
            sm.update(make_standing_lms(hip_y=y))
    assert sm.count == 0


def test_rep_state_machine_rejects_half_rep_below_threshold():
    """Person descends past descent_threshold but not past min_rep_displacement."""
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))

    # Goes down by 0.075 (past descent_threshold=0.07) but min_rep is 0.08
    for y in _simulate_squat_trajectory(0.40, 0.475, 20, 20):
        sm.update(make_standing_lms(hip_y=y))
    # Phase progressed through DESCENDING/ASCENDING but rep didn't count
    assert sm.count == 0


def test_rep_state_machine_does_not_double_count():
    """A full cycle must produce exactly 1 rep, not 2 (down + up)."""
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))

    trajectory = _simulate_squat_trajectory(0.40, 0.62, 30, 30)
    counts_after_each_frame = []
    for y in trajectory:
        sm.update(make_standing_lms(hip_y=y))
        counts_after_each_frame.append(sm.count)

    # Count should reach 1 at most once during the trajectory (at the very end)
    assert sm.count == 1
    # Verify count was monotonically non-decreasing (no decrement)
    assert counts_after_each_frame == sorted(counts_after_each_frame)


def test_rep_state_machine_phase_transitions_correctly():
    """Walk through a single squat and verify phase changes."""
    sm = RepStateMachine(SQUAT_RULES["rep_tracker"])
    # Calibrate
    for _ in range(20):
        sm.update(make_standing_lms(hip_y=0.40))
    assert sm.phase == PHASE_STANDING

    # Going down past descent threshold
    for y in np.linspace(0.40, 0.55, 20):
        sm.update(make_standing_lms(hip_y=y))
    assert sm.phase == PHASE_DESCENDING

    # Now go down more, then start coming back up
    for y in np.linspace(0.55, 0.62, 10):
        sm.update(make_standing_lms(hip_y=y))
    # Reverse direction
    for y in np.linspace(0.62, 0.55, 10):
        sm.update(make_standing_lms(hip_y=y))
    assert sm.phase == PHASE_ASCENDING

    # Return to standing
    for y in np.linspace(0.55, 0.40, 20):
        sm.update(make_standing_lms(hip_y=y))
    assert sm.phase == PHASE_STANDING
    assert sm.count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Top-level GeometricFormValidator integration
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_disabled_for_unknown_exercise():
    v = GeometricFormValidator(selected_exercise="bicep_curl")
    assert not v.enabled
    snap = v.snapshot()
    assert snap["enabled"] is False
    assert snap["rep_count_geometric"] == 0
    # Update should be a no-op
    v.update(make_standing_lms())
    assert v.snapshot()["rep_count_geometric"] == 0


def test_validator_squat_lifecycle_clean_form():
    v = GeometricFormValidator(selected_exercise="squat")
    assert v.enabled

    # Calibrate at standing, with clean form
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40, trunk_lean_deg=10.0))

    # Two clean reps
    for _ in range(2):
        for y in _simulate_squat_trajectory(0.40, 0.62, 30, 30):
            # During descent, lean increases naturally (max ~40°)
            lean = 10.0 + (y - 0.40) / (0.62 - 0.40) * 30.0
            v.update(make_standing_lms(hip_y=y, trunk_lean_deg=lean))

    snap = v.snapshot()
    assert snap["rep_count_geometric"] == 2
    # Active flags should be empty now (we're back to standing)
    assert snap["active_flags"] == []


def test_validator_fires_back_lean_red_flag():
    v = GeometricFormValidator(selected_exercise="squat")
    # Calibrate
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40, trunk_lean_deg=0.0))

    # Excessive forward lean — past 85° red_flag threshold
    v.update(make_standing_lms(hip_y=0.50, trunk_lean_deg=90.0))
    snap = v.snapshot()
    active_names = [f["name"] for f in snap["active_flags"]]
    assert "back_excessive_lean" in active_names
    assert snap["joint_errors_geometric"][8] > 0.5  # G_TRUNK
    assert snap["quality_geometric"] < 0.8
    # The active_flag should carry a categorical label, not just the raw name
    lean_flag = next(f for f in snap["active_flags"] if f["name"] == "back_excessive_lean")
    assert "Lean" in lean_flag["label"]


def test_validator_fires_knee_valgus():
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40, knee_x_offset=0.0))

    # Severe caving: knee_x_offset=0.06 brings knees very close together
    # knee_dist = 0.14 - 2*0.06 = 0.02, ratio = 0.02/0.14 = 0.14 (very low)
    v.update(make_standing_lms(hip_y=0.55, knee_x_offset=0.06))
    snap = v.snapshot()
    active_names = [f["name"] for f in snap["active_flags"]]
    assert "knees_caving" in active_names
    # Both knees should light up
    assert snap["joint_errors_geometric"][4] > 0.0   # L_knee
    assert snap["joint_errors_geometric"][5] > 0.0   # R_knee
    # Categorical label should be a "Caving" variant
    cave_flag = next(f for f in snap["active_flags"] if f["name"] == "knees_caving")
    assert "Caving" in cave_flag["label"]


def test_validator_quality_clamps_at_zero_with_multiple_flags():
    """Multiple simultaneous severe red flags shouldn't drive quality negative."""
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40))

    # Severe lean + severe knee cave at once
    v.update(make_standing_lms(hip_y=0.55, trunk_lean_deg=90.0,
                                knee_x_offset=0.06))
    snap = v.snapshot()
    assert snap["quality_geometric"] >= 0.0
    assert snap["quality_geometric"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# New: severity labels + categorical quality + relevant_joint_groups filtering
# ─────────────────────────────────────────────────────────────────────────────


def test_snapshot_includes_quality_label():
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40))
    snap = v.snapshot()
    assert "quality_label" in snap
    # Clean form → Excellent
    assert snap["quality_label"] == "Excellent Form"


def test_snapshot_quality_label_changes_with_quality():
    """Drop quality with a serious red flag, label should reflect it."""
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40))
    # 90° lean = severity 1.0 with penalty 0.30 → quality 0.70 → "Good Form"
    v.update(make_standing_lms(hip_y=0.50, trunk_lean_deg=90.0))
    snap = v.snapshot()
    assert snap["quality_label"] in {"Good Form", "Fair Form"}


def test_snapshot_relevant_joint_groups_for_squat():
    """For squats, only knees/hips/trunk are relevant; shoulders excluded."""
    v = GeometricFormValidator(selected_exercise="squat")
    snap = v.snapshot()
    relevant = set(snap["relevant_joint_groups"])
    # 4=L_knee, 5=R_knee, 6=L_hip, 7=R_hip, 8=Trunk
    assert {4, 5, 6, 7, 8}.issubset(relevant)
    # 0/1=elbows, 2/3=shoulders, 9=neck must NOT be in the relevant set
    assert {0, 1, 2, 3, 9}.isdisjoint(relevant)


def test_active_flags_carry_categorical_labels():
    """Each active flag must include a name, label, and severity."""
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40))
    # Trigger a knee-caving flag with mid severity
    v.update(make_standing_lms(hip_y=0.55, knee_x_offset=0.05))
    snap = v.snapshot()
    assert len(snap["active_flags"]) >= 1
    f = snap["active_flags"][0]
    assert "name" in f
    assert "label" in f
    assert "severity" in f
    assert 0.0 <= f["severity"] <= 1.0
    assert isinstance(f["label"], str) and len(f["label"]) > 0


def test_severity_label_mild_for_borderline_caving():
    """Just past the safe_min threshold → 'Small' label, not 'Dangerous'."""
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40, knee_x_offset=0.0))
    # knee_x_offset=0.04 → knee_dist=0.06, ratio=0.43, just past safe_min 0.55
    v.update(make_standing_lms(hip_y=0.55, knee_x_offset=0.04))
    snap = v.snapshot()
    caving = next((f for f in snap["active_flags"] if f["name"] == "knees_caving"),
                  None)
    assert caving is not None
    # Should be the mildest tier
    assert caving["label"].startswith("Small")


def test_severity_label_dangerous_for_extreme_caving():
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40, knee_x_offset=0.0))
    # Knees pressed all the way together
    v.update(make_standing_lms(hip_y=0.55, knee_x_offset=0.065))
    snap = v.snapshot()
    caving = next((f for f in snap["active_flags"] if f["name"] == "knees_caving"),
                  None)
    assert caving is not None
    assert caving["label"] == "Dangerous Knee Caving"


def test_validator_rep_history_grows_on_completion():
    v = GeometricFormValidator(selected_exercise="squat")
    for _ in range(20):
        v.update(make_standing_lms(hip_y=0.40))

    assert len(v.rep_history) == 0
    for y in _simulate_squat_trajectory(0.40, 0.62, 30, 30):
        v.update(make_standing_lms(hip_y=y))
    assert len(v.rep_history) == 1
    assert v.rep_history[0]["rep_idx"] == 1


def test_validator_handles_missing_landmarks():
    """None / empty landmarks must not crash."""
    v = GeometricFormValidator(selected_exercise="squat")
    v.update(None)
    v.update([])
    v.update([[0.0, 0.0, 0.0]])   # only 1 joint
    snap = v.snapshot()
    assert snap["rep_count_geometric"] == 0
    assert snap["enabled"] is True  # rules loaded; just no signal


def test_validator_qevd_squats_alias_works():
    """v6's QEVD class name is 'squats' (plural). Should also work."""
    v = GeometricFormValidator(selected_exercise="squats")
    assert v.enabled


def test_validator_handles_uppercase_exercise_name():
    v = GeometricFormValidator(selected_exercise="Squat")
    assert v.enabled
