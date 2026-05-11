"""Unit tests for backend.training.evaluation.reality_check_v6.

Run from repo root:
    pytest backend/training/evaluation/test_reality_check_v6.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.evaluation.reality_check_v6 import (
    BOUNDARY_SPIKE_THRESHOLD,
    JOINT_GROUP_NAMES_V6,
    KNEE_HIP_BAD_MIN,
    KNEE_HIP_GOOD_MAX,
    KNEE_HIP_INDICES,
    QUALITY_GAP_MIN,
    _evaluate_gates,
    _resolve_squat_indices,
    _sliding_windows,
)


# ─────────────────────────────────────────────────────────────────────────────
# Joint-group constants
# ─────────────────────────────────────────────────────────────────────────────


def test_knee_hip_indices_resolve_to_correct_group_names():
    expected_names = {"L_knee", "R_knee", "L_hip", "R_hip"}
    actual_names = {JOINT_GROUP_NAMES_V6[i] for i in KNEE_HIP_INDICES}
    assert actual_names == expected_names


def test_joint_group_names_v6_match_form_session():
    """Master plan section II.10 says these MUST match form_session.py
    so the live Flutter UI receives the right channels in the right order."""
    from backend.services.form_session import JOINT_GROUP_NAMES as live_names
    # Live names use display strings like "Left Elbow"; we use canonical
    # snake-case. Ensure ORDER and MEANING match.
    expected_pairs = [
        ("L_elbow",    "Left Elbow"),
        ("R_elbow",    "Right Elbow"),
        ("L_shoulder", "Left Shoulder"),
        ("R_shoulder", "Right Shoulder"),
        ("L_knee",     "Left Knee"),
        ("R_knee",     "Right Knee"),
        ("L_hip",      "Left Hip"),
        ("R_hip",      "Right Hip"),
        ("Trunk",      "Trunk / Spine"),
        ("Neck",       "Neck"),
    ]
    for i, (canonical, live) in enumerate(expected_pairs):
        assert JOINT_GROUP_NAMES_V6[i] == canonical, (i, canonical)
        assert live_names[i] == live, (i, live)


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_squat_indices
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_squat_indices_finds_squat():
    ex_map = {"squat": 3, "pushup": 0, "plank": 1, "__other__": 24}
    assert _resolve_squat_indices(ex_map) == [3]


def test_resolve_squat_indices_finds_variants():
    ex_map = {
        "squats": 0, "squat punches": 1, "wall squat": 2,
        "pushup": 3, "__other__": 24,
    }
    out = sorted(_resolve_squat_indices(ex_map))
    assert out == [0, 1, 2]


def test_resolve_squat_indices_excludes_other():
    ex_map = {"pushup": 0, "__other__": 24}
    assert _resolve_squat_indices(ex_map) == []


# ─────────────────────────────────────────────────────────────────────────────
# _sliding_windows
# ─────────────────────────────────────────────────────────────────────────────


def test_sliding_windows_basic():
    arr = np.arange(100).reshape(100, 1)
    windows = list(_sliding_windows(arr, target_frames=64, stride=8))
    # First window starts at 0; subsequent at 8, 16, 24, ..., 32
    starts = [s for s, _ in windows]
    assert starts[0] == 0
    # Last window MUST end at the last frame (so we never miss the tail)
    last_start, last_w = windows[-1]
    assert last_start + last_w.shape[0] == 100


def test_sliding_windows_short_clip_is_padded():
    arr = np.arange(40).reshape(40, 1)
    windows = list(_sliding_windows(arr, target_frames=64, stride=8))
    # Only one window, padded to 64
    assert len(windows) == 1
    start, w = windows[0]
    assert w.shape[0] == 64
    # First 40 values match; rest are zero
    assert np.array_equal(w[:40, 0], np.arange(40))
    assert np.all(w[40:, 0] == 0)


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_gates
# ─────────────────────────────────────────────────────────────────────────────


def _fake_clip_info(quality, knee_hip_max, action_idx,
                     boundary_seq=None, n_frames=360):
    """Construct a minimal clip info dict matching what _infer_clip returns."""
    if boundary_seq is None:
        boundary_seq = [0.0] * n_frames
    return {
        "n_windows":         5,
        "duration_s":        12.0,
        "quality_mean":      float(quality),
        "quality_std":       0.05,
        "action_top1_idx":   int(action_idx),
        "action_top1_conf":  0.9,
        "boundary_max":      float(max(boundary_seq) if boundary_seq else 0.0),
        "boundary_seq":      list(boundary_seq),
        "joint_err_max":     [0.0] * 10,
        "joint_err_mean":    [0.0] * 10,
        "knee_hip_max":      float(knee_hip_max),
        "rep_count_mean":    1.0,
    }


def test_evaluate_gates_all_pass():
    """Healthy v6 result: squat on all 4, gap = 0.20, knee_hip 0.4 vs 0.05,
    boundary = 4 spikes in 12 s."""
    boundary_good = [0.0] * 30 + [0.7] * 5 + [0.0] * 60 + [0.7] * 5 + \
                    [0.0] * 60 + [0.7] * 5 + [0.0] * 60 + [0.7] * 5 + [0.0] * 130
    results = {
        "Good_Squats":  _fake_clip_info(0.85, 0.05, 7,
                                         boundary_seq=boundary_good),
        "Good_Squats2": _fake_clip_info(0.83, 0.04, 7,
                                         boundary_seq=boundary_good),
        "Bad_Squats":   _fake_clip_info(0.65, 0.45, 7),
        "Bad_Squats2":  _fake_clip_info(0.60, 0.40, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    assert gates["all_squat"]
    assert gates["quality_gap_pass"]
    assert gates["knee_hip_pass"]
    assert gates["boundary_pass"]
    assert gates["overall_pass"]


def test_evaluate_gates_quality_gap_too_small():
    results = {
        "Good_Squats":  _fake_clip_info(0.50, 0.05, 7),
        "Good_Squats2": _fake_clip_info(0.52, 0.04, 7),
        "Bad_Squats":   _fake_clip_info(0.48, 0.45, 7),
        "Bad_Squats2":  _fake_clip_info(0.45, 0.40, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    # Gap is ~0.045, less than 0.15
    assert not gates["quality_gap_pass"]
    assert not gates["overall_pass"]


def test_evaluate_gates_action_mismatch():
    """Top-1 action is something OTHER than squat — should fail gate 1."""
    results = {
        "Good_Squats":  _fake_clip_info(0.85, 0.05, 11),  # not 'squat'
        "Good_Squats2": _fake_clip_info(0.83, 0.04, 7),
        "Bad_Squats":   _fake_clip_info(0.55, 0.45, 7),
        "Bad_Squats2":  _fake_clip_info(0.50, 0.40, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    assert not gates["all_squat"]
    assert not gates["overall_pass"]


def test_evaluate_gates_knee_hip_signal_collapsed():
    """knee_hip stays low on bad — joint-err head is dead. Should fail."""
    results = {
        "Good_Squats":  _fake_clip_info(0.85, 0.05, 7),
        "Good_Squats2": _fake_clip_info(0.83, 0.04, 7),
        "Bad_Squats":   _fake_clip_info(0.55, 0.05, 7),  # bad but no signal
        "Bad_Squats2":  _fake_clip_info(0.50, 0.05, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    assert not gates["knee_hip_pass"]
    assert not gates["overall_pass"]


def test_evaluate_gates_knee_hip_good_too_loud():
    """knee_hip max in Good clips is HIGH (false positives) — should fail."""
    results = {
        "Good_Squats":  _fake_clip_info(0.85, 0.30, 7),  # >0.10, fail
        "Good_Squats2": _fake_clip_info(0.83, 0.20, 7),
        "Bad_Squats":   _fake_clip_info(0.55, 0.40, 7),
        "Bad_Squats2":  _fake_clip_info(0.50, 0.45, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    assert not gates["knee_hip_pass"]


def test_evaluate_gates_boundary_silent():
    """Boundary head produces no spikes in Good_Squats — fail gate 4."""
    quiet_seq = [0.0] * 360
    results = {
        "Good_Squats":  _fake_clip_info(0.85, 0.05, 7, boundary_seq=quiet_seq),
        "Good_Squats2": _fake_clip_info(0.83, 0.04, 7),
        "Bad_Squats":   _fake_clip_info(0.55, 0.45, 7),
        "Bad_Squats2":  _fake_clip_info(0.50, 0.40, 7),
    }
    gates = _evaluate_gates(results, squat_indices=[7])
    assert not gates["boundary_pass"]


def test_evaluate_gates_thresholds_are_master_plan_locked():
    assert QUALITY_GAP_MIN          == 0.15
    assert KNEE_HIP_BAD_MIN         == 0.20
    assert KNEE_HIP_GOOD_MAX        == 0.10
    assert BOUNDARY_SPIKE_THRESHOLD == 0.5
