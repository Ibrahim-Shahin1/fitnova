"""Tests for v6.1 D9 reality-check gate logic."""
from __future__ import annotations

import numpy as np
import pytest

from backend.training.evaluation.reality_check_v6_1 import (
    BAD_TOPK_PRECISION_MIN,
    GOOD_TOPK_PRECISION_MIN,
    SQUAT_DEFECT_VARIATIONS,
    SQUAT_GOOD_VARIATIONS,
    _evaluate_gates,
    _sliding_windows,
)


# ---------------------------------------------------------------------------
# Variation partition sanity
# ---------------------------------------------------------------------------


def test_good_and_defect_sets_disjoint():
    assert SQUAT_GOOD_VARIATIONS.isdisjoint(SQUAT_DEFECT_VARIATIONS)


def test_good_set_real_qevd_strings():
    # All four entries must exist verbatim in the v6.1 class space
    # (otherwise the metric is unreachable on real predictions).
    from backend.training.preprocessing.qevd_class_space import (
        get_default_class_space,
    )
    cs = get_default_class_space()
    squat_indices = cs.class_indices_for_prefix("squats")
    squat_variations = {cs.decode(i)[1] for i in squat_indices}
    missing = SQUAT_GOOD_VARIATIONS - squat_variations
    assert not missing, (
        f"SQUAT_GOOD_VARIATIONS contains strings not in the v6.1 class space: "
        f"{missing}. Update the set or the class space, not both."
    )


def test_defect_set_real_qevd_strings():
    from backend.training.preprocessing.qevd_class_space import (
        get_default_class_space,
    )
    cs = get_default_class_space()
    squat_indices = cs.class_indices_for_prefix("squats")
    squat_variations = {cs.decode(i)[1] for i in squat_indices}
    missing = SQUAT_DEFECT_VARIATIONS - squat_variations
    assert not missing, (
        f"SQUAT_DEFECT_VARIATIONS contains strings not in the v6.1 class space: "
        f"{missing}. Update the set or the class space, not both."
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _mk_clip(kind: str, good_hit: bool, defect_hit: bool) -> dict:
    return {
        "kind": kind,
        "good_present_in_topk":   good_hit,
        "defect_present_in_topk": defect_hit,
    }


def test_gate_all_good_clips_hit_passes_good():
    results = {
        "Good_1": _mk_clip("good", True,  False),
        "Good_2": _mk_clip("good", True,  False),
    }
    gates = _evaluate_gates(results)
    assert gates["good_topk_precision"] == 1.0
    assert gates["good_topk_precision_pass"] is True


def test_gate_all_bad_clips_defect_hit_passes_bad():
    results = {
        "Bad_1": _mk_clip("bad", False, True),
        "Bad_2": _mk_clip("bad", False, True),
    }
    gates = _evaluate_gates(results)
    assert gates["bad_topk_precision"] == 1.0
    assert gates["bad_topk_precision_pass"] is True


def test_gate_zero_hits_fails():
    results = {
        "Good_1": _mk_clip("good", False, True),  # predicted defect on good
        "Good_2": _mk_clip("good", False, True),
        "Bad_1":  _mk_clip("bad",  True,  False), # predicted good on bad
        "Bad_2":  _mk_clip("bad",  True,  False),
    }
    gates = _evaluate_gates(results)
    assert gates["good_topk_precision"] == 0.0
    assert gates["bad_topk_precision"] == 0.0
    assert gates["overall_pass"] is False


def test_gate_threshold_below_passes():
    # 50% precision is below the 0.6 minimum -> should fail.
    results = {
        "Good_1": _mk_clip("good", True,  False),
        "Good_2": _mk_clip("good", False, False),
        "Bad_1":  _mk_clip("bad", False, True),
        "Bad_2":  _mk_clip("bad", False, True),
    }
    gates = _evaluate_gates(results)
    assert gates["good_topk_precision"] == 0.5
    assert gates["good_topk_precision_pass"] is False
    assert gates["overall_pass"] is False


def test_gate_threshold_exactly_at_minimum_passes():
    # Exact-at-threshold (>=) should pass.
    assert GOOD_TOPK_PRECISION_MIN == 0.6
    results = {
        "Good_1": _mk_clip("good", True, False),
        "Good_2": _mk_clip("good", True, False),
        "Good_3": _mk_clip("good", True, False),
        "Good_4": _mk_clip("good", False, False),
        "Good_5": _mk_clip("good", False, False),
        "Bad_1":  _mk_clip("bad", False, True),
        "Bad_2":  _mk_clip("bad", False, True),
        "Bad_3":  _mk_clip("bad", False, True),
        "Bad_4":  _mk_clip("bad", False, False),
        "Bad_5":  _mk_clip("bad", False, False),
    }
    gates = _evaluate_gates(results)
    assert gates["good_topk_precision"] == 0.6
    assert gates["bad_topk_precision"]  == 0.6
    assert gates["good_topk_precision_pass"] is True
    assert gates["bad_topk_precision_pass"]  is True
    assert gates["overall_pass"] is True


def test_gate_skips_clips_with_errors():
    # error clips should not be counted in the denominator
    results = {
        "Good_1": _mk_clip("good", True, False),
        "Good_2": {"kind": "good", "error": "file missing"},
        "Bad_1":  _mk_clip("bad",  False, True),
    }
    gates = _evaluate_gates(results)
    assert gates["n_good_clips"] == 1
    assert gates["n_bad_clips"]  == 1
    assert gates["good_topk_precision"] == 1.0
    assert gates["bad_topk_precision"]  == 1.0


def test_gate_empty_results():
    gates = _evaluate_gates({})
    assert gates["n_good_clips"] == 0
    assert gates["n_bad_clips"]  == 0
    assert gates["overall_pass"] is False


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------


def test_sliding_windows_exact_multiple():
    arr = np.arange(128).reshape(128, 1)
    windows = list(_sliding_windows(arr, target_frames=64, stride=64))
    assert len(windows) == 2
    starts = [s for s, _ in windows]
    assert starts == [0, 64]
    for _, w in windows:
        assert w.shape == (64, 1)


def test_sliding_windows_non_multiple_adds_tail():
    # 100 frames, window=64, stride=8 → starts at 0, 8, 16, 24, 32; last_start = 36
    arr = np.arange(100).reshape(100, 1)
    windows = list(_sliding_windows(arr, target_frames=64, stride=8))
    starts = [s for s, _ in windows]
    assert starts[0] == 0
    assert max(starts) == 36     # tail window added to cover frames 36-99


def test_sliding_windows_too_short_zero_pads():
    arr = np.ones((30, 4), dtype=np.float32)
    windows = list(_sliding_windows(arr, target_frames=64, stride=8))
    assert len(windows) == 1
    start, w = windows[0]
    assert start == 0
    assert w.shape == (64, 4)
    assert np.all(w[:30] == 1.0)
    assert np.all(w[30:] == 0.0)
