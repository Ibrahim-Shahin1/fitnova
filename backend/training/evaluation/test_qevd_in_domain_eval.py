"""Unit tests for backend.training.evaluation.qevd_in_domain_eval.

Covers metric helpers + the npz resolver + threshold constants.
Skips the actual model-inference path (needs a trained model + .npz).

Run from repo root:
    pytest backend/training/evaluation/test_qevd_in_domain_eval.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.training.evaluation.qevd_in_domain_eval import (
    ACTION_TOP1_MIN, JOINT_ERR_MAP_MIN, QUALITY_SPEARMAN_MIN,
    _resolve_npz, action_top1_accuracy, joint_err_mean_ap,
    quality_spearman,
)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold constants (master-plan locked)
# ─────────────────────────────────────────────────────────────────────────────


def test_thresholds_match_master_plan():
    assert ACTION_TOP1_MIN      == 0.55
    assert QUALITY_SPEARMAN_MIN == 0.40
    assert JOINT_ERR_MAP_MIN    == 0.30


# ─────────────────────────────────────────────────────────────────────────────
# action_top1_accuracy
# ─────────────────────────────────────────────────────────────────────────────


def test_action_top1_perfect():
    assert action_top1_accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_action_top1_zero():
    assert action_top1_accuracy([0, 1, 2], [3, 4, 5]) == 0.0


def test_action_top1_partial():
    # 2 out of 4 correct
    acc = action_top1_accuracy([0, 1, 2, 3], [0, 5, 2, 6])
    assert math.isclose(acc, 0.5, abs_tol=1e-9)


def test_action_top1_empty_returns_nan():
    assert math.isnan(action_top1_accuracy([], []))


def test_action_top1_length_mismatch_truncates():
    # Should compare only on the overlapping prefix
    acc = action_top1_accuracy([0, 1, 2, 3], [0, 1])
    assert math.isclose(acc, 1.0, abs_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# quality_spearman (delegated from qevd_fitcoach_eval, but local copy)
# ─────────────────────────────────────────────────────────────────────────────


def test_spearman_perfect():
    rho = quality_spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert math.isclose(rho, 1.0, abs_tol=1e-9)


def test_spearman_anti_correlated():
    rho = quality_spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert math.isclose(rho, -1.0, abs_tol=1e-9)


def test_spearman_constant_input_is_nan():
    assert math.isnan(quality_spearman([0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4]))


def test_spearman_too_few_points_is_nan():
    assert math.isnan(quality_spearman([1.0], [2.0]))
    assert math.isnan(quality_spearman([], []))


# ─────────────────────────────────────────────────────────────────────────────
# joint_err_mean_ap
# ─────────────────────────────────────────────────────────────────────────────


def test_joint_err_mean_ap_perfect():
    """Pred scores are perfectly aligned with GT — AP should be 1.0
    on every group with a positive."""
    rng = np.random.default_rng(0)
    n, j = 50, 10
    gt = (rng.random((n, j)) > 0.7).astype(np.float32)
    # Pred = gt (after a tiny noise floor so ties don't dominate)
    pred = gt + rng.random((n, j)) * 0.01
    info = joint_err_mean_ap(pred, gt)
    # Every group with at least one positive should have AP == 1.0
    for ap, n_pos in zip(info["per_group_ap"], info["per_group_n_pos"]):
        if n_pos > 0 and n_pos < n:
            assert math.isclose(ap, 1.0, abs_tol=1e-6)
    assert math.isclose(info["mean_ap"], 1.0, abs_tol=1e-6)


def test_joint_err_mean_ap_random_around_baseline():
    """Pred uncorrelated with GT → mean AP roughly equals base rate."""
    rng = np.random.default_rng(1)
    n, j = 200, 10
    gt = (rng.random((n, j)) > 0.7).astype(np.float32)
    pred = rng.random((n, j))
    info = joint_err_mean_ap(pred, gt)
    # Base rate ~0.3 per group; AP for random scores is approximately the
    # base rate. Allow loose bounds since 200 samples × 10 groups is small.
    assert 0.10 < info["mean_ap"] < 0.55


def test_joint_err_mean_ap_excludes_groups_without_positives():
    """Groups with all-zero GT should appear as None in per_group_ap and
    NOT contribute to the mean."""
    n, j = 20, 4
    gt = np.zeros((n, j), dtype=np.float32)
    gt[:5, 0] = 1.0   # only group 0 has positives
    gt[:3, 1] = 1.0   # group 1 has fewer positives
    # Group 2 + 3 are all-zero (excluded)
    pred = np.tile(np.linspace(0.0, 1.0, n)[:, None], (1, j))
    info = joint_err_mean_ap(pred, gt)
    assert info["per_group_ap"][2] is None
    assert info["per_group_ap"][3] is None
    assert info["per_group_ap"][0] is not None
    assert info["per_group_ap"][1] is not None
    assert info["n_valid_groups"] == 2


def test_joint_err_mean_ap_all_groups_undefined_returns_nan():
    """Every group all-zero → mean is NaN, n_valid_groups=0."""
    n, j = 10, 3
    gt = np.zeros((n, j), dtype=np.float32)
    pred = np.random.default_rng(0).random((n, j))
    info = joint_err_mean_ap(pred, gt)
    assert info["n_valid_groups"] == 0
    assert math.isnan(info["mean_ap"])
    assert all(v is None for v in info["per_group_ap"])


def test_joint_err_mean_ap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        joint_err_mean_ap(
            np.zeros((10, 4), dtype=np.float32),
            np.zeros((10, 5), dtype=np.float32),
        )


def test_joint_err_mean_ap_propagates_group_names():
    n, j = 12, 3
    gt = np.zeros((n, j), dtype=np.float32)
    gt[:4, 0] = 1.0
    gt[:6, 1] = 1.0
    gt[:5, 2] = 1.0
    pred = np.random.default_rng(0).random((n, j))
    info = joint_err_mean_ap(pred, gt, group_names=["a", "b", "c"])
    assert info["group_names"] == ["a", "b", "c"]


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_npz
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_npz_finds_in_first_dir(tmp_path):
    d1 = tmp_path / "a"; d1.mkdir()
    d2 = tmp_path / "b"; d2.mkdir()
    (d1 / "00000005.npz").write_bytes(b"")
    found = _resolve_npz("00000005", [d1, d2])
    assert found == d1 / "00000005.npz"


def test_resolve_npz_falls_back_to_second_dir(tmp_path):
    d1 = tmp_path / "a"; d1.mkdir()
    d2 = tmp_path / "b"; d2.mkdir()
    (d2 / "00000007.npz").write_bytes(b"")
    found = _resolve_npz("00000007", [d1, d2])
    assert found == d2 / "00000007.npz"


def test_resolve_npz_zero_pads_numeric_id(tmp_path):
    """Pass cid='5' but file is stored as '00000005.npz' — should still find it."""
    d = tmp_path / "x"; d.mkdir()
    (d / "00000005.npz").write_bytes(b"")
    found = _resolve_npz("5", [d])
    assert found == d / "00000005.npz"


def test_resolve_npz_returns_none_when_missing(tmp_path):
    d = tmp_path / "x"; d.mkdir()
    found = _resolve_npz("99999999", [d])
    assert found is None


def test_resolve_npz_keeps_non_numeric_id_unchanged(tmp_path):
    """Non-numeric clip ID (e.g. fitcoach 'human_002_clip_5') is matched
    as-is, no zero-padding."""
    d = tmp_path / "x"; d.mkdir()
    cid = "human_42_clip_3"
    (d / f"{cid}.npz").write_bytes(b"")
    assert _resolve_npz(cid, [d]) == d / f"{cid}.npz"
