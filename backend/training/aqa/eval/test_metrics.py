"""pytest unit suite for `backend.training.aqa.eval.metrics`.

Covers F1, threshold sweep, PR-AUC, and the 2x2-shape regression for
`confusion_matrix_per_error`. The model head shape test lives in
`backend/training/aqa/harness/test_supervised_train.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ─────────────────────────────────────────────────────────────────────────────
# F1 per error
# ─────────────────────────────────────────────────────────────────────────────


def test_f1_known_values() -> None:
    # TP=2 (idx 0, 2), FP=0, FN=1 (idx 4) → P=1.0, R=2/3, F1=0.8.
    y = _arr(1, 0, 1, 0, 1)
    p = _arr(1, 0, 1, 0, 0)
    assert np.isclose(f1_per_error(y, p), 0.8, atol=1e-6)


def test_f1_all_negative_predictions() -> None:
    # zero_division=0 path: all-negative predictions → F1 == 0.0, no warning.
    y = _arr(1, 0, 1, 1)
    p = _arr(0, 0, 0, 0)
    assert f1_per_error(y, p) == 0.0


def test_f1_perfect() -> None:
    y = _arr(1, 0, 1, 0)
    p = _arr(1, 0, 1, 0)
    assert f1_per_error(y, p) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# threshold sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_threshold_sweep_known_max() -> None:
    # Top 3 scores all match positives → F1=1.0 reachable at threshold 0.4.
    y = _arr(1, 0, 1, 0, 1)
    s = _arr(0.9, 0.1, 0.8, 0.2, 0.4)
    threshold, f1 = threshold_sweep(y, s)
    assert np.isclose(threshold, 0.4, atol=1e-3), threshold
    assert np.isclose(f1, 1.0, atol=1e-3), f1


def test_threshold_sweep_separable() -> None:
    # Perfectly separable: F1=1.0 at any threshold in [0.3, 0.8].
    y = _arr(1, 1, 0, 0)
    s = _arr(0.9, 0.8, 0.2, 0.1)
    threshold, f1 = threshold_sweep(y, s)
    assert np.isclose(f1, 1.0, atol=1e-3)
    assert 0.2 < threshold <= 0.8


def test_threshold_sweep_degenerate_all_tied() -> None:
    # All-tied scores: sklearn returns one threshold; F1 at that threshold
    # = 2 * P * R / (P + R) = 2 * 0.5 * 1.0 / 1.5 = 0.667.
    # The (0.5, 0.0) defensive fallback fires only when len(thresholds) == 0,
    # which sklearn typically prevents from valid arrays.
    y = _arr(1, 0, 1, 0)
    s = _arr(0.5, 0.5, 0.5, 0.5)
    threshold, f1 = threshold_sweep(y, s)
    assert np.isclose(threshold, 0.5, atol=1e-3), threshold
    assert np.isclose(f1, 2.0 / 3.0, atol=1e-3), f1


# ─────────────────────────────────────────────────────────────────────────────
# PR-AUC
# ─────────────────────────────────────────────────────────────────────────────


def test_pr_auc_matches_sklearn() -> None:
    # pr_auc_per_error is a thin wrapper — must match sklearn exactly.
    y = _arr(1, 0, 1, 0, 1)
    s = _arr(0.9, 0.1, 0.8, 0.2, 0.4)
    assert pr_auc_per_error(y, s) == float(average_precision_score(y, s))


def test_pr_auc_perfect_separation() -> None:
    y = _arr(1, 1, 0, 0)
    s = _arr(0.9, 0.8, 0.2, 0.1)
    assert pr_auc_per_error(y, s) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# confusion matrix (2x2 shape regression for `labels=[0, 1]` defensive arg)
# ─────────────────────────────────────────────────────────────────────────────


def test_confusion_matrix_always_2x2() -> None:
    # All-positive labels AND predictions: sklearn would normally return 1x1
    # without explicit labels=[0, 1]. The regression test guards the defensive arg.
    y = _arr(1, 1, 1, 1)
    p = _arr(1, 1, 1, 1)
    cm = confusion_matrix_per_error(y, p)
    assert cm.shape == (2, 2), cm.shape
    assert cm.tolist() == [[0, 0], [0, 4]], cm.tolist()
