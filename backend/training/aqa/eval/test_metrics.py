"""pytest unit suite for `backend.training.aqa.eval.metrics`.

Covers SQUAT-03-b (F1), SQUAT-03-c (threshold sweep), SQUAT-03-d (PR-AUC), and the
2x2-shape regression for `confusion_matrix_per_error` per the 03-VALIDATION.md per-task
verification map. SQUAT-03-a (model head shape) lives in
`backend/training/aqa/harness/test_supervised_train.py` per PATTERNS §3 Note.

Stub bodies in Task 1 (all `pytest.skip("Task 3")`); real test bodies fill in Task 3.

See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D7 / D9.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# (placeholder for shared helpers in Task 3)


# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-b: F1 per error
# ─────────────────────────────────────────────────────────────────────────────


def test_f1_known_values() -> None:
    pytest.skip("Task 3")


def test_f1_all_negative_predictions() -> None:
    pytest.skip("Task 3")


def test_f1_perfect() -> None:
    pytest.skip("Task 3")


# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-c: threshold sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_threshold_sweep_known_max() -> None:
    pytest.skip("Task 3")


def test_threshold_sweep_separable() -> None:
    pytest.skip("Task 3")


def test_threshold_sweep_degenerate_all_tied() -> None:
    pytest.skip("Task 3")


# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-d: PR-AUC
# ─────────────────────────────────────────────────────────────────────────────


def test_pr_auc_matches_sklearn() -> None:
    pytest.skip("Task 3")


def test_pr_auc_perfect_separation() -> None:
    pytest.skip("Task 3")


# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03 confusion matrix (2x2 shape regression)
# ─────────────────────────────────────────────────────────────────────────────


def test_confusion_matrix_always_2x2() -> None:
    pytest.skip("Task 3")
