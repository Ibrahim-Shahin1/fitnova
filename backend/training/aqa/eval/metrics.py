"""Pure-function evaluation metrics for the Phase 3 Squat KIE/KFE supervised baseline.

Wraps sklearn primitives (`f1_score`, `precision_recall_curve`, `average_precision_score`,
`confusion_matrix`) behind the Phase 3 contracts:

- `f1_per_error(y_true, y_pred) -> float` — binary F1 with `zero_division=0`
- `pr_auc_per_error(y_true, y_score) -> float` — threshold-free average precision
- `threshold_sweep(y_true, y_score) -> (threshold, f1)` — F1-maximising threshold via
  `precision_recall_curve` (RESEARCH §5 — replaces CONTEXT D7's linspace sweep)
- `confusion_matrix_per_error(y_true, y_pred) -> ndarray` — 2x2 with explicit
  `labels=[0, 1]` so downstream figure code always sees a fixed shape

No torch, no I/O, no model construction. Phase 4+ may reuse this module unchanged.

This file is the Task 1 scaffold — every public function raises
`NotImplementedError("Task 2")`. Task 2 fills the bodies.

See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D7 / D9.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

logger = logging.getLogger("aqa.phase03")


def f1_per_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 for a single binary error head.

    Args:
        y_true: int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_pred: int array shape `(N,)` of `{0, 1}` thresholded predictions.

    Returns:
        F1 score as a Python `float`. `zero_division=0` returns `0.0` for all-negative
        predictions instead of warning (RESEARCH §8).
    """
    raise NotImplementedError("Task 2")


def pr_auc_per_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC (average precision) for a single binary error head.

    Args:
        y_true:  int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_score: float array shape `(N,)` of sigmoid-output scores in `[0, 1]`.

    Returns:
        Average-precision score as a Python `float`. Asserts scores are in `[0, 1]` —
        passing raw logits is a caller bug.
    """
    raise NotImplementedError("Task 2")


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Sweep decision thresholds via `precision_recall_curve` and return F1-maximising threshold.

    Per RESEARCH §5: uses sklearn's `precision_recall_curve` (all unique thresholds, no
    linspace discretisation). Drops sklearn's `(p=1, r=0)` sentinel from the curve before
    F1 computation. Defensive fallback returns `(0.5, 0.0)` on degenerate all-tied-score
    input.

    Args:
        y_true:  int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_score: float array shape `(N,)` of sigmoid-output scores in `[0, 1]`.

    Returns:
        Tuple `(best_threshold, best_f1)` both as Python `float`s.
    """
    raise NotImplementedError("Task 2")


def confusion_matrix_per_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 2x2 confusion matrix using explicit `labels=[0, 1]`.

    Args:
        y_true: int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_pred: int array shape `(N,)` of `{0, 1}` thresholded predictions.

    Returns:
        Integer ndarray shape `(2, 2)`. Explicit `labels=[0, 1]` guarantees a 2x2
        matrix even when `y_pred` is single-class — downstream figure code (Task 14)
        requires the fixed shape.
    """
    raise NotImplementedError("Task 2")
