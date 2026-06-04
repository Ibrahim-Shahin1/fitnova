"""Pure-function evaluation metrics for the Squat KIE/KFE supervised baseline.

Wraps sklearn primitives (`f1_score`, `precision_recall_curve`, `average_precision_score`,
`confusion_matrix`) behind these contracts:

- `f1_per_error(y_true, y_pred) -> float` — binary F1 with `zero_division=0`
- `pr_auc_per_error(y_true, y_score) -> float` — threshold-free average precision
- `threshold_sweep(y_true, y_score) -> (threshold, f1)` — F1-maximising threshold via
  `precision_recall_curve` (all unique thresholds, no linspace discretisation)
- `confusion_matrix_per_error(y_true, y_pred) -> ndarray` — 2x2 with explicit
  `labels=[0, 1]` so downstream figure code always sees a fixed shape

No torch, no I/O, no model construction. Reusable unchanged by downstream code.
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
        predictions instead of warning.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"f1_per_error: shape mismatch y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )
    # zero_division=0 — all-negative predictions return 0.0 silently
    return float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))


def pr_auc_per_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC (average precision) for a single binary error head.

    Args:
        y_true:  int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_score: float array shape `(N,)` of sigmoid-output scores in `[0, 1]`.

    Returns:
        Average-precision score as a Python `float`. Asserts scores are in `[0, 1]` —
        passing raw logits is a caller bug.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"pr_auc_per_error: shape mismatch y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}"
        )
    if y_score_arr.size:
        assert y_score_arr.min() >= 0.0 and y_score_arr.max() <= 1.0, (
            f"pr_auc_per_error: y_score must be in [0, 1] (sigmoid output) — "
            f"got min={y_score_arr.min()}, max={y_score_arr.max()}. "
            "Did you forget torch.sigmoid?"
        )
    return float(average_precision_score(y_true_arr, y_score_arr))


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Sweep decision thresholds via `precision_recall_curve` and return F1-maximising threshold.

    Uses sklearn's `precision_recall_curve` (all unique thresholds, no linspace
    discretisation). Drops sklearn's `(p=1, r=0)` sentinel from the curve before
    F1 computation. Defensive fallback returns `(0.5, 0.0)` on degenerate all-tied-score
    input.

    Args:
        y_true:  int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_score: float array shape `(N,)` of sigmoid-output scores in `[0, 1]`.

    Returns:
        Tuple `(best_threshold, best_f1)` both as Python `float`s.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    # precision_recall_curve returns one extra (p=1, r=0) sentinel point at the end;
    # thresholds excludes the sentinel by construction. Slice precision/recall to
    # align with thresholds before F1 computation.
    if len(thresholds) == 0:
        # Defensive fallback: all-tied y_score → no thresholds → degenerate PR curve.
        logger.warning(
            "threshold_sweep: degenerate input (all y_score tied or empty); "
            "returning fallback (0.5, 0.0)"
        )
        return (0.5, 0.0)
    p = precision[:-1]
    r = recall[:-1]
    f1_values = 2.0 * p * r / (p + r + 1e-9)
    best_idx = int(np.argmax(f1_values))
    return (float(thresholds[best_idx]), float(f1_values[best_idx]))


def confusion_matrix_per_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 2x2 confusion matrix using explicit `labels=[0, 1]`.

    Args:
        y_true: int array shape `(N,)` of `{0, 1}` ground-truth labels.
        y_pred: int array shape `(N,)` of `{0, 1}` thresholded predictions.

    Returns:
        Integer ndarray shape `(2, 2)`. Explicit `labels=[0, 1]` guarantees a 2x2
        matrix even when `y_pred` is single-class — downstream figure code requires
        the fixed shape.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    # Explicit labels=[0, 1] forces 2x2 even when y_pred contains only one class.
    return confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
