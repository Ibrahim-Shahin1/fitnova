"""Multi-seed ensemble aggregation for Phase 4 MD-SSL (D4).

Mean-of-sigmoids across the 3 fine-tune seeds (42 / 1337 / 7). Pure numpy — no
torch, no I/O, no model construction. Output is constrained to [0, 1] so it
satisfies the ``pr_auc_per_error`` assertion in ``eval/metrics.py`` and feeds the
same threshold sweep the single-model path uses. ``eval/metrics.py`` is reused
unchanged (D9); this module only aggregates score arrays.

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D4.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("aqa.phase04")


def aggregate_sigmoid_mean(per_seed_logits: list[np.ndarray]) -> np.ndarray:
    """Mean of per-seed sigmoid scores across seeds (D4: mean-of-sigmoids).

    Args:
        per_seed_logits: list of float ndarrays, each shape ``(N, 2)`` — raw
            logits (NOT sigmoids) from each seed's model for the same N clips,
            columns ``(KIE, KFE)``. Sigmoid is applied here, then averaged.

    Returns:
        float ndarray shape ``(N, 2)`` — ensemble sigmoid scores in ``[0, 1]``.

    Raises:
        ValueError: empty list, or inconsistent ``(N, 2)`` shapes across seeds.
    """
    if len(per_seed_logits) == 0:
        raise ValueError("aggregate_sigmoid_mean: empty per_seed_logits list")
    arrays = [np.asarray(a, dtype=float) for a in per_seed_logits]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"aggregate_sigmoid_mean: inconsistent shapes across seeds: {sorted(shapes)}"
        )
    # D4: mean of SIGMOIDS (NOT mean of logits). The [0,1] output satisfies the
    # pr_auc_per_error assertion at metrics.py:72.
    sigmoid_scores = [1.0 / (1.0 + np.exp(-a)) for a in arrays]
    ensemble = np.mean(sigmoid_scores, axis=0)
    assert ensemble.min() >= 0.0 and ensemble.max() <= 1.0, (ensemble.min(), ensemble.max())
    return ensemble
