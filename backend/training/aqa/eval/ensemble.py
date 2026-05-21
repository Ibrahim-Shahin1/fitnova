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
    raise NotImplementedError("Task 10 — implement aggregate_sigmoid_mean (D4 / RESEARCH §13)")
