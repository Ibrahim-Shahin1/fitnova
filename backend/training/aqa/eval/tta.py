"""Test-time augmentation forward pass + val-tuned recipe selection (D5).

TTA is applied at inference: each test clip is run through the model under
several augmentations and the logits are averaged. The augmentation recipe is
NOT fixed — it is selected by maximizing macro-F1 on the val split
(``select_tta_recipe``), because Phase 3 trained with horizontal flip OFF, so
flip is out-of-distribution and must be validated, not assumed (RESEARCH §14 /
CONTEXT D5: measure, don't guess).

5-crop is implemented inline here; ``datasets/transforms.py`` is reused unchanged
(D9). ``tta_forward`` returns mean LOGITS (the caller applies sigmoid before
ensemble aggregation in ``eval/ensemble.py``).

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D5.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # torch is only needed at call time (inside tta_forward); keep
    import torch  # the module importable torch-free so select_tta_recipe is testable.

logger = logging.getLogger("aqa.phase04")


def tta_forward(
    model: torch.nn.Module,
    clip: torch.Tensor,
    recipe: list[str],
    device: torch.device,
) -> np.ndarray:
    """Apply a TTA recipe, run each augmented copy through the model, mean logits.

    Args:
        model: fine-tuned model in eval mode.
        clip: float tensor ``[3, 32, 112, 112]`` — already spatial_val-normalized.
        recipe: aug names drawn from ``{"temporal_jitter", "spatial_5crop", "flip"}``.
            Empty recipe == original clip only (the identity TTA).
        device: inference device.

    Returns:
        float ndarray shape ``(2,)`` — mean logits over the TTA copies (NOT
        sigmoid; the caller sigmoids before ensemble aggregation, D4/D5).
    """
    raise NotImplementedError("Task 11 — implement tta_forward (D5 / RESEARCH §14)")


def select_tta_recipe(
    val_logits_per_recipe: dict[tuple[str, ...], np.ndarray],
    val_labels: np.ndarray,
) -> tuple[str, ...]:
    """Return the val-macro-F1-maximizing TTA recipe (D5 — measure, don't guess).

    Args:
        val_logits_per_recipe: ``{recipe_tuple: logits (N, 2)}`` for each candidate
            combo evaluated on the val split.
        val_labels: int ndarray ``(N, 2)`` ground-truth ``(KIE, KFE)``.

    Returns:
        The recipe tuple with the highest val macro-F1 (per-error threshold tuned
        via ``eval/metrics.py``'s ``threshold_sweep`` + ``f1_per_error``).
    """
    raise NotImplementedError("Task 11 — implement select_tta_recipe (D5 / RESEARCH §14)")
