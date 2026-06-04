"""Test-time augmentation forward pass + val-tuned recipe selection.

TTA is applied at inference: each test clip is run through the model under several
augmentations and the logits are averaged. The augmentation recipe is NOT fixed —
it is selected by maximizing macro-F1 on the val split (``select_tta_recipe``),
because the supervised baseline trained with horizontal flip OFF, so flip is
out-of-distribution and must be validated, not assumed (measure, don't guess).

5-crop is implemented inline here; ``datasets/transforms.py`` is reused unchanged.
``tta_forward`` returns mean LOGITS (the caller applies sigmoid before ensemble
aggregation in ``eval/ensemble.py``). ``select_tta_recipe`` is pure-numpy (uses
``eval/metrics.py``) and is importable + testable without torch; ``tta_forward``
imports torch lazily at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from backend.training.aqa.eval.metrics import f1_per_error, threshold_sweep

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
        sigmoid; the caller sigmoids before ensemble aggregation).
    """
    import torch  # lazy — keeps the module importable torch-free for select_tta_recipe

    views = [clip]  # the original clip is always one of the TTA views
    for aug in recipe:
        if aug == "temporal_jitter":
            views.append(torch.roll(clip, shifts=1, dims=1))  # roll along T of [C,T,H,W]
        elif aug == "flip":
            views.append(torch.flip(clip, dims=(-1,)))  # mirror W (KIE/KFE ~bilaterally symmetric)
        elif aug == "spatial_5crop":
            views.extend(_tta_five_crop(clip))  # center + 4 corners (inline)
        else:
            raise ValueError(f"tta_forward: unknown TTA aug {aug!r}")

    model.eval()
    logits = []
    with torch.no_grad():
        for v in views:
            out = model(v.unsqueeze(0).to(device))  # [1, 2]
            logits.append(out.squeeze(0).detach().cpu().numpy())
    # Mean LOGITS, not sigmoid — aggregation order is per-seed TTA mean ->
    # mean across seeds -> val-tuned threshold.
    return np.mean(np.stack(logits, axis=0), axis=0).astype(float)


def _tta_five_crop(clip: torch.Tensor) -> list:
    """Center + 4 corner crops of a ``[C,T,H,W]`` clip (5-crop, inline).

    The clip arrives at crop_size; we upsample the spatial dims ~1.14x then take five
    crop_size views (center + 4 corners), mirroring spatial_val's resize_short=128 ->
    crop=112 convention WITHOUT modifying transforms.py. NOTE: for full fidelity the
    5-crop should run on the pre-crop frame; this self-contained upsample form is kept
    only if select_tta_recipe finds it improves val macro-F1.
    """
    import torch.nn.functional as F

    _c, _t, h, w = clip.shape
    up_h, up_w = round(h * 128 / 112), round(w * 128 / 112)
    big = F.interpolate(clip, size=(up_h, up_w), mode="bilinear", align_corners=False)
    top, left = up_h - h, up_w - w
    offsets = [
        (top // 2, left // 2),  # center
        (0, 0),                 # top-left
        (0, left),              # top-right
        (top, 0),               # bottom-left
        (top, left),            # bottom-right
    ]
    return [big[:, :, oy : oy + h, ox : ox + w] for (oy, ox) in offsets]


def select_tta_recipe(
    val_logits_per_recipe: dict[tuple[str, ...], np.ndarray],
    val_labels: np.ndarray,
) -> tuple[str, ...]:
    """Return the val-macro-F1-maximizing TTA recipe (measure, don't guess).

    Args:
        val_logits_per_recipe: ``{recipe_tuple: logits (N, 2)}`` for each candidate
            combo evaluated on the val split.
        val_labels: int ndarray ``(N, 2)`` ground-truth ``(KIE, KFE)``.

    Returns:
        The recipe tuple with the highest val macro-F1 (per-error threshold tuned
        via ``threshold_sweep`` + ``f1_per_error``). Ties broken deterministically:
        shortest recipe first, then lexicographic (the first/shortest wins).
    """
    val_labels = np.asarray(val_labels)
    best_recipe: tuple[str, ...] | None = None
    best_macro = -1.0
    for recipe in sorted(val_logits_per_recipe, key=lambda r: (len(r), tuple(r))):
        logits = np.asarray(val_logits_per_recipe[recipe], dtype=float)
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> [0, 1]
        per_head_f1 = []
        for head in range(scores.shape[1]):
            thr, _ = threshold_sweep(val_labels[:, head], scores[:, head])
            preds = (scores[:, head] >= thr).astype(int)
            per_head_f1.append(f1_per_error(val_labels[:, head], preds))
        macro = float(np.mean(per_head_f1))
        if macro > best_macro:  # strict > -> first (shortest) recipe wins ties
            best_macro = macro
            best_recipe = recipe
    return best_recipe  # type: ignore[return-value]
