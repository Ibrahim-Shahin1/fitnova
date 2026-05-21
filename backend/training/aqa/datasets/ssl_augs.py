"""Per-branch SSL augmentation primitives for the MD-SSL pretext task (§7).

Each function operates on a ``[T, 3, H, W]`` uint8 clip (the output of
``decode_clip``, BEFORE ``spatial_train`` normalizes it) and returns the same
dtype/shape so the existing spatial pipeline chain is unchanged. They are applied
INDEPENDENTLY per branch in ``squat_ssl.SquatSSLDataset.__getitem__`` — the
anchor's positive is a differently-augmented copy of the same half-cycle.

SSL augmentations are independent of the fine-tune augmentations: the paper
applies "strong augmentations" during MD pretraining (RESEARCH §7), including
horizontal flip — which is deliberately OFF for the supervised fine-tune
(Phase 2 D5). Magnitudes are conservative to preserve the lower-body knee region
that carries the form-error signal (RESEARCH §7 form-signal tension).

See: .planning/phases/04-squat-motion-disentangling-ssl/04-RESEARCH.md — §7.
"""

from __future__ import annotations

import logging
import random

import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger("aqa.phase04")


def temporal_shift(
    clip_tchw: torch.Tensor,
    *,
    max_shift: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Roll frames along the temporal axis by a random offset in [-max_shift, max_shift]."""
    raise NotImplementedError("Task 3 — implement temporal_shift (RESEARCH §7)")


def horizontal_flip(
    clip_tchw: torch.Tensor,
    *,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """With probability p, horizontally flip every frame (TF.hflip)."""
    raise NotImplementedError("Task 3 — implement horizontal_flip (RESEARCH §7)")


def top_mask(
    clip_tchw: torch.Tensor,
    *,
    mask_frac: float = 0.4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero the top ``mask_frac`` rows of every frame (occlusion; preserves lower body)."""
    raise NotImplementedError("Task 3 — implement top_mask (RESEARCH §7)")


def color_jitter(
    clip_tchw: torch.Tensor,
    *,
    brightness: float = 0.2,
    contrast: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random brightness/contrast in [1-x, 1+x] (appearance invariance; joints unmoved)."""
    raise NotImplementedError("Task 3 — implement color_jitter (RESEARCH §7)")


def translation(
    clip_tchw: torch.Tensor,
    *,
    max_px: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Small random (dx, dy) translation in [-max_px, max_px] (large would crop the knees)."""
    raise NotImplementedError("Task 3 — implement translation (RESEARCH §7)")
