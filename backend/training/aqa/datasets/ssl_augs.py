"""Per-branch SSL augmentation primitives for the MD-SSL pretext task (§7).

Each function operates on a ``[T, 3, H, W]`` uint8 clip (the output of
``decode_clip``, BEFORE ``spatial_train`` normalizes it) and returns the same
dtype/shape so the existing spatial pipeline chain is unchanged. They are applied
INDEPENDENTLY per branch in ``squat_ssl.SquatSSLDataset.__getitem__`` — the
anchor's positive is a differently-augmented copy of the same half-cycle.

This is the conservative SAFE-CORE subset of the paper's 8 augmentations
(RESEARCH §7): temporal shift, horizontal flip, partial (top) masking, mild
color jitter, and mild translation. Rotation / zoom / blur are deliberately NOT
implemented here — they are ablation toggles added in Plan 02 only if the
linear-probe improves with them (RESEARCH §7: rotation can destroy the
knee-valgus KIE signal). Magnitudes are conservative to preserve the lower-body
region that carries the form-error signal.

A clip-consistent transform (one random draw applied to all T frames) is used so
the augmentation does not flicker within a clip. Each function takes an optional
``torch.Generator`` for reproducible/deterministic-resume draws.

See: .planning/phases/04-squat-motion-disentangling-ssl/04-RESEARCH.md — §7.
"""

from __future__ import annotations

import logging

import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger("aqa.phase04")


def temporal_shift(
    clip_tchw: torch.Tensor,
    *,
    max_shift: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Roll frames along the temporal axis by a random offset in [-max_shift, max_shift].

    The half-cycle is a temporal segment, so a small temporal roll is the core
    MD augmentation (matches Phase 3's ±2-frame jitter). [CITED §3.2 / §7]
    """
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
    return torch.roll(clip_tchw, shifts=shift, dims=0)


def horizontal_flip(
    clip_tchw: torch.Tensor,
    *,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """With probability ``p``, horizontally flip every frame (TF.hflip).

    KIE/KFE are bilaterally ~symmetric, so a mirror preserves the label; the
    paper includes flip in the MD aug set. NOTE: SSL augs are independent of the
    fine-tune augs (Phase 3 trained flip OFF — that is unrelated). [CITED §3.2]
    """
    if float(torch.rand(1, generator=generator).item()) < p:
        return TF.hflip(clip_tchw)
    return clip_tchw


def top_mask(
    clip_tchw: torch.Tensor,
    *,
    mask_frac: float = 0.4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero the top ``mask_frac`` rows of every frame (occlusion robustness).

    Masks the TOP of the frame so the lower-body knee region (the KIE/KFE signal)
    is preserved (RESEARCH §7). The only *active* CVCSPC augmentation. ``generator``
    is accepted for API uniformity; the mask itself is deterministic per call.
    """
    h = clip_tchw.shape[-2]
    n_rows = int(round(mask_frac * h))
    out = clip_tchw.clone()
    out[..., :n_rows, :] = 0
    return out


def color_jitter(
    clip_tchw: torch.Tensor,
    *,
    brightness: float = 0.2,
    contrast: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mild random brightness/contrast in [1-x, 1+x] (appearance invariance).

    Targets clothing/lighting invariance — does not move joints, so it is form-safe
    (RESEARCH §7). One factor per clip (clip-consistent).
    """
    b_factor = 1.0 + (float(torch.rand(1, generator=generator).item()) * 2.0 - 1.0) * brightness
    c_factor = 1.0 + (float(torch.rand(1, generator=generator).item()) * 2.0 - 1.0) * contrast
    out = TF.adjust_brightness(clip_tchw, b_factor)
    out = TF.adjust_contrast(out, c_factor)
    return out.to(clip_tchw.dtype)  # guarantee uint8 in == uint8 out


def translation(
    clip_tchw: torch.Tensor,
    *,
    max_px: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Small random (dx, dy) translation in [-max_px, max_px] (large would crop the knees).

    NEAREST interpolation + zero fill → uint8 preserved. Clip-consistent shift
    (same (dx, dy) for all frames). Magnitude kept small per RESEARCH §7. [§7]
    """
    dx = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
    dy = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
    return TF.affine(clip_tchw, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])


if __name__ == "__main__":
    fake = torch.randint(0, 256, (16, 3, 112, 112), dtype=torch.uint8)
    for fn in (temporal_shift, horizontal_flip, top_mask, color_jitter, translation):
        out = fn(fake)
        assert out.shape == fake.shape, (fn.__name__, tuple(out.shape))
        assert out.dtype == fake.dtype, (fn.__name__, out.dtype)
    print("ssl_augs.py smoke: ok")
