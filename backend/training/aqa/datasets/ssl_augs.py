"""Per-branch SSL augmentation primitives for the MD-SSL pretext task.

Each function operates on a ``[T, 3, H, W]`` uint8 clip (the output of
``decode_clip``, BEFORE ``spatial_train`` normalizes it) and returns the same
dtype/shape so the existing spatial pipeline chain is unchanged. They are applied
INDEPENDENTLY per branch in ``squat_ssl.SquatSSLDataset.__getitem__`` — the
anchor's positive is a differently-augmented copy of the same half-cycle.

This is the conservative SAFE-CORE subset of the paper's 8 augmentations:
temporal shift, horizontal flip, partial (top) masking, mild color jitter, and
mild translation. Rotation / zoom / blur are deliberately NOT enabled by default
— they are ablation toggles, used only if the linear-probe improves with them
(rotation can destroy the knee-valgus KIE signal). Magnitudes are conservative
to preserve the lower-body region that carries the form-error signal.

A clip-consistent transform (one random draw applied to all T frames) is used so
the augmentation does not flicker within a clip. Each function takes an optional
``torch.Generator`` for reproducible/deterministic-resume draws.
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
    MD augmentation (a ±2-frame jitter). [CITED §3.2 / §7]
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
    fine-tune augs. [CITED §3.2]
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
    is preserved (§7). The only *active* CVCSPC augmentation. ``generator`` is
    accepted for API uniformity; the mask itself is deterministic per call.
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
    (§7). One factor per clip (clip-consistent).
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
    (same (dx, dy) for all frames). Magnitude kept small. [§7]
    """
    dx = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
    dy = int(torch.randint(-max_px, max_px + 1, (1,), generator=generator).item())
    return TF.affine(clip_tchw, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])


def zoom(
    clip_tchw: torch.Tensor,
    *,
    min_scale: float = 0.9,
    max_scale: float = 1.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mild center zoom by a random factor in [min_scale, max_scale] (paper MD aug, §3.2).

    Clip-consistent (one scale for all frames). NEAREST + zero fill → uint8 preserved.
    Magnitude kept small — aggressive zoom changes apparent squat depth
    (the KFE cue). [§7]
    """
    scale = min_scale + float(torch.rand(1, generator=generator).item()) * (max_scale - min_scale)
    return TF.affine(clip_tchw, angle=0.0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])


def gaussian_blur(
    clip_tchw: torch.Tensor,
    *,
    min_sigma: float = 0.1,
    max_sigma: float = 1.2,
    kernel_size: int = 5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random Gaussian blur, sigma in [min_sigma, max_sigma] (paper MD aug, §3.2).

    Clip-consistent (one sigma for all frames). Low form-signal risk — blur does not move
    joints; targets sharpness/appearance invariance. [§7]
    """
    sigma = min_sigma + float(torch.rand(1, generator=generator).item()) * (max_sigma - min_sigma)
    return TF.gaussian_blur(clip_tchw, kernel_size=kernel_size, sigma=sigma).to(clip_tchw.dtype)


def channel_swap(
    clip_tchw: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly permute the RGB channels (the paper's 'color channel swapping', §3.2).

    Clip-consistent (one permutation for all frames). Pure index op → dtype/shape preserved.
    Strong appearance invariance that does not move joints (form-safe). [§7]
    """
    perm = torch.randperm(clip_tchw.shape[1], generator=generator)
    return clip_tchw[:, perm, :, :]


def rotation(
    clip_tchw: torch.Tensor,
    *,
    max_deg: float = 10.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Small random rotation in [-max_deg, max_deg] degrees. RISKIEST aug — DEFAULT OFF.

    Rotation changes the apparent knee-valgus angle — the KIE signal — so it is gated behind
    ``SquatSSLDataset(use_rotation=...)`` and enabled ONLY if the linear-probe (esp. KIE) does
    not regress (§7). Clip-consistent, NEAREST + zero fill → uint8 preserved. [§7]
    """
    deg = (float(torch.rand(1, generator=generator).item()) * 2.0 - 1.0) * max_deg
    return TF.affine(clip_tchw, angle=deg, translate=[0, 0], scale=1.0, shear=[0.0, 0.0])


if __name__ == "__main__":
    fake = torch.randint(0, 256, (16, 3, 112, 112), dtype=torch.uint8)
    for fn in (temporal_shift, horizontal_flip, top_mask, color_jitter, translation,
               zoom, gaussian_blur, channel_swap, rotation):
        out = fn(fake)
        assert out.shape == fake.shape, (fn.__name__, tuple(out.shape))
        assert out.dtype == fake.dtype, (fn.__name__, out.dtype)
    print("ssl_augs.py smoke: ok")
