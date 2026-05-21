"""Unlabeled Squat SSL dataset for the Motion-Disentangling pretext task.

Yields a ``{anchor, positive, negative}`` dict of three float32 clips
``[3, 16, H, W]`` from the 4,970 unlabeled Squat clips + their pre-computed
barbell trajectories. Per RESEARCH §3 (paper §3.2): the anchor is the descent
half-cycle, the negative is the ascent half-cycle of the SAME rep, and the
positive is a differently-augmented copy of the anchor. Either ``{anchor,
positive}`` or ``{negative}`` is randomly temporal-reversed so the global
down/up motion is identical across all three and only the local (anomalous)
motion distinguishes them.

The only genuinely new sampling logic is ``split_half_cycles`` (RESEARCH §1);
everything else reuses ``decode_clip`` / ``uniform_sample_indices`` /
``spatial_train`` from ``transforms.py`` (D9, unchanged) and the augmentation
primitives in ``ssl_augs.py``.

NOTE: ``_load_trajectory`` and ``__getitem__`` bodies are finalized in Plan 02
Task 2, AFTER the gated trajectory-format + half-cycle-sign probe (§8) — the
on-disk format and the argmax-vs-argmin sign are [ASSUMED] until probed.

See: .planning/phases/04-squat-motion-disentangling-ssl/04-RESEARCH.md — §1, §3, §8.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import ssl_augs
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    spatial_train,
    uniform_sample_indices,
)

logger = logging.getLogger("aqa.phase04")


def split_half_cycles(
    traj_y: np.ndarray,
    *,
    frames_per_half: int = 16,
    smooth_sigma: float = 2.0,
    bottom_is_argmax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a single-rep barbell-y trajectory into descent + ascent frame indices.

    Args:
        traj_y: 1-D barbell y-coordinate trajectory (one per video frame).
        frames_per_half: frames to sample from each half-cycle (16, CITED §5).
        smooth_sigma: gaussian_filter1d sigma for de-noising the trajectory.
        bottom_is_argmax: True → bottom of rep = argmax of smoothed y; False →
            argmin. The SIGN is resolved empirically in Plan 02's probe (§1 pt.3)
            because Fig.3 plots amplitude (possibly inverted) and image-y grows
            downward — it is a PARAM here, never hard-coded (T-04-02 mitigation).

    Returns:
        ``(descent_indices, ascent_indices)`` — both int ndarrays of length
        ``frames_per_half``, with ``descent[-1] <= bottom <= ascent[0]``.

    Raises:
        ValueError: trajectory shorter than 2 frames.
    """
    raise NotImplementedError("Task 2 — implement split_half_cycles (D1 / RESEARCH §1)")


class SquatSSLDataset(Dataset):
    """Triplet dataset over the unlabeled Squat clips (anchor/positive/negative).

    __getitem__ returns ``{"anchor", "positive", "negative"}`` — three float32
    ``[3, 16, H, W]`` Kinetics-normalized clips. No labels, no ``pos_weight``
    (SSL is unsupervised).
    """

    def __init__(
        self,
        *,
        videos_root: str,
        trajectories_root: str,
        frames_per_half: int = 16,
        crop_size: int = 112,
        seed: int = 42,
    ) -> None:
        self.videos_root = videos_root
        self.trajectories_root = trajectories_root
        self.frames_per_half = frames_per_half
        self.crop_size = crop_size
        self.seed = seed
        self._spatial_fn: Callable[..., torch.Tensor] = spatial_train  # always train-aug in SSL
        self._clip_ids: list[str] = []  # populated by Plan 02 Task 2 after the probe

    def __len__(self) -> int:
        raise NotImplementedError("Plan 02 Task 2 — finalize after trajectory-format probe (§8)")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Plan 02 Task 2 — finalize after trajectory-format probe (§8)")

    def _load_trajectory(self, clip_id: str) -> np.ndarray:
        raise NotImplementedError("Plan 02 Task 2 — finalize after trajectory-format probe (§8)")


def build_ssl_loader(
    *,
    videos_root: str,
    trajectories_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    frames_per_half: int = 16,
    crop_size: int = 112,
    seed: int = 42,
) -> DataLoader:
    """Construct the SSL DataLoader (shuffle=True, persistent_workers when num_workers>0)."""
    raise NotImplementedError("Plan 02 — finalize after the probe-confirmed dataset (§8)")
