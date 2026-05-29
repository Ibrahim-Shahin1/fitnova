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

import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import ssl_augs
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    decode_clip_cached,
    spatial_train,
    uniform_sample_indices,
)

if TYPE_CHECKING:  # avoid a runtime import cycle (md_pretrain imports this module)
    from backend.training.aqa.harness.md_pretrain import MDConfig

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
    traj_y = np.asarray(traj_y, dtype=float)
    if traj_y.ndim != 1 or traj_y.shape[0] < 2:
        raise ValueError(
            f"split_half_cycles: traj_y must be 1-D with >=2 samples, got shape {traj_y.shape}"
        )
    # 1. Smooth to suppress YOLO-detection jitter before extremum detection (RESEARCH §1 pt.2).
    sm = scipy.ndimage.gaussian_filter1d(traj_y, sigma=smooth_sigma)
    # 2. Bottom-of-rep = the global extremum of the smoothed y-curve (RESEARCH §1 pt.3).
    #    The SIGN is a PARAM, never hard-coded: Fig.3 plots amplitude (possibly inverted)
    #    and image-y grows downward, so the real argmax-vs-argmin sign is resolved
    #    empirically in Plan 02's checkpoint:human-verify probe on real clips (T-04-02).
    bottom = int(np.argmax(sm)) if bottom_is_argmax else int(np.argmin(sm))
    # 3. Descent = frames [0..bottom]; Ascent = frames [bottom..end]. Reuse the existing
    #    uniform_sample_indices (D9 — no new sampler). The bottom frame is the shared
    #    turning point: descent[-1] == bottom == ascent[0].
    descent = uniform_sample_indices(num_frames=bottom + 1, target=frames_per_half)
    ascent = uniform_sample_indices(num_frames=len(sm) - bottom, target=frames_per_half) + bottom
    # NOTE: multi-rep find_peaks handling is deferred to Plan 02, after the probe measures
    # multi-rep frequency across the 4,970 trajectories (RESEARCH §1 pt.4 — paper is
    # nominally single-rep, so argmax/argmin covers the bulk).
    return descent.numpy(), ascent.numpy()


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
        strong_augs: bool = True,
        use_rotation: bool = False,
        aug_prob: float = 0.5,
        cache_dir: str | None = None,
    ) -> None:
        self.videos_root = videos_root
        self.trajectories_root = trajectories_root
        self.frames_per_half = frames_per_half
        self.crop_size = crop_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.strong_augs = strong_augs       # v2: paper-faithful strong aug set (§3.2/§7); False = v1 safe-core
        self.use_rotation = use_rotation     # rotation OFF by default — distorts the knee-valgus KIE signal (§7)
        self.aug_prob = aug_prob             # per-aug independent application probability (strong set)
        self._spatial_fn: Callable[..., torch.Tensor] = spatial_train  # always train-aug in SSL
        # Clip IDs = stems present in BOTH the videos dir and the trajectories tree.
        # The mp4s are flattened to the top by the video extractor, but the trajectory
        # JSONs keep the zip's internal folder (extractall) — so RECURSIVE-glob them and
        # keep a stem->path map. (The probe used a recursive glob and saw 4970; a
        # non-recursive glob here returned 0 — this is the len(ds)==0 fix.)
        vid_stems = {p.stem for p in Path(videos_root).glob("*.mp4")}
        self._traj_paths: dict[str, Path] = {
            p.stem: p for p in Path(trajectories_root).rglob("*.json")
        }
        self._clip_ids: list[str] = sorted(vid_stems & set(self._traj_paths))
        logger.info(
            "SquatSSLDataset: %d clips (videos=%d, trajectories=%d) under %s",
            len(self._clip_ids), len(vid_stems), len(self._traj_paths), videos_root,
        )

    def __len__(self) -> int:
        return len(self._clip_ids)

    def _load_trajectory(self, clip_id: str) -> np.ndarray:
        """Load the barbell-y trajectory for ``clip_id`` (Task-1-confirmed: per-clip JSON flat float list).

        The trajectory is 1:1 with video frames (Task 1 probe: frames/traj == 1.000), so its
        index space IS the video-frame space — no rescaling. In-file NaN/null values (detection
        gaps — FOUND by the probe in e.g. 25707_3, contradicting Phase 1's "0% NaN") are linearly
        interpolated over so ``split_half_cycles``' extremum detection is robust (RESEARCH §1/§8).
        """
        path = self._traj_paths[clip_id]  # actual (possibly nested) path from the rglob map
        with open(path, encoding="utf-8") as fh:
            y = np.asarray(json.load(fh), dtype=float)
        nan_mask = np.isnan(y)
        if nan_mask.any():
            valid = ~nan_mask
            if int(valid.sum()) < 2:
                raise ValueError(f"trajectory {clip_id}: <2 non-NaN samples ({int(valid.sum())})")
            idx = np.arange(len(y))
            y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
        return y

    def _augment(self, clip_u8: torch.Tensor) -> torch.Tensor:
        """Apply the SSL augmentations per-branch (§3.2/§7) — independent default-RNG draws per call.

        ``strong_augs=False`` reproduces md_pretrain_v1's SAFE-CORE set (the weak-aug ablation
        point). ``strong_augs=True`` is the paper-faithful set: each aug is applied INDEPENDENTLY
        with probability ``aug_prob`` so the anchor and its positive are genuinely different views
        — preventing the trivial-positive contrastive collapse seen in v1 (positive ≈ anchor →
        eff_rank 11.8→3.3, probe peaked at ep5). Rotation is gated behind ``use_rotation`` (default
        OFF) because it distorts the knee-valgus KIE signal — our largest SSL lift (§7).
        """
        if not self.strong_augs:
            # md_pretrain_v1 safe-core (reproduces the weak-aug run for the ablation).
            out = ssl_augs.temporal_shift(clip_u8)
            out = ssl_augs.horizontal_flip(out)
            out = ssl_augs.top_mask(out)
            out = ssl_augs.color_jitter(out)
            return out

        # md_pretrain_v2 paper-faithful strong set — each applied independently with prob aug_prob.
        def _coin() -> bool:
            return float(torch.rand(1).item()) < self.aug_prob

        out = clip_u8
        if _coin():
            out = ssl_augs.temporal_shift(out, max_shift=3)
        out = ssl_augs.horizontal_flip(out, p=0.5)
        if _coin():
            out = ssl_augs.top_mask(out, mask_frac=0.4)
        if _coin():
            out = ssl_augs.translation(out, max_px=10)
        if _coin():
            out = ssl_augs.zoom(out)
        if _coin():
            out = ssl_augs.gaussian_blur(out)
        if _coin():
            out = ssl_augs.channel_swap(out)
        if self.use_rotation and _coin():
            out = ssl_augs.rotation(out)
        out = ssl_augs.color_jitter(out, brightness=0.3, contrast=0.3)
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip_id = self._clip_ids[idx]
        traj_y = self._load_trajectory(clip_id)
        # 1. Split into descent/ascent. bottom_is_argmax=False (ARGMIN) is the Task-1-confirmed
        #    sign: the rep-bottom is the MIDDLE minimum; argmax lands at the standing endpoints and
        #    degenerates the ascent (Plan 02 Task 1 probe — argmax ascent collapsed to one frame).
        descent_idx, ascent_idx = split_half_cycles(
            traj_y, frames_per_half=self.frames_per_half, bottom_is_argmax=False,
        )
        # 2. Decode the two half-cycles (1:1 traj->frame, Task 1). decode_clip -> [16,3,H,W] uint8.
        video_path = os.path.join(self.videos_root, f"{clip_id}.mp4")
        n = self.frames_per_half
        descent_u8 = decode_clip_cached(self.cache_dir, f"{clip_id}_desc{n}", video_path, torch.as_tensor(descent_idx, dtype=torch.long))
        ascent_u8 = decode_clip_cached(self.cache_dir, f"{clip_id}_asc{n}", video_path, torch.as_tensor(ascent_idx, dtype=torch.long))
        # 3. anchor + positive = two independent augmented views of the DESCENT; negative = an
        #    augmented view of the ASCENT (RESEARCH §3 / paper §3.2).
        anchor_u8 = self._augment(descent_u8)
        positive_u8 = self._augment(descent_u8)
        negative_u8 = self._augment(ascent_u8)
        # 4. Temporal-reverse either {anchor,positive} OR {negative} so the GLOBAL down/up motion is
        #    identical across all three and only the LOCAL (anomalous) motion distinguishes them
        #    (RESEARCH §3). Coin flip on the default RNG (harness capture/restore covers it).
        if float(torch.rand(1).item()) < 0.5:
            anchor_u8 = anchor_u8.flip(0)
            positive_u8 = positive_u8.flip(0)
        else:
            negative_u8 = negative_u8.flip(0)
        # 5. Spatial pipeline -> [3,16,112,112] float32 Kinetics-normalized (transforms.py, D9).
        return {
            "anchor": self._spatial_fn(anchor_u8, crop_size=self.crop_size),
            "positive": self._spatial_fn(positive_u8, crop_size=self.crop_size),
            "negative": self._spatial_fn(negative_u8, crop_size=self.crop_size),
        }


def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker (PyTorch reproducibility idiom; pairs with persistent_workers, D7)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_ssl_loader(dataset: SquatSSLDataset, config: "MDConfig", *, seed: int = 42) -> DataLoader:
    """Construct the SSL DataLoader (shuffle=True; persistent_workers when num_workers>0, D7).

    Args:
        dataset: a constructed ``SquatSSLDataset``.
        config:  ``MDConfig`` — supplies ``batch_size`` and ``num_workers``.
        seed:    seeds the DataLoader generator (worker re-seeding via ``seed_worker``).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    _persistent = config.num_workers > 0  # D7 / [[reference_pytorch_persistent_workers]]
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=True,  # contrastive + BatchNorm: never feed a size-1 final batch
        persistent_workers=_persistent,
    )
