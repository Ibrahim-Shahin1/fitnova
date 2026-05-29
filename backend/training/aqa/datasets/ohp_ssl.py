"""Unlabeled OHP SSL dataset for the Motion-Disentangling pretext task.

Yields {anchor, positive, negative} float32 clips [3, 16, H, W] from the unlabeled
OHP clips + their barbell BBox trajectories. _load_trajectory parses the per-frame
BBox JSON (frame = [region_0, region_1, region_2]; region_0 = barbell bbox
[x1, y1, x2, y2, conf]; y_center = (y1 + y2) / 2; empty region_0 -> NaN -> interp).
Trajectory index space is 1:1 with video frames; half-cycle split uses argMIN
(overhead = lowest y pixel).
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
    spatial_train,
    uniform_sample_indices,
)

if TYPE_CHECKING:  # avoid a runtime import cycle (md_pretrain imports this module)
    from backend.training.aqa.harness.md_pretrain import MDConfig

logger = logging.getLogger("aqa.phase06")

# The half-cycle split needs a reliable barbell trajectory; a clip where the raw tracker
# missed the bar in most frames can't be split (interpolating across >50% gaps misplaces
# the overhead extremum). Exclude clips below this detection coverage from the SSL set.
_MIN_VALID_FRAC: float = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Half-cycle splitter — COPIED VERBATIM from squat_ssl.py (lines 51-96)
# Same argMIN sign (bottom_is_argmax=False) applies to OHP: overhead = barbell
# at top of image = lowest y pixel = argMIN. (D6-reuse / RESEARCH §3 / Pitfall 2)
# ──────────────────────────────────────────────────────────────────────────────


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
    # multi-rep frequency across the 5,490 trajectories (RESEARCH §1 pt.4 — paper is
    # nominally single-rep, so argmax/argmin covers the bulk).
    return descent.numpy(), ascent.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# OHP SSL Dataset
# ──────────────────────────────────────────────────────────────────────────────


class OHPSSLDataset(Dataset):
    """Triplet dataset over the unlabeled OHP clips (anchor/positive/negative)."""

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
    ) -> None:
        self.videos_root = videos_root
        self.trajectories_root = trajectories_root
        self.frames_per_half = frames_per_half
        self.crop_size = crop_size
        self.seed = seed
        self.strong_augs = strong_augs       # v2: paper-faithful strong aug set (§3.2/§7)
        self.use_rotation = use_rotation     # rotation OFF by default
        self.aug_prob = aug_prob             # per-aug independent application probability
        self._spatial_fn: Callable[..., torch.Tensor] = spatial_train  # always train-aug in SSL

        vid_stems = {p.stem for p in Path(videos_root).glob("*.mp4")}
        self._traj_paths: dict[str, Path] = {
            p.stem: p for p in Path(trajectories_root).rglob("*.json")
        }
        _candidates = sorted(vid_stems & set(self._traj_paths))
        self._clip_ids: list[str] = [c for c in _candidates if self._barbell_coverage(c) >= _MIN_VALID_FRAC]
        logger.info(
            "OHPSSLDataset: %d clips (dropped %d with <%.0f%% barbell coverage; videos=%d, traj=%d)",
            len(self._clip_ids), len(_candidates) - len(self._clip_ids),
            _MIN_VALID_FRAC * 100, len(vid_stems), len(self._traj_paths),
        )

    def __len__(self) -> int:
        return len(self._clip_ids)

    def _barbell_coverage(self, clip_id: str) -> float:
        with open(self._traj_paths[clip_id], encoding="utf-8") as fh:
            data = json.load(fh)
        if not data:
            return 0.0
        return sum(1 for fr in data if len(fr) > 0 and fr[0]) / len(data)

    def _load_trajectory(self, clip_id: str) -> np.ndarray:
        """Load OHP barbell trajectory from BBox JSON and return y_center array.

        OHP format (ReadMe.md.docx + offline inspection of 20 files, RESEARCH §3):
        each JSON is a list of frames; each frame = [region_0, region_1, region_2];
        each region is a list of 0 or 1 bboxes [x1, y1, x2, y2, conf]. Region 0 =
        barbell (wide horizontal bbox covering full image width). y_center = (y1+y2)/2
        per ReadMe formula.

        Missing frames (empty region-0 list, ~40% of files have some) are written as
        np.nan and linearly interpolated, analogous to squat_ssl's NaN interpolation.
        The Squat loader's np.isnan() check is REUSED; only the extraction step differs.

        Defensive A6: access frame[0] if len(frame)>0 else [] — in case a file has
        fewer than 3 regions (not observed in 20-file inspection, but guarded).

        Returns:
            1-D float64 ndarray, length == number of frames in the clip.
            1:1 with video frames [ASSUMED — MUST be confirmed in the Colab probe,
            Plan 02 Task 1, before the SSL GPU burn].

        Raises:
            ValueError: if fewer than 2 non-NaN samples after empty-frame fill.
        """
        path = self._traj_paths[clip_id]
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)  # list of frames

        y = np.empty(len(raw), dtype=float)
        for i, frame in enumerate(raw):
            # A6 defensive guard: access region_0 = frame[0] if frame has >=1 region.
            region_0 = frame[0] if len(frame) > 0 else []
            if region_0:
                # Region 0 has exactly 1 bbox: [x1, y1, x2, y2, conf].
                # y_center = (y1 + y2) / 2 per ReadMe formula (RESEARCH §3 / Pitfall 1).
                x1, y1, x2, y2, conf = region_0[0]
                y[i] = (y1 + y2) / 2.0
            else:
                # Empty region-0 frame (detection gap) — write NaN sentinel;
                # interpolated below. (Pitfall 3 / RESEARCH §3 "~40% have some empty frames")
                y[i] = np.nan

        # Interpolate NaN sentinels with linear interpolation (reuses Squat's np.isnan path).
        nan_mask = np.isnan(y)
        if nan_mask.any():
            valid = ~nan_mask
            if int(valid.sum()) < 2:
                raise ValueError(
                    f"trajectory {clip_id}: <2 non-NaN samples ({int(valid.sum())})"
                )
            idx = np.arange(len(y))
            y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
        return y

    def _augment(self, clip_u8: torch.Tensor) -> torch.Tensor:
        """Apply the SSL augmentations per-branch (§3.2/§7) — independent default-RNG draws per call.

        ``strong_augs=False`` reproduces the SAFE-CORE set (weak-aug ablation point).
        ``strong_augs=True`` is the paper-faithful set: each aug is applied INDEPENDENTLY
        with probability ``aug_prob`` so the anchor and its positive are genuinely different
        views — preventing trivial-positive contrastive collapse (Phase 4 established this).
        Rotation is gated behind ``use_rotation`` (default OFF).
        """
        if not self.strong_augs:
            out = ssl_augs.temporal_shift(clip_u8)
            out = ssl_augs.horizontal_flip(out)
            out = ssl_augs.top_mask(out)
            out = ssl_augs.color_jitter(out)
            return out

        # md_pretrain v2 paper-faithful strong set — each applied independently with prob aug_prob.
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
        # 1. Split into descent/ascent. bottom_is_argmax=False (ARGMIN) — OHP barbell at
        #    overhead = highest in image = lowest y pixel = argMIN. Same sign as Squat
        #    (also argMIN) for different physical reason. (RESEARCH §3 / D6-reuse / Pitfall 2)
        descent_idx, ascent_idx = split_half_cycles(
            traj_y, frames_per_half=self.frames_per_half, bottom_is_argmax=False,
        )
        # 2. Decode the two half-cycles (1:1 traj->frame [ASSUMED, confirmed at probe]).
        video_path = os.path.join(self.videos_root, f"{clip_id}.mp4")
        descent_u8 = decode_clip(video_path, torch.as_tensor(descent_idx, dtype=torch.long))
        ascent_u8  = decode_clip(video_path, torch.as_tensor(ascent_idx,  dtype=torch.long))
        # 3. anchor + positive = two independent augmented views of the DESCENT;
        #    negative = an augmented view of the ASCENT (RESEARCH §3 / paper §3.2).
        anchor_u8   = self._augment(descent_u8)
        positive_u8 = self._augment(descent_u8)
        negative_u8 = self._augment(ascent_u8)
        # 4. Temporal-reverse either {anchor,positive} OR {negative} so the GLOBAL down/up
        #    motion is identical across all three (RESEARCH §3). Coin flip on default RNG.
        if float(torch.rand(1).item()) < 0.5:
            anchor_u8   = anchor_u8.flip(0)
            positive_u8 = positive_u8.flip(0)
        else:
            negative_u8 = negative_u8.flip(0)
        # 5. Spatial pipeline -> [3,16,112,112] float32 Kinetics-normalized.
        return {
            "anchor":   self._spatial_fn(anchor_u8,   crop_size=self.crop_size),
            "positive": self._spatial_fn(positive_u8, crop_size=self.crop_size),
            "negative": self._spatial_fn(negative_u8, crop_size=self.crop_size),
        }


def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker (PyTorch reproducibility idiom; pairs with persistent_workers, D7)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(0)  # 1 cv2 thread per worker; the DataLoader provides parallelism
    except ImportError:
        pass


def build_ssl_loader(dataset: OHPSSLDataset, config: "MDConfig", *, seed: int = 42) -> DataLoader:
    """Construct the SSL DataLoader (shuffle=True; persistent_workers when num_workers>0, D7).

    Args:
        dataset: a constructed ``OHPSSLDataset``.
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
