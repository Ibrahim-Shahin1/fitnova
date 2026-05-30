"""CVCSPC phase-contrastive SSL dataset over the unlabeled Back-Squat frames + bar trajectories.

The triplet pairs frames at the same bar-trajectory phase from two different reps (anchor from
clip v0, positive from clip v1) and a frame a phase-gap away on v1 (negative). Phase is the
per-clip bar-trajectory y-position normalized to [0,1] then scaled to degrees (×360). The official
NaN-trajectory clips (traj_nan.json) and any degenerate (empty / all-NaN / constant) trajectory are
excluded at construction so phase computation never divides by zero. Masking is the only active
augmentation (top 40-50% of rows blacked out with ~50% probability); all other augmentations are
commented out in the shipped code.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("aqa.phase07")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _traj2phase(traj) -> np.ndarray:
    """Normalize a per-clip bar-trajectory (y-position) to [0,1] then scale to degrees ([0,360]).

    NaN values are linearly interpolated. Raises ValueError on a degenerate trajectory
    (empty, fewer than 2 valid samples, or constant) so the clip is excluded rather than
    producing a divide-by-zero / flat phase.
    """
    y = np.asarray(traj, dtype=float)
    if y.ndim == 2 and y.shape[1] >= 2:
        y = y[:, 1]
    y = y.reshape(-1)
    if y.size < 2:
        raise ValueError("degenerate trajectory: <2 samples")
    nan_mask = np.isnan(y)
    if nan_mask.any():
        valid = ~nan_mask
        if int(valid.sum()) < 2:
            raise ValueError("degenerate trajectory: <2 non-NaN samples")
        idx = np.arange(len(y))
        y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
    y_min = float(y.min())
    span = float(y.max()) - y_min
    if span < 1e-8:
        raise ValueError("degenerate trajectory: constant")
    return (y - y_min) / span * 360.0


class ShallowSquatSSLDataset(Dataset):
    """Phase-matched cross-rep triplet dataset over the unlabeled Squat frame dirs + trajectories.

    __getitem__ returns ``{"anchor", "positive", "negative"}`` — three float32 [3,224,224]
    ImageNet-normalized frames. No labels.
    """

    def __init__(
        self,
        *,
        frames_root: str,
        trajectories_root: str,
        traj_nan_path: str | None = None,
        ssl_contrastive_phase_gap: float = 30.0,
        mask_prob: float = 0.5,
        mask_amt_lo: float = 0.4,
        mask_amt_hi: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.frames_root = Path(frames_root)
        self.phase_gap = ssl_contrastive_phase_gap
        self.mask_prob = mask_prob
        self.mask_amt_lo = mask_amt_lo
        self.mask_amt_hi = mask_amt_hi
        self.seed = seed

        self._traj_paths: dict[str, Path] = {
            p.stem: p for p in Path(trajectories_root).rglob("*.json")
        }
        frame_ids = {p.name for p in self.frames_root.iterdir() if p.is_dir()}

        if traj_nan_path is not None and Path(traj_nan_path).exists():
            nan_stems = {Path(x).stem for x in json.loads(Path(traj_nan_path).read_text(encoding="utf-8"))}
        else:
            nan_stems = set()
            if traj_nan_path is not None:
                logger.warning("traj_nan exclusion file not found: %s", traj_nan_path)
            else:
                logger.warning("no traj_nan_path provided; relying on the degenerate-trajectory guard")

        candidates = sorted((frame_ids & set(self._traj_paths)) - nan_stems)
        self._clip_ids: list[str] = []
        dropped = 0
        for cid in candidates:
            try:
                self._load_phase(cid)
            except (ValueError, json.JSONDecodeError, OSError):
                dropped += 1
                continue
            self._clip_ids.append(cid)

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        logger.info(
            "ShallowSquatSSLDataset: %d clips (excluded %d traj_nan, %d degenerate)",
            len(self._clip_ids), len(nan_stems), dropped,
        )

    def __len__(self) -> int:
        return len(self._clip_ids)

    def _load_phase(self, clip_id: str) -> np.ndarray:
        with open(self._traj_paths[clip_id], encoding="utf-8") as fh:
            return _traj2phase(json.load(fh))

    def _frame_files(self, clip_id: str) -> list[str]:
        return sorted(os.listdir(self.frames_root / clip_id))

    def _load_frame(self, clip_id: str, frame_idx: int) -> torch.Tensor:
        files = self._frame_files(clip_id)
        frame_idx = max(0, min(frame_idx, len(files) - 1))
        img = Image.open(self.frames_root / clip_id / files[frame_idx]).convert("RGB")
        return self.transform(img)

    def _mask(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.mask_prob:
            h = img.shape[1]
            k = int(h * random.uniform(self.mask_amt_lo, self.mask_amt_hi))
            img[:, :k, :] = 0
        return img

    def _select_triplet(self, idx: int) -> dict:
        v0 = self._clip_ids[idx]
        j = idx
        if len(self._clip_ids) > 1:
            while j == idx:
                j = random.randint(0, len(self._clip_ids) - 1)
        v1 = self._clip_ids[j]

        phase0 = self._load_phase(v0)
        phase1 = self._load_phase(v1)
        lo = max(float(phase0.min()), float(phase1.min()))
        hi = min(float(phase0.max()), float(phase1.max()))
        if hi <= lo:
            lo, hi = float(phase0.min()), float(phase0.max())
        p_anchor = random.uniform(lo, hi)

        anchor_idx = int(np.argmin(np.abs(phase0 - p_anchor)))
        positive_idx = int(np.argmin(np.abs(phase1 - p_anchor)))
        cand = np.flatnonzero(np.abs(phase1 - p_anchor) >= self.phase_gap)
        if cand.size > 0:
            negative_idx = int(cand[random.randint(0, cand.size - 1)])
        else:
            negative_idx = int(np.argmax(np.abs(phase1 - p_anchor)))

        return {
            "v0": v0, "v1": v1,
            "anchor_idx": anchor_idx, "positive_idx": positive_idx, "negative_idx": negative_idx,
            "p_anchor": p_anchor, "phase0": phase0, "phase1": phase1,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sel = self._select_triplet(idx)
        anchor = self._mask(self._load_frame(sel["v0"], sel["anchor_idx"]))
        positive = self._mask(self._load_frame(sel["v1"], sel["positive_idx"]))
        negative = self._mask(self._load_frame(sel["v1"], sel["negative_idx"]))
        return {"anchor": anchor, "positive": positive, "negative": negative}


def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker (PyTorch reproducibility idiom; pairs with persistent_workers)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def build_cvcspc_loader(dataset: ShallowSquatSSLDataset, config, *, seed: int = 42) -> DataLoader:
    """Construct the CVCSPC SSL DataLoader (shuffle=True; drop_last=True; persistent workers when N>0)."""
    g = torch.Generator()
    g.manual_seed(seed)
    _persistent = config.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=True,
        persistent_workers=_persistent,
    )
