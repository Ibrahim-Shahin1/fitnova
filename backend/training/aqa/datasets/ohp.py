"""OHPElbowsKneesDataset + build_loaders factory — joint multi-label dataset over the official OHP splits.

Yields `(video[3, T, H, W] float32, labels[2] float32)` from official train/val/test splits.
Computes and exposes `pos_weight` (Tensor of length 2 for Elbows/Knees) at construction,
sourced from the **train** split regardless of which split this instance represents — the
`BCEWithLogitsLoss(pos_weight=...)` uses the train-derived weights for all loaders.
Measured class balance: Elbows+ 25.7% (w ≈ 2.887), Knees+ 34.2% (w ≈ 1.924).

Joint multi-label is justified by the shared MD-SSL representation plus efficiency:
OHP errors are largely independent (only 85/407 Elbow+ are also Knees+), so NO
co-occurrence lift is expected — each head learns its own signal independently.
BCEWithLogitsLoss with per-output pos_weight handles independent multi-label correctly.

`build_loaders(...)` returns `{"train": DataLoader, "val": DataLoader, "test": DataLoader}`.
**num_workers=0 is clamped** in the convenience factory — trainers use `_build_dataloaders`
directly with num_workers>0 for production runs.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
import torchvision.io
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import splits
from backend.training.aqa.datasets.transforms import (
    count_frames,
    decode_clip,
    decode_clip_cached,
    spatial_train,
    spatial_val,
    uniform_sample_indices,
)

logger = logging.getLogger("aqa.phase06")


# ──────────────────────────────────────────────────────────────────────────────
# pos_weight helper (train-derived, same formula as squat._compute_pos_weight)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_pos_weight(train_records: list[splits.OHPClipRecord]) -> torch.Tensor:
    """Compute `pos_weight = (N - pos) / max(pos, 1)` for Elbows and Knees from train records.

    Returns:
        Float32 tensor of shape [2] — `[w_elbows, w_knees]`. Expected values:
        Elbows ≈ 2.887, Knees ≈ 1.924 (tolerance ±0.05, from 1582 train clips).
    """
    n_total = len(train_records)
    pos_elbows = sum(r.label_elbows for r in train_records)
    pos_knees  = sum(r.label_knees  for r in train_records)
    w_elbows = (n_total - pos_elbows) / max(pos_elbows, 1)
    w_knees  = (n_total - pos_knees)  / max(pos_knees,  1)
    logger.info(
        "pos_weight from train (N=%d): Elbows+ %d -> w=%.4f, Knees+ %d -> w=%.4f",
        n_total, pos_elbows, w_elbows, pos_knees, w_knees,
    )
    return torch.tensor([w_elbows, w_knees], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class (mirrors SquatKIEKFEDataset; labels elbows/knees)
# ──────────────────────────────────────────────────────────────────────────────


class OHPElbowsKneesDataset(Dataset):
    """Joint multi-label OHP Elbows/Knees dataset over the official splits.

    On each `__getitem__`:
      1. Cheap `read_video_timestamps` probe to learn the clip's frame count.
      2. `uniform_sample_indices` picks `num_frames` frame positions (with optional
         ±jitter on train).
      3. `decode_clip` window-bounded decodes those frames.
      4. `spatial_train` (train) or `spatial_val` (val/test) applies resize + crop +
         Kinetics norm.
      5. Returns `(clip[3, T, crop_size, crop_size] float32, labels[2] float32)` where
         `labels = [label_elbows, label_knees]`.

    The joint 2-output head is justified by the shared MD-SSL representation plus
    efficiency. Errors are largely independent (85/407 co-occur); each head converges
    on its own signal.

    Determinism: jitter (in `uniform_sample_indices`) and random-crop offsets (in
    `spatial_train`) pull from torch's default RNG (`generator=None`). Harness
    `capture_rng_state` / `restore_rng_state` handles epoch-boundary resume (same
    contract as `SquatKIEKFEDataset`).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        *,
        drive_root: str,
        videos_root: str,
        num_frames: int = 32,
        crop_size: int = 112,
        train_aug: bool = True,
        train_jitter_frames: int = 2,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> None:
        self.split = split
        self.drive_root = drive_root
        self.videos_root = videos_root
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.train_aug = train_aug
        self.train_jitter_frames = train_jitter_frames
        self.seed = seed
        # Cache only helps the deterministic (train_aug=False) path; jittered train
        # resamples frames per epoch, so it is never cached.
        self.cache_dir = cache_dir if not train_aug else None

        # Records for THIS split (val_dataset has val records, etc.).
        self.records: list[splits.OHPClipRecord] = splits.index_ohp(
            split, drive_root=drive_root, videos_root=videos_root,
        )

        # pos_weight always derives from TRAIN, regardless of which split this instance
        # represents — so val_dataset.pos_weight == train_dataset.pos_weight, and the
        # trainer can build a single loss with weights computed once.
        if split == "train":
            train_records = self.records
        else:
            train_records = splits.index_ohp(
                "train", drive_root=drive_root, videos_root=videos_root,
            )
        self.pos_weight: torch.Tensor = _compute_pos_weight(train_records)

        # Cache the spatial pipeline choice so __getitem__ stays tight.
        self._spatial_fn: Callable[..., torch.Tensor] = (
            spatial_train if train_aug else spatial_val
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # Cheap header probe — no frames decoded here. count_frames is version-robust:
        # read_video_timestamps on torchvision <0.26, cv2 CAP_PROP_FRAME_COUNT on >=0.26.
        num_frames = count_frames(rec.video_path)

        # Index sampling — jitter only on train. generator=None so torch's default RNG
        # drives jitter; harness capture/restore covers it across resume.
        jitter = self.train_jitter_frames if self.train_aug else 0
        indices = uniform_sample_indices(
            num_frames,
            target=self.num_frames,
            jitter=jitter,
            generator=None,
        )

        # Window-bounded decode then spatial pipeline. cache_dir is set only on the
        # deterministic path (train_aug=False), where the 32 indices are stable per clip.
        if self.cache_dir:
            clip_tchw = decode_clip_cached(self.cache_dir, f"{rec.clip_id}_lbl{self.num_frames}", rec.video_path, indices)
        else:
            clip_tchw = decode_clip(rec.video_path, indices)
        clip = self._spatial_fn(clip_tchw, crop_size=self.crop_size)

        # labels=[label_elbows, label_knees] — the joint 2-output head target.
        label = torch.tensor([rec.label_elbows, rec.label_knees], dtype=torch.float32)
        return clip, label


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory (notebook smoke only — trainers use _build_dataloaders)
# ──────────────────────────────────────────────────────────────────────────────


def build_loaders(
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    """Build the three DataLoaders against the official OHP splits.

    Convenience / notebook-smoke factory only — trainers use `_build_dataloaders`
    directly with `num_workers>0` for production runs. This factory clamps
    `num_workers=0` (same contract as `squat.build_loaders`).

    Args:
        batch_size:     batch size for all three loaders.
        num_workers:    clamped to 0 (convenience factory; trainers use num_workers>0).
        dataset_kwargs: forwarded to `OHPElbowsKneesDataset(**dataset_kwargs)` for each
                        split. Must include `drive_root` and `videos_root`.

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"` mapping to `DataLoader` instances.
        - train: shuffle=True, drop_last=False
        - val / test: shuffle=False, drop_last=False
    """
    if num_workers != 0:
        logger.warning(
            "num_workers=%d overridden to 0 (convenience factory — trainers use "
            "_build_dataloaders directly with num_workers>0)",
            num_workers,
        )

    # persistent_workers: clamp=0 here, so _persistent is always False.
    _persistent = 0 > 0  # always False; guard is here for clarity

    train_ds = OHPElbowsKneesDataset(split="train", train_aug=True,  **dataset_kwargs)
    val_ds   = OHPElbowsKneesDataset(split="val",   train_aug=False, **dataset_kwargs)
    test_ds  = OHPElbowsKneesDataset(split="test",  train_aug=False, **dataset_kwargs)

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=0,
        ),
        "val": DataLoader(
            val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        ),
        "test": DataLoader(
            test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        ),
    }
