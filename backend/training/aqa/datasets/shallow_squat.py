"""ShallowSquatDataset + build_loaders — single-image Shallow-Squat depth-error dataset.

Yields ``(image[3, 224, 224] float32 ImageNet-normalized, label float32 scalar)`` from the
official Shallow-Squat splits (train/val/test = 2542/529/540). Labels are read directly from
the flat ``{id: 0/1}`` ``labels_shallow_depth.json`` (3611 crops, 43.9% positive). Shallow-Squat
is a SINGLE binary error ("shallow depth"), so ``pos_weight`` is a length-1 tensor and the label
is a scalar — unlike the two-head Squat KIE/KFE and OHP Elbows/Knees datasets.

Image pipeline: JPEG crop -> 224² -> ImageNet mean/std (the ResNet-18 native contract). No video
decode, no temporal axis, no Kinetics norm.

``build_loaders(...)`` returns ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("aqa.phase07")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _compute_pos_weight(train_records: list[tuple[str, int]]) -> torch.Tensor:
    """Compute ``pos_weight = (N - pos) / max(pos, 1)`` from train records.

    Returns a length-1 float32 tensor ``[w]`` — single binary head, so ``[w]`` not
    ``[w_kie, w_kfe]``. Shallow-Squat is ~44% positive, so ``w`` ≈ 1.28 (near-balanced).
    """
    n_total = len(train_records)
    pos = sum(label for _, label in train_records)
    w = (n_total - pos) / max(pos, 1)
    logger.info("pos_weight from train (N=%d): pos=%d -> w=%.4f", n_total, pos, w)
    return torch.tensor([w], dtype=torch.float32)


class ShallowSquatDataset(Dataset):
    """Single-image Shallow-Squat dataset over the official splits.

    ``pos_weight`` always derives from the train split regardless of which split this
    instance represents, so a single loss can be built once from any loader.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        *,
        images_root: str,
        labels_path: str,
        splits_root: str,
        train_aug: bool = True,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.seed = seed
        self.images_root = Path(images_root)

        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        ids = json.loads(Path(splits_root, f"{split}_ids.json").read_text(encoding="utf-8"))
        train_ids = json.loads(Path(splits_root, "train_ids.json").read_text(encoding="utf-8"))

        self.records: list[tuple[str, int]] = [(i, int(labels[i])) for i in ids if i in labels]
        train_records = [(i, int(labels[i])) for i in train_ids if i in labels]
        self.pos_weight: torch.Tensor = _compute_pos_weight(train_records)

        self.transform = (
            T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            if train_aug
            else T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        logger.info("ShallowSquatDataset[%s]: %d records", split, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        id_, label = self.records[idx]
        img = Image.open(self.images_root / f"{id_}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


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


def build_loaders(
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    """Build the three DataLoaders against the official Shallow-Squat splits.

    ``dataset_kwargs`` must include ``images_root``, ``labels_path``, ``splits_root``.
    train: shuffle=True; val/test: shuffle=False; drop_last=False on all.
    """
    _persistent = num_workers > 0
    train_ds = ShallowSquatDataset(split="train", train_aug=True, **dataset_kwargs)
    val_ds = ShallowSquatDataset(split="val", train_aug=False, **dataset_kwargs)
    test_ds = ShallowSquatDataset(split="test", train_aug=False, **dataset_kwargs)
    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=_persistent, worker_init_fn=seed_worker, drop_last=False,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            persistent_workers=_persistent, worker_init_fn=seed_worker, drop_last=False,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            persistent_workers=_persistent, worker_init_fn=seed_worker, drop_last=False,
        ),
    }
