"""SquatKIEKFEDataset + build_loaders factory — joint multi-label dataset over the official Squat splits.

Yields `(video[3, T, H, W] float32, labels[2] float32)` from official train/val/test splits.
Computes and exposes `pos_weight` (Tensor of length 2 for KIE/KFE) at construction, sourced
from the **train** split regardless of which split this instance represents — Phase 3's
`BCEWithLogitsLoss(pos_weight=...)` uses the train-derived weights for all loaders.

Phase 1 measured KIE = 14.29% positive (severe imbalance, w ≈ 6.0) and KFE = 68.33%
positive (mild reverse imbalance, w ≈ 0.47). Joint multi-label is justified by Phase 1's
co-occurrence finding: 192/232 KIE+ clips are also KFE+.

`build_loaders(...)` returns `{"train": DataLoader, "val": DataLoader, "test": DataLoader}`.
**Phase 2 constraint: `num_workers=0` is hardcoded** — the bitwise-determinism guarantee
(D14, Task 14) requires single-worker access. Phase 3 may bump it with a `worker_init_fn`.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 6 / D9 / D14 / interfaces block.
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

logger = logging.getLogger("aqa.phase02")


def _compute_pos_weight(train_records: list[splits.ClipRecord]) -> torch.Tensor:
    """Compute `pos_weight = (N - pos) / max(pos, 1)` for KIE and KFE from train records.

    Returns:
        Float32 tensor of shape [2] — `[w_KIE, w_KFE]`. Used by Phase 3's
        `BCEWithLogitsLoss(pos_weight=...)`. Phase 1 expected values: KIE ≈ 6.0,
        KFE ≈ 0.47 (tolerance ±0.05 per Task 6 acceptance).
    """
    n_total = len(train_records)
    pos_kie = sum(r.label_kie for r in train_records)
    pos_kfe = sum(r.label_kfe for r in train_records)
    w_kie = (n_total - pos_kie) / max(pos_kie, 1)
    w_kfe = (n_total - pos_kfe) / max(pos_kfe, 1)
    logger.info(
        "pos_weight from train (N=%d): KIE+ %d -> w=%.4f, KFE+ %d -> w=%.4f",
        n_total, pos_kie, w_kie, pos_kfe, w_kfe,
    )
    return torch.tensor([w_kie, w_kfe], dtype=torch.float32)


class SquatKIEKFEDataset(Dataset):
    """Joint multi-label Squat KIE/KFE dataset over the official splits.

    On each `__getitem__`:
      1. Cheap `read_video_timestamps` probe to learn the clip's frame count.
      2. `uniform_sample_indices` picks 32 frame positions (with optional ±jitter on train).
      3. `decode_clip` window-bounded decodes those frames.
      4. `spatial_train` (train) or `spatial_val` (val/test) applies resize + crop + Kinetics norm.
      5. Returns `(clip[3, T, crop_size, crop_size] float32, labels[2] float32)`.

    Determinism: jitter (in `uniform_sample_indices`) and random-crop offsets (in
    `spatial_train`) pull from **torch's default RNG** (`generator=None`). Torch's
    default RNG is captured/restored by `harness.colab.capture_rng_state` /
    `restore_rng_state`, which is what makes Task 14's bitwise-resume assertion work
    — the resumed run restores torch's default RNG to end-of-epoch-0 state, so the
    dataset's subsequent sampling matches the fresh-run baseline exactly.

    **The `seed` kwarg is reserved (Phase 3+ may bind it to a per-instance generator
    when num_workers > 0).** In Phase 2 it's a no-op — the trainer's
    `_set_global_seed(seed)` is what actually controls determinism.
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
        self.cache_dir = cache_dir if not train_aug else None

        # Records for THIS split (val_dataset has val records, etc.).
        self.records: list[splits.ClipRecord] = splits.index(
            split, drive_root=drive_root, videos_root=videos_root,
        )

        # Phase 2 constraint: pos_weight always derives from TRAIN, regardless of which
        # split this instance represents. So val_dataset.pos_weight == train_dataset.pos_weight,
        # and Phase 3 can build a single loss with weights computed once.
        if split == "train":
            train_records = self.records
        else:
            train_records = splits.index(
                "train", drive_root=drive_root, videos_root=videos_root,
            )
        self.pos_weight: torch.Tensor = _compute_pos_weight(train_records)

        # Phase 2: NO per-instance generator (reserved for Phase 3+ with num_workers > 0).
        # __getitem__ uses torch's default RNG (generator=None) so harness.colab's
        # capture_rng_state / restore_rng_state can snapshot and replay sampling state
        # across resume boundaries — load-bearing for Task 14's bitwise assertion.

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

        # Index sampling — jitter only on train. generator=None so torch's default
        # RNG drives jitter; harness capture/restore covers it across resume.
        jitter = self.train_jitter_frames if self.train_aug else 0
        indices = uniform_sample_indices(
            num_frames,
            target=self.num_frames,
            jitter=jitter,
            generator=None,
        )

        # Window-bounded decode (F11) then spatial pipeline. cache_dir is set only on the
        # deterministic path (train_aug=False).
        if self.cache_dir:
            clip_tchw = decode_clip_cached(self.cache_dir, f"{rec.clip_id}_lbl{self.num_frames}", rec.video_path, indices)
        else:
            clip_tchw = decode_clip(rec.video_path, indices)
        clip = self._spatial_fn(clip_tchw, crop_size=self.crop_size)

        label = torch.tensor([rec.label_kie, rec.label_kfe], dtype=torch.float32)
        return clip, label


def build_loaders(
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    """Build the three DataLoaders against the official splits.

    Args:
        batch_size:     batch size for all three loaders (Phase 2 tiny: 2).
        num_workers:    **clamped to 0 in Phase 2** (D14). A caller passing >0 gets a
                        warning + the 0 override. Phase 3 can lift this with a
                        `worker_init_fn` that re-seeds per-worker RNGs.
        dataset_kwargs: forwarded to `SquatKIEKFEDataset(**dataset_kwargs)` for each split.
                        Must include `drive_root` and `videos_root`. Caller-controllable:
                        `num_frames`, `crop_size`, `train_jitter_frames`, `seed`.

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"` mapping to `DataLoader` instances.
        - train: shuffle=True, drop_last=False
        - val / test: shuffle=False, drop_last=False
    """
    # Phase 2 constraint: num_workers = 0 (D14 bitwise-determinism precondition 5).
    if num_workers != 0:
        logger.warning(
            "num_workers=%d overridden to 0 (Phase 2 D14 constraint — bitwise-determinism "
            "requires single-worker access; Phase 3 may lift this with worker_init_fn)",
            num_workers,
        )

    train_ds = SquatKIEKFEDataset(split="train", train_aug=True, **dataset_kwargs)
    val_ds = SquatKIEKFEDataset(split="val", train_aug=False, **dataset_kwargs)
    test_ds = SquatKIEKFEDataset(split="test", train_aug=False, **dataset_kwargs)

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        ),
    }
