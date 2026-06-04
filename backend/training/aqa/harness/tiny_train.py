"""Toy 3D-conv model + multi-epoch training loop wired to the Colab harness — drives the end-to-end smoke validation.

The toy model is intentionally minimal — `Conv3d(3, 4, 3, padding=1) → ReLU →
AdaptiveAvgPool3d(1) → Flatten → Linear(4, 2)` — its only job is exercising the
shape contract (input `[B, 3, 32, 112, 112]` → output `[B, 2]`), the
forward / loss / backward / optimizer step / atomic checkpoint write / RNG restore /
resumed-forward-pass cycle. No learning, no accuracy.

`run_tiny_epoch(*, run_name, ..., resume=True, max_epochs=1) -> dict` is the single
public entrypoint. With `max_epochs >= 2` and `resume=True`, the bitwise
assertion test pattern is supported: a fresh `max_epochs=2` run produces a 2-epoch
loss trajectory; a resumed-from-epoch-0 run training the same epoch 1 produces a
**bitwise-identical** epoch-1 loss trajectory.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)

logger = logging.getLogger("aqa.phase02")


class TinyModel(nn.Module):
    """`Conv3d(3, 4, 3, padding=1) → ReLU → AdaptiveAvgPool3d(1) → Flatten → Linear(4, 2)`.

    Outputs `[B, 2]` logits — joint multi-label head over `(KIE, KFE)`. Intentionally
    minimal but shape-strict: a wrong input shape will throw at the `Conv3d` step (3
    input channels expected) or at the final pool/linear collapse.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)        # (B, 4, 1, 1, 1)
        x = self.flatten(x)     # (B, 4)
        return self.fc(x)       # (B, 2)


def _set_global_seed(seed: int) -> None:
    """Seed all 4 RNG sources. Determinism precondition 2."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_tiny_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_videos",
    seed: int = 42,
    max_train_clips: int = 16,
    max_val_clips: int = 4,
    batch_size: int = 2,
    crop_size: int = 112,
    num_frames: int = 32,
    learning_rate: float = 1e-4,
    resume: bool = True,
    max_epochs: int = 1,
) -> dict:
    """Train through epoch `max_epochs - 1` from a checkpointed prior state (or fresh).

    Args:
        run_name:        Logical run identifier. Checkpoint dir is
                         `{drive_root}/FitNova/checkpoints/phase02/{run_name}/`.
        drive_root:      Drive MyDrive path forwarded to datasets.
        videos_root:     Local extracted videos dir forwarded to datasets.
        seed:            Global RNG seed (Python random / NumPy / torch CPU / CUDA).
        max_train_clips: Clamp train Subset to this many clips.
        max_val_clips:   Same for val.
        batch_size:      DataLoader batch size.
        crop_size:       Spatial crop (default 112).
        num_frames:      Temporal sample target (default 32).
        learning_rate:   Adam learning rate (CVCSPC default 1e-4).
        resume:          If True, call `load_latest_checkpoint` and continue from
                         `payload['epoch'] + 1`.
        max_epochs:      Train through epoch `max_epochs - 1` (exclusive range).
                         If start_epoch >= max_epochs, log and early-return.

    Returns:
        Dict with the contract keys:
          `epoch`                — last completed epoch this call (or start-1 on early-return)
          `train_loss_per_batch` — last epoch's per-batch BCE losses (or [] on early-return)
          `val_loss_mean`        — last epoch's mean val BCE (or nan on early-return)
          `checkpoint_path`      — path to the LAST checkpoint written this call
          `config_hash`          — 16-char sha256 of the hashed-config keys
    """
    _set_global_seed(seed)

    # Hash-relevant config (excludes run_name, paths, max_epochs — those vary across
    # resume/restart cycles without invalidating the underlying recipe).
    config = {
        "seed": seed,
        "max_train_clips": max_train_clips,
        "max_val_clips": max_val_clips,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "num_frames": num_frames,
        "learning_rate": learning_rate,
        "model": "TinyModel(Conv3d3-4-3,ReLU,AdaptiveAvgPool3d1,Linear4-2)",
        "loss": "BCEWithLogitsLoss(no pos_weight)",  # tiny smoke config: no pos_weight
    }
    config_hash_str = hash_config(config)

    run_dir = os.path.join(drive_root, "FitNova/checkpoints/phase02", run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)

    # Build full datasets then Subset-wrap to clamp wall time.
    train_ds_full = SquatKIEKFEDataset(
        split="train", drive_root=drive_root, videos_root=videos_root,
        num_frames=num_frames, crop_size=crop_size, train_aug=True, seed=seed,
    )
    val_ds_full = SquatKIEKFEDataset(
        split="val", drive_root=drive_root, videos_root=videos_root,
        num_frames=num_frames, crop_size=crop_size, train_aug=False, seed=seed,
    )

    # Subset preserves the rest of the pipeline (decode, sampler, transforms)
    # but caps the number of real videos touched per epoch.
    train_subset = Subset(train_ds_full, range(min(max_train_clips, len(train_ds_full))))
    val_subset = Subset(val_ds_full, range(min(max_val_clips, len(val_ds_full))))
    # num_workers=0 — bitwise-determinism precondition.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()  # tiny smoke config: no pos_weight

    # Resume branch
    metrics_history: list[dict] = []
    start_epoch = 0
    if resume:
        loaded = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str)
        if loaded is not None:
            model.load_state_dict(loaded["model_state_dict"])
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
            restore_rng_state(loaded["rng_state"])
            metrics_history = list(loaded["metrics_history"])  # copy so we don't mutate the loaded dict
            start_epoch = int(loaded["epoch"]) + 1
            logger.info(
                "resumed from epoch_%03d.pt; start_epoch=%d",
                int(loaded["epoch"]), start_epoch,
            )

    if start_epoch >= max_epochs:
        logger.info(
            "resumed from epoch_%03d.pt; max_epochs=%d already reached; exiting without training",
            start_epoch - 1, max_epochs,
        )
        return {
            "epoch": start_epoch - 1,
            "train_loss_per_batch": [],
            "val_loss_mean": float("nan"),
            "checkpoint_path": os.path.join(run_dir, f"epoch_{start_epoch-1:03d}.pt"),
            "config_hash": config_hash_str,
        }

    # Multi-epoch loop
    last_train_losses: list[float] = []
    last_val_mean: float = float("nan")
    last_ckpt_path: str = ""

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_losses: list[float] = []
        for batch_idx, (clip, label) in enumerate(train_loader):
            clip = clip.to(device)
            label = label.to(device)
            logits = model(clip)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            train_losses.append(loss_val)
            logger.info("epoch=%d batch=%d train_loss=%.6f", epoch, batch_idx, loss_val)

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for clip, label in val_loader:
                clip = clip.to(device)
                label = label.to(device)
                logits = model(clip)
                val_losses.append(float(loss_fn(logits, label).item()))
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        logger.info("epoch=%d val_loss_mean=%.6f", epoch, val_mean)

        # Metrics history schema
        metrics_history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(train_losses)),
            "train_loss_per_batch": train_losses,
            "val_loss_mean": val_mean,
        })

        # Checkpoint payload schema
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,  # no scheduler in the tiny smoke config
            "rng_state": capture_rng_state(),
            "metrics_history": metrics_history,
            "config_hash": config_hash_str,
            "config_repr": config,
            "code_version": "phase02-tiny",
        }
        ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, ckpt_path)
        prune_checkpoints(run_dir, keep_last=3, keep_best=False)

        last_train_losses = train_losses
        last_val_mean = val_mean
        last_ckpt_path = ckpt_path

    return {
        "epoch": max_epochs - 1,
        "train_loss_per_batch": last_train_losses,
        "val_loss_mean": last_val_mean,
        "checkpoint_path": last_ckpt_path,
        "config_hash": config_hash_str,
    }
