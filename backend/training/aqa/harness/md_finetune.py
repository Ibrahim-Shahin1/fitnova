"""Fine-tune trainer for the MD-pretrained backbone.

A near-exact parallel of ``supervised_train.run_supervised_epoch`` with exactly
four deltas:
  1. ``build_finetune_model`` loads the MD backbone from ``backbone.pt``
     (``strict=False``), NOT Kinetics weights.
  2. Optimizer is ``torch.optim.AdamW`` (NOT Adam) — true decoupled weight decay.
  3. The head is ``nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))`` (head-only dropout).
  4. A runtime overfit monitor aborts the seed if the train/val BCE loss ratio
     exceeds 10x before epoch 10 (an earlier run hit 32x by epoch 8).

Everything else (the labeled ``SquatKIEKFEDataset`` pipeline, ``_val_pass``,
``BCEWithLogitsLoss(pos_weight)``, the ``best.pt`` write contract, early-stop,
``prune_checkpoints``) reuses the supervised-training contracts unchanged.
``supervised_train.py`` is NOT edited — this is a new module.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
from backend.training.aqa.eval.metrics import f1_per_error, pr_auc_per_error
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)
# Reuse the dataloader / val-pass / seed helpers verbatim (do NOT edit
# supervised_train.py). _build_dataloaders + _val_pass are config-duck-typed — they read
# batch_size/num_workers/num_frames/crop_size/train_jitter_frames/flip_aug, all of which
# FinetuneConfig supplies — so a FinetuneConfig passes through unchanged.
from backend.training.aqa.harness.supervised_train import (
    _build_dataloaders,
    _set_global_seed,
    _val_pass,
)

logger = logging.getLogger("aqa.phase04")


@dataclasses.dataclass
class FinetuneConfig:
    """Fine-tune hyperparameters."""

    learning_rate: float = 1e-4          # paper specifies no separate downstream LR
    weight_decay: float = 1e-4           # AdamW decoupled wd
    dropout: float = 0.2                 # head-only dropout
    batch_size: int = 16                 # VRAM measured OK
    num_workers: int = 4                 # persistent_workers guard below
    max_epochs: int = 50                 # supervised-vs-finetune parity
    early_stop_patience: int = 8         # supervised-vs-finetune parity
    num_frames: int = 32                 # NOT the 16-frame half-cycle
    crop_size: int = 112
    train_jitter_frames: int = 2
    flip_aug: bool = False               # flip OFF
    scheduler_name: str = "cosine"
    model_arch: str = "r2plus1d_18_md_finetuned"
    loss_name: str = "BCEWithLogitsLoss(pos_weight=dataset.pos_weight)"
    md_backbone_path: str = ""           # path to backbone.pt from md_pretrain

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs


def build_finetune_model(md_backbone_path: str, *, dropout: float = 0.2) -> nn.Module:
    """Load the MD backbone (NOT Kinetics) + a fresh Dropout(0.2)+Linear(512,2) head.

    Constructs ``r2plus1d_18(weights=None)`` (architecture only), loads
    ``backbone_state_dict`` from the MD pretrain checkpoint with
    ``map_location='cpu'`` + ``strict=False`` (projector keys absent), asserts
    ``fc.in_features == 512``, then replaces ``fc`` with the dropout head.
    """
    from torchvision.models.video import r2plus1d_18

    # weights=None — NOT Kinetics. The MD-pretrained backbone IS the initialization;
    # loading Kinetics here would discard the SSL pretraining.
    model = r2plus1d_18(weights=None)
    assert model.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={model.fc.in_features}, expected 512"
    )
    # map_location='cpu' avoids the CUDA-ByteTensor landmine: CUDA ByteTensors break
    # set_rng_state_all. strict=False — the backbone-only state_dict has no fc/projector
    # keys (the MD checkpoint stored fc=Identity + a separate projector).
    ckpt = torch.load(md_backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    # head-only dropout before the joint KIE/KFE Linear(512, 2). Replaces fc AFTER
    # the backbone load so the random Kinetics-400 fc is discarded.
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))
    return model


def _d6_overfit_abort(
    epoch: int,
    train_loss_mean: float,
    val_loss_mean: float,
    *,
    ratio_threshold: float = 10.0,
    before_epoch: int = 10,
) -> bool:
    """Runtime overfit monitor — a pure, unit-testable decision.

    Overfitting drives the VAL loss far ABOVE the train loss, so the signal is val/train
    (an earlier run hit ~32x by epoch 8 with wd=0/no-dropout — val loss ~32x train). Returns
    True iff ``val_loss / train_loss`` exceeds ``ratio_threshold`` BEFORE ``before_epoch``.

    NOTE: writing this ratio as train/val would be INVERTED — train/val>10 flags under-fitting,
    never the catastrophic overfit it must catch. It is val/train here (the academically correct
    direction). The val-macro-F1 early-stop stays the primary guard; this is the fast
    catastrophic-overfit abort. Pure helper so it's testable without a GPU.
    """
    val_train_ratio = val_loss_mean / (train_loss_mean + 1e-9)
    return epoch < before_epoch and val_train_ratio > ratio_threshold


def run_md_finetune_epoch(
    *,
    run_name: str,
    md_backbone_path: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_videos",
    seed: int = 42,
    config: FinetuneConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    dataset_cls=SquatKIEKFEDataset,
    checkpoint_phase: str = "phase04",
) -> dict:
    """Fine-tune the MD backbone on labeled Squat KIE/KFE (parallels run_supervised_epoch + 4 deltas).

    Deltas vs supervised training:
      1. ``build_finetune_model(md_backbone_path)`` — MD-pretrained backbone init, NOT Kinetics.
      2. ``torch.optim.AdamW`` (true decoupled weight decay), NOT Adam.
      3. Head is ``Dropout(config.dropout) + Linear(512, 2)`` (inside ``build_finetune_model``).
      4. Runtime overfit monitor: val/train BCE ratio > 10x before epoch 10 -> abort the seed
         (early-return ``aborted_overfit=True``) so the notebook can bump reg + restart.

    Auto-resumes from ``{drive_root}/FitNova/checkpoints/phase04/{run_name}/latest.txt``. Writes
    ``epoch_NNN.pt`` every epoch + ``best.pt`` on val-macro-F1 improvement, with a payload whose
    schema matches the supervised checkpoints (``code_version`` = ``"phase04-md-finetune"``) so the
    same tools load it. Use ``max_epochs=1`` for the timing/VRAM probe before the full 50-epoch run.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    config = config or FinetuneConfig()
    effective_max_epochs = max_epochs if max_epochs is not None else config.max_epochs
    _set_global_seed(seed)

    # config_repr is the ordered hash + the fine-tune-specific keys (dropout, the MD backbone
    # path) so a resume with a different backbone / dropout correctly mismatches.
    config_repr = {
        "seed": seed,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "crop_size": config.crop_size,
        "num_frames": config.num_frames,
        "train_jitter_frames": config.train_jitter_frames,
        "flip_aug": config.flip_aug,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "scheduler_name": config.scheduler_name,
        "scheduler_t_max": config.scheduler_t_max,
        "model_arch": config.model_arch,
        "loss_name": config.loss_name,
        "md_backbone_path": md_backbone_path,
    }
    config_hash_str = hash_config(config_repr)

    run_dir = os.path.join(drive_root, "FitNova/checkpoints", checkpoint_phase, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run_dir: %s (config_hash=%s, md_backbone=%s)", run_dir, config_hash_str, md_backbone_path)

    loaders = _build_dataloaders(seed, config, drive_root, videos_root, dataset_cls=dataset_cls)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    pos_weight = train_loader.dataset.pos_weight  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_finetune_model(md_backbone_path, dropout=config.dropout).to(device)  # MD init + dropout head
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(  # AdamW (decoupled wd), NOT Adam
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)

    metrics_history: list[dict] = []
    best_f1_val = -1.0
    epochs_since_improve = 0
    start_epoch = 0

    if resume:
        # map_location='cpu' REQUIRED — CUDA ByteTensors break set_rng_state_all.
        prior = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str, map_location="cpu")
        if prior is not None:
            model.load_state_dict(prior["model_state_dict"])  # full fine-tuned model (backbone+head)
            optimizer.load_state_dict(prior["optimizer_state_dict"])
            if prior.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(prior["scheduler_state_dict"])
            restore_rng_state(prior["rng_state"])
            metrics_history = list(prior["metrics_history"])
            best_f1_val = float(prior.get("best_f1_val", -1.0))
            start_epoch = int(prior["epoch"]) + 1
            epochs_since_improve = 0
            for entry in reversed(metrics_history):
                if float(entry.get("val_macro_f1", -1.0)) >= best_f1_val:
                    break
                epochs_since_improve += 1
            logger.info("Resumed epoch_%03d.pt; start_epoch=%d best_f1_val=%.4f since_improve=%d",
                        int(prior["epoch"]), start_epoch, best_f1_val, epochs_since_improve)

    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if start_epoch >= effective_max_epochs:
        logger.info("Already at/past effective_max_epochs=%d (start_epoch=%d)", effective_max_epochs, start_epoch)
        return {
            "epoch": start_epoch - 1, "metrics_history": metrics_history,
            "best_f1_val": best_f1_val, "best_thresholds": None,
            "checkpoint_path": os.path.join(run_dir, f"epoch_{start_epoch - 1:03d}.pt"),
            "best_checkpoint_path": best_ckpt_path, "config_hash": config_hash_str,
            "aborted_overfit": False,
        }

    last_ckpt_path = ""
    aborted_overfit = False

    for epoch in range(start_epoch, effective_max_epochs):
        epoch_start_t = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        train_iter = (
            train_loader if tqdm is None
            else tqdm(train_loader, desc=f"seed{seed} epoch {epoch}/{effective_max_epochs - 1} train", leave=False)
        )
        for batch_idx, (clip, label) in enumerate(train_iter):
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(clip)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if batch_idx % 10 == 0:
                logger.info("seed=%d epoch=%d batch=%d train_loss=%.6f", seed, epoch, batch_idx, train_losses[-1])
        scheduler.step()  # cosine, per-epoch

        val_loss_mean, val_scores, val_labels = _val_pass(model, val_loader, criterion, device)
        val_pred = (val_scores >= 0.5).astype(int)  # 0.5 proxy for best.pt selection; thresholds swept later
        val_f1_kie = f1_per_error(val_labels[:, 0], val_pred[:, 0])
        val_f1_kfe = f1_per_error(val_labels[:, 1], val_pred[:, 1])
        val_pr_auc_kie = pr_auc_per_error(val_labels[:, 0], val_scores[:, 0])
        val_pr_auc_kfe = pr_auc_per_error(val_labels[:, 1], val_scores[:, 1])
        val_macro_f1 = (val_f1_kie + val_f1_kfe) / 2.0
        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        # Overfit signal = val/train (the "~32x" case): val loss diverging ABOVE train.
        val_train_loss_ratio = val_loss_mean / (train_loss_mean + 1e-9)
        epoch_wall_time_s = time.perf_counter() - epoch_start_t
        logger.info(
            "seed=%d epoch=%d train_loss=%.6f val_loss=%.6f val/train=%.2f val_macro_f1=%.4f "
            "(kie=%.4f kfe=%.4f) wall=%.1fs",
            seed, epoch, train_loss_mean, val_loss_mean, val_train_loss_ratio,
            val_macro_f1, val_f1_kie, val_f1_kfe, epoch_wall_time_s,
        )

        metrics_history.append({
            "epoch": epoch,
            "train_loss_mean": train_loss_mean,
            "train_loss_per_batch": train_losses,
            "val_loss_mean": val_loss_mean,
            "val_f1_kie": val_f1_kie,
            "val_f1_kfe": val_f1_kfe,
            "val_pr_auc_kie": val_pr_auc_kie,
            "val_pr_auc_kfe": val_pr_auc_kfe,
            "val_macro_f1": val_macro_f1,
            "val_train_loss_ratio": val_train_loss_ratio,  # overfit monitor evidence (val/train; >10 = overfit)
            "epoch_wall_time_s": epoch_wall_time_s,
        })

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "metrics_history": metrics_history,
            "best_f1_val": best_f1_val,
            "best_thresholds": None,  # filled by the threshold sweep
            "config_hash": config_hash_str,
            "config_repr": config_repr,
            "code_version": "phase04-md-finetune",  # distinct version tag, same checkpoint schema
        }
        ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, ckpt_path)
        last_ckpt_path = ckpt_path

        if val_macro_f1 > best_f1_val:
            best_f1_val = val_macro_f1
            payload["best_f1_val"] = best_f1_val
            atomic_save_checkpoint(payload, best_ckpt_path, update_latest=False)
            logger.info("seed=%d new best val_macro_f1=%.4f -> wrote best.pt", seed, best_f1_val)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        prune_checkpoints(run_dir, keep_last=3, keep_best=True)

        # Runtime overfit monitor — abort the seed (notebook bumps reg + restarts).
        if _d6_overfit_abort(epoch, train_loss_mean, val_loss_mean):
            logger.warning(
                "ABORT seed=%d epoch=%d: val/train loss ratio %.2f > 10.0 before epoch 10 — "
                "aborting seed; bump reg (wd=5e-4, dropout=0.3) + restart, lock the recipe.",
                seed, epoch, val_train_loss_ratio,
            )
            aborted_overfit = True
            break

        if epochs_since_improve >= config.early_stop_patience:
            logger.info("seed=%d early stop at epoch %d (best=%.4f, %d epochs no improve)",
                        seed, epoch, best_f1_val, epochs_since_improve)
            break

    return {
        "epoch": epoch,
        "metrics_history": metrics_history,
        "best_f1_val": best_f1_val,
        "best_thresholds": None,
        "checkpoint_path": last_ckpt_path,
        "best_checkpoint_path": best_ckpt_path,
        "config_hash": config_hash_str,
        "aborted_overfit": aborted_overfit,
    }
