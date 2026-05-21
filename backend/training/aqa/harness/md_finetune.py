"""Fine-tune trainer for the MD-pretrained backbone (SQUAT-05).

A near-exact parallel of Phase 3's ``supervised_train.run_supervised_epoch`` with
exactly four deltas (CONTEXT D3 + RESEARCH §9):
  1. ``build_finetune_model`` loads the MD backbone from Plan 02's ``backbone.pt``
     (``strict=False``), NOT Kinetics weights.
  2. Optimizer is ``torch.optim.AdamW`` (NOT Adam) — true decoupled weight decay.
  3. The head is ``nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))`` (head-only dropout).
  4. A D6 runtime overfit monitor aborts the seed if the train/val BCE loss ratio
     exceeds 10x before epoch 10 (Phase 3 hit 32x by epoch 8).

Everything else (the labeled ``SquatKIEKFEDataset`` pipeline, ``_val_pass``,
``BCEWithLogitsLoss(pos_weight)``, the ``best.pt`` write contract, early-stop,
``prune_checkpoints``) reuses the Phase 2/3 contracts unchanged (D9).
``supervised_train.py`` is NOT edited — this is a new module.

The training-loop body is finalized in Plan 03; ``build_finetune_model`` and
``FinetuneConfig`` are landed + unit-tested in Wave 0 (SQUAT-05-a).

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D3 / D6.
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

logger = logging.getLogger("aqa.phase04")


@dataclasses.dataclass
class FinetuneConfig:
    """Fine-tune hyperparameters — LOCKED by CONTEXT D3, confirmed by RESEARCH §9."""

    learning_rate: float = 1e-4          # [CITED §5/§9 — paper specifies no separate downstream LR]
    weight_decay: float = 1e-4           # [LOCKED D3 — AdamW decoupled wd]
    dropout: float = 0.2                 # [LOCKED D3 — head-only dropout]
    batch_size: int = 16                 # Phase 3 contract (VRAM measured OK)
    num_workers: int = 4                 # Phase 3 contract; persistent_workers guard (D7)
    max_epochs: int = 50                 # [LOCKED D3 — P3-vs-P4 parity]
    early_stop_patience: int = 8         # [LOCKED D3 — P3-vs-P4 parity]
    num_frames: int = 32                 # Phase 2/3 contract (NOT the 16-frame half-cycle)
    crop_size: int = 112                 # Phase 2/3 contract
    train_jitter_frames: int = 2         # Phase 3 contract
    flip_aug: bool = False               # [LOCKED Phase 2 D5 — flip OFF]
    scheduler_name: str = "cosine"
    model_arch: str = "r2plus1d_18_md_finetuned"
    loss_name: str = "BCEWithLogitsLoss(pos_weight=dataset.pos_weight)"
    md_backbone_path: str = ""           # path to backbone.pt from md_pretrain (Plan 02)

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs


def build_finetune_model(md_backbone_path: str, *, dropout: float = 0.2) -> nn.Module:
    """Load the MD backbone (NOT Kinetics) + a fresh Dropout(0.2)+Linear(512,2) head.

    Constructs ``r2plus1d_18(weights=None)`` (architecture only), loads
    ``backbone_state_dict`` from the MD pretrain checkpoint with
    ``map_location='cpu'`` + ``strict=False`` (projector keys absent), asserts
    ``fc.in_features == 512``, then replaces ``fc`` with the dropout head (D3 / §9).
    """
    from torchvision.models.video import r2plus1d_18

    # ANTI-PATTERN GUARD (§9): weights=None — NOT Kinetics. The MD-pretrained backbone
    # IS the initialization; loading Kinetics here would discard the SSL pretraining.
    model = r2plus1d_18(weights=None)
    assert model.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={model.fc.in_features}, expected 512"
    )
    # map_location='cpu' — D7 landmine (Phase 3 fix a0841b4: CUDA ByteTensors break
    # set_rng_state_all). strict=False — the backbone-only state_dict has no fc/projector
    # keys (the MD checkpoint stored fc=Identity + a separate projector).
    ckpt = torch.load(md_backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    # D3: head-only dropout before the joint KIE/KFE Linear(512, 2). Replaces fc AFTER
    # the backbone load so the random Kinetics-400 fc is discarded.
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))
    return model


def run_md_finetune_epoch(*args: Any, **kwargs: Any) -> dict:
    """One fine-tune epoch (parallels supervised_train.run_supervised_epoch + D6 monitor).

    Finalized in Plan 03.
    """
    raise NotImplementedError("Plan 03 — finalize; parallels supervised_train.run_supervised_epoch + D6 monitor")
