"""Motion-Disentangling SSL pretraining trainer for Phase 4 (SQUAT-04).

Pretrains an R(2+1)D-18 backbone on the 4,970 unlabeled Squat clips with the
paper's half-cycle contrast pretext task. The loss is the triplet distance-ratio
loss (Hoffer & Ailon 2015 / paper Eq.1 §3.2 / official train_test.py:60-71) —
NOT NT-Xent (RESEARCH §2 CITED, overriding CONTEXT D2's provisional default):
one structural negative per anchor (the opposite half-cycle), no temperature, no
stop-gradient, no in-batch negatives. Batch 5-8 fits L4 (RESEARCH §5) — no
gradient accumulation. A linear-probe monitor (§6) tracks downstream convergence
and an embedding-std/effective-rank monitor (§12) detects representation
collapse, both written into the checkpoint history per epoch.

The training-loop bodies (``run_md_pretrain_epoch``, ``_linear_probe``,
collapse detection) are finalized in Plan 02, after the VRAM probe (§5) and the
probe-confirmed SSL dataset. The pure-function cores (``md_triplet_loss``,
``ProjectionHead``, ``build_md_model``) are implemented + unit-tested in Wave 0
(Tasks 5/6/8) BEFORE the 12-24h GPU burn.

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D2 / D7 / D8.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.training.aqa.datasets.squat_ssl import SquatSSLDataset, build_ssl_loader
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

# SSL checkpoint payload key set (RESEARCH §10). backbone.pt / best.pt writes use
# atomic_save_checkpoint(..., update_latest=False) so latest.txt is not clobbered (D7).
_SSL_CHECKPOINT_KEYS: tuple[str, ...] = (
    "epoch",
    "backbone_state_dict",
    "projector_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
    "metrics_history",
    "linear_probe_history",
    "config_hash",
    "config_repr",
    "code_version",
)


@dataclasses.dataclass
class MDConfig:
    """Hyperparameters for Phase 4 MD-SSL pretraining.

    Every value is cited inline to a paper section or labeled [ASSUMED] with its
    SSL-literature / official-code basis ([[feedback_ai_correctness]]).
    """

    learning_rate: float = 1e-4          # [CITED §5 p.9 — ADAM lr 1e-4]
    weight_decay: float = 1e-4           # [ASSUMED — AdamW decoupled wd, D2/D3]
    batch_size: int = 8                  # [ASSUMED; paper 5 CITED §5; 8 fits L4 ~11.7GB]
    num_workers: int = 4                 # persistent_workers guard applies (D7)
    max_epochs: int = 60                 # [CITED baseline 20 §5; ASSUMED extension <=60 via linear-probe]
    frames_per_half: int = 16            # [CITED §5 p.9 — 16 frames/half-cycle]
    crop_size: int = 112                 # Phase 2/3 contract
    linear_probe_cadence: int = 5        # [ASSUMED §6 — probe every N=5 epochs]
    projector_hidden: int = 512          # [ASSUMED §4 — SimCLR 2-layer MLP]
    projector_out_dim: int = 128         # [ASSUMED §4 — SimCLR/MoCo default]
    loss_squared: bool = True            # [§2 — official code uses squared L2; flag exposes Eq.1 non-squared]
    loss_three_term: bool = True         # [§2 — official active line is 3-term; flag exposes Eq.1 2-term]
    scheduler_name: str = "cosine"       # [ASSUMED — SSL standard, paper silent]
    model_arch: str = "r2plus1d_18_md_ssl_v1"

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs


class ProjectionHead(nn.Module):
    """2-layer MLP 512->512->128 (BN+ReLU), L2-normalized output (RESEARCH §4).

    L2-norm CITED from official train_test.py:51; dims SimCLR-standard [ASSUMED].
    Discarded at fine-tune time (backbone-only transfer).
    """

    def __init__(self, in_dim: int = 512, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        # 2-layer MLP — dims SimCLR-standard [ASSUMED §4]; discarded at fine-tune.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # L2-normalize before the distance-ratio loss. CITED: official train_test.py:51
        # (F.normalize(..., dim=-1, p=2)).
        return F.normalize(self.net(x), dim=-1, p=2)


def md_triplet_loss(
    phi_a: torch.Tensor,
    phi_p: torch.Tensor,
    phi_n: torch.Tensor,
    *,
    squared: bool = True,
    three_term: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Triplet distance-ratio loss (paper Eq.1 §3.2 p.6 + official train_test.py:60-71).

    ``d_ap = ((phi_a - phi_p)**2).sum(-1)``; ``num = exp(-d_ap)``;
    ``den = num + exp(-d_an) [+ exp(-d_pn) if three_term]``;
    ``loss = (-log(num / (den + eps))).mean()``. Inputs are L2-normalized
    embeddings ``[B, D]``. NO temperature, NO stop-gradient (RESEARCH §2).

    Args:
        phi_a, phi_p, phi_n: L2-normalized anchor/positive/negative embeddings ``[B, D]``.
        squared: squared L2 (official code) vs non-squared (paper Eq.1) — ablation flag.
        three_term: 3-term denominator (official active line) vs 2-term (Eq.1) — ablation flag.
        eps: numerical-stability epsilon inside the log.
    """
    # Paper Eq.1 §3.2 p.6 + official train_test.py:60-71 (the ACTIVE line is 3-term,
    # squared). Two documented discrepancies vs the paper text, exposed as flags:
    #   squared:    official code uses squared L2 (sum of squares); Eq.1 shows ||·||
    #               (non-squared). Default True = as-shipped code; False = exact Eq.1.
    #   three_term: official active line adds exp(-d_pn) (pos-neg repulsion); Eq.1's
    #               denominator is 2-term. Default True = as-shipped; False = exact Eq.1.
    # NO temperature, NO stop-gradient (RESEARCH §2). Vectorized (not the per-sample loop).
    d_ap = ((phi_a - phi_p) ** 2).sum(-1)
    d_an = ((phi_a - phi_n) ** 2).sum(-1)
    if not squared:
        d_ap = d_ap.sqrt()
        d_an = d_an.sqrt()
    num = torch.exp(-d_ap)
    den = num + torch.exp(-d_an)
    if three_term:
        d_pn = ((phi_p - phi_n) ** 2).sum(-1)
        if not squared:
            d_pn = d_pn.sqrt()
        den = den + torch.exp(-d_pn)
    return (-torch.log(num / (den + eps))).mean()


def build_md_model() -> tuple[nn.Module, nn.Module]:
    """Construct R(2+1)D-18 (Kinetics-V1 init, fc=Identity) + ProjectionHead.

    Returns ``(backbone, projector)`` kept separate so fine-tune loads only the
    backbone (the projector is a pretext-only artifact, discarded at fine-tune).
    """
    # Lazy import keeps module-level import of md_pretrain fast on CPU-only machines
    # (PATTERNS §4 — the slow model-build test is @pytest.mark.slow).
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    # Kinetics-V1 init matches Phase 3 for a clean P3-vs-P4 comparison (CITED §5 p.9).
    backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    assert backbone.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={backbone.fc.in_features}, expected 512"
    )
    backbone.fc = nn.Identity()  # expose the 512-d pooled features (RESEARCH §4)
    projector = ProjectionHead(in_dim=512, hidden=512, out_dim=128)
    n_params = sum(p.numel() for p in backbone.parameters()) + sum(
        p.numel() for p in projector.parameters()
    )
    logger.info(
        "build_md_model: R(2+1)D-18 (Kinetics-V1, fc=Identity) + ProjectionHead — %d params",
        n_params,
    )
    return backbone, projector


def build_ssl_checkpoint_payload(
    *,
    epoch: int,
    backbone: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metrics_history: list,
    linear_probe_history: list,
    config_hash: str,
    config_repr: dict,
) -> dict:
    """Assemble the SSL checkpoint payload (RESEARCH §10). Pure assembly — no I/O.

    Returns a dict with exactly the keys in ``_SSL_CHECKPOINT_KEYS``. Plan 02's
    ``run_md_pretrain_epoch`` calls this each epoch, then writes it via
    ``atomic_save_checkpoint``; ``backbone.pt`` writes pass ``update_latest=False``
    so the ``latest.txt`` epoch-resume pointer is never clobbered (D7 / Pitfall 4).
    """
    return {
        "epoch": epoch,
        "backbone_state_dict": backbone.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "metrics_history": metrics_history,
        "linear_probe_history": linear_probe_history,
        "config_hash": config_hash,
        "config_repr": config_repr,
        "code_version": "phase04-md-pretrain",
    }


def run_md_pretrain_epoch(*args: Any, **kwargs: Any) -> dict:
    """One MD-SSL pretraining epoch (3-branch forward + triplet loss + checkpoint).

    Finalized in Plan 02 after the VRAM probe (§5) + the probe-confirmed dataset.
    """
    raise NotImplementedError("Plan 02 — finalize after VRAM probe (§5) + probe-confirmed dataset")


def _linear_probe(*args: Any, **kwargs: Any) -> dict:
    """Freeze backbone, fit a linear head on labeled features, return val macro-F1 (§6).

    The SSL convergence monitor. Finalized in Plan 02.
    """
    raise NotImplementedError("Plan 02 — finalize linear-probe monitor (RESEARCH §6)")
