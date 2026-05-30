"""CVCSPC pose-contrastive SSL pretraining trainer for Phase 7 (IMG-01).

Self-supervisedly pretrains a 2D ResNet-18 backbone (ImageNet init) on the unlabeled
Back-Squat set with the CVCSPC pose-contrastive pretext task: frames at the same
bar-trajectory phase from two different reps are pulled together, frames a phase-gap
apart are pushed apart. The contrastive loss is a three-term distance-ratio loss on
L2-normalized features; the convergence monitor is triplet accuracy (anchor-positive
distance < anchor-negative distance), not a linear probe.

This is a self-contained trainer: it reuses the colab checkpoint/RNG primitives but does
NOT import the R(2+1)D-18 video SSL trainer (md_pretrain). The pure-function cores
(cvcspc_triplet_loss, CVCSPCProjectionHead, build_cvcspc_model) are unit-tested on CPU
before any GPU run; run_cvcspc_pretrain_epoch wires them into the epoch loop.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.training.aqa.datasets.cvcspc_ssl import (
    ShallowSquatSSLDataset,
    build_cvcspc_loader,
)
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)

logger = logging.getLogger("aqa.phase07")

_CVCSPC_CHECKPOINT_KEYS: tuple[str, ...] = (
    "epoch",
    "backbone_state_dict",
    "projector_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
    "metrics_history",
    "triplet_acc_history",
    "config_hash",
    "config_repr",
    "code_version",
)


@dataclasses.dataclass
class CVCSPCConfig:
    """Hyperparameters for CVCSPC SSL pretraining (paper §5: Adam 1e-4, batch 25, 100 epochs)."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 25
    num_workers: int = 4
    max_epochs: int = 100
    projector_hidden: int = 128
    projector_out_dim: int = 128
    phase_gap_start: float = 30.0
    phase_gap_decay_every: int = 4
    mask_prob: float = 0.5
    mask_amt_lo: float = 0.4
    mask_amt_hi: float = 0.5
    triplet_acc_cadence: int = 5
    scheduler_name: str = "cosine"
    model_arch: str = "resnet18_cvcspc_ssl_v1"

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs


class CVCSPCProjectionHead(nn.Module):
    """2-layer MLP projection head with L2-normalized output, no BatchNorm.

    BatchNorm is omitted: the CVCSPC batch size is 25 and the head has no BN in the
    shipped code. ``hidden``/``out_dim`` come from the config so the head is ablatable
    (e.g. 512->512) without a code edit.
    """

    def __init__(self, in_dim: int = 512, hidden: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, p=2)


def cvcspc_triplet_loss(
    phi_anc: torch.Tensor,
    phi_pos: torch.Tensor,
    phi_neg: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Three-term distance-ratio loss on L2-normalized features ``[B, D]``.

    ``-log( e^-d_ap / (e^-d_ap + e^-d_an + e^-d_pn) )`` averaged over the batch, with
    squared-L2 distances. The three-term denominator is the shipped code (train_test.py);
    the paper's Eq.1 shows a two-term denominator (no pos-neg term) — the code version is
    implemented here and the deviation is documented in FINDINGS.
    """
    d_ap = ((phi_anc - phi_pos) ** 2).sum(-1)
    d_an = ((phi_anc - phi_neg) ** 2).sum(-1)
    d_pn = ((phi_pos - phi_neg) ** 2).sum(-1)
    num = torch.exp(-d_ap)
    den = num + torch.exp(-d_an) + torch.exp(-d_pn)
    return (-torch.log(num / (den + eps))).mean()


def build_cvcspc_model(config: CVCSPCConfig) -> tuple[nn.Module, nn.Module]:
    """Construct ResNet-18 (ImageNet init, fc=Identity) + a CVCSPC projection head.

    Returns ``(backbone, projector)`` kept separate so fine-tune loads only the backbone.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    backbone = resnet18(weights=None)
    pretrained_sd = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    backbone.load_state_dict(pretrained_sd, strict=False)
    assert backbone.fc.in_features == 512, (
        f"ResNet-18 fc.in_features={backbone.fc.in_features}, expected 512"
    )
    backbone.fc = nn.Identity()
    projector = CVCSPCProjectionHead(
        in_dim=512, hidden=config.projector_hidden, out_dim=config.projector_out_dim,
    )
    n_params = sum(p.numel() for p in backbone.parameters()) + sum(
        p.numel() for p in projector.parameters()
    )
    logger.info("build_cvcspc_model: ResNet-18 (ImageNet, fc=Identity) + projection head — %d params", n_params)
    return backbone, projector


def build_cvcspc_checkpoint_payload(
    *,
    epoch: int,
    backbone: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics_history: list,
    triplet_acc_history: list,
    config_hash: str,
    config_repr: dict,
) -> dict:
    """Assemble the SSL checkpoint payload (the _CVCSPC_CHECKPOINT_KEYS set). Pure assembly — no I/O."""
    return {
        "epoch": epoch,
        "backbone_state_dict": backbone.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "metrics_history": metrics_history,
        "triplet_acc_history": triplet_acc_history,
        "config_hash": config_hash,
        "config_repr": config_repr,
        "code_version": "phase07-cvcspc-pretrain",
    }


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


def _set_global_seed(seed: int) -> None:
    """Seed all 4 RNG sources (determinism precondition)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _triplet_accuracy(
    backbone: nn.Module,
    projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Fraction of triplets where the anchor is closer to its positive than its negative.

    The CVCSPC convergence monitor (replaces the MD-SSL linear probe): frozen, no-grad.
    """
    backbone.eval()
    projector.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            phi_a = projector(backbone(batch["anchor"].to(device)))
            phi_p = projector(backbone(batch["positive"].to(device)))
            phi_n = projector(backbone(batch["negative"].to(device)))
            d_ap = ((phi_a - phi_p) ** 2).sum(-1)
            d_an = ((phi_a - phi_n) ** 2).sum(-1)
            correct += int((d_ap < d_an).sum().item())
            total += int(d_ap.shape[0])
    return correct / max(total, 1)


def run_cvcspc_pretrain_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    frames_root: str = "/content/squat_ssl_frames",
    trajectories_root: str = "/content/squat_trajectories",
    traj_nan_path: str | None = None,
    exclude_ids: list[str] | None = None,
    seed: int = 42,
    config: CVCSPCConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    ssl_dataset_cls=ShallowSquatSSLDataset,
    checkpoint_phase: str = "phase07",
) -> dict:
    """CVCSPC SSL pretraining — 3-branch triplet forward + Adam + triplet-accuracy monitor.

    Auto-resumes from {drive_root}/FitNova/checkpoints/{checkpoint_phase}/{run_name}/latest.txt;
    writes epoch_NNN.pt every epoch + backbone.pt on triplet-accuracy improvement
    (update_latest=False). The phase gap anneals every config.phase_gap_decay_every epochs
    (floor 30°) and is re-applied on resume. Uses Adam to match the official code; with the
    default weight_decay=0 this is identical to AdamW.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    config = config or CVCSPCConfig()
    effective_max_epochs = max_epochs if max_epochs is not None else config.max_epochs
    _set_global_seed(seed)

    config_repr = {
        "seed": seed,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler_name": config.scheduler_name,
        "scheduler_t_max": config.scheduler_t_max,
        "projector_hidden": config.projector_hidden,
        "projector_out_dim": config.projector_out_dim,
        "phase_gap_start": config.phase_gap_start,
        "phase_gap_decay_every": config.phase_gap_decay_every,
        "mask_prob": config.mask_prob,
        "mask_amt_lo": config.mask_amt_lo,
        "mask_amt_hi": config.mask_amt_hi,
        "model_arch": config.model_arch,
        "exclude_ids_hash": hash_config({"ids": sorted(str(x) for x in (exclude_ids or []))}),
    }
    config_hash_str = hash_config(config_repr)
    run_dir = os.path.join(drive_root, "FitNova/checkpoints", checkpoint_phase, run_name)
    os.makedirs(run_dir, exist_ok=True)
    backbone_path = os.path.join(run_dir, "backbone.pt")
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)

    loader = build_cvcspc_loader(
        ssl_dataset_cls(
            frames_root=frames_root,
            trajectories_root=trajectories_root,
            traj_nan_path=traj_nan_path,
            exclude_ids=exclude_ids,
            ssl_contrastive_phase_gap=config.phase_gap_start,
            mask_prob=config.mask_prob,
            mask_amt_lo=config.mask_amt_lo,
            mask_amt_hi=config.mask_amt_hi,
            seed=seed,
        ),
        config,
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, projector = build_cvcspc_model(config)
    backbone = backbone.to(device)
    projector = projector.to(device)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)

    metrics_history: list[dict] = []
    triplet_acc_history: list[dict] = []
    best_triplet_acc = -1.0
    start_epoch = 0

    if resume:
        prior = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str, map_location="cpu")
        if prior is not None:
            backbone.load_state_dict(prior["backbone_state_dict"])
            projector.load_state_dict(prior["projector_state_dict"])
            optimizer.load_state_dict(prior["optimizer_state_dict"])
            if prior.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(prior["scheduler_state_dict"])
            restore_rng_state(prior["rng_state"])
            metrics_history = list(prior["metrics_history"])
            triplet_acc_history = list(prior.get("triplet_acc_history", []))
            start_epoch = int(prior["epoch"]) + 1
            best_triplet_acc = max(
                (e.get("triplet_acc", -1.0) for e in triplet_acc_history), default=-1.0,
            )
            n_decays = start_epoch // config.phase_gap_decay_every
            config.phase_gap_start = max(30.0, config.phase_gap_start - n_decays)
            loader.dataset.phase_gap = config.phase_gap_start
            logger.info(
                "Resumed epoch_%03d.pt; start_epoch=%d best_triplet_acc=%.4f phase_gap=%.1f",
                int(prior["epoch"]), start_epoch, best_triplet_acc, config.phase_gap_start,
            )

    if start_epoch >= effective_max_epochs:
        logger.info("Already at/past effective_max_epochs=%d (start_epoch=%d)", effective_max_epochs, start_epoch)
        return {
            "epoch": start_epoch - 1,
            "metrics_history": metrics_history,
            "triplet_acc_history": triplet_acc_history,
            "backbone_path": backbone_path,
            "checkpoint_path": os.path.join(run_dir, f"epoch_{start_epoch - 1:03d}.pt"),
            "config_hash": config_hash_str,
        }

    last_ckpt_path = ""
    for epoch in range(start_epoch, effective_max_epochs):
        epoch_start_t = time.perf_counter()
        backbone.train()
        projector.train()
        ssl_losses: list[float] = []
        it = loader if tqdm is None else tqdm(loader, desc=f"CVCSPC epoch {epoch}/{effective_max_epochs - 1}", leave=False)
        for batch in it:
            phi_a = projector(backbone(batch["anchor"].to(device, non_blocking=True)))
            phi_p = projector(backbone(batch["positive"].to(device, non_blocking=True)))
            phi_n = projector(backbone(batch["negative"].to(device, non_blocking=True)))
            loss = cvcspc_triplet_loss(phi_a, phi_p, phi_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ssl_losses.append(float(loss.item()))
        scheduler.step()

        triplet_acc = None
        if epoch % config.triplet_acc_cadence == 0:
            triplet_acc = _triplet_accuracy(backbone, projector, loader, device)
            triplet_acc_history.append({"epoch": epoch, "triplet_acc": triplet_acc})
            logger.info("epoch=%d triplet_acc=%.4f", epoch, triplet_acc)

        epoch_wall = time.perf_counter() - epoch_start_t
        metrics_history.append({
            "epoch": epoch,
            "ssl_loss_mean": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
            "ssl_loss_per_batch": ssl_losses,
            "phase_gap": config.phase_gap_start,
            "epoch_wall_time_s": epoch_wall,
        })
        logger.info("epoch=%d ssl_loss=%.6f phase_gap=%.1f wall=%.1fs",
                    epoch, metrics_history[-1]["ssl_loss_mean"], config.phase_gap_start, epoch_wall)

        payload = build_cvcspc_checkpoint_payload(
            epoch=epoch, backbone=backbone, projector=projector, optimizer=optimizer,
            scheduler=scheduler, metrics_history=metrics_history,
            triplet_acc_history=triplet_acc_history, config_hash=config_hash_str, config_repr=config_repr,
        )
        last_ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, last_ckpt_path)

        if triplet_acc is not None and triplet_acc > best_triplet_acc:
            best_triplet_acc = triplet_acc
            atomic_save_checkpoint(payload, backbone_path, update_latest=False)
            logger.info("New best triplet_acc=%.4f -> wrote backbone.pt", best_triplet_acc)

        prune_checkpoints(run_dir, keep_last=3, keep_best=True)

        if (epoch + 1) % config.phase_gap_decay_every == 0 and config.phase_gap_start > 30.0:
            config.phase_gap_start = max(30.0, config.phase_gap_start - 1.0)
            loader.dataset.phase_gap = config.phase_gap_start
            logger.info("phase_gap annealed to %.1f", config.phase_gap_start)

    return {
        "epoch": epoch,
        "metrics_history": metrics_history,
        "triplet_acc_history": triplet_acc_history,
        "backbone_path": backbone_path,
        "checkpoint_path": last_ckpt_path,
        "config_hash": config_hash_str,
    }
