"""Motion-Disentangling SSL pretraining trainer.

Pretrains an R(2+1)D-18 backbone on the 4,970 unlabeled Squat clips with the
paper's half-cycle contrast pretext task. The loss is the triplet distance-ratio
loss (Hoffer & Ailon 2015 / paper Eq.1 §3.2 / official train_test.py:60-71) —
NOT NT-Xent: one structural negative per anchor (the opposite half-cycle), no
temperature, no stop-gradient, no in-batch negatives. Batch 5-8 fits L4 — no
gradient accumulation. A linear-probe monitor tracks downstream convergence and
an embedding-std/effective-rank monitor detects representation collapse, both
written into the checkpoint history per epoch.

The pure-function cores (``md_triplet_loss``, ``ProjectionHead``,
``build_md_model``) are unit-tested independently of the 12-24h GPU run.
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

from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
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

# SSL checkpoint payload key set. backbone.pt / best.pt writes use
# atomic_save_checkpoint(..., update_latest=False) so latest.txt is not clobbered.
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
    """Hyperparameters for MD-SSL pretraining.

    Each value is annotated inline with its paper-section or SSL-literature /
    official-code basis.
    """

    learning_rate: float = 1e-4          # paper: ADAM lr 1e-4
    weight_decay: float = 1e-4           # AdamW decoupled wd
    batch_size: int = 8                  # paper uses 5; 8 fits L4 ~11.7GB
    num_workers: int = 8                 # 8 cv2-decode streams (cv2.setNumThreads(0) per worker); fits L4/A100 vCPUs
    max_epochs: int = 60                 # paper baseline 20; extended to <=60 via linear-probe
    frames_per_half: int = 16            # paper: 16 frames/half-cycle
    crop_size: int = 112
    linear_probe_cadence: int = 5        # probe every N=5 epochs
    projector_hidden: int = 512          # SimCLR 2-layer MLP
    projector_out_dim: int = 128         # SimCLR/MoCo default
    loss_squared: bool = True            # official code uses squared L2; flag exposes Eq.1 non-squared
    loss_three_term: bool = True         # official active line is 3-term; flag exposes Eq.1 2-term
    scheduler_name: str = "cosine"       # SSL standard, paper silent
    model_arch: str = "r2plus1d_18_md_ssl_v1"
    strong_augs: bool = True             # paper-faithful strong aug set (a safe-core set collapsed)
    use_rotation: bool = False           # rotation OFF — distorts the knee-valgus KIE signal (our largest lift)
    aug_prob: float = 0.5                # per-aug independent application probability (strong set)

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs


class ProjectionHead(nn.Module):
    """2-layer MLP 512->512->128 (BN+ReLU), L2-normalized output.

    L2-norm from official train_test.py:51; dims SimCLR-standard.
    Discarded at fine-tune time (backbone-only transfer).
    """

    def __init__(self, in_dim: int = 512, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        # 2-layer MLP — dims SimCLR-standard; discarded at fine-tune.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # L2-normalize before the distance-ratio loss (official train_test.py:51:
        # F.normalize(..., dim=-1, p=2)).
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
    embeddings ``[B, D]``. NO temperature, NO stop-gradient.

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
    # NO temperature, NO stop-gradient. Vectorized (not the per-sample loop).
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
    # (the slow model-build test is @pytest.mark.slow).
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    # Kinetics-V1 init matches the supervised baseline for a clean comparison (paper §5 p.9).
    backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    assert backbone.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={backbone.fc.in_features}, expected 512"
    )
    backbone.fc = nn.Identity()  # expose the 512-d pooled features
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
    """Assemble the SSL checkpoint payload. Pure assembly — no I/O.

    Returns a dict with exactly the keys in ``_SSL_CHECKPOINT_KEYS``.
    ``run_md_pretrain_epoch`` calls this each epoch, then writes it via
    ``atomic_save_checkpoint``; ``backbone.pt`` writes pass ``update_latest=False``
    so the ``latest.txt`` epoch-resume pointer is never clobbered.
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


def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker (PyTorch reproducibility idiom). Copied, not imported."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(0)  # 1 cv2 thread per worker; the DataLoader provides parallelism
    except ImportError:
        pass


def _set_global_seed(seed: int) -> None:
    """Seed all 4 RNG sources (determinism precondition). Copied from supervised_train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _embedding_collapse_metrics(
    backbone: nn.Module, probe_batch: torch.Tensor, device: torch.device, *, chunk: int = 16,
) -> dict:
    """Representation-collapse metrics on a FIXED probe SET.

    embedding_std = mean per-dim std of L2-normalized BACKBONE embeddings (healthy ≈ 1/√512;
    collapse → 0). effective_rank = exp(entropy(normalized covariance singular values)) — the
    dimensional-collapse signal (arXiv:2110.09348). The probe SET must have MANY samples
    (≥ ~128): effective_rank is bounded by N−1, so a small batch (e.g. 8) caps it at ~7 and
    makes the dimensional-collapse signal meaningless. Forwarded in ``chunk``-sized slices to
    bound VRAM.
    """
    backbone.eval()
    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, probe_batch.shape[0], chunk):
            z_i = F.normalize(backbone(probe_batch[i:i + chunk].to(device)), dim=-1, p=2)
            embs.append(z_i.detach().cpu())
    z = torch.cat(embs, dim=0)  # [N, 512]
    embedding_std = float(z.std(dim=0).mean().item())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    sv = torch.linalg.svdvals(cov)
    sv = sv[sv > 1e-12]
    p = sv / sv.sum()
    effective_rank = float(np.exp(float(-(p * p.log()).sum().item())))
    return {"embedding_std": embedding_std, "effective_rank": effective_rank}


def _extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Forward each labeled batch through the FROZEN backbone -> (feats [N,512], labels [N,2])."""
    backbone.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for clip, label in loader:
            feats.append(backbone(clip.to(device, non_blocking=True)).detach().cpu().numpy())  # fc=Identity -> [B,512]
            labels.append(label.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


def _linear_probe(*, backbone: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> dict:
    """Freeze backbone, fit a per-head logistic head on labeled TRAIN features, report VAL macro-F1.

    Evaluates the BACKBONE (not the projector) — fine-tune reuses the backbone. sklearn
    LogisticRegression per error head (class_weight='balanced' for the KIE/KFE imbalance).
    The SSL convergence monitor: rising-then-plateau ⇒ converged; flat-at-random ⇒ collapse.
    """
    from sklearn.linear_model import LogisticRegression

    tr_feats, tr_labels = _extract_features(backbone, train_loader, device)
    va_feats, va_labels = _extract_features(backbone, val_loader, device)
    f1s: dict[str, float] = {}
    for head, name in ((0, "kie"), (1, "kfe")):
        y_tr = tr_labels[:, head].astype(int)
        if len(set(y_tr.tolist())) < 2:  # degenerate single-class -> skip fit
            f1s[name] = 0.0
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(tr_feats, y_tr)
        f1s[name] = float(f1_per_error(va_labels[:, head].astype(int), clf.predict(va_feats)))
    return {
        "linear_probe_f1_kie": f1s["kie"],
        "linear_probe_f1_kfe": f1s["kfe"],
        "linear_probe_f1_macro": (f1s["kie"] + f1s["kfe"]) / 2.0,
    }


def run_md_pretrain_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_unlabeled_videos",
    trajectories_root: str = "/content/squat_trajectories",
    labeled_videos_root: str = "/content/squat_videos",
    seed: int = 42,
    config: MDConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    ssl_dataset_cls=SquatSSLDataset,
    probe_dataset_cls=SquatKIEKFEDataset,
    checkpoint_phase: str = "phase04",
    cache_dir: str | None = None,
) -> dict:
    """MD-SSL pretraining — 3-branch triplet forward + AdamW + linear-probe + collapse detection.

    Mirrors supervised_train.run_supervised_epoch with the SSL deltas.
    Auto-resumes from {drive_root}/FitNova/checkpoints/phase04/{run_name}/latest.txt. Writes
    epoch_NNN.pt every epoch (update_latest=True) + backbone.pt on linear-probe-macro improvement
    (update_latest=False). Aborts on representation collapse before epoch 10.
    Use max_epochs=1 for the epoch-0 timing probe before authorizing the full run.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    config = config or MDConfig()
    effective_max_epochs = max_epochs if max_epochs is not None else config.max_epochs
    _set_global_seed(seed)

    config_repr = {
        "seed": seed, "batch_size": config.batch_size, "num_workers": config.num_workers,
        "crop_size": config.crop_size, "frames_per_half": config.frames_per_half,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "scheduler_name": config.scheduler_name, "scheduler_t_max": config.scheduler_t_max,
        "projector_hidden": config.projector_hidden, "projector_out_dim": config.projector_out_dim,
        "loss_squared": config.loss_squared, "loss_three_term": config.loss_three_term,
        "model_arch": config.model_arch,
        "strong_augs": config.strong_augs, "use_rotation": config.use_rotation, "aug_prob": config.aug_prob,
    }
    config_hash_str = hash_config(config_repr)
    run_dir = os.path.join(drive_root, "FitNova/checkpoints", checkpoint_phase, run_name)
    os.makedirs(run_dir, exist_ok=True)
    backbone_path = os.path.join(run_dir, "backbone.pt")
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)

    loader = build_ssl_loader(
        ssl_dataset_cls(
            videos_root=videos_root, trajectories_root=trajectories_root,
            frames_per_half=config.frames_per_half, crop_size=config.crop_size, seed=seed,
            strong_augs=config.strong_augs, use_rotation=config.use_rotation, aug_prob=config.aug_prob,
            cache_dir=cache_dir,
        ),
        config, seed=seed,
    )

    # Labeled loaders for the linear-probe monitor — val transform (deterministic), 32-frame.
    _persistent = config.num_workers > 0
    def _labeled_loader(split: str) -> DataLoader:
        return DataLoader(
            probe_dataset_cls(
                split=split, train_aug=False, drive_root=drive_root,
                videos_root=labeled_videos_root, num_frames=32, crop_size=config.crop_size,
                cache_dir=cache_dir,
            ),
            batch_size=config.batch_size, num_workers=config.num_workers,
            worker_init_fn=seed_worker, shuffle=False, drop_last=False, persistent_workers=_persistent,
        )
    probe_train_loader = _labeled_loader("train")
    probe_val_loader = _labeled_loader("val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, projector = build_md_model()
    backbone = backbone.to(device)
    projector = projector.to(device)
    optimizer = torch.optim.AdamW(  # AdamW NOT Adam — true decoupled weight decay
        list(backbone.parameters()) + list(projector.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)

    metrics_history: list[dict] = []
    linear_probe_history: list[dict] = []
    best_lp_macro = -1.0
    start_epoch = 0

    if resume:
        # map_location='cpu' REQUIRED — CUDA ByteTensors break set_rng_state_all.
        prior = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str, map_location="cpu")
        if prior is not None:
            backbone.load_state_dict(prior["backbone_state_dict"])
            projector.load_state_dict(prior["projector_state_dict"])
            optimizer.load_state_dict(prior["optimizer_state_dict"])
            if prior.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(prior["scheduler_state_dict"])
            restore_rng_state(prior["rng_state"])
            metrics_history = list(prior["metrics_history"])
            linear_probe_history = list(prior.get("linear_probe_history", []))
            start_epoch = int(prior["epoch"]) + 1
            best_lp_macro = max(
                (e.get("linear_probe_f1_macro", -1.0) for e in linear_probe_history), default=-1.0,
            )
            logger.info("Resumed epoch_%03d.pt; start_epoch=%d best_lp_macro=%.4f",
                        int(prior["epoch"]), start_epoch, best_lp_macro)

    if start_epoch >= effective_max_epochs:
        logger.info("Already at/past effective_max_epochs=%d (start_epoch=%d)", effective_max_epochs, start_epoch)
        return {
            "epoch": start_epoch - 1, "metrics_history": metrics_history,
            "linear_probe_history": linear_probe_history, "backbone_path": backbone_path,
            "checkpoint_path": os.path.join(run_dir, f"epoch_{start_epoch - 1:03d}.pt"),
            "config_hash": config_hash_str, "collapsed": False,
        }

    # Fixed probe SET for the collapse monitor — accumulate ~256 anchors (one batch of 8 is too
    # few: effective_rank is bounded by N−1, so a batch of 8 caps it at ~7 and makes the
    # dimensional-collapse signal meaningless). SAME clips every epoch.
    _probe_clips: list[torch.Tensor] = []
    _probe_n = 0
    for _pb in loader:
        _probe_clips.append(_pb["anchor"])
        _probe_n += int(_pb["anchor"].shape[0])
        if _probe_n >= 256:
            break
    probe_batch = torch.cat(_probe_clips, dim=0)[:256]
    last_ckpt_path = ""
    collapsed = False

    for epoch in range(start_epoch, effective_max_epochs):
        epoch_start_t = time.perf_counter()
        backbone.train()
        projector.train()
        ssl_losses: list[float] = []
        it = loader if tqdm is None else tqdm(loader, desc=f"SSL epoch {epoch}/{effective_max_epochs - 1}", leave=False)
        for batch in it:
            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            negative = batch["negative"].to(device, non_blocking=True)
            phi_a = projector(backbone(anchor))
            phi_p = projector(backbone(positive))
            phi_n = projector(backbone(negative))
            loss = md_triplet_loss(
                phi_a, phi_p, phi_n, squared=config.loss_squared, three_term=config.loss_three_term,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ssl_losses.append(float(loss.item()))
        scheduler.step()  # cosine, per-epoch (mirrors supervised_train)

        collapse = _embedding_collapse_metrics(backbone, probe_batch, device)
        epoch_wall = time.perf_counter() - epoch_start_t
        metrics_history.append({
            "epoch": epoch,
            "ssl_loss_mean": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
            "ssl_loss_per_batch": ssl_losses,
            "embedding_std": collapse["embedding_std"],
            "effective_rank": collapse["effective_rank"],
            "epoch_wall_time_s": epoch_wall,
        })
        logger.info("epoch=%d ssl_loss=%.6f emb_std=%.5f eff_rank=%.1f wall=%.1fs",
                    epoch, metrics_history[-1]["ssl_loss_mean"], collapse["embedding_std"],
                    collapse["effective_rank"], epoch_wall)

        if epoch % config.linear_probe_cadence == 0:  # convergence monitor
            lp = _linear_probe(backbone=backbone, train_loader=probe_train_loader,
                               val_loader=probe_val_loader, device=device)
            lp["epoch"] = epoch
            linear_probe_history.append(lp)
            logger.info("epoch=%d linear-probe macro=%.4f (kie=%.4f kfe=%.4f)",
                        epoch, lp["linear_probe_f1_macro"], lp["linear_probe_f1_kie"], lp["linear_probe_f1_kfe"])

        payload = build_ssl_checkpoint_payload(
            epoch=epoch, backbone=backbone, projector=projector, optimizer=optimizer,
            scheduler=scheduler, metrics_history=metrics_history,
            linear_probe_history=linear_probe_history, config_hash=config_hash_str, config_repr=config_repr,
        )
        last_ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, last_ckpt_path)  # update_latest=True (default)

        # backbone.pt-best on linear-probe-macro improvement — update_latest=False.
        if linear_probe_history and linear_probe_history[-1]["epoch"] == epoch:
            cur = linear_probe_history[-1]["linear_probe_f1_macro"]
            if cur > best_lp_macro:
                best_lp_macro = cur
                atomic_save_checkpoint(payload, backbone_path, update_latest=False)
                logger.info("New best linear-probe macro=%.4f -> wrote backbone.pt", best_lp_macro)

        prune_checkpoints(run_dir, keep_last=3, keep_best=True)

        # Collapse abort: embedding_std -> 0 before epoch 10.
        if epoch < 10 and collapse["embedding_std"] < 0.1 / (512 ** 0.5):
            logger.warning(
                "COLLAPSE: embedding_std=%.5f < %.5f at epoch %d — ABORT; raise SSL aug strength (§7/§12).",
                collapse["embedding_std"], 0.1 / (512 ** 0.5), epoch,
            )
            collapsed = True
            break

    return {
        "epoch": epoch,
        "metrics_history": metrics_history,
        "linear_probe_history": linear_probe_history,
        "backbone_path": backbone_path,
        "checkpoint_path": last_ckpt_path,
        "config_hash": config_hash_str,
        "collapsed": collapsed,
    }
