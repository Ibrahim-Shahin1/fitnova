"""R(2+1)D-18 supervised baseline trainer for Squat KIE/KFE error detection.

Drives end-to-end fine-tuning of torchvision's `r2plus1d_18`
(Kinetics-400-V1 init) on the labeled Squat split, reusing the dataset
surface verbatim and the resumable Colab harness, with a checkpoint payload
schema carrying `best_f1_val`, `best_thresholds`, `scheduler_state_dict`,
and per-epoch metric keys.

Public API:

- `SupervisedConfig` — frozen-at-construction hyperparameter dataclass
  (Adam lr=1e-4, weight_decay=0, batch_size=16, num_workers=4, 50 epochs
  with 8-epoch val-macro-F1 patience, CosineAnnealingLR, no AMP).
- `build_model()` — constructs R(2+1)D-18 + `nn.Linear(512, 2)` head with a
  hard assertion that torchvision's `fc.in_features` stays at 512.
- `seed_worker(worker_id)` — PyTorch official `torch.initial_seed() % 2**32`
  worker re-seed.
- `run_supervised_epoch(*, run_name, ...)` — public entry point. Atomic
  checkpoint per epoch + `best.pt` on val-macro-F1 improvement. Auto-resumes
  from `latest.txt` via the atomic-write contract.

Loss: `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)` — train-derived
imbalance weights from `SquatKIEKFEDataset`.

Optimizer: Adam lr=1e-4 (paper §5 CITED), weight_decay=0 (ASSUMED — paper
silent), CosineAnnealingLR with T_max=max_epochs (ASSUMED). No mixed
precision (paper fidelity + determinism + decode-bound).

DataLoader: constructed directly (NOT via `squat.build_loaders` which
clamps `num_workers=0`); per-worker re-seed via `seed_worker`.

Determinism: relaxed from bitwise-identical to functional correctness.
Main-process 4-RNG capture/restore wraps every epoch boundary; worker RNGs
are NOT captured.
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
from backend.training.aqa.eval.metrics import (
    f1_per_error,
    pr_auc_per_error,
)
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)

logger = logging.getLogger("aqa.phase03")


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter dataclass (fields hashed into config_hash for resume)
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class SupervisedConfig:
    """Hyperparameters for the supervised baseline run.

    All values cited inline to paper § (`[CITED]` or `[ASSUMED]`) or
    PyTorch/torchvision docs. The trainer enters this dataclass into
    `hash_config` to produce the resume-protection `config_hash`.
    """

    learning_rate: float = 1e-4               # paper §5 [CITED]
    weight_decay: float = 0.0                 # paper silent [ASSUMED]
    batch_size: int = 16                      # L4 24 GB fits fp32 batch 16 with headroom
    num_workers: int = 4                      # decode-bound; 4 workers on L4 Colab
    max_epochs: int = 50                      # paper silent for downstream [ASSUMED]
    early_stop_patience: int = 8              # 8-epoch val-macro-F1 patience
    num_frames: int = 32
    crop_size: int = 112                      # locked default
    train_jitter_frames: int = 2
    flip_aug: bool = False                    # OFF by default (paper line 576 omits flip)
    scheduler_name: str = "cosine"            # paper silent [ASSUMED]
    scheduler_t_max: int | None = None        # defaults to max_epochs at __post_init__
    model_arch: str = "r2plus1d_18_kinetics400_v1"  # Stable identifier for config_hash
    loss_name: str = "BCEWithLogitsLoss(pos_weight=dataset.pos_weight)"

    def __post_init__(self) -> None:
        # Freeze scheduler_t_max into the hash so a resume with a different
        # max_epochs does NOT silently change the LR trajectory of restored epochs.
        if self.scheduler_t_max is None:
            self.scheduler_t_max = self.max_epochs


# ──────────────────────────────────────────────────────────────────────────────
# Model construction (lazy torchvision import; pytest can import this module
# without paying the torchvision import cost)
# ──────────────────────────────────────────────────────────────────────────────


def build_model() -> nn.Module:
    """Construct R(2+1)D-18 with Kinetics-400-V1 weights and replace the head.

    Returns:
        `nn.Module` with `model.fc = nn.Linear(512, 2)`. All backbone layers
        trainable (end-to-end fine-tune). Verifies `fc.in_features == 512`
        before replacement.

    Raises:
        AssertionError: if `model.fc.in_features != 512` — catches upstream
            torchvision API drift (head-replacement assumption breaks).
    """
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    assert model.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={model.fc.in_features}, expected 512 "
        "(upstream torchvision API may have changed)"
    )
    model.fc = nn.Linear(model.fc.in_features, 2)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: R(2+1)D-18 + Linear(512, 2) head, %d params", n_params)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader worker re-seed (verbatim from PyTorch docs)
# ──────────────────────────────────────────────────────────────────────────────


def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker from the main-process RNG.

    Verbatim from PyTorch reproducibility docs:
    https://docs.pytorch.org/docs/2.12/notes/randomness.html

    The body uses `torch.initial_seed()` (which PyTorch derives per-worker
    from the trainer's `torch.Generator`) modulo `2**32` for NumPy's 32-bit
    seed requirement.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(0)  # 1 cv2 thread per worker; the DataLoader provides parallelism
    except ImportError:
        pass


def _set_global_seed(seed: int) -> None:
    """Seed all 4 RNG sources. Determinism precondition.

    Tries to reuse `tiny_train._set_global_seed` first; falls back to the
    verbatim 7-line body on the Python private-name import edge case.
    """
    try:
        from backend.training.aqa.harness.tiny_train import (
            _set_global_seed as _phase2_seed,
        )

        _phase2_seed(seed)
        return
    except ImportError:
        # Mirrors tiny_train._set_global_seed.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_dataloaders(
    seed: int, config: SupervisedConfig, drive_root: str, videos_root: str,
    *, dataset_cls=SquatKIEKFEDataset,  # OHP callers pass OHPElbowsKneesDataset
) -> dict[str, DataLoader]:
    """Construct three DataLoaders directly (bypasses `build_loaders`).

    `squat.build_loaders` clamps `num_workers=0` to preserve bitwise
    determinism. This relaxes that constraint and constructs `DataLoader`
    directly with `worker_init_fn=seed_worker` and a seeded `torch.Generator`
    so the multi-worker decode pipeline is per-worker deterministic without
    modifying the dataset module.

    Args:
        seed: Single integer seed driving the DataLoader generator.
        config: Hyperparameters — pulls batch_size, num_workers, num_frames,
            crop_size, train_jitter_frames, flip_aug.
        drive_root: Path to Drive `MyDrive` (forwards to dataset).
        videos_root: Local staged-mp4 directory (forwards to dataset).

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"` mapping to `DataLoader`.
        train: shuffle=True; val/test: shuffle=False. drop_last=False on all.
    """
    # NOTE: `SquatKIEKFEDataset` does NOT accept a `flip_aug` kwarg
    # (flip=False is hardcoded). The `flip_aug` field on `SupervisedConfig`
    # is an ablation hook reserved for later; for the locked-default run it's
    # a no-op. Passing it would break the dataset signature contract.
    common_kwargs: dict[str, Any] = {
        "drive_root": drive_root,
        "videos_root": videos_root,
        "num_frames": config.num_frames,
        "crop_size": config.crop_size,
        "train_jitter_frames": config.train_jitter_frames,
        "seed": seed,
    }
    train_ds = dataset_cls(split="train", train_aug=True,  **common_kwargs)
    val_ds   = dataset_cls(split="val",   train_aug=False, **common_kwargs)
    test_ds  = dataset_cls(split="test",  train_aug=False, **common_kwargs)

    g = torch.Generator()
    g.manual_seed(seed)

    # `persistent_workers=True` (when num_workers > 0) keeps workers alive across
    # iterations instead of fork-spawning per epoch. Two wins: (1) suppresses the
    # known PyTorch #40157 spam at end-of-epoch DataLoader iterator __del__
    # (AssertionError "can only test a child process" from is_alive() called via
    # a child process context — harmless but floods stderr); (2) per-epoch
    # restart overhead drops (no fresh worker fork). Not in the config_hash so
    # adding it doesn't invalidate resume from existing checkpoints.
    # PyTorch requires `num_workers > 0` for `persistent_workers=True`; the
    # `config.num_workers > 0` guard preserves the num_workers=0 fallback path
    # for CPU-only smoke runs.
    _persistent = config.num_workers > 0

    # Direct DataLoader construction with seed_worker re-seed.
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=False,
        persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        persistent_workers=_persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        persistent_workers=_persistent,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def _val_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run a full val (or test) pass and return (loss_mean, scores, labels).

    Accumulates logits + labels across the full pass on CPU, then computes
    sigmoid scores once at the end. No per-batch F1 — the ~15-batch val pass
    aliases on small denominators with 14% positive rate.

    Args:
        model: Model in eval mode (caller sets `.eval()`; this helper enforces).
        loader: DataLoader (val or test).
        criterion: BCEWithLogitsLoss used for the mean-loss return value.
        device: cuda/cpu target.

    Returns:
        Tuple `(loss_mean, scores, labels)`:
            - `loss_mean`: float — mean per-batch BCE loss.
            - `scores`: float ndarray shape `(N, 2)` — sigmoid scores in `[0, 1]`.
            - `labels`: int ndarray shape `(N, 2)` — ground-truth {0, 1}.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    model.eval()
    per_batch_losses: list[float] = []
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []

    iterator = loader if tqdm is None else tqdm(loader, desc="val", leave=False)
    with torch.no_grad():
        for clip, label in iterator:
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(clip)
            loss = criterion(logits, label)
            per_batch_losses.append(float(loss.item()))
            logits_chunks.append(logits.detach().cpu())
            labels_chunks.append(label.detach().cpu())

    loss_mean = float(np.mean(per_batch_losses)) if per_batch_losses else float("nan")
    # Gather once at the end, no per-batch F1.
    scores = torch.sigmoid(torch.cat(logits_chunks)).numpy()
    labels_arr = torch.cat(labels_chunks).numpy().astype(int)
    return loss_mean, scores, labels_arr


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_supervised_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_videos",
    seed: int = 42,
    config: SupervisedConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    dataset_cls=SquatKIEKFEDataset,
    checkpoint_phase: str = "phase03",
) -> dict:
    """Train through `effective_max_epochs - 1` from a checkpointed prior state (or fresh).

    Public training entry point. Auto-resumes from
    `{drive_root}/FitNova/checkpoints/phase03/{run_name}/latest.txt` if
    `resume=True` and the checkpoint config_hash matches the current config.
    Writes `epoch_NNN.pt` every epoch and `best.pt` whenever val macro-F1
    improves.

    Args:
        run_name: Subdirectory under
            `{drive_root}/FitNova/checkpoints/phase03/{run_name}/`.
        drive_root: Colab Drive mount point (default `/content/drive/MyDrive`).
        videos_root: Locally-staged Squat videos (default `/content/squat_videos`).
        seed: Single integer seed feeding `_set_global_seed` and the
            DataLoader `torch.Generator`.
        config: `SupervisedConfig` instance; defaults to `SupervisedConfig()` if
            `None`.
        resume: If `True` (default), call `load_latest_checkpoint` and restore
            model + optimizer + scheduler + main-process RNG + metrics_history +
            best_f1_val. If `False`, start fresh from epoch 0.
        max_epochs: Override `config.max_epochs` for this call. Used by the
            notebook to run the epoch-0 timing probe (`max_epochs=1`) before
            authorising the full run.

    Returns:
        Dict with keys:
            - `epoch`: int — last completed epoch (0-indexed)
            - `metrics_history`: list[dict] — one entry per completed epoch
            - `best_f1_val`: float — best val macro-F1 seen this run
            - `best_thresholds`: dict | None — `{"kie", "kfe"}` post-sweep,
              else `None`
            - `checkpoint_path`: str — path to last `epoch_NNN.pt` written
            - `best_checkpoint_path`: str — path to `best.pt`
            - `config_hash`: str — sha256[:16] of the hashed config
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    config = config or SupervisedConfig()
    effective_max_epochs = max_epochs if max_epochs is not None else config.max_epochs

    _set_global_seed(seed)

    # Ordered config_repr — every key hashed for resume protection.
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
        "scheduler_name": config.scheduler_name,
        "scheduler_t_max": config.scheduler_t_max,
        "model_arch": config.model_arch,
        "loss_name": config.loss_name,
    }
    config_hash_str = hash_config(config_repr)

    run_dir = os.path.join(drive_root, "FitNova/checkpoints", checkpoint_phase, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)

    loaders = _build_dataloaders(seed, config, drive_root, videos_root, dataset_cls=dataset_cls)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # pos_weight from the train dataset (identical across all three loaders
    # since computed from train regardless of split).
    pos_weight = train_loader.dataset.pos_weight  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(  # Adam lr=1e-4, weight_decay=0
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # CosineAnnealingLR with T_max frozen into config_hash.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_t_max,
    )

    metrics_history: list[dict] = []
    best_f1_val = -1.0
    epochs_since_improve = 0
    start_epoch = 0

    # Resume branch.
    if resume:
        # map_location='cpu' is REQUIRED — torch.cuda.set_rng_state_all rejects
        # CUDA ByteTensors with "RNG state must be a torch.ByteTensor" (the type
        # system distinguishes CPU vs CUDA ByteTensors). Loading to CPU keeps
        # RNG ByteTensors on CPU; model.load_state_dict / optimizer.load_state_dict
        # handle their own device transfer to `device` automatically. Mirrors
        # tiny_train.py which uses the load_latest_checkpoint default.
        prior = load_latest_checkpoint(
            run_dir, expected_config_hash=config_hash_str, map_location="cpu",
        )
        if prior is not None:
            model.load_state_dict(prior["model_state_dict"])
            optimizer.load_state_dict(prior["optimizer_state_dict"])
            if prior.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(prior["scheduler_state_dict"])
            restore_rng_state(prior["rng_state"])
            metrics_history = list(prior["metrics_history"])  # copy — don't mutate loaded dict
            best_f1_val = float(prior.get("best_f1_val", -1.0))
            start_epoch = int(prior["epoch"]) + 1
            # Recompute epochs_since_improve from metrics_history tail.
            epochs_since_improve = 0
            for entry in reversed(metrics_history):
                if float(entry.get("val_macro_f1", -1.0)) >= best_f1_val:
                    break
                epochs_since_improve += 1
            logger.info(
                "Resumed from epoch_%03d.pt; start_epoch=%d, best_f1_val=%.4f, "
                "epochs_since_improve=%d",
                int(prior["epoch"]),
                start_epoch,
                best_f1_val,
                epochs_since_improve,
            )

    if start_epoch >= effective_max_epochs:
        logger.info(
            "Resumed from epoch_%03d.pt; effective_max_epochs=%d already reached",
            start_epoch - 1, effective_max_epochs,
        )
        last_ckpt = os.path.join(run_dir, f"epoch_{start_epoch - 1:03d}.pt")
        best_ckpt = os.path.join(run_dir, "best.pt")
        return {
            "epoch": start_epoch - 1,
            "metrics_history": metrics_history,
            "best_f1_val": best_f1_val,
            "best_thresholds": None,
            "checkpoint_path": last_ckpt,
            "best_checkpoint_path": best_ckpt,
            "config_hash": config_hash_str,
        }

    last_ckpt_path = ""
    best_ckpt_path = os.path.join(run_dir, "best.pt")

    # Main training loop.
    for epoch in range(start_epoch, effective_max_epochs):
        epoch_start_t = time.perf_counter()

        # Train pass.
        model.train()
        train_losses: list[float] = []
        train_iter = (
            train_loader if tqdm is None
            else tqdm(train_loader, desc=f"epoch {epoch}/{effective_max_epochs - 1} train", leave=False)
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
                logger.info(
                    "epoch=%d batch=%d train_loss=%.6f", epoch, batch_idx, train_losses[-1],
                )

        # scheduler.step() at EPOCH end (cosine annealing per-epoch schedule).
        scheduler.step()

        # Val pass.
        val_loss_mean, val_scores, val_labels = _val_pass(
            model, val_loader, criterion, device,
        )

        # Per-error F1 + PR-AUC computed on the full val pass. Use
        # threshold=0.5 during training (proxy for best.pt selection); the
        # post-training threshold sweep picks per-error optimal thresholds.
        val_pred = (val_scores >= 0.5).astype(int)
        val_f1_kie = f1_per_error(val_labels[:, 0], val_pred[:, 0])
        val_f1_kfe = f1_per_error(val_labels[:, 1], val_pred[:, 1])
        val_pr_auc_kie = pr_auc_per_error(val_labels[:, 0], val_scores[:, 0])
        val_pr_auc_kfe = pr_auc_per_error(val_labels[:, 1], val_scores[:, 1])
        val_macro_f1 = (val_f1_kie + val_f1_kfe) / 2.0

        epoch_wall_time_s = time.perf_counter() - epoch_start_t
        logger.info(
            "epoch=%d val_loss=%.6f val_f1_kie=%.4f val_f1_kfe=%.4f "
            "val_macro_f1=%.4f wall_time=%.1fs",
            epoch, val_loss_mean, val_f1_kie, val_f1_kfe, val_macro_f1, epoch_wall_time_s,
        )

        # Per-epoch metrics_history entry.
        metrics_history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_loss_per_batch": train_losses,
            "val_loss_mean": val_loss_mean,
            "val_f1_kie": val_f1_kie,
            "val_f1_kfe": val_f1_kfe,
            "val_pr_auc_kie": val_pr_auc_kie,
            "val_pr_auc_kfe": val_pr_auc_kfe,
            "val_macro_f1": val_macro_f1,
            "epoch_wall_time_s": epoch_wall_time_s,
        })

        # Build payload + write epoch_NNN.pt always; write best.pt on improvement.
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # actual state, not None
            "rng_state": capture_rng_state(),
            "metrics_history": metrics_history,
            "best_f1_val": best_f1_val,
            "best_thresholds": None,  # filled by post-training sweep
            "config_hash": config_hash_str,
            "config_repr": config_repr,
            "code_version": "phase03-supervised-baseline",
        }
        ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, ckpt_path)
        last_ckpt_path = ckpt_path

        # best.pt write contract — on val-macro-F1 improvement, write a
        # SECOND atomic checkpoint to fixed-name target. latest.txt is NOT
        # updated by the best write (so it can never point to a corrupt file).
        if val_macro_f1 > best_f1_val:
            best_f1_val = val_macro_f1
            payload["best_f1_val"] = best_f1_val
            atomic_save_checkpoint(payload, best_ckpt_path)
            logger.info("New best val_macro_f1=%.4f → wrote best.pt", best_f1_val)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        prune_checkpoints(run_dir, keep_last=3, keep_best=True)  # note keep_best=True

        # Early-stop on val-macro-F1 plateau.
        if epochs_since_improve >= config.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (best=%.4f after %d epochs without improvement)",
                epoch, best_f1_val, epochs_since_improve,
            )
            break

    return {
        "epoch": epoch,
        "metrics_history": metrics_history,
        "best_f1_val": best_f1_val,
        "best_thresholds": None,  # filled by post-training threshold sweep
        "checkpoint_path": last_ckpt_path,
        "best_checkpoint_path": best_ckpt_path,
        "config_hash": config_hash_str,
    }
