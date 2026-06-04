"""ResNet-18 supervised image trainer for Shallow-Squat depth-error detection.

A self-contained 2D-image trainer: ImageNet-pretrained ResNet-18 + a single binary head,
end-to-end fine-tune on the labeled Shallow-Squat crops, with the checkpoint/resume/early-stop
contract carried from the video trainers. It reuses only eval/metrics.py + the colab checkpoint
primitives; it does NOT import the R(2+1)D-18 video trainer (which assumes 5-D clip tensors and
has no model seam).

The ``model_builder`` seam on ``run_image_epoch`` lets the CVCSPC fine-tune inject a
backbone-loading builder without editing this module; the default builds the ImageNet baseline.
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
from torch.utils.data import DataLoader

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.eval.metrics import f1_per_error, pr_auc_per_error
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)

logger = logging.getLogger("aqa.phase07")


@dataclasses.dataclass
class ImageConfig:
    """Hyperparameters for the Shallow-Squat ResNet-18 fine-tune (Adam 1e-4, cosine, 50 epochs)."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 50
    early_stop_patience: int = 8
    scheduler_name: str = "cosine"
    scheduler_t_max: int | None = None
    model_arch: str = "resnet18_imagenet_v1"
    loss_name: str = "BCEWithLogitsLoss(pos_weight=dataset.pos_weight)"

    def __post_init__(self) -> None:
        if self.scheduler_t_max is None:
            self.scheduler_t_max = self.max_epochs


def build_resnet18() -> nn.Module:
    """Construct ImageNet ResNet-18 + a single binary head (Linear(512, 1))."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert model.fc.in_features == 512, (
        f"ResNet-18 fc.in_features={model.fc.in_features}, expected 512"
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: ResNet-18 + Linear(512, 1) head, %d params", n_params)
    return model


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


def _build_dataloaders(
    seed: int,
    config: ImageConfig,
    images_root: str,
    labels_path: str,
    splits_root: str,
    *,
    dataset_cls=ShallowSquatDataset,
) -> dict[str, DataLoader]:
    common_kwargs = {
        "images_root": images_root,
        "labels_path": labels_path,
        "splits_root": splits_root,
        "seed": seed,
    }
    train_ds = dataset_cls(split="train", train_aug=True, **common_kwargs)
    val_ds = dataset_cls(split="val", train_aug=False, **common_kwargs)
    test_ds = dataset_cls(split="test", train_aug=False, **common_kwargs)

    g = torch.Generator()
    g.manual_seed(seed)
    _persistent = config.num_workers > 0

    def _loader(ds, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=_persistent,
        )

    return {
        "train": _loader(train_ds, shuffle=True),
        "val": _loader(val_ds, shuffle=False),
        "test": _loader(test_ds, shuffle=False),
    }


def _val_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run a full val/test pass; return (loss_mean, scores[N], labels[N]) for the single binary head."""
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
        for img, label in iterator:
            img = img.to(device, non_blocking=True)
            target = label.view(-1, 1).to(device, non_blocking=True)
            logits = model(img)
            loss = criterion(logits, target)
            per_batch_losses.append(float(loss.item()))
            logits_chunks.append(logits.detach().cpu())
            labels_chunks.append(label.detach().cpu())

    loss_mean = float(np.mean(per_batch_losses)) if per_batch_losses else float("nan")
    scores = torch.sigmoid(torch.cat(logits_chunks)).squeeze(-1).numpy()
    labels_arr = torch.cat(labels_chunks).numpy().astype(int)
    return loss_mean, scores, labels_arr


def run_image_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    images_root: str = "/content/squat_shallow_images/crops_unaligned",
    labels_path: str = "/content/squat_shallow_images/labels_shallow_depth.json",
    splits_root: str = "/content/squat_shallow_images/splits",
    seed: int = 42,
    config: ImageConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    dataset_cls=ShallowSquatDataset,
    model_builder=build_resnet18,
    checkpoint_phase: str = "phase07",
) -> dict:
    """Fine-tune a ResNet-18 image classifier with per-epoch checkpoint/resume + early stop.

    Auto-resumes from {drive_root}/FitNova/checkpoints/{checkpoint_phase}/{run_name}/latest.txt.
    Writes epoch_NNN.pt every epoch + best.pt on val-F1 improvement (update_latest=False).
    ``model_builder`` is the seam the CVCSPC fine-tune uses to inject a backbone-loading builder.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    config = config or ImageConfig()
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
        "model_arch": config.model_arch,
        "loss_name": config.loss_name,
    }
    config_hash_str = hash_config(config_repr)
    run_dir = os.path.join(drive_root, "FitNova/checkpoints", checkpoint_phase, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)

    loaders = _build_dataloaders(
        seed, config, images_root, labels_path, splits_root, dataset_cls=dataset_cls,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    pos_weight = train_loader.dataset.pos_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_builder().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)

    metrics_history: list[dict] = []
    best_f1_val = -1.0
    epochs_since_improve = 0
    start_epoch = 0

    if resume:
        prior = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str, map_location="cpu")
        if prior is not None:
            model.load_state_dict(prior["model_state_dict"])
            optimizer.load_state_dict(prior["optimizer_state_dict"])
            if prior.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(prior["scheduler_state_dict"])
            restore_rng_state(prior["rng_state"])
            metrics_history = list(prior["metrics_history"])
            best_f1_val = float(prior.get("best_f1_val", -1.0))
            start_epoch = int(prior["epoch"]) + 1
            epochs_since_improve = 0
            for entry in reversed(metrics_history):
                if float(entry.get("val_f1", -1.0)) >= best_f1_val:
                    break
                epochs_since_improve += 1
            logger.info("Resumed epoch_%03d.pt; start_epoch=%d best_f1_val=%.4f",
                        int(prior["epoch"]), start_epoch, best_f1_val)

    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if start_epoch >= effective_max_epochs:
        logger.info("Already at/past effective_max_epochs=%d (start_epoch=%d)", effective_max_epochs, start_epoch)
        return {
            "epoch": start_epoch - 1,
            "metrics_history": metrics_history,
            "best_f1_val": best_f1_val,
            "best_thresholds": None,
            "checkpoint_path": os.path.join(run_dir, f"epoch_{start_epoch - 1:03d}.pt"),
            "best_checkpoint_path": best_ckpt_path,
            "config_hash": config_hash_str,
        }

    last_ckpt_path = ""
    for epoch in range(start_epoch, effective_max_epochs):
        epoch_start_t = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        train_iter = (
            train_loader if tqdm is None
            else tqdm(train_loader, desc=f"epoch {epoch}/{effective_max_epochs - 1} train", leave=False)
        )
        for img, label in train_iter:
            img = img.to(device, non_blocking=True)
            target = label.view(-1, 1).to(device, non_blocking=True)
            logits = model(img)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        val_loss_mean, val_scores, val_labels = _val_pass(model, val_loader, criterion, device)
        val_pred = (val_scores >= 0.5).astype(int)
        val_f1 = f1_per_error(val_labels, val_pred)
        val_pr_auc = pr_auc_per_error(val_labels, val_scores)
        epoch_wall = time.perf_counter() - epoch_start_t
        logger.info("epoch=%d val_loss=%.6f val_f1=%.4f val_pr_auc=%.4f wall=%.1fs",
                    epoch, val_loss_mean, val_f1, val_pr_auc, epoch_wall)

        metrics_history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_loss_per_batch": train_losses,
            "val_loss_mean": val_loss_mean,
            "val_f1": val_f1,
            "val_pr_auc": val_pr_auc,
            "epoch_wall_time_s": epoch_wall,
        })

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "metrics_history": metrics_history,
            "best_f1_val": best_f1_val,
            "best_thresholds": None,
            "config_hash": config_hash_str,
            "config_repr": config_repr,
            "code_version": "phase07-image-baseline",
        }
        last_ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        atomic_save_checkpoint(payload, last_ckpt_path)

        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            payload["best_f1_val"] = best_f1_val
            atomic_save_checkpoint(payload, best_ckpt_path, update_latest=False)
            logger.info("New best val_f1=%.4f -> wrote best.pt", best_f1_val)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        prune_checkpoints(run_dir, keep_last=3, keep_best=True)

        if epochs_since_improve >= config.early_stop_patience:
            logger.info("Early stopping at epoch %d (best=%.4f after %d without improvement)",
                        epoch, best_f1_val, epochs_since_improve)
            break

    return {
        "epoch": epoch,
        "metrics_history": metrics_history,
        "best_f1_val": best_f1_val,
        "best_thresholds": None,
        "checkpoint_path": last_ckpt_path,
        "best_checkpoint_path": best_ckpt_path,
        "config_hash": config_hash_str,
    }
