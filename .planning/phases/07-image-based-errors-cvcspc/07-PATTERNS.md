# Phase 7: Image-Based Errors (CVCSPC) — Pattern Map

**Mapped:** 2026-05-30
**Files analyzed:** 6 new/modified files
**Analogs found:** 6 / 6

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `datasets/shallow_squat.py` | dataset | request-response (CPU, image I/O) | `datasets/squat.py` | role-match (video→image; same split/label/pos_weight contract) |
| `datasets/cvcspc_ssl.py` | dataset | event-driven (triplet sampling) | `datasets/squat_ssl.py` | role-match (triplet contract; phase signal differs) |
| `harness/image_train.py` | trainer | request-response (GPU, CRUD loop) | `harness/supervised_train.py` | exact (same checkpoint/resume/early-stop/threshold-sweep seam; 2D vs 3D backbone only) |
| `harness/cvcspc_pretrain.py` | trainer | event-driven (triplet SSL loop) | `harness/md_pretrain.py` | exact (same SSL payload/backbone.pt/linear-probe seam; triplet-accuracy replaces linear-probe) |
| `harness/colab.py` | utility | file-I/O (staging + frame extract) | `harness/colab.py` (self) | self-extension (add two new staging fns; all primitives unchanged) |
| `datasets/test_shallow_squat.py` | test | request-response (pytest) | `datasets/test_ohp.py` | exact (pos_weight, shape, split-count, dataset_cls seam pattern) |

---

## Pattern Assignments

### `datasets/shallow_squat.py` (dataset, image I/O)

**Analog:** `backend/training/aqa/datasets/squat.py`

**Imports pattern** (squat.py lines 19–37):
```python
from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import splits
```
For `shallow_squat.py`: drop all `transforms.py` imports (no video decode). Add PIL and torchvision.transforms instead. Do NOT import from `transforms.py` — it carries Kinetics norm.

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
```

**pos_weight pattern** (squat.py lines 41–58, ohp.py lines 48–64):
```python
def _compute_pos_weight(train_records: list) -> torch.Tensor:
    n_total = len(train_records)
    pos = sum(r[1] for r in train_records)   # label is the second element of each (id, label) tuple
    w = (n_total - pos) / max(pos, 1)
    logger.info("pos_weight from train (N=%d): pos=%d -> w=%.4f", n_total, pos, w)
    return torch.tensor([w], dtype=torch.float32)
```
`pos_weight` is computed from the **train split regardless of which split this instance represents** — same invariant as squat.py lines 114–120 and ohp.py. The scalar form `[w]` (not `[w_kie, w_kfe]`) because Shallow-Squat is binary single-label.

**Dataset class structure** (squat.py lines 61–161, adapted):
```python
class ShallowSquatDataset(Dataset):
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
        labels = json.loads(Path(labels_path).read_text())
        ids = json.loads(Path(splits_root, f"{split}_ids.json").read_text())
        train_ids = json.loads(Path(splits_root, "train_ids.json").read_text())

        self.records: list[tuple[str, int]] = [
            (i, labels[i]) for i in ids if i in labels
        ]
        train_records = [(i, labels[i]) for i in train_ids if i in labels]
        self.pos_weight: torch.Tensor = _compute_pos_weight(train_records)
        self.images_root = Path(images_root)

        _imagenet_mean = [0.485, 0.456, 0.406]
        _imagenet_std  = [0.229, 0.224, 0.225]
        self.transform = (
            T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2),
                T.ToTensor(),
                T.Normalize(_imagenet_mean, _imagenet_std),
            ])
            if train_aug else
            T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(_imagenet_mean, _imagenet_std),
            ])
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        id_, label = self.records[idx]
        img = Image.open(self.images_root / f"{id_}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
```

**Key difference from squat.py:** No `count_frames`, no `decode_clip`, no `uniform_sample_indices`, no Kinetics norm. The `images_root` points to the `crops_unaligned/` subdirectory (after zip extraction), so `__getitem__` does `images_root / f"{id_}.jpg"` directly.

**`build_loaders` factory** (squat.py lines 164–208):
```python
def build_loaders(
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    _persistent = num_workers > 0
    train_ds = ShallowSquatDataset(split="train", train_aug=True,  **dataset_kwargs)
    val_ds   = ShallowSquatDataset(split="val",   train_aug=False, **dataset_kwargs)
    test_ds  = ShallowSquatDataset(split="test",  train_aug=False, **dataset_kwargs)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, persistent_workers=_persistent,
                            worker_init_fn=seed_worker, drop_last=False),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=_persistent,
                            worker_init_fn=seed_worker, drop_last=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=_persistent,
                            worker_init_fn=seed_worker, drop_last=False),
    }
```
Note: unlike squat.py's Phase 2 constraint, Phase 7 does NOT clamp `num_workers=0`. Use the supervised_train.py pattern of passing `num_workers` and `persistent_workers` directly.

---

### `datasets/cvcspc_ssl.py` (dataset, event-driven triplet sampling)

**Analog:** `backend/training/aqa/datasets/squat_ssl.py`

**Imports pattern** (squat_ssl.py lines 1–49):
```python
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
```
Drop `scipy.ndimage`, `decode_clip`, `ssl_augs` (video-specific). Add PIL and T for still-frame augmentation.

**Dataset `__init__` structure** (squat_ssl.py lines 107–145, adapted):

The CVCSPC dataset pairs two different videos (not two half-cycles of one video). The clip IDs come from the unlabeled set; the trajectory JSONs provide the phase signal.

```python
class ShallowSquatSSLDataset(Dataset):
    def __init__(
        self,
        *,
        frames_root: str,          # /content/squat_ssl_frames/{video_id}/*.jpg
        trajectories_root: str,    # /content/squat_trajectories/*.json
        ssl_contrastive_phase_gap: float = 30.0,
        mask_prob: float = 0.5,
        mask_amt_lo: float = 0.4,
        mask_amt_hi: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.frames_root = Path(frames_root)
        self._traj_paths: dict[str, Path] = {
            p.stem: p for p in Path(trajectories_root).rglob("*.json")
        }
        # Clip IDs = video_id dirs present in frames_root AND have trajectory JSONs.
        frame_ids = {p.name for p in self.frames_root.iterdir() if p.is_dir()}
        self._clip_ids: list[str] = sorted(frame_ids & set(self._traj_paths))
        self.phase_gap = ssl_contrastive_phase_gap
        self.mask_prob = mask_prob
        self.mask_amt_lo = mask_amt_lo
        self.mask_amt_hi = mask_amt_hi
        ...
```
Mirror squat_ssl.py's `vid_stems & set(self._traj_paths)` intersection pattern (lines 136–140).

**Trajectory phase computation** (RESEARCH §CVCSPC; adapted from official dataloader.py):
```python
def _load_phase(self, clip_id: str) -> np.ndarray:
    path = self._traj_paths[clip_id]
    with open(path, encoding="utf-8") as fh:
        traj = np.asarray(json.load(fh), dtype=float)
    # Normalize to [0,1] then convert to degrees — official dataloader.py lines 35-39
    traj_min, traj_max = np.nanmin(traj), np.nanmax(traj)
    denom = max(traj_max - traj_min, 1e-8)
    phase = ((traj - traj_min) / denom) * 360.0
    return phase
```
Mirror squat_ssl.py's `_load_trajectory` NaN-interpolation pattern (lines 149–167) before calling `_load_phase`.

**Triplet `__getitem__`** (squat_ssl.py lines 210–242, adapted):

The CVCSPC contrast differs from MD-SSL: anchor and positive come from **two different videos at the same phase**; negative comes from the same second video at a phase `>= phase_gap` away.

```python
def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    # Anchor video: self._clip_ids[idx]
    # Sample a second video randomly (different from anchor)
    # Find the shared phase overlap between the two trajectories
    # Anchor frame: frame from v0 nearest to a random shared phase p_anchor
    # Positive frame: frame from v1 nearest to p_anchor
    # Negative frame: frame from v1 at a phase >= phase_gap from p_anchor
    # Apply masking augmentation independently to each
    # Return {"anchor": tensor, "positive": tensor, "negative": tensor}
    ...
```
The masking augmentation (top 40–50% blackout) is applied after transform. Mirror squat_ssl.py's `_augment` coin-flip pattern (lines 169–208) but with only masking active (all other augs are commented out in the official code).

**`build_ssl_loader`** (squat_ssl.py lines 252–272):
```python
def build_cvcspc_loader(dataset, config, *, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    _persistent = config.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=True,
        persistent_workers=_persistent,
    )
```
Copy from squat_ssl.py lines 252–272 verbatim, renaming `build_ssl_loader` → `build_cvcspc_loader`.

---

### `harness/image_train.py` (trainer, CRUD loop)

**Analog:** `backend/training/aqa/harness/supervised_train.py`

**Imports pattern** (supervised_train.py lines 40–68):
```python
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
```
Replace `SquatKIEKFEDataset` with `ShallowSquatDataset`. All colab harness imports are identical.

**Config dataclass** (supervised_train.py lines 77–105):
```python
@dataclasses.dataclass
class ImageConfig:
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
```
Mirror `SupervisedConfig.__post_init__` (supervised_train.py lines 101–105) verbatim.

**`build_model()`** (supervised_train.py lines 114–136, adapted):
```python
def build_baseline_model() -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert model.fc.in_features == 512, (
        f"ResNet-18 fc.in_features={model.fc.in_features}, expected 512"
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: ResNet-18 + Linear(512, 1) head, %d params", n_params)
    return model
```
Key differences from `build_model()`: `resnet18` not `r2plus1d_18`; `IMAGENET1K_V1` not `KINETICS400_V1`; `Linear(512, 1)` not `Linear(512, 2)` (binary).

**`_build_dataloaders`** (supervised_train.py lines 192–278):

Copy the function signature with `dataset_cls=ShallowSquatDataset` as the default. The `common_kwargs` block must drop video-specific keys (`num_frames`, `crop_size`, `train_jitter_frames`) and instead pass `images_root`, `labels_path`, `splits_root`:

```python
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
    train_ds = dataset_cls(split="train", train_aug=True,  **common_kwargs)
    val_ds   = dataset_cls(split="val",   train_aug=False, **common_kwargs)
    test_ds  = dataset_cls(split="test",  train_aug=False, **common_kwargs)
    ...  # DataLoader construction identical to supervised_train.py lines 233–278
```
The `persistent_workers` guard (supervised_train.py lines 244–246) and `seed_worker` (lines 144–161) are copied verbatim.

**`_val_pass`** (supervised_train.py lines 281–330):

Copy verbatim with one change: `logits` shape is `[B, 1]` not `[B, 2]`, so the squeeze and metric calls change:
```python
# In _val_pass for binary:
scores = torch.sigmoid(torch.cat(logits_chunks)).squeeze(-1).numpy()   # [N]
labels_arr = torch.cat(labels_chunks).numpy().astype(int)               # [N]
```
The `f1_per_error` / `pr_auc_per_error` calls then operate on 1-D arrays (single error head).

**`run_image_epoch` entry point** (supervised_train.py lines 338–610):

The signature and body mirror `run_supervised_epoch` closely:
```python
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
    checkpoint_phase: str = "phase07",
) -> dict:
```

**Checkpoint payload schema** (supervised_train.py lines 562–575):
```python
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
```
Identical key set to supervised_train.py. `best_thresholds` is filled in post-training by a threshold sweep cell, not inside the loop. `best.pt` is written with `atomic_save_checkpoint(payload, best_ckpt_path)` (update_latest default True is wrong for best.pt — copy the supervised_train.py pattern which uses the same call; latest.txt points to epoch_NNN.pt, not best.pt, because best.pt is written on val-F1 improvement as a second save alongside the epoch checkpoint).

**Val metrics for binary** — replace the two-head macro-F1 with single-head F1:
```python
val_pred = (val_scores >= 0.5).astype(int)
val_f1   = f1_per_error(val_labels, val_pred)
val_pr_auc = pr_auc_per_error(val_labels, val_scores)
# best.pt guard uses val_f1 directly (not macro average of two heads)
```

**Return dict** (supervised_train.py lines 602–610):
```python
return {
    "epoch": epoch,
    "metrics_history": metrics_history,
    "best_f1_val": best_f1_val,
    "best_thresholds": None,
    "checkpoint_path": last_ckpt_path,
    "best_checkpoint_path": best_ckpt_path,
    "config_hash": config_hash_str,
}
```
Identical key set. The `dataset_cls` seam (supervised_train.py line 194, 347) is preserved so CVCSPC fine-tune can pass a patched dataset class that loads SSL-pretrained weights.

---

### `harness/cvcspc_pretrain.py` (trainer, event-driven SSL)

**Analog:** `backend/training/aqa/harness/md_pretrain.py`

**Imports pattern** (md_pretrain.py lines 22–48):
```python
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

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.datasets.cvcspc_ssl import ShallowSquatSSLDataset, build_cvcspc_loader
from backend.training.aqa.eval.metrics import f1_per_error, pr_auc_per_error
from backend.training.aqa.harness.colab import (
    atomic_save_checkpoint,
    capture_rng_state,
    hash_config,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
)
```

**SSL checkpoint key set** (md_pretrain.py lines 52–65):
```python
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
```
Replace `linear_probe_history` with `triplet_acc_history` — CVCSPC uses triplet accuracy (AP < AN) as the convergence monitor, not a linear probe (no video labeled loader required at SSL time).

**Config dataclass** (md_pretrain.py lines 68–95):
```python
@dataclasses.dataclass
class CVCSPCConfig:
    learning_rate: float = 1e-4       # [CITED: paper §5]
    weight_decay: float = 0.0         # [ASSUMED — paper silent; AdamW not mentioned]
    batch_size: int = 25              # [CITED: paper §5]
    num_workers: int = 4
    max_epochs: int = 100             # [CITED: paper §5]
    projector_hidden: int = 128       # [ASSUMED — 512->128->128 SimCLR-style]
    projector_out_dim: int = 128      # [ASSUMED]
    phase_gap_start: float = 30.0     # [CITED: train_test.py lines 172-179]
    phase_gap_decay_every: int = 4    # [CITED: train_test.py lines 172-179]
    mask_prob: float = 0.5            # [CITED: dataloader.py lines 219-231]
    mask_amt_lo: float = 0.4          # [CITED: dataloader.py lines 219-231]
    mask_amt_hi: float = 0.5          # [CITED: dataloader.py lines 219-231]
    scheduler_name: str = "cosine"    # [ASSUMED]
    model_arch: str = "resnet18_cvcspc_ssl_v1"

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs
```
Mirror MDConfig's `__post_init__` (md_pretrain.py lines 94–95).

**`ProjectionHead`** (md_pretrain.py lines 98–118, adapted):
```python
class CVCSPCProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, p=2)
```
Difference from MD-SSL's `ProjectionHead`: NO BatchNorm1d between layers (BatchNorm causes issues with small batch sizes like 25; CVCSPC paper does not mention BN in the head). The L2-normalize call (md_pretrain.py line 118) is kept.

**`cvcspc_triplet_loss`** (md_pretrain.py lines 121–162, adapted):
```python
def cvcspc_triplet_loss(
    phi_anc: torch.Tensor,
    phi_pos: torch.Tensor,
    phi_neg: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Three-term denominator — official train_test.py lines 64-68.
    # Squared L2 on L2-normalized inputs.
    d_ap = ((phi_anc - phi_pos) ** 2).sum(-1)
    d_an = ((phi_anc - phi_neg) ** 2).sum(-1)
    d_pn = ((phi_pos - phi_neg) ** 2).sum(-1)
    num = torch.exp(-d_ap)
    den = num + torch.exp(-d_an) + torch.exp(-d_pn)
    return (-torch.log(num / (den + eps))).mean()
```
Mirror `md_triplet_loss` (md_pretrain.py lines 121–162) but without the `squared`/`three_term` ablation flags — Phase 7 implements the code version (3-term, squared) as canonical and documents the paper deviation in FINDINGS. Flags can be added later if needed.

**`build_cvcspc_model()`** (md_pretrain.py lines 165–189, adapted):
```python
def build_cvcspc_model(config: CVCSPCConfig) -> tuple[nn.Module, nn.Module]:
    from torchvision.models import ResNet18_Weights, resnet18
    backbone = resnet18(weights=None)
    # Load ImageNet weights manually, filtering to matching keys
    # (mirrors train_test.py lines 235-241)
    pretrained_sd = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    backbone.load_state_dict(pretrained_sd, strict=False)
    assert backbone.fc.in_features == 512
    backbone.fc = nn.Identity()
    projector = CVCSPCProjectionHead(
        in_dim=512,
        hidden=config.projector_hidden,
        out_dim=config.projector_out_dim,
    )
    ...
    return backbone, projector
```
Mirror `build_md_model()` (md_pretrain.py lines 165–189): separate backbone+projector, backbone.fc=Identity, return both. Key difference: `resnet18` (2D) not `r2plus1d_18` (3D).

**`build_ssl_checkpoint_payload()`** (md_pretrain.py lines 192–223):

Copy verbatim, replacing `linear_probe_history` with `triplet_acc_history` and `"phase04-md-pretrain"` with `"phase07-cvcspc-pretrain"`:
```python
def build_cvcspc_checkpoint_payload(
    *,
    epoch: int,
    backbone: nn.Module,
    projector: nn.Module,
    optimizer,
    scheduler,
    metrics_history: list,
    triplet_acc_history: list,
    config_hash: str,
    config_repr: dict,
) -> dict:
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
```

**Triplet-accuracy convergence monitor** (replaces `_linear_probe` from md_pretrain.py lines 288–312):
```python
def _triplet_accuracy(
    backbone: nn.Module,
    projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Fraction of triplets where ||phi_anc - phi_pos|| < ||phi_anc - phi_neg||.

    Monitor from train_test.py lines 130-134. Runs frozen, no gradient.
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
            total += d_ap.shape[0]
    return correct / max(total, 1)
```

**`run_cvcspc_pretrain_epoch` entry point** (md_pretrain.py lines 315–525):

Mirror `run_md_pretrain_epoch` exactly. Key substitutions:
- `ssl_dataset_cls=SquatSSLDataset` → `ssl_dataset_cls=ShallowSquatSSLDataset`
- `probe_dataset_cls` → removed (no labeled-video probe loader needed)
- `build_ssl_loader` → `build_cvcspc_loader`
- `md_triplet_loss` → `cvcspc_triplet_loss`
- `build_md_model` → `build_cvcspc_model(config)`
- `_linear_probe` call every `N` epochs → `_triplet_accuracy` call every `N` epochs (default every 5, same cadence)
- `backbone.pt` on `triplet_acc > best_triplet_acc` (replacing `linear_probe_macro`) with `update_latest=False` (md_pretrain.py lines 498–504)
- Phase-gap anneal inside the epoch loop (no analog in md_pretrain — new behavior):
  ```python
  # After each epoch:
  if epoch > 0 and config.phase_gap_start > 30.0 and epoch % config.phase_gap_decay_every == 0:
      config.phase_gap_start = max(30.0, config.phase_gap_start - 1.0)
      loader.dataset.phase_gap = config.phase_gap_start
  ```
- Collapse abort NOT needed (ResNet-18 with ImageNet init is far more stable than R(2+1)D-18 from Kinetics on a small unlabeled set)

**`run_cvcspc_pretrain_epoch` return dict** (md_pretrain.py lines 517–525):
```python
return {
    "epoch": epoch,
    "metrics_history": metrics_history,
    "triplet_acc_history": triplet_acc_history,
    "backbone_path": backbone_path,
    "checkpoint_path": last_ckpt_path,
    "config_hash": config_hash_str,
}
```

---

### `harness/colab.py` — ADDITIONS ONLY (utility, file-I/O)

**Analog:** `harness/colab.py` itself (self-extension). All existing primitives are unchanged.

**New function 1: `stage_shallow_squat_images`** (analog: `stage_ohp_videos`, colab.py lines 552–650):
```python
_SHALLOW_SQUAT_IMAGES_EXPECT_COUNT = 3738  # images.zip ships 3,738 JPEGs

def stage_shallow_squat_images(
    drive_root_3001: str,
    *,
    local_root: str = "/content/squat_shallow_images",
    expect_count: int = _SHALLOW_SQUAT_IMAGES_EXPECT_COUNT,
) -> str:
    """Copy Shallow-Squat images.zip from Drive (-3-001) → /content/, extract, verify count.

    Source: {drive_root_3001}/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/
                              Shallow_Squat_Error_Dataset/images.zip
    Extracts to local_root/crops_unaligned/{id}.jpg

    Uses zipfile.extractall (JPEG members — NOT _extract_with_resume_and_progress which
    is mp4-only). Mirror stage_unlabeled_squat_videos' trajectory-zip path for JSON/JPEG zips.
    """
```
Key difference from `stage_ohp_videos`: uses `zipfile.extractall` (not `_extract_with_resume_and_progress`) because the zip contains JPEG files, not mp4. Mirror the trajectory-zip branch of `stage_unlabeled_squat_videos` (colab.py lines 518–522). The expect_count check counts `*.jpg` files recursively under `local_root`, not `*.mp4`.

**New function 2: `extract_frames_for_ssl`** (no direct analog; closest is the body of `_extract_with_resume_and_progress`, colab.py lines 168–244):
```python
def extract_frames_for_ssl(
    videos_root: str,
    frames_root: str,
    *,
    skip_existing: bool = True,
) -> int:
    """Decode each unlabeled Squat mp4 → per-clip JPEG frame dirs for CVCSPC.

    Output layout: {frames_root}/{video_id}/frame_{:06d}.jpg
    Required by the CVCSPC dataloader which indexes frames by directory
    (dataloader.py lines 130-131: os.listdir(ssl_frames_dir + video_id + '/')).

    skip_existing: if {video_id}/ dir already exists with at least 1 frame, skip.
    Uses cv2.VideoCapture (already in requirements.txt). tqdm progress (cell >2 min).
    Returns total frames extracted.
    """
    import cv2
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    videos_root_p = Path(videos_root)
    frames_root_p = Path(frames_root)
    frames_root_p.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(videos_root_p.glob("*.mp4"))
    total_frames = 0
    ...
    # for each mp4: check skip_existing, create subdir, cv2.VideoCapture read loop,
    # write JPEG with cv2.imwrite, tqdm update
```
Mirror the `_do_one` skip-existing pattern from `_extract_with_resume_and_progress` (colab.py lines 218–226). The disconnect-safe skip check is: `if (frames_root_p / video_id).exists() and any((frames_root_p / video_id).iterdir()): skip`.

---

### `datasets/test_shallow_squat.py` (test, pytest)

**Analog:** `backend/training/aqa/datasets/test_ohp.py`

**Imports + local archive guard** (test_ohp.py lines 1–26):
```python
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_LOCAL_SQUAT_3001 = Path(
    "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
)
_LOCAL_ARCHIVE_EXISTS = _LOCAL_SQUAT_3001.exists()
```
Mirror the `_LOCAL_ARCHIVE_EXISTS` guard (test_ohp.py lines 23–26). The Shallow-Squat images are in the `-3-001` release folder.

**Test 1 — split sizes** (test_ohp.py `test_ohp_splits_count` lines 28–55):
```python
@pytest.mark.skipif(not _LOCAL_ARCHIVE_EXISTS, reason="Local archive not found")
def test_split_sizes() -> None:
    """IMG-01: ShallowSquatDataset returns 2542/529/540 records for train/val/test."""
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
    images_root = str(_LOCAL_SQUAT_3001 / "...")
    labels_path = str(_LOCAL_SQUAT_3001 / ".../labels_shallow_depth.json")
    splits_root = str(_LOCAL_SQUAT_3001 / ".../splits")
    for split_name, expected in [("train", 2542), ("val", 529), ("test", 540)]:
        ds = ShallowSquatDataset(split=split_name, images_root=images_root,
                                 labels_path=labels_path, splits_root=splits_root,
                                 train_aug=False)
        assert len(ds) == expected
```

**Test 2 — pos_weight formula** (test_ohp.py `test_ohp_pos_weight` lines 62–77):
```python
def test_pos_weight() -> None:
    """IMG-01: pos_weight = (N - pos) / max(pos, 1); approx 1.28 on synthetic near-balanced data."""
    from backend.training.aqa.datasets.shallow_squat import _compute_pos_weight
    # 4 records: 2 positive -> w = (4-2)/2 = 1.0
    records = [("a", 1), ("b", 0), ("c", 1), ("d", 0)]
    pw = _compute_pos_weight(records)
    assert pw.shape == (1,)
    assert float(pw[0]) == pytest.approx(1.0, rel=1e-6)
```

**Test 3 — `__getitem__` tensor shape** (test_ohp.py `test_ohp_dataset_shape` lines 99–145, adapted):
```python
def test_getitem_shape(tmp_path: Path) -> None:
    """IMG-01: __getitem__ returns (tensor[3,224,224] float32, scalar float32)."""
    from PIL import Image
    import numpy as np
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset

    crops_dir = tmp_path / "crops_unaligned"
    crops_dir.mkdir()
    ids = ["v1_0_0", "v1_0_1"]
    for id_ in ids:
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(crops_dir / f"{id_}.jpg")

    import json
    labels = {id_: i % 2 for i, id_ in enumerate(ids)}
    (tmp_path / "labels_shallow_depth.json").write_text(json.dumps(labels))
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split_name in ("train", "val", "test"):
        (splits_dir / f"{split_name}_ids.json").write_text(json.dumps(ids))

    ds = ShallowSquatDataset(
        split="train",
        images_root=str(crops_dir),
        labels_path=str(tmp_path / "labels_shallow_depth.json"),
        splits_root=str(splits_dir),
        train_aug=False,
    )
    img_tensor, label = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert img_tensor.dtype == torch.float32
    assert label.shape == ()
    assert float(label) in (0.0, 1.0)
```

**Test 4 — ImageNet norm range** (no direct analog; pattern implied by test_ohp.py):
```python
def test_imagenet_norm() -> None:
    """IMG-01: val transform produces pixel values in [-2.5, 2.5] (ImageNet norm)."""
    # synthetic all-white image (255,255,255) -> after ImageNet norm: (1-mean)/std
    # Rough bounds: pixel mean in [-3, 3] is sufficient
    ...
    img_tensor, _ = ds[0]
    assert float(img_tensor.min()) > -3.0
    assert float(img_tensor.max()) < 3.0
```

**Test 5 — `dataset_cls` injection seam** (test_ohp.py `test_dataset_cls_injection` lines 152–178):
```python
def test_dataset_cls_injection() -> None:
    """run_image_epoch and _build_dataloaders expose dataset_cls as keyword-only kwarg."""
    import inspect
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
    from backend.training.aqa.harness.image_train import _build_dataloaders, run_image_epoch

    sig_bd = inspect.signature(_build_dataloaders)
    assert "dataset_cls" in sig_bd.parameters
    assert sig_bd.parameters["dataset_cls"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_bd.parameters["dataset_cls"].default is ShallowSquatDataset

    sig_ri = inspect.signature(run_image_epoch)
    assert "dataset_cls" in sig_ri.parameters
    assert sig_ri.parameters["dataset_cls"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_ri.parameters["dataset_cls"].default is ShallowSquatDataset
```
Mirror test_ohp.py lines 152–178 exactly, substituting the Phase 7 symbols.

**Test 6 — `build_loaders` returns dict** (test_ohp.py pattern, pure/no I/O):
```python
def test_build_loaders(tmp_path: Path) -> None:
    """IMG-01: build_loaders returns dict with train/val/test DataLoader keys."""
    ...
    loaders = build_loaders(images_root=..., labels_path=..., splits_root=...,
                            batch_size=2, num_workers=0)
    assert set(loaders.keys()) == {"train", "val", "test"}
    from torch.utils.data import DataLoader
    for k, v in loaders.items():
        assert isinstance(v, DataLoader), f"{k} is not DataLoader"
```

---

## Shared Patterns

### Checkpoint / Resume Contract
**Source:** `harness/colab.py` (all phases)
**Apply to:** `image_train.py`, `cvcspc_pretrain.py`

The full contract is: `atomic_save_checkpoint(payload, epoch_path)` every epoch (updates `latest.txt`); `atomic_save_checkpoint(payload, best_path, update_latest=False)` for best/backbone writes (does NOT update `latest.txt`). `load_latest_checkpoint(run_dir, expected_config_hash=..., map_location="cpu")` on resume. `map_location="cpu"` is required (CUDA ByteTensors break `set_rng_state_all`). `prune_checkpoints(run_dir, keep_last=3, keep_best=True)` at end of every epoch.

```python
# The exact epoch-ckpt + best.pt pattern (supervised_train.py lines 576-591):
ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
atomic_save_checkpoint(payload, ckpt_path)              # update_latest=True (default)
if val_f1 > best_f1_val:
    best_f1_val = val_f1
    payload["best_f1_val"] = best_f1_val
    atomic_save_checkpoint(payload, best_ckpt_path)     # update_latest=True is fine here
    epochs_since_improve = 0
else:
    epochs_since_improve += 1
prune_checkpoints(run_dir, keep_last=3, keep_best=True)
```

### `seed_worker` / `_set_global_seed`
**Source:** `harness/supervised_train.py` lines 144–184
**Apply to:** `image_train.py`, `cvcspc_pretrain.py`, `shallow_squat.py`

Copy `seed_worker` and `_set_global_seed` verbatim. Do NOT import from `supervised_train.py` — copy per the Phase 4 D9 precedent (md_pretrain.py lines 226–244 copies instead of importing).

### `persistent_workers` guard
**Source:** `harness/supervised_train.py` lines 244–246
**Apply to:** `image_train.py`, `cvcspc_pretrain.py`, `shallow_squat.build_loaders`

```python
_persistent = config.num_workers > 0
```
Always pair with `worker_init_fn=seed_worker`. Required per `[[reference_pytorch_persistent_workers]]`.

### Config hash + resume protection
**Source:** `harness/supervised_train.py` lines 395–415
**Apply to:** `image_train.py`, `cvcspc_pretrain.py`

Every training-relevant field must be in `config_repr` and hashed via `hash_config`. Fields that do NOT invalidate resume (e.g., `run_name`, `drive_root`) must NOT be in `config_repr`. The `config_hash_str` goes into every checkpoint payload under the key `"config_hash"`.

### ImageNet norm constants
**Source:** RESEARCH.md §Code Examples; official dataloader.py lines 95–96
**Apply to:** `shallow_squat.py`, `cvcspc_ssl.py`

```python
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
```
These are module-level constants. Never import from `transforms.py` (Kinetics norm). Never mix with Kinetics stats.

### `tqdm` optional import
**Source:** `harness/supervised_train.py` lines 386–389; `harness/md_pretrain.py` lines 339–342
**Apply to:** `image_train.py`, `cvcspc_pretrain.py`, `harness/colab.py` new functions

```python
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
```
All cells taking >2 min must use this pattern. Iterator wrap: `loader if tqdm is None else tqdm(loader, desc=..., leave=False)`.

---

## No Analog Found

All files have analogs. No entries in this section.

---

## Metadata

**Analog search scope:** `backend/training/aqa/datasets/`, `backend/training/aqa/harness/`
**Files scanned:** `squat.py`, `squat_ssl.py`, `ohp.py`, `supervised_train.py`, `md_pretrain.py`, `colab.py`, `test_ohp.py`, `splits.py` (partial)
**Pattern extraction date:** 2026-05-30
