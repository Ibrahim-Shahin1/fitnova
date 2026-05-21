# Phase 04: Squat Motion-Disentangling SSL — Pattern Map

**Mapped:** 2026-05-21
**Files analyzed:** 10 new/modified files + 4 test files
**Analogs found:** 14 / 14

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `datasets/squat_ssl.py` | dataset | event-driven (triplet) | `datasets/squat.py` | role-match |
| `datasets/ssl_augs.py` | utility | transform | `datasets/transforms.py` | role-match |
| `harness/md_pretrain.py` | service | batch/training loop | `harness/supervised_train.py` | role-match |
| `harness/md_finetune.py` | service | batch/training loop | `harness/supervised_train.py` | exact |
| `eval/ensemble.py` | utility | transform | `eval/metrics.py` | role-match |
| `eval/tta.py` | utility | request-response | `eval/metrics.py` + `datasets/transforms.py` | role-match |
| `harness/colab.py` (MODIFY) | utility | file-I/O | `harness/colab.py` existing functions | exact |
| `notebooks/04_squat_md_ssl.py` + `.ipynb` | notebook | batch | `notebooks/03_squat_supervised_baseline.py` | exact |
| `harness/test_md_pretrain.py` | test | — | `harness/test_supervised_train.py` | exact |
| `harness/test_md_finetune.py` | test | — | `harness/test_supervised_train.py` | exact |
| `eval/test_ensemble.py` | test | — | `eval/test_metrics.py` | exact |
| `eval/test_tta.py` | test | — | `eval/test_metrics.py` | exact |

---

## Pattern Assignments

---

### `datasets/squat_ssl.py` (dataset, triplet/event-driven)

**Analog:** `backend/training/aqa/datasets/squat.py`

**Imports pattern** (`squat.py` lines 19–36):
```python
from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
import torchvision.io
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import splits
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    spatial_train,
    spatial_val,
    uniform_sample_indices,
)

logger = logging.getLogger("aqa.phase02")
```
Phase 4 analog: replace `splits` import with trajectory-JSON loading; add `scipy.ndimage`, `numpy` imports; add `ssl_augs` import. Logger name: `"aqa.phase04"`.

**Module docstring pattern** (`squat.py` lines 1–17):
```python
"""[Module role in pipeline — one paragraph].

Yields `{anchor, positive, negative}` dict of three float32 clips `[3, 16, H, W]`
from the 4,970 unlabeled Squat clips + barbell trajectories.

`build_ssl_loader(...)` returns a DataLoader with `shuffle=True`.
...
"""
```

**Dataset class skeleton** (`squat.py` lines 59–156):
```python
class SquatSSLDataset(Dataset):
    def __init__(
        self,
        *,
        videos_root: str,
        trajectories_root: str,
        frames_per_half: int = 16,
        crop_size: int = 112,
        seed: int = 42,
    ) -> None:
        ...
        self._spatial_fn: Callable[..., torch.Tensor] = spatial_train  # always train-aug in SSL

    def __len__(self) -> int:
        return len(self._clip_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1. Load trajectory JSON → split_half_cycles → descent/ascent indices
        # 2. decode_clip(path, descent_indices) → anchor_tchw (uint8)
        # 3. Clone anchor_tchw → apply ssl_aug independently → positive_tchw
        # 4. temporal-reverse either {anchor,positive} OR {negative} randomly (§3)
        # 5. spatial_train(anchor_tchw, ...) → anchor [3,16,H,W]; same for pos, neg
        # Returns {"anchor": tensor, "positive": tensor, "negative": tensor}
```

**DataLoader construction** (`squat.py` lines 186–203 — use this exact `persistent_workers` guard):
```python
# persistent_workers=True when num_workers>0 (D7 carry-forward landmine)
_persistent = config.num_workers > 0

ssl_loader = DataLoader(
    ssl_ds,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=True,
    drop_last=False,
    persistent_workers=_persistent,
)
```

**pos_weight analog:** the SSL dataset has no labels, so no `pos_weight` field. The fine-tune `build_finetune_model` caller gets `pos_weight` from `SquatKIEKFEDataset` (unchanged Phase 2 contract).

**Key difference from `squat.py`:** `__getitem__` returns `dict[str, torch.Tensor]` (not `tuple`). No `label` tensor. Trajectory-split logic is the only genuinely new code; everything else reuses `decode_clip`, `uniform_sample_indices`, `spatial_train` from `transforms.py`.

---

### `datasets/ssl_augs.py` (utility, transform)

**Analog:** `backend/training/aqa/datasets/transforms.py`

**Imports pattern** (`transforms.py` lines 10–18):
```python
from __future__ import annotations

import logging
import warnings
from typing import Final

import torch
import torchvision.io
import torchvision.transforms.functional as TF

logger = logging.getLogger("aqa.phase02")
```
Phase 4 analog: `"aqa.phase04"` logger; add `import random` for probabilistic aug application.

**Function signature pattern** (`transforms.py` lines 33–60, `uniform_sample_indices`):
```python
def <aug_name>(
    clip_tchw: torch.Tensor,          # [T, 3, H, W] uint8 — input BEFORE spatial_*
    *,
    <aug_param>: <type> = <default>,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """[one-line docstring].

    Args:
        clip_tchw: uint8 tensor [T, 3, H, W] from decode_clip.
        ...

    Returns:
        uint8 tensor [T, 3, H, W] (same dtype/shape — spatial_* applies afterward).
    """
```
All augmentations operate on `[T, 3, H, W]` uint8 **before** `spatial_train`. They return the same dtype so the `spatial_*` pipeline chain is unchanged. Applied **independently per branch** in `squat_ssl.py.__getitem__`.

**Key augmentations to implement** (§7 of RESEARCH):
- `temporal_shift(clip, max_shift=2)` — shift frame indices ±2 (matches Phase 3 jitter)
- `horizontal_flip(clip, p=0.5)` — `TF.hflip` on each frame
- `top_mask(clip, mask_frac=0.4)` — zero out top `mask_frac` rows
- `color_jitter(clip, brightness=0.2, contrast=0.2)` — `TF.adjust_brightness` / `adjust_contrast`
- `translation(clip, max_px=10)` — `TF.affine(translate=[dx,dy])`

**Smoke test pattern** (`transforms.py` lines 218–244):
```python
if __name__ == "__main__":
    fake = torch.randint(0, 255, (16, 3, 112, 112), dtype=torch.uint8)
    out = horizontal_flip(fake)
    assert out.shape == fake.shape and out.dtype == fake.dtype
    print("ssl_augs.py smoke: ok")
```

---

### `harness/md_pretrain.py` (service, batch/SSL training loop)

**Analog:** `backend/training/aqa/harness/supervised_train.py`

**Imports pattern** (`supervised_train.py` lines 41–68):
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
```
Phase 4 changes: add `torch.nn.functional as F` (for `F.normalize`); replace `squat` import with `squat_ssl.SquatSSLDataset`; add labeled-split import for linear-probe.

**Config dataclass pattern** (`supervised_train.py` lines 77–106):
```python
@dataclasses.dataclass
class MDConfig:
    """Hyperparameters for Phase 4 MD-SSL pretraining.

    All values cited inline to paper § or labeled [ASSUMED].
    """
    learning_rate: float = 1e-4           # [CITED: §5 p.9]
    weight_decay: float = 1e-4            # [ASSUMED — AdamW decoupled, D2/D3]
    batch_size: int = 8                   # [ASSUMED — paper 5 CITED; 8 for throughput]
    num_workers: int = 4                  # D7 carry; persistent_workers guard applies
    max_epochs: int = 60                  # [CITED baseline 20; ASSUMED extension ≤60]
    frames_per_half: int = 16             # [CITED: §5 p.9]
    crop_size: int = 112                  # Phase 2/3 contract
    linear_probe_cadence: int = 5         # [ASSUMED — §6 N=5]
    projector_hidden: int = 512           # [ASSUMED — SimCLR 2-layer MLP]
    projector_out_dim: int = 128          # [ASSUMED — SimCLR/MoCo default]
    scheduler_name: str = "cosine"        # [ASSUMED — SSL standard]
    model_arch: str = "r2plus1d_18_md_ssl_v1"

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs
```

**Model construction pattern** (`supervised_train.py` lines 114–136):
```python
def build_md_model() -> tuple[nn.Module, nn.Module]:
    """Construct R(2+1)D-18 backbone (fc=Identity) + ProjectionHead.

    Returns:
        (backbone, projector) — kept separate so fine-tune loads only backbone.
    """
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    assert backbone.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={backbone.fc.in_features}, expected 512"
    )
    backbone.fc = nn.Identity()           # expose 512-d pooled features
    projector = ProjectionHead(in_dim=512, hidden=512, out_dim=128)
    return backbone, projector
```

**SSL checkpoint payload schema** (RESEARCH §10 + `supervised_train.py` lines 554–567):
```python
payload = {
    "epoch": epoch,
    "backbone_state_dict": backbone.state_dict(),
    "projector_state_dict": projector.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "rng_state": capture_rng_state(),
    "metrics_history": metrics_history,          # SSL loss + embedding_std per epoch
    "linear_probe_history": linear_probe_history,  # linear-probe F1 every N epochs
    "config_hash": config_hash_str,
    "config_repr": config_repr,
    "code_version": "phase04-md-pretrain",
}
```
`atomic_save_checkpoint(payload, ckpt_path)` — per epoch, per the Phase 2 contract.
`atomic_save_checkpoint(backbone_payload, backbone_path, update_latest=False)` — for `backbone.pt` (the new kwarg added to `colab.py`; prevents clobbering `latest.txt` resume pointer).

**Resume pattern** (`supervised_train.py` lines 436–468):
```python
if resume:
    prior = load_latest_checkpoint(
        run_dir, expected_config_hash=config_hash_str, map_location="cpu",
        # map_location='cpu' REQUIRED — D7 landmine; Phase 3 fix a0841b4
    )
    if prior is not None:
        backbone.load_state_dict(prior["backbone_state_dict"])
        projector.load_state_dict(prior["projector_state_dict"])
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        scheduler.load_state_dict(prior["scheduler_state_dict"])
        restore_rng_state(prior["rng_state"])
        metrics_history = list(prior["metrics_history"])
        linear_probe_history = list(prior.get("linear_probe_history", []))
        start_epoch = int(prior["epoch"]) + 1
```

**Train epoch pattern** (`supervised_train.py` lines 491–551 — adapt for 3-branch SSL):
```python
# Train pass — SSL version: dict batch not (clip, label) tuple
model.train()   # backbone + projector both in train mode
for batch in train_iter:
    anchor = batch["anchor"].to(device, non_blocking=True)    # [B, 3, 16, 112, 112]
    positive = batch["positive"].to(device, non_blocking=True)
    negative = batch["negative"].to(device, non_blocking=True)

    phi_anc = projector(backbone(anchor))    # L2-norm inside ProjectionHead
    phi_pos = projector(backbone(positive))
    phi_neg = projector(backbone(negative))

    loss = md_triplet_loss(phi_anc, phi_pos, phi_neg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(float(loss.item()))

scheduler.step()  # per-epoch, AFTER train pass (same as supervised_train.py:516)
```

**prune_checkpoints call** (`supervised_train.py` line 584):
```python
prune_checkpoints(run_dir, keep_last=3, keep_best=True)  # keep backbone.pt as "best"
```

**seed_worker and _set_global_seed:** copy verbatim from `supervised_train.py` lines 144–179. Do not re-import from `supervised_train.py` — new module, same body (D9 rule: do NOT bloat the Phase 3 module).

---

### `harness/md_finetune.py` (service, batch/training loop)

**Analog:** `backend/training/aqa/harness/supervised_train.py` — near-exact copy with 4 deltas.

**Config dataclass** (`supervised_train.py` lines 77–106 — copy with these changes):
```python
@dataclasses.dataclass
class FinetuneConfig:
    learning_rate: float = 1e-4            # [CITED: §5/§9 — same LR as SSL, no separate downstream LR]
    weight_decay: float = 1e-4             # [LOCKED D3 — AdamW decoupled wd=1e-4]
    dropout: float = 0.2                   # [LOCKED D3 — head-only dropout]
    batch_size: int = 16                   # Phase 3 contract (VRAM measured OK)
    num_workers: int = 4                   # same as Phase 3
    max_epochs: int = 50                   # [LOCKED D3 — P3-vs-P4 parity]
    early_stop_patience: int = 8           # [LOCKED D3 — P3-vs-P4 parity]
    num_frames: int = 32                   # Phase 2/3 contract
    crop_size: int = 112                   # Phase 2/3 contract
    train_jitter_frames: int = 2           # Phase 3 contract
    flip_aug: bool = False                 # [LOCKED Phase 2 D5 — flip OFF]
    scheduler_name: str = "cosine"
    model_arch: str = "r2plus1d_18_md_finetuned"
    loss_name: str = "BCEWithLogitsLoss(pos_weight=dataset.pos_weight)"
    md_backbone_path: str = ""             # path to backbone.pt from md_pretrain

    def __post_init__(self) -> None:
        self.scheduler_t_max = self.max_epochs
```

**Model construction** — Delta 1 (vs `supervised_train.py` lines 114–136):
```python
def build_finetune_model(md_backbone_path: str) -> nn.Module:
    """Load MD backbone (NOT Kinetics) + fresh Dropout(0.2)+Linear(512,2) head.

    Deviation from Phase 3 build_model(): weights=None (no Kinetics); load
    backbone_state_dict from MD pretrain checkpoint; head is Dropout+Linear.
    """
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(weights=None)      # architecture only — NO Kinetics init
    assert model.fc.in_features == 512
    ckpt = torch.load(md_backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"], strict=False)  # projector keys absent
    model.fc = nn.Sequential(              # D3: head-only dropout before Linear
        nn.Dropout(config.dropout),
        nn.Linear(512, 2),
    )
    return model
```

**Optimizer** — Delta 2 (`supervised_train.py` line 420):
```python
# Phase 3 (DO NOT USE):
# optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Phase 4 (md_finetune.py — use AdamW):
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,    # true decoupled weight decay — semantic difference from Adam
)
```

**D6 runtime overfit monitor** — Delta 3 (INSERT after val_loss_mean computation):
```python
# D6 Monitor 1: train/val BCE loss ratio > 10× before epoch 10 → abort + bump reg
train_val_ratio = train_loss_mean / (val_loss_mean + 1e-9)
metrics_history[-1]["train_val_loss_ratio"] = train_val_ratio
if epoch < 10 and train_val_ratio > 10.0:
    logger.warning(
        "D6 ABORT: train/val loss ratio %.1f > 10 at epoch %d "
        "(seed=%d). Abort this seed; bump wd=5e-4, dropout=0.3.",
        train_val_ratio, epoch, seed,
    )
    # Return early with a flag so the notebook cell can abort and reconfigure.
    return {
        "epoch": epoch, "metrics_history": metrics_history,
        "best_f1_val": best_f1_val, "aborted_overfit": True,
        "checkpoint_path": last_ckpt_path, "best_checkpoint_path": best_ckpt_path,
        "config_hash": config_hash_str,
    }
```

**Checkpoint payload** — Delta 4: replace `"model_state_dict"` key with the same name (fine-tune checkpoint mirrors Phase 3 schema exactly so Phase 3 tools can load it). Change `code_version`:
```python
payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),    # same key as Phase 3 — tools compatible
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "rng_state": capture_rng_state(),
    "metrics_history": metrics_history,
    "best_f1_val": best_f1_val,
    "best_thresholds": None,
    "config_hash": config_hash_str,
    "config_repr": config_repr,
    "code_version": "phase04-md-finetune",     # changed from phase03
}
```

**Everything else** (DataLoader construction, `_val_pass`, early-stop, `best.pt` write, `prune_checkpoints`) copies verbatim from `supervised_train.py`. The `_build_dataloaders` function is identical — it constructs `SquatKIEKFEDataset` (Phase 2 labeled pipeline, unchanged per D9).

---

### `eval/ensemble.py` (utility, transform)

**Analog:** `backend/training/aqa/eval/metrics.py`

**Module pattern** (`metrics.py` lines 1–29):
```python
"""[one-sentence role].

No torch, no I/O, no model construction. Pure aggregation over numpy score arrays.
Phase 4+ reuses this alongside `eval/metrics.py` (unchanged, D9).

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D4.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("aqa.phase04")
```

**Core function signature** (mirrors `f1_per_error` style, `metrics.py` lines 33–51):
```python
def aggregate_sigmoid_mean(
    per_seed_logits: list[np.ndarray],
) -> np.ndarray:
    """Mean of sigmoid scores across seeds. Output ∈ [0,1] — satisfies pr_auc_per_error assertion.

    Args:
        per_seed_logits: list of 3 float ndarrays, each shape (N, 2) — raw logits from
            each seed's model. Sigmoid applied here (D4: mean-of-sigmoids, NOT mean-of-logits).

    Returns:
        float ndarray shape (N, 2) — ensemble sigmoid scores in [0, 1].

    Raises:
        ValueError: inconsistent (N, 2) shapes across seeds.
    """
    if len(per_seed_logits) == 0:
        raise ValueError("aggregate_sigmoid_mean: empty per_seed_logits list")
    sigmoid_scores = [1.0 / (1.0 + np.exp(-logits)) for logits in per_seed_logits]
    ensemble = np.mean(sigmoid_scores, axis=0)
    assert ensemble.min() >= 0.0 and ensemble.max() <= 1.0  # satisfies pr_auc_per_error assertion
    return ensemble
```

**Threshold contract:** the single per-error threshold is tuned via `threshold_sweep` (reused from `eval/metrics.py`) on the ensemble val scores — NOT implemented in `ensemble.py`.

---

### `eval/tta.py` (utility, request-response)

**Analog:** `backend/training/aqa/eval/metrics.py` (structure) + `datasets/transforms.py` (augmentation primitives)

**Module pattern** (same as `ensemble.py` above — pure functions, no torch, no I/O):
```python
"""TTA forward pass + val-tuned recipe selection for Phase 4 ensemble.

See: .planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md — D5.
"""
from __future__ import annotations
import logging
import numpy as np
import torch
logger = logging.getLogger("aqa.phase04")
```

**Core function signatures:**
```python
def tta_forward(
    model: torch.nn.Module,
    clip: torch.Tensor,
    recipe: list[str],
    device: torch.device,
) -> np.ndarray:
    """Apply TTA augmentations, run each through model, return mean logits [2].

    Args:
        model:  fine-tuned model in eval mode.
        clip:   float tensor [3, 32, 112, 112] — already spatial_val-normalized.
        recipe: list of aug names from {"temporal_jitter", "spatial_5crop", "flip"}.
        device: inference device.

    Returns:
        float ndarray shape (2,) — mean logits over TTA copies (NOT sigmoid — caller
        applies sigmoid before ensemble aggregation).
    """

def select_tta_recipe(
    val_logits_per_recipe: dict[tuple[str, ...], np.ndarray],
    val_labels: np.ndarray,
) -> tuple[str, ...]:
    """Return the val-F1-maximising TTA recipe (D5 — measure, don't guess).

    Args:
        val_logits_per_recipe: {recipe_tuple: logits (N,2)} for each candidate combo.
        val_labels: int ndarray (N, 2) ground-truth.

    Returns:
        Tuple of aug names corresponding to the best macro-F1 on val ensemble scores.
    """
```

**5-crop extension** — DO NOT modify `transforms.py` (D9). Implement corner crops inline in `tta.py`:
```python
# Analog: spatial_val center-crop in transforms.py lines 190–215.
# 5-crop = center + top-left, top-right, bottom-left, bottom-right.
# Use TF.resized_crop or manual slice — same resize_short=128 then crop_size=112.
```

---

### `harness/colab.py` — MODIFY (add `update_latest` kwarg + `stage_unlabeled_squat_videos`)

**Analog:** existing `atomic_save_checkpoint` (lines 538–592) and `stage_squat_videos` (lines 317–443)

**Change 1 — `update_latest` kwarg** (minimal, backward-compatible):
```python
# BEFORE (colab.py line 538):
def atomic_save_checkpoint(payload: dict, target_path: str) -> None:

# AFTER:
def atomic_save_checkpoint(
    payload: dict,
    target_path: str,
    *,
    update_latest: bool = True,          # NEW kwarg — default True preserves Phase 2/3 behavior
) -> None:
    ...
    # Step 4: latest.txt LAST — guarded by new kwarg (D7 landmine fix).
    if update_latest:                    # NEW guard
        latest_path = os.path.join(target_dir, "latest.txt")
        _atomic_write_text(latest_path, target_name + "\n")
        logger.info("atomic_save_checkpoint: %s (+ latest.txt -> %s)", target_path, target_name)
    else:
        logger.info("atomic_save_checkpoint: %s (latest.txt NOT updated)", target_path)
```
Call site in `md_pretrain.py`: `atomic_save_checkpoint(backbone_payload, backbone_path, update_latest=False)`.

**Change 2 — `stage_unlabeled_squat_videos`** (new function, same file — copy `stage_squat_videos` pattern, lines 317–443):
```python
_SQUAT_UNLABELED_VIDEOS_EXPECT_COUNT = 4970   # Phase 1 verified count

def stage_unlabeled_squat_videos(
    drive_root: str,
    *,
    local_videos_root: str = "/content/squat_unlabeled_videos",
    local_traj_root: str = "/content/squat_trajectories",
    expect_count: int = _SQUAT_UNLABELED_VIDEOS_EXPECT_COUNT,
) -> tuple[str, str]:
    """Stage unlabeled Squat videos.zip + bar_trajectories_raw.zip from Drive.

    Mirrors stage_squat_videos three-layer resume design exactly:
    1. Cache hit — if local_videos_root has expect_count .mp4s, return immediately.
    2. Byte-level resume copy via _copy_with_resume_and_progress.
    3. Per-member extract resume via _extract_with_resume_and_progress.

    Returns:
        (local_videos_root, local_traj_root) — paths to staged assets.
    """
    # src_zip = Path(drive_root) / "Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset/videos.zip"
    # traj_zip = Path(drive_root) / "Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset/bar_trajectories_raw.zip"
    # Reuse _copy_with_resume_and_progress + _extract_with_resume_and_progress verbatim.
    ...
```

---

### `notebooks/04_squat_md_ssl.py` + `04_squat_md_ssl.ipynb` (notebook)

**Analog:** `backend/training/aqa/notebooks/03_squat_supervised_baseline.py`

**Cell A bootstrap** (lines 29–58 — copy verbatim, change branch comment):
```python
# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH = "fresh-start"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

_head = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(f"\nRepo ready: {REPO_DIR} @ {_head}")
```

**Step 0 environment setup** (lines 83–163 — copy with Phase 4 additions):
```python
# %%
from backend.training.aqa.harness import _envinit  # F8 constraint: FIRST line

# Phase 3 carry: PyAV before torch
try:
    import av
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av

# Phase 4 dep probe — add scipy to the list
_PHASE4_DEPS = ["torch", "torchvision", "scikit-learn", "matplotlib", "tqdm", "numpy", "scipy"]
# ... same pattern as Phase 3

# GPU check — same L4 assert
assert "L4" in _gpu_name or _vram_gb >= 22.0

# Phase 4 staging — use stage_unlabeled_squat_videos
from backend.training.aqa.harness.colab import stage_squat_videos, stage_unlabeled_squat_videos
VIDEOS_ROOT = stage_squat_videos(MYDRIVE)              # labeled (1739 mp4s, Phase 2/3 carry)
UNLABELED_VIDEOS_ROOT, TRAJ_ROOT = stage_unlabeled_squat_videos(MYDRIVE)  # NEW for SSL
```

**Jupytext percent format:** every cell starts with `# %%` or `# %% [markdown]`. Same convention as Phase 3. The `.ipynb` is the deliverable for Colab (`[[feedback_deliver_colab_as_ipynb]]`); the `.py` is the version-controlled source.

**`importlib.reload` pattern** (carry-forward — insert before every trainer import cell after a `git pull`):
```python
# After git pull in a running notebook, sys.modules is stale.
# Restart the runtime OR importlib.reload the changed module:
import importlib
import backend.training.aqa.harness.md_pretrain as _md
importlib.reload(_md)
from backend.training.aqa.harness.md_pretrain import run_md_pretrain_epoch
```

---

### Test files — all four

**Analog:** `harness/test_supervised_train.py` + `eval/test_metrics.py`

**File-top skip pattern** (`test_supervised_train.py` lines 27–28 — copy verbatim):
```python
# Skip the whole file on torch-less environments (Windows orchestrator shell).
torch = pytest.importorskip("torch")
```
Apply to `test_md_pretrain.py`, `test_md_finetune.py`. For `test_ensemble.py` and `test_tta.py`, torch is optional (pure numpy logic); use `pytest.importorskip("torch")` only in the slow model-forward test, not at module top.

**@pytest.mark.slow pattern** (`test_supervised_train.py` lines 129–145):
```python
@pytest.mark.slow
def test_md_model_build() -> None:  # downloads ~120 MB weights on first call
    pytest.importorskip("torchvision")
    backbone, projector = build_md_model()
    assert isinstance(backbone.fc, torch.nn.Identity)
    x = torch.zeros(2, 3, 16, 112, 112)
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape == (2, 512), feat.shape
    # Projector: L2-norm output
    proj_out = projector(feat)
    assert proj_out.shape == (2, 128)
    norms = proj_out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), norms
```

**Checkpoint schema test pattern** (`test_supervised_train.py` lines 72–121 — copy structure):
```python
_EXPECTED_SSL_KEYS: tuple[str, ...] = (
    "epoch", "backbone_state_dict", "projector_state_dict",
    "optimizer_state_dict", "scheduler_state_dict",
    "rng_state", "metrics_history", "linear_probe_history",
    "config_hash", "config_repr", "code_version",
)

def test_ssl_checkpoint_schema() -> None:
    payload = {
        "epoch": 0,
        "backbone_state_dict": {},
        "projector_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "rng_state": {"python": (), "numpy": (), "torch_cpu": torch.tensor([0], dtype=torch.uint8), "torch_cuda_all": []},
        "metrics_history": [],
        "linear_probe_history": [],
        "config_hash": "abc123",
        "config_repr": {},
        "code_version": "phase04-md-pretrain",
    }
    missing = [k for k in _EXPECTED_SSL_KEYS if k not in payload]
    assert missing == [], f"Missing keys: {missing}"
    assert payload["code_version"] == "phase04-md-pretrain"
```

**Helper function pattern** (`test_metrics.py` lines 30–32):
```python
def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)
```
Use this idiom in all four test files for synthetic data construction.

**Triplet loss test pattern** (for `test_md_pretrain.py`):
```python
def test_triplet_loss_direction() -> None:
    # Anchor≈positive and far from negative → loss should be lower than the reverse.
    d = 128
    anc = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    pos_close = anc + 0.01 * torch.randn(4, d)
    pos_close = torch.nn.functional.normalize(pos_close, dim=-1)
    neg_far = torch.nn.functional.normalize(-anc + 0.01 * torch.randn(4, d), dim=-1)
    loss_good = md_triplet_loss(anc, pos_close, neg_far)
    loss_bad = md_triplet_loss(anc, neg_far, pos_close)  # anchor≈negative, pos far
    assert loss_good < loss_bad, f"Expected loss_good {loss_good:.4f} < loss_bad {loss_bad:.4f}"

def test_triplet_loss_known() -> None:
    # Hand-computed: d_ap=0, d_an=∞ → loss = -log(1/(1+0)) = -log(1) = 0.
    d = 4
    same = torch.ones(1, d) / (d ** 0.5)  # unit vector
    ortho = torch.zeros(1, d); ortho[0, 0] = 1.0  # different direction
    # When anchor == positive (d_ap=0) and negative very far: loss → 0
    loss = md_triplet_loss(same, same, -same)   # anc=pos, neg is antipodal
    assert loss.item() < 0.05, loss.item()
```

---

## Shared Patterns

### `map_location='cpu'` on checkpoint load (D7 landmine — mandatory everywhere)

**Source:** `harness/supervised_train.py` lines 443–445 + comment lines 437–442
**Apply to:** `md_pretrain.py` and `md_finetune.py` resume branches
```python
prior = load_latest_checkpoint(
    run_dir, expected_config_hash=config_hash_str, map_location="cpu",
    # map_location='cpu' REQUIRED — Phase 3 fix a0841b4.
    # CUDA ByteTensors break set_rng_state_all with "RNG state must be a
    # torch.ByteTensor". Loading to CPU keeps RNG tensors on CPU; model/
    # optimizer state-dicts handle device transfer automatically.
)
```
Also in `md_finetune.build_finetune_model`:
```python
ckpt = torch.load(md_backbone_path, map_location="cpu", weights_only=False)
```

### `persistent_workers` guard (D7 landmine — mandatory everywhere)

**Source:** `harness/supervised_train.py` lines 229–251
**Apply to:** every `DataLoader` construction in `md_pretrain.py`, `md_finetune.py`, `squat_ssl.py`
```python
_persistent = config.num_workers > 0

DataLoader(
    dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=<True/False>,
    drop_last=False,
    persistent_workers=_persistent,  # REQUIRED when num_workers > 0
)
```

### Atomic checkpoint write + `prune_checkpoints` (Phase 2 contract)

**Source:** `harness/colab.py` lines 538–699
**Apply to:** `md_pretrain.py` and `md_finetune.py` every epoch
```python
atomic_save_checkpoint(payload, ckpt_path)           # epoch_NNN.pt — update_latest=True (default)
atomic_save_checkpoint(backbone_payload, backbone_path, update_latest=False)  # backbone.pt only
prune_checkpoints(run_dir, keep_last=3, keep_best=True)
```

### Seed worker + global seed (Phase 2/3 carry)

**Source:** `harness/supervised_train.py` lines 144–179
**Apply to:** `md_pretrain.py`, `md_finetune.py`
```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 4-RNG capture/restore (Phase 2 contract)

**Source:** `harness/colab.py` lines 454–488
**Apply to:** every epoch payload in `md_pretrain.py` and `md_finetune.py`
```python
"rng_state": capture_rng_state(),   # in payload at save time
restore_rng_state(prior["rng_state"])  # on resume
```

### `from __future__ import annotations` + `logger = logging.getLogger("aqa.phase04")`

**Source:** every file in `backend/training/aqa/`
**Apply to:** all 10 new/modified files
```python
from __future__ import annotations
...
logger = logging.getLogger("aqa.phase04")
```

### `BCEWithLogitsLoss(pos_weight=...)` pattern

**Source:** `harness/supervised_train.py` lines 415–419
**Apply to:** `md_finetune.py` only (SSL pretraining has no classification loss)
```python
pos_weight = train_loader.dataset.pos_weight   # from SquatKIEKFEDataset — Phase 2 contract
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

### `metrics_history` from `latest.txt`-pointed checkpoint (D7 landmine)

**Source:** `harness/supervised_train.py` lines 452–453 + RESEARCH note
**Apply to:** `md_pretrain.py`, `md_finetune.py`, notebook visualization cells
```python
# Load metrics_history from the latest.txt-pointed checkpoint (NOT best.pt).
# best.pt freezes metrics_history at the best epoch — training-curve viz requires
# the full history from the latest epoch.
prior = load_latest_checkpoint(run_dir, ..., map_location="cpu")
metrics_history = list(prior["metrics_history"])  # copy, don't mutate
```

---

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `harness/md_pretrain.py` → `ProjectionHead` class | model component | transform | No projection head exists in the codebase; use RESEARCH §4 SimCLR-standard 2-layer MLP pattern |
| `harness/md_pretrain.py` → `md_triplet_loss` function | utility | transform | No contrastive loss exists; use RESEARCH §2 (Eq. 1 + official `train_test.py:60-71` pattern) |
| `harness/md_pretrain.py` → `_linear_probe` helper | utility | batch | No linear-probe monitor exists; use RESEARCH §6 protocol (freeze backbone → sklearn LogReg on labeled features → `eval/metrics.py`) |
| `harness/md_pretrain.py` → collapse detection | utility | batch | No embedding-std monitor exists; use RESEARCH §12 (`z.std(dim=0).mean()` + `effective_rank` formula) |
| `datasets/squat_ssl.py` → `split_half_cycles` function | utility | transform | No half-cycle splitting exists; use RESEARCH §1 (`scipy.ndimage.gaussian_filter1d` + `np.argmax`, with argmax sign resolved empirically) |

---

## Metadata

**Analog search scope:** `backend/training/aqa/` (datasets/, harness/, eval/, notebooks/)
**Files scanned:** 8 source files + 2 test files (all existing Phase 2/3 modules)
**Pattern extraction date:** 2026-05-21

---

## PATTERN MAPPING COMPLETE

**Phase:** 04 - Squat Motion-Disentangling SSL
**Files classified:** 14 (10 production + 4 test)
**Analogs found:** 14 / 14 (5 partial — matched to existing analog but contain genuinely novel logic with no analog)

### Coverage
- Files with exact analog: 8 (md_finetune.py, colab.py modification, notebook, all 4 tests, ensemble.py structure, metrics.py structure)
- Files with role-match analog: 6 (squat_ssl.py, ssl_augs.py, md_pretrain.py, tta.py — all have a structural analog but contain novel logic)
- Files with no analog (new logic): 5 sub-components within md_pretrain.py and squat_ssl.py (ProjectionHead, md_triplet_loss, _linear_probe, collapse detection, split_half_cycles)

### Key Patterns Identified
- All trainers use `supervised_train.py`'s checkpoint payload schema, 4-RNG capture/restore, `atomic_save_checkpoint`/`load_latest_checkpoint` with `map_location='cpu'`, and `prune_checkpoints(keep_last=3, keep_best=True)`
- All DataLoaders use the `_persistent = num_workers > 0` guard + `persistent_workers=_persistent` + `seed_worker` worker_init_fn
- `md_finetune.py` is a near-exact copy of `supervised_train.py` with 4 deltas: AdamW (not Adam), `Dropout(0.2)+Linear(512,2)` head, MD backbone loading (not Kinetics), and D6 runtime overfit monitor
- `atomic_save_checkpoint` gains an `update_latest: bool = True` kwarg (backward-compatible); `backbone.pt` writes pass `update_latest=False` to prevent clobbering the epoch-resume pointer
- The notebook Cell A bootstrap is a verbatim copy from `03_squat_supervised_baseline.py` lines 29–58; Step 0 extends it with `stage_unlabeled_squat_videos` and scipy added to the dep probe list
- All test files use `pytest.importorskip("torch")` at module top; slow model-build tests use `@pytest.mark.slow`

### File Created
`.planning/phases/04-squat-motion-disentangling-ssl/04-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
