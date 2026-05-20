# Phase 03 — File Pattern Map

**Mapped:** 2026-05-20
**Source files read:**
- `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md`
- `.planning/phases/03-squat-supervised-baseline/03-RESEARCH.md`
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` (lines 95–243, interfaces block)
- `backend/training/aqa/harness/tiny_train.py` (full, 257 lines)
- `backend/training/aqa/harness/colab.py` (full, 700 lines)
- `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` (lines 1–220)
- `backend/training/aqa/datasets/squat.py` (full, 204 lines)
- `backend/services/test_form_geometry.py` (lines 1–129)
- `backend/config/test_defect_to_region.py` (lines 1–60)
- `backend/training/aqa/datasets/__init__.py` (empty, 0 bytes)
- `backend/training/aqa/harness/__init__.py` (empty, 0 bytes)
- `backend/training/aqa/harness/_envinit.py` (3 lines)

---

## Summary

Phase 3 creates 6 new files: one empty package marker (`eval/__init__.py`), one pure-function metrics library (`eval/metrics.py`), one pytest test module for the metrics library (`eval/test_metrics.py`), one full trainer module (`harness/supervised_train.py`), one pytest test module for the trainer (`harness/test_supervised_train.py`), and one Colab notebook (`notebooks/03_squat_supervised_baseline.py`). The primary analog for the trainer is `backend/training/aqa/harness/tiny_train.py` — `supervised_train.py` mirrors its `run_tiny_epoch` signature shape, checkpoint payload schema, `_set_global_seed` helper, and logger naming, extending the payload with the new D8 keys and adding the real R(2+1)D-18 model. The primary analog for the notebook is `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` — the Phase 3 notebook copies its jupytext percent-format, the `_envinit`-first-cell constraint, the PyAV-before-torch bootstrap, the `# %% [markdown] ## Step N` heading convention, and the paste-back pattern. The one structural break is the new `eval/` package directory: there is no existing analog in `backend/training/aqa/` for a pure-function evaluation library, so its closest model is the `backend/config/test_defect_to_region.py` and `backend/services/test_form_geometry.py` patterns for test conventions, and `backend/training/aqa/datasets/__init__.py` for the empty `__init__.py`.

---

## File Map

### 1. `backend/training/aqa/eval/__init__.py`

**Role:** package marker
**Data flow:** none (declarative)
**Closest analog:** `backend/training/aqa/datasets/__init__.py` and `backend/training/aqa/harness/__init__.py`
**Excerpt from analog:**
Both existing `__init__.py` files are completely empty (0 bytes). The shell `ls` and glob confirm each file exists with content shorter than 1 line.

**Phase 3 spec:** Empty file — 0 bytes. No docstring, no imports. Identical to both Phase 2 `__init__.py` files.

**Rules:**
- Do NOT add any imports or re-exports. The `eval/` package is accessed via fully-qualified imports (`from backend.training.aqa.eval.metrics import f1_per_error`).
- Matches Phase 2 convention exactly.

---

### 2. `backend/training/aqa/eval/metrics.py`

**Role:** metrics library (pure functions)
**Data flow:** receives numpy arrays (scores, labels); produces floats and dicts; no model loading, no I/O, no side effects
**Closest analog:** No direct analog in `backend/training/aqa/`. The co-located test convention is modeled on `backend/services/test_form_geometry.py` and `backend/config/test_defect_to_region.py`. The module-top docstring style and `from __future__ import annotations` are modeled on `backend/training/aqa/datasets/squat.py`.

**Excerpt — module-top boilerplate to replicate** (`squat.py` lines 1–19):
```python
"""SquatKIEKFEDataset + build_loaders factory — joint multi-label dataset over the official Squat splits.

Yields `(video[3, T, H, W] float32, labels[2] float32)` from official train/val/test splits.
...

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 6 / D9 / D14 / interfaces block.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
...

logger = logging.getLogger("aqa.phase02")
```

**Phase 3 spec:**
```python
"""Evaluation primitives for binary multi-label error classification — F1, PR-AUC, threshold sweep, confusion matrix.

Pure functions; no model loading, no disk I/O, no side effects. All inputs are numpy
arrays gathered from full val/test passes (243 and 244 clips respectively — fits in
CPU memory without streaming). Uses sklearn primitives per CONTEXT.md D9.

Function index:
  - f1_per_error(y_true, y_pred) -> float           — binary F1 at a fixed threshold
  - pr_auc_per_error(y_true, y_score) -> float       — threshold-free AP
  - threshold_sweep(y_true, y_score) -> (float, float) — best (threshold, F1) via PR curve
  - confusion_matrix_per_error(y_true, y_pred) -> np.ndarray  — 2x2 confusion matrix

See: `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md` — D7 / D9.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

logger = logging.getLogger("aqa.phase03")
```

**Rules:**
- ONLY use `sklearn.metrics` primitives: `f1_score`, `precision_recall_curve`, `average_precision_score`, `confusion_matrix`. No homegrown implementations.
- `threshold_sweep` uses `precision_recall_curve` approach from RESEARCH §5, NOT the linspace approach in CONTEXT D7. `precision_recall_curve` returns all unique thresholds — more precise than 91-point grid. The linspace is demoted to a notebook-side sanity check.
- No torch imports in this module — all inputs are numpy arrays.
- `logger = logging.getLogger("aqa.phase03")` — note the phase03 suffix, NOT phase02.
- All functions are pure (no mutation, no side effects on module state).
- The `[:-1]` slice on `precision_recall_curve` output is mandatory: the last element is the artificial `p=1, r=0` sentinel with no corresponding threshold. Code: `f1_vals = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)`.

---

### 3. `backend/training/aqa/eval/test_metrics.py`

**Role:** pytest unit test module
**Data flow:** creates synthetic numpy arrays; calls `eval/metrics.py` functions; asserts outputs
**Closest analog:** `backend/services/test_form_geometry.py` (co-located convention, section separators, helper functions), `backend/config/test_defect_to_region.py` (parametrize pattern, minimal imports)

**Excerpt — module-top and helper pattern** (`test_form_geometry.py` lines 1–29):
```python
"""Unit tests for the geometric form validator.

Covers:
  - Rep state machine: counts completion-only, rejects half-reps, ...
  - Form rule evaluator: trunk lean, knee valgus, depth.
  - GeometricFormValidator integration: full lifecycle on synthetic squat trajectories.

Run from repo root:
    pytest backend/services/test_form_geometry.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.services.form_geometry import (
    GeometricFormValidator, ...
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build synthetic 15-joint landmark frames
# ─────────────────────────────────────────────────────────────────────────────


def make_standing_lms(...) -> np.ndarray:
    """..."""
```

**Excerpt — parametrize pattern** (`test_defect_to_region.py` lines 25–41):
```python
@pytest.mark.parametrize("variation,expected", [
    ("knees over toes",        {REGION_KNEES}),
    ("back not straight",      {REGION_TRUNK}),
    ...
])
def test_named_body_part_maps_to_region(variation, expected):
    assert regions_for_defect(variation) == expected
```

**Phase 3 spec:**
```python
"""Unit tests for backend.training.aqa.eval.metrics — SQUAT-03-a through SQUAT-03-d.

Covers:
  - SQUAT-03-a: model head outputs [B, 2] logits (lives here per RESEARCH §4 Wave 0 mapping)
  - SQUAT-03-b: f1_per_error matches manual calculation on known arrays
  - SQUAT-03-c: threshold_sweep selects the F1-maximizing threshold
  - SQUAT-03-d: pr_auc_per_error matches sklearn.average_precision_score on known arrays

Run from repo root:
    python -m pytest backend/training/aqa/eval/test_metrics.py -x -v

See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D9.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)

# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-a: model head shape (import torchvision here, not in metrics.py)
# ─────────────────────────────────────────────────────────────────────────────
```

**Rules:**
- File location is `backend/training/aqa/eval/test_metrics.py` — co-located next to `metrics.py` per the project convention (`test_form_geometry.py` lives next to `form_geometry.py`).
- SQUAT-03-a (model head shape `[B, 2]`) belongs here, not in `harness/test_supervised_train.py`. RESEARCH §4 maps it to `test_metrics.py`.
- `@pytest.mark.slow` on any test that imports torchvision and constructs the full R(2+1)D-18 model (heavy download + forward pass).
- `# ─────────` section separators between the four test groups (SQUAT-03-a, -b, -c, -d) as in `test_form_geometry.py`.
- No `from __future__ import annotations` needed in the test file per existing test conventions (both analog test files omit it from function signatures; only the source modules use it). Actually `test_defect_to_region.py` line 1 DOES have it — include it.

---

### 4. `backend/training/aqa/harness/supervised_train.py`

**Role:** trainer module
**Data flow:** reads Phase 2 dataset (via direct DataLoader construction over `SquatKIEKFEDataset`), trains R(2+1)D-18 end-to-end, writes checkpoints to Drive via `colab.py`'s `atomic_save_checkpoint`
**Closest analog:** `backend/training/aqa/harness/tiny_train.py` — PRIMARY, exact structural match

**Excerpt — module-top docstring** (`tiny_train.py` lines 1–16):
```python
"""Toy 3D-conv model + multi-epoch training loop wired to the Colab harness — drives Phase 2's end-to-end smoke validation.

...

`run_tiny_epoch(*, run_name, ..., resume=True, max_epochs=1) -> dict` is the single
public entrypoint. ...

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 11/12/13/14 / D14 / interfaces block.
"""
```

**Excerpt — imports and logger** (`tiny_train.py` lines 18–39):
```python
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
```

**Excerpt — `_set_global_seed` helper** (`tiny_train.py` lines 66–73):
```python
def _set_global_seed(seed: int) -> None:
    """Seed all 4 RNG sources. Determinism precondition 2."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Excerpt — function signature shape** (`tiny_train.py` lines 75–116):
```python
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
        run_name: ...
        ...

    Returns:
        Dict with the F7 contract keys: ...
    """
    _set_global_seed(seed)

    config = {
        "seed": seed,
        "batch_size": batch_size,
        ...
        "model": "TinyModel(...)",
        "loss": "BCEWithLogitsLoss(no pos_weight)",
    }
    config_hash_str = hash_config(config)

    run_dir = os.path.join(drive_root, "FitNova/checkpoints/phase02", run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run_dir: %s (config_hash=%s)", run_dir, config_hash_str)
```

**Excerpt — checkpoint payload schema** (`tiny_train.py` lines 230–241):
```python
payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": None,   # no scheduler in Phase 2 tiny
    "rng_state": capture_rng_state(),
    "metrics_history": metrics_history,
    "config_hash": config_hash_str,
    "config_repr": config,
    "code_version": "phase02-tiny",
}
ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
atomic_save_checkpoint(payload, ckpt_path)
prune_checkpoints(run_dir, keep_last=3, keep_best=False)
```

**Phase 3 spec for `supervised_train.py`:**

The module introduces a `SupervisedConfig` dataclass and `build_model()` function alongside `run_supervised_epoch()`. Key differences from `tiny_train.py`:

```python
"""R(2+1)D-18 supervised baseline trainer for Squat KIE/KFE error detection.

Full fine-tuning of torchvision R(2+1)D-18 (Kinetics-400-V1 init) with a joint
two-output multi-label head. Extends Phase 2's tiny_train.py patterns — does NOT
replace it; tiny_train.py stays as the Phase 2 reference.

Architecture: torchvision.models.video.r2plus1d_18(weights=KINETICS400_V1),
  fc replaced with nn.Linear(512, 2) per CONTEXT.md D1.
Loss: BCEWithLogitsLoss(pos_weight=dataset.pos_weight) per CONTEXT.md D2.
Optimizer: Adam lr=1e-4, weight_decay=0 per RESEARCH §1 (paper §5, Adam confirmed).
Scheduler: CosineAnnealingLR(T_max=max_epochs) per CONTEXT.md D6 provisional.
DataLoader: direct construction (NOT build_loaders) with worker_init_fn per RESEARCH §2.

Checkpoint payload extends Phase 2 schema (D8):
  + scheduler_state_dict (actual, not None)
  + best_f1_val (best val macro-F1 seen so far)
  + best_thresholds ({"kie": float, "kfe": float} — set post-training, None until then)
  + metrics_history[i] gains: val_f1_kie, val_f1_kfe, val_pr_auc_kie, val_pr_auc_kfe,
    val_macro_f1 alongside Phase 2's loss fields
  code_version = "phase03-supervised-baseline"

best.pt written via atomic_save_checkpoint whenever val macro-F1 improves.

See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D1–D8 / D10.
"""

from __future__ import annotations
...
logger = logging.getLogger("aqa.phase03")
```

```python
@dataclasses.dataclass
class SupervisedConfig:
    """Hyperparameters for the Phase 3 supervised baseline run.

    All values cited to paper or justified in 03-RESEARCH.md.
    """
    learning_rate: float = 1e-4          # paper §5: "ADAM optimizer with initial lr of 1e-4"
    weight_decay: float = 0.0            # RESEARCH §1 A2: paper silent; default 0 for fidelity
    batch_size: int = 16                 # RESEARCH §1: L4 24GB fits batch 16 fp32 with headroom
    num_workers: int = 4                 # RESEARCH §2: decode-bound; 4 workers on L4 Colab
    max_epochs: int = 50                 # RESEARCH §1 A4: paper silent on downstream count; 50 + early-stop
    early_stop_patience: int = 8         # RESEARCH §1: 8-epoch patience
    num_frames: int = 32                 # CONTEXT D3: Phase 2 contract
    crop_size: int = 112                 # CONTEXT D5: locked default
    train_jitter_frames: int = 2         # CONTEXT D3: Phase 2 contract
    flip_aug: bool = False               # CONTEXT D5: OFF by default (paper omits flip)
```

```python
def build_model() -> torch.nn.Module:
    """Construct R(2+1)D-18 with Kinetics-400-V1 weights and replace the 400-class head.

    Returns:
        nn.Module with fc=Linear(512, 2). All backbone layers trainable (D1: end-to-end fine-tune).

    Raises:
        AssertionError: if torchvision's fc.in_features != 512 (catches upstream API change).
    """
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    assert model.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={model.fc.in_features}, expected 512 (RESEARCH §3)"
    )
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model
```

```python
def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker from the main-process RNG.

    PyTorch docs: https://docs.pytorch.org/docs/2.12/notes/randomness.html
    Cited in RESEARCH §2.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

`run_supervised_epoch` signature mirrors `run_tiny_epoch` shape:
```python
def run_supervised_epoch(
    *,
    run_name: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_videos",
    seed: int = 42,
    config: SupervisedConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
) -> dict:
```

**Rules:**
- `logger = logging.getLogger("aqa.phase03")` — NOT `"aqa.phase02"`.
- Import `_set_global_seed` from `tiny_train` rather than redefining: `from backend.training.aqa.harness.tiny_train import _set_global_seed`. If import fails (e.g., private name), copy the 7-line function body verbatim — do NOT modify `tiny_train.py`.
- `code_version = "phase03-supervised-baseline"`.
- Run dir pattern: `{drive_root}/FitNova/checkpoints/phase03/{run_name}/` (phase03, NOT phase02).
- `config_hash` must include all `SupervisedConfig` fields plus `model` identifier string and `seed`. Exclude `drive_root`, `videos_root`, `run_name`, `max_epochs` (they vary across resume cycles without invalidating the training recipe — same rule as Phase 2).
- `best.pt` written by `atomic_save_checkpoint(payload, os.path.join(run_dir, "best.pt"))` whenever current `val_macro_f1 > best_f1_val`. Called AFTER the epoch `latest.pt` write.
- `prune_checkpoints(run_dir, keep_last=3, keep_best=True)` — note `keep_best=True` (differs from Phase 2's `keep_best=False`).
- Val pass: accumulate `(logits_list, labels_list)` across all val batches, then `torch.cat` → `.numpy()` ONCE after the loop, then call `f1_score` from sklearn. No per-batch F1 (RESEARCH §8 aliasing guard).
- No `torch.cuda.amp.autocast` — RESEARCH §11 disables for baseline (determinism conflict with `torch.use_deterministic_algorithms(True, warn_only=True)` set by `colab.py`).
- DataLoader construction is direct (bypass `build_loaders`) per RESEARCH §2 recommendation: `DataLoader(ds, batch_size=..., num_workers=..., worker_init_fn=seed_worker, generator=g, shuffle=True)`.

---

### 5. `backend/training/aqa/harness/test_supervised_train.py`

**Role:** pytest unit test module for trainer
**Data flow:** constructs `SupervisedConfig`, calls `build_model()`, checks output shape; inspects checkpoint schema from a dummy payload
**Closest analog:** `backend/services/test_form_geometry.py` (module docstring convention, `# ─────────` separators, section structure), `backend/config/test_defect_to_region.py` (import style)

**Excerpt — docstring and section separator pattern** (`test_form_geometry.py` lines 1–30):
```python
"""Unit tests for the geometric form validator.

Covers:
  - Rep state machine: ...

Run from repo root:
    pytest backend/services/test_form_geometry.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

...

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build synthetic 15-joint landmark frames
# ─────────────────────────────────────────────────────────────────────────────
```

**Phase 3 spec:**
```python
"""Unit tests for backend.training.aqa.harness.supervised_train — SQUAT-03-a and SQUAT-03-e.

Covers:
  - SQUAT-03-a: R(2+1)D-18 head outputs [B, 2] logits from [B, 3, 32, 112, 112] input
    (marked slow — constructs full model + forward pass)
  - SQUAT-03-e: checkpoint schema smoke — payload dict contains all Phase 3 D8 keys

Run from repo root:
    python -m pytest backend/training/aqa/harness/test_supervised_train.py -x -v
    python -m pytest backend/training/aqa/harness/test_supervised_train.py -x -v -m "not slow"

See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D8.
"""

from __future__ import annotations

import pytest
import torch

from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    build_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-a: model head shape
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_model_head_shape():
    ...

# ─────────────────────────────────────────────────────────────────────────────
# SQUAT-03-e: checkpoint schema keys
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_schema_keys():
    ...
```

**Rules:**
- Co-located at `backend/training/aqa/harness/test_supervised_train.py` — next to `supervised_train.py`.
- `@pytest.mark.slow` on `test_model_head_shape` — model construction downloads ~120 MB and does a forward pass; should not block fast CI.
- SQUAT-03-e (checkpoint schema) does NOT require actual training — build a minimal dict with all required Phase 3 keys and assert their presence. This is a pure-Python test, no GPU needed.
- `# ─────────` section separator between the two test groups.

---

### 6. `backend/training/aqa/notebooks/03_squat_supervised_baseline.py`

**Role:** Colab notebook (jupytext percent-format)
**Data flow:** drives `supervised_train.run_supervised_epoch` interactively; calls `eval/metrics.py` for post-training threshold sweep + test evaluation; saves all figures to `figures/`
**Closest analog:** `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` — PRIMARY, exact format match

**Excerpt — notebook header comment + Cell 0 F8 constraint** (lines 1–61):
```python
# Phase 2 — Squat Data Pipeline & Resumable Colab Harness
#
# Companion notebook to `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md`.
# Reference: `02-CONTEXT.md` (locked decisions), ...
#
# Run in Colab. Mounts the user's Drive shortcut `My Drive/Fitness-AQA_dataset_release`.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Open in Colab via
# `pip install jupytext && jupytext --to ipynb 02_squat_pipeline_harness.py`,
# or paste cells one at a time.

# %% [markdown]
# ## Step 0 — environment + Drive mount
#
# **F8 import-order constraint (PLAN.md D13, `<determinism_checklist>`):** the first
# line of the next cell **MUST** be `from backend.training.aqa.harness import _envinit`.
# ...

# %%
from backend.training.aqa.harness import _envinit  # F8: sets CUBLAS_WORKSPACE_CONFIG before torch

import os
import sys

# torchvision.io.read_video requires PyAV as its FFmpeg backend on Colab. ...
import subprocess
try:
    import av  # noqa: F401
    _pyav_status = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av  # noqa: F401
    _pyav_status = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav_status})")

import torch
import torchvision
...
from google.colab import drive
drive.mount('/content/drive')

MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
```

**Excerpt — Step header markdown cell pattern** (lines 70–78):
```python
# %% [markdown]
# ## Step 1 — official splits + labels (Task 4)
#
# Loads `{train,val,test}_keys.json` + `error_knees_{inward,forward}.json` via
# `backend.training.aqa.datasets.splits.index`. ...
```

**Excerpt — paste-back verification pattern** (lines 178–198):
Each Step has an assertion after the main call that surfaces failures immediately for paste-back:
```python
assert tuple(clip.shape) == (2, 3, 32, 112, 112), clip.shape
assert tuple(labels.shape) == (2, 2), labels.shape
print("batch shape contract: OK")
```

**Phase 3 spec:** The notebook header comment changes to reference Phase 3 files; all Step content changes. The F8 / PyAV / Drive-mount bootstrap in Step 0 is copied verbatim. Step numbering continues from Step 0:

```
Step 0 — environment + Drive mount (COPY VERBATIM from Phase 2 notebook)
Step 1 — stage squat videos (idempotent — cache hit if /content/ survived)
Step 2 — verify dataset loaders (pos_weight + batch shape spot-check)
Step 3 — build model + VRAM probe (build_model(), dummy forward, print VRAM)
Step 4 — epoch 0 timing probe (1 epoch, paste back per-batch time + VRAM)
Step 5 — supervised training run (run_supervised_epoch, tqdm per-batch)
Step 6 — training curves visualization (loss + val F1, saved to figures/)
Step 7 — threshold sweep on val (threshold_sweep per error, tqdm)
Step 8 — test evaluation (f1 + pr_auc on test split at val-tuned thresholds)
Step 9 — confusion matrices (KIE + KFE, saved to figures/)
Step 10 — PR curves (KIE + KFE, saved to figures/)
Step 11 — sample-prediction grid (TP/FP/FN frames, saved to figures/)
Step 12 — results summary (print final F1 + PR-AUC table vs. paper targets)
```

**Rules:**
- File: `backend/training/aqa/notebooks/03_squat_supervised_baseline.py`
- `from backend.training.aqa.harness import _envinit` MUST be the FIRST import on Cell 0 (Line 1 of the `# %%` code block). No other import, no comment, can precede it inside that cell.
- PyAV-before-torch bootstrap (the `import av` try/except with `subprocess.run pip install`) MUST come before `import torch` in Cell 0. This is Phase 2's F8 ordering constraint and carries forward.
- Every Step is exactly ONE runnable cell (`# %%` + code) plus one markdown header cell (`# %% [markdown]` + `## Step N —`). Each cell is one paste-back unit per D11.
- Every figure: `Path(out).parent.mkdir(parents=True, exist_ok=True)` then `plt.savefig(out, dpi=150, bbox_inches="tight")` BEFORE `plt.show()`. Figures saved to `figures/` relative path in the planning directory: `.planning/phases/03-squat-supervised-baseline/figures/`.
- `stage_squat_videos` call in Step 1 is a cache hit if `/content/squat_videos/` already has 1739 mp4s — the function is idempotent per Phase 2 contract (no re-download needed).
- tqdm progress bars on: per-batch training loop, per-batch val/test loop, threshold sweep if a visible linspace sweep is retained as a sanity-check.
- `DRIVE_ROOT` and `VIDEOS_ROOT` constants defined once in Step 0 or Step 1; reused across all subsequent steps via Python variable reference.
- `run_name = "r2plus1d18_squat_supervised_v1"` hardcoded in Step 5 per CONTEXT specifics.

---

## Cross-Cutting Patterns

### Module-top boilerplate (apply to every new `.py` source file)

**Source:** `backend/training/aqa/harness/tiny_train.py` lines 1–39 and `backend/training/aqa/datasets/squat.py` lines 1–36.

**Rule:** Every new `.py` module (NOT test files, NOT `__init__.py`) follows this top-of-file structure:
1. Module docstring (triple-quoted, multi-line). Opens with one-line summary, blank line, then detailed description. Last line `See: .planning/phases/03-squat-supervised-baseline/03-CONTEXT.md — D<N> / D<N>`.
2. `from __future__ import annotations` — blank line.
3. Standard library imports (alphabetical).
4. Blank line.
5. Third-party imports (alphabetical: numpy, sklearn, torch, torchvision).
6. Blank line.
7. Local imports from `backend.training.aqa.*`.
8. Blank line.
9. `logger = logging.getLogger("aqa.phase03")`.

Test files (`test_*.py`) also include `from __future__ import annotations` (confirmed: `test_defect_to_region.py` line 1, `test_form_geometry.py` line 14).

### Logger naming

**Source:** `backend/training/aqa/harness/tiny_train.py` line 39, `backend/training/aqa/harness/colab.py` line 36, `backend/training/aqa/datasets/squat.py` line 36.

All Phase 2 modules use:
```python
logger = logging.getLogger("aqa.phase02")
```

**Phase 3 rule:** All Phase 3 modules use `logger = logging.getLogger("aqa.phase03")`. This propagates to the `aqa` parent logger so any handler attached at `logging.getLogger("aqa")` captures all phase output. The phase suffix distinguishes Phase 3 log lines from Phase 2 log lines when both modules are loaded in the same process.

### Section-separator convention

**Source:** `backend/training/aqa/harness/colab.py` lines 446–451:
```python
# ──────────────────────────────────────────────────────────────────────────────
# Task 9 — RNG capture/restore. Required for Task 14's bitwise-resume assertion
# ...
# ──────────────────────────────────────────────────────────────────────────────
```

**Rule:** Use `# ──────────────────────────────────────────────────────────────────────────────` (80-char em-dash line) as a section separator inside longer source files and test files. The separator is followed by a comment block naming the section (task number, logical group), then another separator line, then a blank line before the code. See `test_form_geometry.py` lines 29–32 for the test-file variant (same pattern, shorter lines acceptable).

### Atomic-write contract (disk writes)

**Source:** `backend/training/aqa/harness/colab.py` lines 538–592.

**Rule:** The ONLY function that writes checkpoint files to disk is `colab.atomic_save_checkpoint(payload, target_path)`. `supervised_train.py` calls it for both `epoch_{NNN}.pt` and `best.pt`. `eval/metrics.py` is pure — zero disk writes. The notebook writes figures via `plt.savefig` (not through `colab.py`) — that is the only exception and is not checkpoint data.

The atomic-write sequence (never bypass):
1. `torch.save(payload, tmp)` → tmp file at `.tmp_{basename}.{pid}`.
2. `torch.load(tmp)` → round-trip verify.
3. `os.replace(tmp, target)` → POSIX rename.
4. Write `latest.txt` LAST.

A crash at any step leaves the prior `latest.txt` pointing to the last good checkpoint.

### F8 import-order constraint (notebook Cell 0 first line)

**Source:** `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` lines 22–23:
```python
# %%
from backend.training.aqa.harness import _envinit  # F8: sets CUBLAS_WORKSPACE_CONFIG before torch
```

**Source:** `backend/training/aqa/harness/_envinit.py` lines 1–3:
```python
# F8 / D13: must execute before any 'import torch' so CUBLAS_WORKSPACE_CONFIG is set before CUDA context creation; otherwise torch.use_deterministic_algorithms(True) raises.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

**Rule:** The first executable line of the first code cell (`# %%`) in `03_squat_supervised_baseline.py` MUST be `from backend.training.aqa.harness import _envinit`. This sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA context creation. `colab.py` then calls `torch.use_deterministic_algorithms(True, warn_only=True)` at import time — if `CUBLAS_WORKSPACE_CONFIG` is absent at that point, the call raises. Even with `warn_only=True`, the workspace config must be set before the CUDA context is created, not before the Python function call.

### Pytest co-location convention

**Source:** `backend/services/test_form_geometry.py` lives next to `backend/services/form_geometry.py`. `backend/config/test_defect_to_region.py` lives next to `backend/config/defect_to_region.py`.

**Rule:** Test files live next to their source modules:
- `backend/training/aqa/eval/test_metrics.py` co-located with `eval/metrics.py`.
- `backend/training/aqa/harness/test_supervised_train.py` co-located with `harness/supervised_train.py`.
- Run command: `python -m pytest backend/training/aqa/ -x -v` discovers both.

### Docstring style — PLAN.md reference in See-line

**Source:** `backend/training/aqa/harness/tiny_train.py` line 15:
```
See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 11/12/13/14 / D14 / interfaces block.
```

**Source:** `backend/training/aqa/harness/colab.py` line 19:
```
See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 7/9/10 / D11 / D13 / interfaces block / risk register R3.
```

**Rule:** Every new `.py` module's docstring ends with a `See:` line citing the relevant CONTEXT.md decision IDs (D-N) that govern the file. Use the format:
```
See: `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md` — D<N> / D<N>.
```

---

## Risks / Anti-Patterns to Avoid

- **Do NOT modify Phase 2 modules** (`splits.py`, `transforms.py`, `squat.py`, `colab.py`, `tiny_train.py`, `_envinit.py`). They are the locked Phase 2 contract. `supervised_train.py` imports from them; it does not extend them.
- **Do NOT redefine `KINETICS_MEAN` / `KINETICS_STD`** in `metrics.py` or `supervised_train.py`. Import from `backend.training.aqa.datasets.transforms` where they are already defined.
- **Do NOT write a homegrown F1 or PR-AUC.** Use `sklearn.metrics.f1_score`, `sklearn.metrics.precision_recall_curve`, `sklearn.metrics.average_precision_score`, `sklearn.metrics.confusion_matrix` only.
- **Do NOT add a per-batch F1 logger.** RESEARCH §4 Aliasing Guards prohibit this. Val F1 is computed once per epoch from the full gathered val-pass arrays. Batch-level F1 on ~16-sample batches with 14% positive rate is numerically meaningless.
- **Do NOT enable `torch.cuda.amp.autocast`.** RESEARCH §11 disables AMP for the baseline: (1) paper does not mention mixed precision; (2) fp16 AMP conflicts with `torch.use_deterministic_algorithms(True, warn_only=True)` set by `colab.py` at import time; (3) L4 has 24 GB VRAM — batch 16 fp32 fits with headroom; (4) training is decode-bound not GPU-compute-bound, so AMP's 2× GPU speedup gives only ~20-30% wall-time gain.
- **Do NOT pass `num_workers > 0` to `build_loaders`.** `build_loaders` overrides to 0 with a warning (Phase 2 D14 constraint). `supervised_train.py` constructs DataLoaders directly with `worker_init_fn=seed_worker` and a seeded `torch.Generator`.
- **Do NOT use `logger.error()` inside an `except` block.** Use `logger.exception()` so the traceback is always captured. (CLAUDE.md project convention.)
- **Do NOT write checkpoint files from `metrics.py`.** It is a pure-function module. All checkpoint I/O goes through `colab.atomic_save_checkpoint` inside `supervised_train.py`.

---

## PATTERN MAPPING COMPLETE
