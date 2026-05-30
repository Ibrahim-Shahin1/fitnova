# Phase 7 — Shallow-Squat Supervised Baseline (ImageNet ResNet-18, IMAGENET1K_V1 init)
#
# Companion notebook to `.planning/phases/07-image-based-errors-cvcspc/07-02-PLAN.md`.
#
# Run in Colab on an L4 GPU (a T4 is acceptable — ResNet-18 on ~3.7k crops is light).
# Mounts the Drive shortcut holding `Fitness-AQA_dataset_release`. The Shallow-Squat data
# lives under the -3-001 release folder:
#   Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/
#     images.zip (3738 crops_unaligned/{id}.jpg) + labels_shallow_depth.json + splits/
#
# Plan 02 arc: Cell A bootstrap -> Step 0 stage (images.zip + labels + splits)
#   -> Step 1 split reconciliation (2542/529/540) + dataset smoke
#   -> Step 2 model + VRAM probe (single-logit head) -> Step 3 epoch-0 timing gate (BLOCKING)
#   -> Step 4 multi-seed baseline training -> Step 5 ensemble + threshold sweep -> Step 6 test eval.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Cells are delivered one at a
# time in chat per the FitNova interactive working agreement (paste-back gating).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in the fresh Colab session.** Pulls `fresh-start` so Colab gets the Phase-7
# Plan-01 modules (`datasets/shallow_squat.py`, `datasets/cvcspc_ssl.py`,
# `harness/image_supervised_train.py`, `harness/colab.py` with `stage_shallow_squat_images`).
# Then `os.chdir` + `sys.path.insert` so `from backend.training.aqa...` resolves.
#
# Idempotent: clones if `/content/fitnova` is missing, otherwise fetch + checkout + ff-only pull.

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH = "fresh-start"

if not os.path.isdir(REPO_DIR):
    print(f"Cloning {REPO_URL} (branch {BRANCH}) -> {REPO_DIR} ...")
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    print(f"Repo already at {REPO_DIR}; pulling latest from origin/{BRANCH} ...")
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
print(f"sys.path[0]: {sys.path[0]}")
print(f"cwd: {os.getcwd()}")

for _m in [
    "backend/training/aqa/datasets/shallow_squat.py",
    "backend/training/aqa/datasets/cvcspc_ssl.py",
    "backend/training/aqa/harness/image_supervised_train.py",
    "backend/training/aqa/harness/cvcspc_pretrain.py",
]:
    print(f"  {'OK ' if os.path.isfile(_m) else 'MISSING'} {_m}")


# %% [markdown]
# ## Step 0 — env + dep probe + GPU check + Drive mount + stage Shallow-Squat images
#
# **F8 import-order constraint:** the first executable line MUST be
# `from backend.training.aqa.harness import _envinit` (sets `CUBLAS_WORKSPACE_CONFIG`
# before any `import torch`). Do not reorder.
#
# No PyAV — crops load via PIL (no video decode). Stages `images.zip` (3738 crops) +
# `labels_shallow_depth.json` + `splits/` from the consolidated Drive root to
# `/content/squat_shallow_images/` (~100 MB copy + extract; resume-safe on disconnect).

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import shutil
import sys

for _pkg in ["torch", "torchvision", "scikit-learn", "numpy", "tqdm"]:
    print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")

import torch
import torchvision

print("\npython     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

assert torch.cuda.is_available(), (
    "GPU required — Runtime > Change runtime type > Hardware accelerator: L4 (T4 also fine)"
)
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

from backend.training.aqa.harness.colab import mount_drive, stage_shallow_squat_images

MYDRIVE = mount_drive()
DRIVE_ROOT_3001 = MYDRIVE  # consolidated Drive root (no -3-001 split on Drive)

_expected = os.path.join(
    DRIVE_ROOT_3001,
    "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images.zip",
)
print(f"\nimages.zip on Drive: {'FOUND' if os.path.isfile(_expected) else 'NOT FOUND'}\n  {_expected}")

SHALLOW_ROOT = stage_shallow_squat_images(DRIVE_ROOT_3001, local_root="/content/squat_shallow_images")
IMAGES_ROOT = os.path.join(SHALLOW_ROOT, "crops_unaligned")
LABELS_PATH = os.path.join(SHALLOW_ROOT, "labels_shallow_depth.json")
SPLITS_ROOT = os.path.join(SHALLOW_ROOT, "splits")

_n_jpg = sum(1 for _r, _d, _fs in os.walk(IMAGES_ROOT) for _f in _fs if _f.endswith(".jpg"))
print(f"\nstaged: {_n_jpg} jpgs at {IMAGES_ROOT}")
print(f"labels: {'OK' if os.path.isfile(LABELS_PATH) else 'MISSING'}   "
      f"splits: {'OK' if os.path.isdir(SPLITS_ROOT) else 'MISSING'}")
print(f"free disk on /content: {shutil.disk_usage('/content').free / 1e9:.1f} GB")


# %% [markdown]
# ## Step 1 — split reconciliation (2542/529/540) + dataset smoke
#
# Builds the loaders against the staged crops, reconciles the official split sizes live, checks
# the single-head `pos_weight` (~1.28, shape (1,)), pulls one train batch (shape (B,3,224,224) +
# label batch (B,) — single binary head, NOT (B,2)), and shows a few denormalized crops so you
# can eyeball that the images load correctly.

# %%
import torch
import matplotlib.pyplot as plt

from backend.training.aqa.datasets.shallow_squat import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_loaders,
)

loaders = build_loaders(
    images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
    batch_size=32, num_workers=2,
)

_EXPECT = {"train": 2542, "val": 529, "test": 540}
for _split, _exp in _EXPECT.items():
    _ds = loaders[_split].dataset
    _pos = sum(_lbl for _, _lbl in _ds.records)
    print(f"{_split:5s}: {len(_ds):4d} crops (expected {_exp})  pos={_pos} "
          f"({100 * _pos / len(_ds):.1f}%)  {'OK' if len(_ds) == _exp else 'MISMATCH'}")

_pw = loaders["train"].dataset.pos_weight
print(f"\npos_weight {tuple(_pw.shape)}: {float(_pw[0]):.4f}")
assert tuple(_pw.shape) == (1,), "pos_weight must be single-head (1,)"

_imgs, _labels = next(iter(loaders["train"]))
print(f"train batch: imgs {tuple(_imgs.shape)} {_imgs.dtype}, "
      f"labels {tuple(_labels.shape)} {_labels.dtype}, "
      f"label values {sorted(set(int(_x) for _x in _labels.tolist()))}")
assert _imgs.shape[1:] == (3, 224, 224) and _imgs.dtype == torch.float32
assert _labels.ndim == 1, f"label batch must be (B,) single-head, got {tuple(_labels.shape)}"

_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for _i, _ax in enumerate(axes):
    _crop = (_imgs[_i] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
    _ax.imshow(_crop)
    _ax.set_title(f"y={int(_labels[_i])}")
    _ax.axis("off")
plt.suptitle("Shallow-Squat train crops (denormalized) — y=1 is shallow-depth error")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Step 2 — model + VRAM probe (single-logit head assert)
#
# Builds the ImageNet ResNet-18 + `Linear(512,1)` head, asserts the single-logit head shape
# (zeros[2,3,224,224] -> [2,1]), and runs one forward+backward at batch 32 to measure peak VRAM.
# First call downloads the IMAGENET1K_V1 weights (~45 MB).

# %%
import torch.nn as nn

from backend.training.aqa.harness.image_supervised_train import build_resnet18

_device = torch.device("cuda")
_model = build_resnet18().to(_device)
assert isinstance(_model.fc, nn.Linear) and _model.fc.out_features == 1, "head must be Linear(512, 1)"
with torch.no_grad():
    _probe = _model(torch.zeros(2, 3, 224, 224, device=_device))
print("head:", _model.fc, "| forward zeros[2,3,224,224] ->", tuple(_probe.shape))
assert tuple(_probe.shape) == (2, 1), f"expected [2,1], got {tuple(_probe.shape)}"

torch.cuda.reset_peak_memory_stats()
_imgs, _labels = next(iter(loaders["train"]))
_target = _labels.view(-1, 1).to(_device)
_crit = nn.BCEWithLogitsLoss(pos_weight=_pw.to(_device))
_logits = _model(_imgs.to(_device))
_loss = _crit(_logits, _target)
_loss.backward()
_peak_gb = torch.cuda.max_memory_allocated() / 1e9
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"forward+backward batch {_imgs.shape[0]}: loss={_loss.item():.4f}, "
      f"peak VRAM={_peak_gb:.2f} GB / {_total_gb:.1f} GB")
assert _peak_gb < _total_gb, "VRAM exceeded — drop batch_size"

del _model, _logits, _loss
torch.cuda.empty_cache()
