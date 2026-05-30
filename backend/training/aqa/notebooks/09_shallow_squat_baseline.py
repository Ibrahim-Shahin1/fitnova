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
