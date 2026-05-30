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
