# Phase 7 — Shallow-Squat CVCSPC SSL Pretrain (ResNet-18 pose-contrastive, unlabeled Back-Squat set)
#
# Companion notebook to `.planning/phases/07-image-based-errors-cvcspc/07-03-PLAN.md`.
#
# Run in Colab on an L4 GPU. The heavy one-time cost is the frame-extraction of the 4,970 unlabeled
# Squat clips (~10-20 min); the SSL pretrain itself is light (ResNet-18). Mounts the Drive shortcut
# holding `Fitness-AQA_dataset_release`. The unlabeled Squat data:
#   Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset/
#     videos.zip (4970 mp4s) + bar_trajectories_raw.zip (4970 flat y-trajectory JSONs)
#
# The labeled val/test CLIPS are EXCLUDED from the SSL set (faithful to the official
# dataloader.py:70-74 — no eval-clip leakage). traj_nan.json is absent from this release; the
# dataset's degenerate-trajectory guard covers NaN trajectories.
#
# Plan 03 arc: Cell A bootstrap -> Step 0 stage(videos+traj) + frame-extract + compute exclude_ids
#   -> PROBE (trajectory<->frame mapping + phase-matched triplet on real frames, BLOCKING human-verify)
#   -> Step 1 SSL dataset smoke -> Step 2 VRAM probe -> Step 3 epoch-0 timing gate (BLOCKING)
#   -> Step 4 full CVCSPC SSL pretrain -> backbone.pt.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Cells delivered one at a time in
# chat per the FitNova interactive working agreement (paste-back gating).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in the fresh L4 Colab session.** Pulls `fresh-start` so Colab gets the Phase-7
# CVCSPC modules (`datasets/cvcspc_ssl.py` with exclude_ids, `harness/cvcspc_pretrain.py`,
# `harness/colab.py` with `extract_frames_for_ssl`). Then `os.chdir` + `sys.path.insert`.

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

for _m in [
    "backend/training/aqa/datasets/cvcspc_ssl.py",
    "backend/training/aqa/harness/cvcspc_pretrain.py",
    "backend/training/aqa/harness/colab.py",
]:
    print(f"  {'OK ' if os.path.isfile(_m) else 'MISSING'} {_m}")
