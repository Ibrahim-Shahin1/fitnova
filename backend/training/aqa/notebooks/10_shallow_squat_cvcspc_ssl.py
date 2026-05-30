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


# %% [markdown]
# ## Step 0 — stage unlabeled videos + trajectories + frame-extract (4970 clips) + compute exclude_ids
#
# F8 `_envinit` first. Stages the unlabeled Squat videos.zip + bar_trajectories_raw.zip, extracts each
# clip's frames to `/content/squat_ssl_frames/{video_id}/` (the ~10-20 min one-time cost, skip-existing,
# tqdm), and computes the val/test holdout clip ids (excluded from the SSL set per the official protocol).

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS_WORKSPACE_CONFIG before torch

import glob
import importlib.metadata
import json
import os
import random
import shutil
import sys

for _pkg in ["torch", "torchvision", "scikit-learn", "numpy", "scipy", "tqdm"]:
    print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")

import torch

print("\ntorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU required — Runtime > Change runtime type > L4"
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

from backend.training.aqa.harness.colab import (
    extract_frames_for_ssl,
    mount_drive,
    stage_unlabeled_squat_videos,
)

MYDRIVE = mount_drive()

VIDEOS_ROOT, TRAJ_ROOT = stage_unlabeled_squat_videos(MYDRIVE)
print(f"\nvideos: {VIDEOS_ROOT}\ntrajectories: {TRAJ_ROOT}")

FRAMES_ROOT = "/content/squat_ssl_frames"
_n_frames = extract_frames_for_ssl(VIDEOS_ROOT, FRAMES_ROOT, skip_existing=True)
_n_dirs = sum(1 for _d in os.scandir(FRAMES_ROOT) if _d.is_dir())
print(f"\nextracted {_n_frames} frames across {_n_dirs} clip dirs at {FRAMES_ROOT}")
print(f"free disk on /content: {shutil.disk_usage('/content').free / 1e9:.1f} GB")

# traj_nan.json is absent from this release -> None (the degenerate-trajectory guard covers NaN).
TRAJ_NAN_PATH = None

# val/test holdout clips excluded from the SSL set (faithful to dataloader.py:70-74).
SHALLOW_SPLITS = (
    f"{MYDRIVE}/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/"
    f"Shallow_Squat_Error_Dataset/splits"
)


def _clip_of(crop_id):
    p = crop_id.split("_")
    return "_".join(p[:2]) if len(p) >= 2 else crop_id


EXCLUDE_IDS = set()
for _split in ("val", "test"):
    _ids = json.loads(open(f"{SHALLOW_SPLITS}/{_split}_ids.json").read())
    EXCLUDE_IDS |= {_clip_of(i) for i in _ids}
_present_holdout = EXCLUDE_IDS & {d.name for d in os.scandir(FRAMES_ROOT) if d.is_dir()}
print(f"\nval/test holdout clips: {len(EXCLUDE_IDS)} ({len(_present_holdout)} present in the SSL set -> excluded)")

# traj<->frame mapping (1:1 expected) on 5 sample clips.
_sample = random.sample([d.name for d in os.scandir(FRAMES_ROOT) if d.is_dir()], k=min(5, _n_dirs))
print("\nclip                 n_frames  traj_len")
for _cid in _sample:
    _nf = len([f for f in os.listdir(f"{FRAMES_ROOT}/{_cid}") if f.endswith(".jpg")])
    _tp = glob.glob(f"{TRAJ_ROOT}/**/{_cid}.json", recursive=True)
    _tl = len(json.loads(open(_tp[0]).read())) if _tp else "NO-TRAJ"
    print(f"  {_cid:18s}  {_nf:6d}    {_tl}")
