# Phase 6 — Overhead Press Supervised Baseline (R(2+1)D-18, Kinetics-400-V1 init)
#
# Companion notebook to `.planning/phases/06-overhead-press/06-02-PLAN.md`.
# Reference: `06-CONTEXT.md` (D1–D8), `06-RESEARCH.md` (§1–§6 + the OHP BBox
# trajectory format), `06-PATTERNS.md`, `06-VALIDATION.md`.
#
# Run in Colab on **L4 GPU** (24 GB VRAM). Mounts the Drive shortcut
# `My Drive/Fitness-AQA_dataset_release` (reused from Phases 2–5 — no new shortcut).
# OHP data is consolidated under that root:
#   Fitness-AQA_dataset_release/OHP/Labeled_Dataset/   (videos.zip 865 MB + Labels/ + Splits/)
#   Fitness-AQA_dataset_release/OHP/Unlabeled_Dataset/ (videos.zip 1765 MB + bar_trajectories_raw.zip)
#
# Plan 02 arc: Cell A bootstrap -> Step 0 stage (OHP labeled 2367 + unlabeled 5490 + traj)
#   -> Step 1 trajectory-format / half-cycle-sign PROBE (BLOCKING human-verify, before any GPU)
#   -> Step 2 SSL dataset smoke -> Step 3 model+VRAM probe -> Step 4 epoch-0 timing gate
#   (BLOCKING decision) -> Step 5 baseline training -> Step 6 threshold sweep -> Step 7 test eval.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Cells are delivered
# one at a time in chat per the FitNova interactive working agreement (paste-back gating).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in the fresh L4 Colab session.** Pulls `fresh-start` so Colab gets
# the Phase-6 Plan-01 OHP modules (`datasets/ohp.py`, `datasets/ohp_ssl.py`,
# `harness/colab.py` with `stage_ohp_videos` / `stage_unlabeled_ohp_videos`). Then
# `os.chdir` + `sys.path.insert` so `from backend.training.aqa...` resolves before
# Step 0 triggers the F8 `_envinit` import.
#
# Idempotent: clones if `/content/fitnova` is missing, otherwise fetch + checkout +
# ff-only pull. If the repo is already open from a prior session, this just
# fast-forwards to the latest `fresh-start`.

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH = "fresh-start"

if not os.path.isdir(REPO_DIR):
    print(f"Cloning {REPO_URL} (branch {BRANCH}) -> {REPO_DIR} ...")
    subprocess.run(
        ["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR],
        check=True,
    )
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

# Sanity: the Phase-6 OHP modules must be present (proves the fresh-start pull worked).
for _m in [
    "backend/training/aqa/datasets/ohp.py",
    "backend/training/aqa/datasets/ohp_ssl.py",
]:
    print(f"  {'OK ' if os.path.isfile(_m) else 'MISSING'} {_m}")
