# Phase 2 — Squat Data Pipeline & Resumable Colab Harness
#
# Companion notebook to `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md`.
# Reference: `02-CONTEXT.md` (locked decisions), `02-RESEARCH.md` (technical findings),
# `CODE-RELEASE-NOTES.md` (gap analysis of the official Code_Release).
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
# This sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** any `import torch` triggers CUDA
# context creation. If torch is imported first, `torch.use_deterministic_algorithms(True)`
# (Task 9) will raise at runtime. Do not reorder.

# %%
from backend.training.aqa.harness import _envinit  # F8: sets CUBLAS_WORKSPACE_CONFIG before torch

import os
import sys
import torch
import torchvision

print("python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device :", torch.cuda.get_device_name(0))

print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

from google.colab import drive
drive.mount('/content/drive')

MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
print("\nROOT :", ROOT)
print("exists:", os.path.exists(ROOT))

# %% [markdown]
# ## Step 1 — official splits + labels (Task 4)
#
# Loads `{train,val,test}_keys.json` + `error_knees_{inward,forward}.json` via
# `backend.training.aqa.datasets.splits.index`. Reconciles per-split counts against
# Phase 1 (1136/243/244) and confirms splits are disjoint (Phase 1 invariant).
# `videos_root` is interpolated into each record's `video_path` but not opened here —
# the .mp4 files don't need to be staged yet (that's Task 7).

# %%
from backend.training.aqa.datasets.splits import index, expected_counts, ClipRecord

DRIVE_ROOT = "/content/drive/MyDrive"
VIDEOS_ROOT = "/content/squat_videos"   # interpolated into record.video_path; not opened here

print("expected counts:", expected_counts())
print()

all_ids: dict[str, set[str]] = {}
for split_name in ("train", "val", "test"):
    recs = index(split_name, drive_root=DRIVE_ROOT, videos_root=VIDEOS_ROOT)
    kie_pos = sum(r.label_kie for r in recs)
    kfe_pos = sum(r.label_kfe for r in recs)
    print(f"{split_name:5s}: {len(recs):4d} records   KIE+ {kie_pos:3d}   KFE+ {kfe_pos:3d}")
    all_ids[split_name] = {r.clip_id for r in recs}

print()
print("=== disjointness check (Phase 1 invariant) ===")
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = all_ids[a] & all_ids[b]
    print(f"{a} ∩ {b}: {len(overlap)} overlapping ids (must be 0)")

print()
print("=== first train record (spot check) ===")
first = index("train", drive_root=DRIVE_ROOT, videos_root=VIDEOS_ROOT)[0]
print(first)

# %% [markdown]
# ## Step 2 — transforms synthetic smoke (Task 5)
#
# Exercises `uniform_sample_indices`, `spatial_train`, `spatial_val`, and Kinetics
# normalization on a synthetic `[T=60, 3, 480, 600]` uint8 tensor. Output shape must be
# `(3, 60, 112, 112)` float32 with normalized mean near 0.
#
# **F11 real-video gate deferred to Task 7** — `decode_clip` against real `.mp4`s needs
# `/content/squat_videos/` staged. The test case in PLAN.md (`indices=[0, 200, 403]`) was
# corrected: those indices span the full clip and don't exercise windowing. The real F11
# proof uses **clustered indices** `[100, 102, 104]` against a long clip, which is what
# windowing is designed to optimize.

# %%
import sys, subprocess
result = subprocess.run(
    [sys.executable, "/content/fitnova/backend/training/aqa/datasets/transforms.py"],
    capture_output=True, text=True,
)
print("--- stdout ---")
print(result.stdout)
if result.returncode != 0:
    print("--- stderr ---")
    print(result.stderr)
    raise RuntimeError(f"transforms.py smoke failed (exit {result.returncode})")

# %% [markdown]
# ## Step 3 — dataset + loaders (Task 6)
#
# Fills in once `squat.py` is implemented. Cell will build `train/val/test` loaders and
# pull a single batch to verify the tensor shape contract `(B, 3, 32, 112, 112)` + labels
# `(B, 2)` + `dataset.pos_weight` field.

# %% [markdown]
# ## Step 4 — Drive mount + zip-stage (Task 7)
#
# Fills in once `harness/colab.py` slice 1 is implemented. Stages
# `Squat/Labeled_Dataset/videos.zip` → `/content/squat_videos/` (1,739 mp4s). Prerequisite
# for Step 5 visualization and Steps 8–11 training.

# %% [markdown]
# ## Step 5 — decoded-batch visualization (Task 8) — supervisor priority
#
# Fills in once Tasks 4–7 are in place. Renders a 2×8 grid of decoded-and-augmented
# frames from the train loader, saves to
# `.planning/phases/02-squat-data-pipeline-colab-harness/figures/decoded_batch.png`. Task
# 15 is a blocking human-verify gate against five concrete visual criteria.

# %% [markdown]
# ## Step 6 — RNG capture/restore + cudnn determinism (Task 9)
#
# Fills in once `harness/colab.py` slice 2 is implemented. Round-trip identity test:
# `restore_rng_state(capture_rng_state())` is bitwise-identity on the RNG state.

# %% [markdown]
# ## Step 7 — atomic checkpoint primitives (Task 10)
#
# Fills in once `harness/colab.py` slice 3 is implemented. Round-trips a fake checkpoint
# through atomic save → load → verify; confirms `CheckpointConfigMismatchError` fires on
# intentional config drift; verifies prune keeps last-3 + best.

# %% [markdown]
# ## Step 8 — tiny end-to-end run + bitwise-resume proof (Tasks 11–14)
#
# Fills in once `harness/tiny_train.py` is implemented. Three cells:
# - 8a: baseline 2-epoch fresh run → `tiny_baseline_2epoch_losses.json`
# - 8b: simulated restart (delete epoch_001.pt, rewind `latest.txt`, `sys.modules` purge,
#   resume into epoch 1) → `tiny_resumed_epoch1_losses.json`
# - 8c: assert byte-exact equality between baseline epoch-1 and resumed epoch-1 losses;
#   save overlay plot `figures/tiny_train_loss.png`.

# %% [markdown]
# ## Step 9 — closeout (Task 15 human-verify + Task 16 SUMMARY)
#
# Open both figures, confirm five visual criteria for `decoded_batch.png` (clip variation,
# frame ordering, aspect handling, no-flip semantics, label correctness), and confirm the
# two `tiny_train_loss.png` curves overlap point-for-point. Authoring `02-01-SUMMARY.md`
# closes Phase 2.
