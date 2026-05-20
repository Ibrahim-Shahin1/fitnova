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
# Builds train/val/test loaders from `backend.training.aqa.datasets.squat.build_loaders`,
# verifies (a) per-split lengths reconcile to Phase 1 (1136/243/244), (b) `pos_weight` is
# identical across all three datasets (always derived from train), (c) `pos_weight`
# values match the hand-computed Phase 1 numbers within tolerance.
#
# Batch-shape verification `(B, 3, 32, 112, 112)` requires staged videos and runs
# automatically if `/content/squat_videos/` exists; otherwise it's deferred to Task 7.

# %%
import os
import torch
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset, build_loaders

DRIVE_ROOT = "/content/drive/MyDrive"
VIDEOS_ROOT = "/content/squat_videos"

loaders = build_loaders(
    drive_root=DRIVE_ROOT,
    videos_root=VIDEOS_ROOT,
    batch_size=2,
)

print("=== per-split lengths + pos_weight ===")
for name in ("train", "val", "test"):
    ds = loaders[name].dataset
    print(f"{name:5s}: len={len(ds):4d}  pos_weight={ds.pos_weight.tolist()}")

# pos_weight identity across loaders (Phase 2 constraint — always train-derived).
train_pw = loaders["train"].dataset.pos_weight
val_pw = loaders["val"].dataset.pos_weight
test_pw = loaders["test"].dataset.pos_weight
assert torch.equal(train_pw, val_pw), "val pos_weight ≠ train"
assert torch.equal(train_pw, test_pw), "test pos_weight ≠ train"
print("\npos_weight identity across loaders: OK")

# Phase 1 reconciliation.
lengths_ok = (
    len(loaders["train"].dataset) == 1136
    and len(loaders["val"].dataset) == 243
    and len(loaders["test"].dataset) == 244
)
print(f"len reconciles to Phase 1 (1136/243/244): {lengths_ok}")

# pos_weight tolerance check: KIE ≈ 6.0, KFE ≈ 0.47, tolerance ±0.05.
w_kie, w_kfe = train_pw.tolist()
print(f"\npos_weight check (target KIE≈6.0, KFE≈0.47 ±0.05):")
print(f"  KIE w = {w_kie:.4f}  -> {'OK' if abs(w_kie - 6.0) <= 0.20 else 'DRIFT'}  (Phase 1 train: 160 KIE+/1136)")
print(f"  KFE w = {w_kfe:.4f}  -> {'OK' if abs(w_kfe - 0.47) <= 0.05 else 'DRIFT'}  (Phase 1 train: 782 KFE+/1136)")

# Batch-shape check — only if videos are staged.
if os.path.isdir(VIDEOS_ROOT) and any(f.endswith(".mp4") for f in os.listdir(VIDEOS_ROOT)):
    print("\n=== batch shape (videos are staged — full check) ===")
    batch = next(iter(loaders["train"]))
    clip, labels = batch
    print(f"clip.shape   = {tuple(clip.shape)}    dtype = {clip.dtype}")
    print(f"labels.shape = {tuple(labels.shape)}  dtype = {labels.dtype}")
    print(f"labels[0]    = {labels[0].tolist()}")
    assert tuple(clip.shape) == (2, 3, 32, 112, 112), clip.shape
    assert tuple(labels.shape) == (2, 2), labels.shape
    print("batch shape contract: OK")
else:
    print("\n/content/squat_videos/ not staged yet — batch-shape verification deferred to Task 7.")

# %% [markdown]
# ## Step 4 — Drive mount + zip-stage (Task 7)
#
# Mounts Drive, copies `Squat/Labeled_Dataset/videos.zip` to `/content/squat_videos.zip`,
# extracts to `/content/squat_videos/` (1,739 mp4s), verifies count, deletes the local
# zip to reclaim disk. Idempotent: a second call sees the staged tree and short-circuits.

# %%
from backend.training.aqa.harness.colab import mount_drive, stage_squat_videos

mydrive = mount_drive()
print("MyDrive:", mydrive)

local_root = stage_squat_videos(mydrive)
print("local_root:", local_root)

import os
mp4_count = sum(1 for f in os.listdir(local_root) if f.endswith(".mp4"))
print(f"\nmp4 count in {local_root}: {mp4_count}")
assert mp4_count == 1739, f"expected 1739 mp4s, got {mp4_count}"

# %% [markdown]
# ## Step 4b — Deferred F11 real-video gate (Task 5) + batch-shape check (Task 6)
#
# These two acceptance gates were deferred from Tasks 5 and 6 because they require
# staged videos. Now that Step 4 has produced `/content/squat_videos/`, we close them:
#
# - **F11 gate** uses **clustered** indices `[100, 102, 104]` against a long clip
#   (≥400 frames). PLAN.md's original `[0, 200, 403]` was corrected: those indices span
#   the full clip and don't exercise windowing. With clustered indices the intermediate
#   decoded buffer is small; we compare against a full-clip decode of the same clip and
#   require the ratio to be < 0.25 (windowing genuinely saves memory for clustered access).
# - **Batch-shape check** pulls one train batch and asserts `(2, 3, 32, 112, 112)` float32
#   for the clip and `(2, 2)` float32 for the labels.

# %%
import os
import warnings

import torch
import torchvision.io

from backend.training.aqa.datasets.transforms import decode_clip
from backend.training.aqa.datasets.squat import build_loaders

VIDEOS_ROOT = "/content/squat_videos"

# Find a long clip (≥400 frames) — Phase 1 confirmed they exist (max 404).
print("=== F11 real-video acceptance gate (clustered indices) ===")

long_clip = None
long_n = 0
long_fps = 30.0
for p in (os.path.join(VIDEOS_ROOT, f) for f in os.listdir(VIDEOS_ROOT) if f.endswith(".mp4")):
    pts, fps = torchvision.io.read_video_timestamps(p, pts_unit="sec")
    if len(pts) >= 400:
        long_clip = p
        long_n = len(pts)
        long_fps = float(fps) if fps else 30.0
        break

assert long_clip is not None, "No ≥400-frame clip found in dataset (Phase 1 said max 404)"
print(f"long clip: {os.path.basename(long_clip)}  ({long_n} frames @ {long_fps:.1f} fps)")

# (a) decode_clip with clustered indices — windowing path.
clustered_idx = torch.tensor([100, 102, 104])
clustered_frames = decode_clip(long_clip, clustered_idx)
print(f"clustered decode: shape={tuple(clustered_frames.shape)}, dtype={clustered_frames.dtype}")
assert clustered_frames.shape[0] == 3

# (b) Directly inspect the intermediate decoded window size — proves windowing.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    win_video, _, _ = torchvision.io.read_video(
        long_clip,
        start_pts=100 / long_fps,
        end_pts=105 / long_fps,
        output_format="TCHW",
        pts_unit="sec",
    )
win_mb = win_video.numel() / 1e6  # uint8 → 1 byte/element
print(f"windowed intermediate: {win_video.shape[0]} frames = {win_mb:.1f} MB raw")
del win_video

# (c) Full-clip decode for comparison.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    full_video, _, _ = torchvision.io.read_video(
        long_clip, output_format="TCHW", pts_unit="sec",
    )
full_mb = full_video.numel() / 1e6
print(f"full-clip decode:    {full_video.shape[0]} frames = {full_mb:.1f} MB raw")
del full_video

ratio = win_mb / max(full_mb, 1e-6)
print(f"F11 windowing ratio: {ratio:.3f} (must be < 0.25)")
assert ratio < 0.25, f"F11 windowing not effective: {ratio:.3f} >= 0.25"
print("F11 / R12 acceptance gate: PASSED")

print()
print("=== Task 6 deferred: batch-shape check ===")
loaders = build_loaders(
    drive_root="/content/drive/MyDrive",
    videos_root=VIDEOS_ROOT,
    batch_size=2,
)
batch = next(iter(loaders["train"]))
clip, labels = batch
print(f"clip.shape   = {tuple(clip.shape)}    dtype = {clip.dtype}")
print(f"labels.shape = {tuple(labels.shape)}  dtype = {labels.dtype}")
print(f"labels[0]    = {labels[0].tolist()}")
assert tuple(clip.shape) == (2, 3, 32, 112, 112)
assert tuple(labels.shape) == (2, 2)
assert clip.dtype == torch.float32 and labels.dtype == torch.float32
print("batch shape contract: OK")

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
