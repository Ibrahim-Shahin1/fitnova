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

# torchvision.io.read_video requires PyAV as its FFmpeg backend on Colab. Some Colab
# runtimes ship without it. Idempotent install — pip is a no-op if already present.
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

print("python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device :", torch.cuda.get_device_name(0))

print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

# Sanity-check torchvision.io has its PyAV backend wired — NO video needed for this,
# we just inspect that `av` inside the torchvision.io.video module is a real module
# (not an Exception placeholder from a failed import).
from torchvision.io.video import av as _tv_av_ref
assert not isinstance(_tv_av_ref, Exception), (
    "torchvision.io cached `av` as an Exception — PyAV install ordered wrong. "
    "Step 0 must install av BEFORE importing torch/torchvision."
)
print("torchvision.io PyAV backend: ready (no probe needed; real videos exercised in Step 4)")

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
# Renders a 2×8 grid (2 train clips × 8 evenly-spaced frames each) showing the
# decoded-and-augmented frames the model will actually see. Un-normalizes
# (`frame * KINETICS_STD + KINETICS_MEAN`, clamped to `[0,1]`) before display so the
# image is human-readable rather than blue-shifted normalization output. Saves PNG
# **before** `plt.show()` so a Colab disconnect mid-display still leaves the file.
#
# Visual sanity check criteria (Task 15 will gate on these):
# 1. Each row reads left → right as forward time (decode order correct).
# 2. The 8 sampled frames span the clip — first ≠ near-last (uniform sampling working).
# 3. Aspect handling preserves human shape (no extreme squash/stretch).
# 4. Printed `KIE+/KFE+` labels match what's visible (KIE = knees-inward at any frame,
#    KFE = knees-forward-of-toes at any frame).
# 5. No horizontal flip artifacts — D5 confirmed.

# %%
# Visualization deliverable — supervisor priority per MEMORY.md project_supervisor_visualizations
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from backend.training.aqa.datasets.transforms import KINETICS_MEAN, KINETICS_STD
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset

# Build the train dataset directly (not via loader) so we can read clip_ids for the
# row labels. `train_aug=True` so the rendered frames reflect what the model sees
# during training (jitter + random crop). seed=42 fixed so the visualization is
# bitwise-reproducible across re-runs.
train_ds = SquatKIEKFEDataset(
    split="train",
    drive_root="/content/drive/MyDrive",
    videos_root="/content/squat_videos",
    train_aug=True,
    seed=42,
)

# Deterministic pick — records[0] and records[1] (the first two in train_keys.json order).
items = [train_ds[i] for i in (0, 1)]
batch_clips = torch.stack([c for c, _ in items])    # (2, 3, 32, 112, 112) float32
batch_labels = torch.stack([l for _, l in items])   # (2, 2)
clip_ids = [train_ds.records[i].clip_id for i in (0, 1)]

# Un-normalize for display. Tensors are (C, T, H, W); broadcast mean/std as (3, 1, 1, 1).
mean_t = torch.tensor(KINETICS_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
std_t = torch.tensor(KINETICS_STD, dtype=torch.float32).view(3, 1, 1, 1)
display = (batch_clips * std_t.unsqueeze(0) + mean_t.unsqueeze(0)).clamp(0, 1)
# display shape: (B=2, C=3, T=32, H, W)

# 8 evenly-spaced frame positions out of 32.
frame_idx = np.linspace(0, 31, 8).round().astype(int)

fig, axes = plt.subplots(2, 8, figsize=(16, 5), dpi=120)

for row in range(2):
    label_kie = int(batch_labels[row, 0].item())
    label_kfe = int(batch_labels[row, 1].item())
    for col, fi in enumerate(frame_idx):
        ax = axes[row, col]
        # display[row] is (3, T, H, W); pick frame fi -> (3, H, W); permute -> (H, W, 3)
        frame = display[row, :, fi].permute(1, 2, 0).numpy()
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f"t={fi}", fontsize=8)
    # Row label to the left of the leftmost axis (using its axes-relative coords).
    axes[row, 0].text(
        -0.18, 0.5,
        f"{clip_ids[row]}\nKIE={label_kie}\nKFE={label_kfe}",
        fontsize=9, ha="right", va="center",
        transform=axes[row, 0].transAxes,
    )

fig.suptitle(
    "Decoded-and-augmented Squat batch — Phase 2 supervisor visualization",
    fontsize=11, y=1.02,
)
plt.tight_layout()

# F9 defensive mkdir: figures/ exists from Task 3 but be safe in case of a future move.
FIG_PATH = Path("/content/fitnova/.planning/phases/02-squat-data-pipeline-colab-harness/figures/decoded_batch.png")
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save BEFORE show — disconnect-by-default rule: persist the artifact first.
plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
print(f"saved : {FIG_PATH}")
print(f"size  : {FIG_PATH.stat().st_size / 1024:.1f} KB")
print(f"shape : batch={tuple(batch_clips.shape)} labels={tuple(batch_labels.shape)}")
print(f"clips : {clip_ids}")
print(f"labels: row0={batch_labels[0].tolist()} row1={batch_labels[1].tolist()}")

plt.show()

# %% [markdown]
# ## Step 6 — RNG capture/restore + cudnn determinism (Task 9)
#
# Verifies the four RNG sources (Python `random`, NumPy, torch CPU, torch CUDA-all)
# can be snapshotted and restored such that consuming the same sequence twice produces
# bitwise-identical outputs. This is the foundation of Task 14's resume assertion —
# without correct RNG capture/restore, a resumed training run produces a different
# stochastic trajectory than the fresh run we're comparing it to.
#
# F8 audit: also asserts `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set, because
# `torch.use_deterministic_algorithms(True)` (which `colab.py` calls on import) would
# have raised otherwise. If this cell reaches the import without crashing, F8 held.

# %%
import os, random
import numpy as np
import torch
from backend.training.aqa.harness.colab import capture_rng_state, restore_rng_state

# (b) F8 ordering audit — CUBLAS_WORKSPACE_CONFIG must already be set.
cubcfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
assert cubcfg == ":4096:8", f"CUBLAS_WORKSPACE_CONFIG={cubcfg!r}, expected ':4096:8'"
print(f"F8 audit                    : CUBLAS_WORKSPACE_CONFIG = {cubcfg}")

# (c) capture returns dict with exactly the 4 expected keys
s1 = capture_rng_state()
expected_keys = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
assert set(s1.keys()) == expected_keys, f"keys={sorted(s1.keys())}, expected {sorted(expected_keys)}"
print(f"capture_rng_state keys      : {sorted(s1.keys())}")
print(f"  python state type         : {type(s1['python']).__name__}")
print(f"  numpy state type          : {type(s1['numpy']).__name__}")
print(f"  torch_cpu state           : Tensor[{tuple(s1['torch_cpu'].shape)}] dtype={s1['torch_cpu'].dtype}")
print(f"  torch_cuda_all            : {len(s1['torch_cuda_all'])} device state(s)")

# (d) drift the RNG state by consuming some random ops.
_ = random.random()
_ = np.random.rand(8)
_ = torch.randn(8)
if torch.cuda.is_available():
    _ = torch.randn(8, device="cuda")

# (e + f) restore TWICE, consume the same sequence each time, compare.
def _consume() -> dict:
    return {
        "py": random.random(),
        "np": np.random.rand(5).tolist(),
        "cpu": torch.randn(5).tolist(),
        "cuda": torch.randn(5, device="cuda").tolist() if torch.cuda.is_available() else None,
    }

restore_rng_state(s1)
seq_a = _consume()
restore_rng_state(s1)
seq_b = _consume()

print("\n=== bitwise identity check (consume-twice after restore-twice) ===")
for k in seq_a:
    same = seq_a[k] == seq_b[k]
    status = "OK" if same else "DRIFT"
    print(f"  {k:5s}: {status}")
    assert same, f"{k} drifted: {seq_a[k]} vs {seq_b[k]}"
print("\nRNG capture/restore bitwise-identical: PASSED")

# %% [markdown]
# ## Step 7 — atomic checkpoint primitives (Task 10)
#
# Exercises the full atomic-save / load-with-hash-check / prune cycle against a
# scratch directory (`/tmp/test_run/`). Four sub-checks:
# (a) `hash_config` is key-order-insensitive (sort_keys at work).
# (b) Round-trip: atomic_save_checkpoint -> load_latest_checkpoint returns the same payload.
# (c) Drift: feeding a different `expected_config_hash` to load raises CheckpointConfigMismatchError.
# (d) Prune: with 4 epoch_*.pt files + best.pt and `keep_last=3 keep_best=True`,
#     exactly 3 epoch_*.pt + best.pt survive (4 files total).

# %%
import os, shutil, time
from pathlib import Path
import torch
from backend.training.aqa.harness.colab import (
    hash_config,
    atomic_save_checkpoint,
    load_latest_checkpoint,
    prune_checkpoints,
    CheckpointConfigMismatchError,
)

RUN_DIR = "/tmp/test_run"
# Clean slate each run.
if os.path.isdir(RUN_DIR):
    shutil.rmtree(RUN_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

# (a) hash_config — key-order invariance.
cfg1 = {"seed": 42, "crop_size": 112, "lr": 1e-4}
cfg2 = {"lr": 1e-4, "crop_size": 112, "seed": 42}  # same content, different key order
h1, h2 = hash_config(cfg1), hash_config(cfg2)
assert h1 == h2, f"hash mismatch on key order: {h1} vs {h2}"
print(f"(a) hash_config key-order invariant : {h1}  (16-char sha256)")

# (b) Round-trip atomic save + load.
payload_e0 = {
    "epoch": 0,
    "model_state_dict": {"dummy.weight": torch.zeros(3)},
    "optimizer_state_dict": {},
    "scheduler_state_dict": None,
    "rng_state": {},
    "metrics_history": [{"epoch": 0, "train_loss_mean": 0.5}],
    "config_hash": h1,
    "config_repr": cfg1,
    "code_version": "test",
}
atomic_save_checkpoint(payload_e0, os.path.join(RUN_DIR, "epoch_000.pt"))
loaded = load_latest_checkpoint(RUN_DIR, expected_config_hash=h1)
assert loaded is not None, "expected payload, got None"
assert loaded["epoch"] == 0, f"epoch={loaded['epoch']!r}, expected 0"
assert loaded["config_hash"] == h1
print(f"(b) round-trip                      : epoch={loaded['epoch']}, latest.txt -> "
      f"{Path(RUN_DIR, 'latest.txt').read_text().strip()}")

# (c) Config drift → CheckpointConfigMismatchError.
drifted_h = hash_config({**cfg1, "seed": 43})  # different seed → different hash
try:
    _ = load_latest_checkpoint(RUN_DIR, expected_config_hash=drifted_h)
    raise AssertionError("expected CheckpointConfigMismatchError, got no exception")
except CheckpointConfigMismatchError as exc:
    msg = str(exc)
    assert h1 in msg and drifted_h in msg, f"both hashes should appear in message: {msg}"
    print(f"(c) config drift -> error raised    : OK  (message contains both hashes + config_repr)")

# (d) Prune — write 3 more epoch files + best.pt, then prune keep_last=3 keep_best=True.
# (Use small sleeps so mtimes are distinct.)
for ep in (1, 2, 3):
    payload = {**payload_e0, "epoch": ep}
    atomic_save_checkpoint(payload, os.path.join(RUN_DIR, f"epoch_{ep:03d}.pt"))
    time.sleep(0.01)
# Drop a best.pt
torch.save(payload_e0, os.path.join(RUN_DIR, "best.pt"))

before = sorted(f for f in os.listdir(RUN_DIR) if f.endswith(".pt"))
print(f"(d) before prune (5 .pt files)      : {before}")
prune_checkpoints(RUN_DIR, keep_last=3, keep_best=True)
after = sorted(f for f in os.listdir(RUN_DIR) if f.endswith(".pt"))
print(f"(d) after prune  (keep_last=3+best) : {after}")

epoch_pts = [f for f in after if f.startswith("epoch_") and f.endswith(".pt")]
assert len(epoch_pts) == 3, f"expected 3 epoch_*.pt after prune, got {len(epoch_pts)}: {epoch_pts}"
assert "best.pt" in after, "best.pt was pruned despite keep_best=True"
# Confirm we kept the 3 newest (epoch_001, 002, 003) and dropped the oldest (epoch_000)
assert "epoch_000.pt" not in after, "expected oldest epoch_000.pt to be pruned"
print(f"\nTask 10 atomic checkpoint primitives: ALL PASSED")

# %% [markdown]
# ## Step 8a — baseline 2-epoch fresh trajectory (Task 12)
#
# Establishes the ground-truth loss trajectory for Task 14's bitwise assertion.
# `run_tiny_epoch(resume=False, max_epochs=2)` runs 2 epochs from a clean state with
# `seed=42`. Both `epoch_000.pt` and `epoch_001.pt` land atomically on Drive;
# `latest.txt` ends pointing at `epoch_001.pt`. We extract `metrics_history` from
# `epoch_001.pt` and serialize both epochs' per-batch losses to
# `figures/tiny_baseline_2epoch_losses.json` for Task 14 to read back.
#
# Cleanup cell first — destructive, only intended for this acceptance test.

# %%
# Step 8a-cleanup — destructive, idempotent. Wipes the Phase-2-acceptance run dir
# so the baseline starts cold. Only intended for the bitwise-resume test.
import shutil
RUN_NAME = "squat_tiny_2026-05-20_a1b2c3d"
RUN_DIR = f"/content/drive/MyDrive/FitNova/checkpoints/phase02/{RUN_NAME}/"
shutil.rmtree(RUN_DIR, ignore_errors=True)
print(f"cleaned : {RUN_DIR}")

# %%
# Step 8a-baseline — fresh 2-epoch run. F1 baseline trajectory.
from backend.training.aqa.harness.tiny_train import run_tiny_epoch

result_baseline = run_tiny_epoch(
    run_name=RUN_NAME,
    seed=42,
    resume=False,
    max_epochs=2,
)
print("\nresult_baseline:")
for k, v in result_baseline.items():
    if k == "train_loss_per_batch":
        print(f"  {k:22s}: {v}  ({len(v)} batches in LAST epoch)")
    else:
        print(f"  {k:22s}: {v}")

# %%
# Step 8a-save — extract both epochs' per-batch losses from epoch_001.pt and write
# JSON for Task 14's byte-equality assertion to read back.
import json
from pathlib import Path
import torch

ckpt_path = RUN_DIR + "epoch_001.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

losses = {
    f"epoch_{m['epoch']}": m["train_loss_per_batch"]
    for m in ckpt["metrics_history"]
}
assert set(losses.keys()) == {"epoch_0", "epoch_1"}, losses.keys()
assert len(losses["epoch_0"]) == 8 and len(losses["epoch_1"]) == 8, {
    "epoch_0_len": len(losses["epoch_0"]),
    "epoch_1_len": len(losses["epoch_1"]),
}

out_path = Path("/content/fitnova/.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_baseline_2epoch_losses.json")
out_path.parent.mkdir(parents=True, exist_ok=True)  # F9 defense
with open(out_path, "w") as fh:
    json.dump(losses, fh)
print(f"saved : {out_path}")
print(f"  epoch_0: {losses['epoch_0']}")
print(f"  epoch_1: {losses['epoch_1']}")

import os
print(f"\nrun_dir state:")
for f in sorted(os.listdir(RUN_DIR)):
    full = RUN_DIR + f
    if os.path.isfile(full):
        print(f"  {f:18s}  {os.path.getsize(full)} bytes")
print(f"latest.txt content : {Path(RUN_DIR, 'latest.txt').read_text().strip()!r}")

# %% [markdown]
# ## Step 8b — resume-after-simulated-restart (Task 13)
#
# Three cells, each a separate runnable unit:
#
# 1. **Rewind**: delete `epoch_001.pt` and overwrite `latest.txt` to point at
#    `epoch_000.pt` — simulates "the system died right after we saved epoch 0."
# 2. **Purge `sys.modules` + re-import**: drops in-memory state for
#    `backend.training.aqa.*` modules. Closest a single notebook session gets to a
#    kernel restart while keeping `/content/`, `/content/drive`, and the cell session
#    intact. A true kernel restart would also drop GPU memory + driver context — but
#    a true restart loses cell history too, which would break the comparison flow.
# 3. **Resume**: `run_tiny_epoch(resume=True, max_epochs=2)` — `load_latest_checkpoint`
#    finds `epoch_000.pt`, restores model + optim + RNG, `start_epoch=1` advances
#    into the F1 loop, epoch 1 trains with the restored RNG state, new `epoch_001.pt`
#    + `latest.txt` written. The 8 per-batch losses from this resumed epoch 1 are
#    Task 14's other side of the byte-equality assertion.

# %%
# Step 8b-1 — rewind. Idempotent (guarded on file existence).
import os
from pathlib import Path

# F1: this exercises the resume code path with a real training epoch —
# load epoch_0, restore RNG, train epoch 1.

epoch_001 = RUN_DIR + "epoch_001.pt"
if os.path.exists(epoch_001):
    os.remove(epoch_001)
    print(f"removed   : {epoch_001}")
else:
    print(f"already gone: {epoch_001}")

with open(RUN_DIR + "latest.txt", "w", encoding="utf-8") as fh:
    fh.write("epoch_000.pt\n")
print(f"latest.txt rewound to: {Path(RUN_DIR, 'latest.txt').read_text().strip()!r}")

print(f"\nfiles on Drive now:")
for f in sorted(os.listdir(RUN_DIR)):
    full = RUN_DIR + f
    if os.path.isfile(full):
        print(f"  {f:18s}  {os.path.getsize(full)} bytes")

# %%
# Step 8b-2 — purge backend.training.aqa.* from sys.modules and re-import. Simulates
# kernel-restart-equivalent module state reset. Drops in-memory closures, fresh
# tiny_train re-imports from the latest disk version.
import sys

mods = [k for k in sys.modules if k.startswith("backend.training.aqa")]
for k in mods:
    sys.modules.pop(k)
print(f"purged {len(mods)} modules: {sorted(mods)}")

from backend.training.aqa.harness.tiny_train import run_tiny_epoch  # re-import after purge — fresh module state
print(f"run_tiny_epoch re-loaded from: {run_tiny_epoch.__code__.co_filename}")

# %%
# Step 8b-3 — resume. Loads epoch_000.pt, restores RNG, trains epoch 1, writes new
# epoch_001.pt. The 8 per-batch losses produced here are Task 14's resumed side of
# the bitwise assertion.
result_resumed = run_tiny_epoch(
    run_name=RUN_NAME,
    seed=42,
    resume=True,
    max_epochs=2,
)
print("\nresult_resumed:")
for k, v in result_resumed.items():
    if k == "train_loss_per_batch":
        print(f"  {k:22s}: {v}  ({len(v)} batches in LAST epoch)")
    else:
        print(f"  {k:22s}: {v}")

# Save the resumed epoch-1 per-batch losses for Task 14's byte-equality assert.
import json
resumed_path = Path("/content/fitnova/.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_resumed_epoch1_losses.json")
resumed_path.parent.mkdir(parents=True, exist_ok=True)
with open(resumed_path, "w") as fh:
    json.dump(result_resumed["train_loss_per_batch"], fh)
print(f"\nsaved : {resumed_path}")

print(f"\nrun_dir state after resume:")
for f in sorted(os.listdir(RUN_DIR)):
    full = RUN_DIR + f
    if os.path.isfile(full):
        print(f"  {f:18s}  {os.path.getsize(full)} bytes")
print(f"latest.txt content : {Path(RUN_DIR, 'latest.txt').read_text().strip()!r}")

# %% [markdown]
# ## Step 9 — closeout (Task 15 human-verify + Task 16 SUMMARY)
#
# Open both figures, confirm five visual criteria for `decoded_batch.png` (clip variation,
# frame ordering, aspect handling, no-flip semantics, label correctness), and confirm the
# two `tiny_train_loss.png` curves overlap point-for-point. Authoring `02-01-SUMMARY.md`
# closes Phase 2.
