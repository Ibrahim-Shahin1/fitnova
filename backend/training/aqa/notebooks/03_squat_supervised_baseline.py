# Phase 3 — Squat Supervised Baseline (R(2+1)D-18, Kinetics-400-V1 init)
#
# Companion notebook to `.planning/phases/03-squat-supervised-baseline/03-01-PLAN.md`.
# Reference: `03-CONTEXT.md` (locked decisions D1–D14), `03-RESEARCH.md` (paper §5
# hyperparameters + RESEARCH §1–§13), `03-PATTERNS.md` (file → analog map),
# `03-VALIDATION.md` (Nyquist contract).
#
# Run in Colab on **L4 GPU** (24 GB VRAM). Mounts Drive shortcut
# `My Drive/Fitness-AQA_dataset_release` (reused from Phase 2 — no new shortcut).
# Estimated wall-time: 2–4 h best case / 6–8 h worst case (RESEARCH §10) —
# Task 10 (Step 3) is a BLOCKING timing gate that asks you to authorise the full
# 50-epoch run after measuring epoch-0 wall time.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Convert via
# `pip install jupytext && jupytext --to ipynb 03_squat_supervised_baseline.py`,
# or paste cells one at a time per the FitNova interactive working agreement.

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in a fresh Colab session.** Phase 2 precedent (02-01-SUMMARY.md):
# "New session bootstrap is Cell A (clone-or-pull) → Cell B (Step 0)." Phase 3
# bakes Cell A into the notebook so `from backend...` resolves before Step 0
# triggers the F8 `_envinit` import.
#
# Idempotent: clones the repo if `/content/fitnova` is missing, otherwise
# `git pull` to fast-forward. Then `os.chdir` + `sys.path.insert` so the
# Phase 2 modules under `backend/training/aqa/...` are importable.

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


# %% [markdown]
# ## Step 0 — environment + dep version probe + GPU/VRAM check + Drive mount + stage videos (Task 7)
#
# **F8 import-order constraint (PLAN.md D10 / CONTEXT.md D10 / Phase 2 D13):** the
# first executable line of the next cell **MUST** be
# `from backend.training.aqa.harness import _envinit`. This sets
# `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** any `import torch` triggers CUDA
# context creation. If torch is imported first, `torch.use_deterministic_algorithms(True)`
# (Phase 2 colab.py at import time) will raise at runtime. Do not reorder.
#
# Three Phase 3 additions on top of the Phase 2 Step 0:
# 1. Dep version probe — confirms torch / torchvision / sklearn / matplotlib /
#    tqdm / numpy are installed (T-03-04 supply-chain guard). No new pip
#    installs in Phase 3 beyond Phase 2's PyAV.
# 2. GPU + VRAM check — asserts we're on L4 (or ≥22 GB VRAM) per CONTEXT D10 /
#    [[feedback_heavy_training_new_notebook]].
# 3. `stage_squat_videos` — idempotent cache-hit per Phase 2 D15 (re-downloads
#    `videos.zip` only on fresh `/content/`).

# %%
from backend.training.aqa.harness import _envinit  # F8: sets CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import subprocess
import sys

# Phase 2 carry-forward: torchvision.io.read_video requires PyAV as its FFmpeg
# backend on Colab; install BEFORE the first torch/torchvision import so the
# cached `av` reference in torchvision.io.video is a real module.
try:
    import av  # noqa: F401
    _pyav_status = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av  # noqa: F401
    _pyav_status = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav_status})")

# Phase 3 addition (1) — Dep version probe (T-03-04). No new pip installs.
_PHASE3_DEPS = ["torch", "torchvision", "scikit-learn", "matplotlib", "tqdm", "numpy"]
print("\n[T-03-04] Dependency versions (supply-chain guard):")
for _pkg in _PHASE3_DEPS:
    try:
        _ver = importlib.metadata.version(_pkg)
        print(f"  {_pkg:<14} {_ver}")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"[T-03-04] Phase 3 dependency '{_pkg}' is NOT installed. Phase 3 "
            "introduces zero new pip dependencies — every package was already "
            "in backend/requirements.txt at Phase 2 close. If this fires, the "
            "Colab runtime is misconfigured."
        )

import torch
import torchvision

print("\npython     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

# Phase 3 addition (2) — GPU + VRAM check (CONTEXT D10 / RESEARCH §10).
assert torch.cuda.is_available(), (
    "GPU required for Phase 3 — Runtime > Change runtime type > Hardware accelerator: L4"
)
_gpu_name = torch.cuda.get_device_name(0)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"CUDA device : {_gpu_name} ({_vram_gb:.1f} GB VRAM)")
assert "L4" in _gpu_name or _vram_gb >= 22.0, (
    f"Phase 3 expects L4 GPU with ≥22 GB VRAM; got {_gpu_name} {_vram_gb:.1f} GB. "
    "RESEARCH §10 sizing assumes 24 GB headroom for fp32 batch 16 on R(2+1)D-18."
)

# Phase 2 carry-forward — verify torchvision.io PyAV backend (no real video needed).
from torchvision.io.video import av as _tv_av_ref
assert not isinstance(_tv_av_ref, Exception), (
    "torchvision.io cached `av` as an Exception — PyAV install ordered wrong. "
    "Step 0 must install av BEFORE importing torch/torchvision."
)
print("torchvision.io PyAV backend: ready")

# Drive mount (Phase 2 idempotent — drive.mount no-ops on a second call).
from google.colab import drive
drive.mount('/content/drive')

MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
print("\nROOT :", ROOT)
print("exists:", os.path.exists(ROOT))

# Phase 3 addition (3) — idempotent video staging (Phase 2 D15 cache-hit path).
from backend.training.aqa.harness.colab import stage_squat_videos
VIDEOS_ROOT = stage_squat_videos(MYDRIVE)
from pathlib import Path
_mp4_count = len(list(Path(VIDEOS_ROOT).glob('*.mp4')))
print(f"\nVIDEOS_ROOT: {VIDEOS_ROOT}")
print(f"mp4 count: {_mp4_count} (expected 1739 per Phase 1)")
assert _mp4_count == 1739, f"Expected 1739 mp4s, got {_mp4_count}"


# %% [markdown]
# ## Step 1 — dataset smoke (Task 8)
#
# Phase 2 pipeline wires up correctly in this fresh Colab session — counts
# reconcile, pos_weight matches train-derivation, one batch loads with the
# expected `(B, 3, T, H, W)` + `(B, 2)` shapes. Fast (no model construction
# yet). If anything drifts, stop here.

# %%
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    _build_dataloaders,
)

config = SupervisedConfig()

train_ds = SquatKIEKFEDataset(
    split="train",
    drive_root=MYDRIVE,
    videos_root=VIDEOS_ROOT,
    num_frames=config.num_frames,
    crop_size=config.crop_size,
    train_aug=True,
    train_jitter_frames=config.train_jitter_frames,
)
val_ds = SquatKIEKFEDataset(
    split="val",
    drive_root=MYDRIVE,
    videos_root=VIDEOS_ROOT,
    num_frames=config.num_frames,
    crop_size=config.crop_size,
    train_aug=False,
    train_jitter_frames=config.train_jitter_frames,
)
test_ds = SquatKIEKFEDataset(
    split="test",
    drive_root=MYDRIVE,
    videos_root=VIDEOS_ROOT,
    num_frames=config.num_frames,
    crop_size=config.crop_size,
    train_aug=False,
    train_jitter_frames=config.train_jitter_frames,
)

print(f"len(train_ds) = {len(train_ds)} (expected 1136)")
print(f"len(val_ds)   = {len(val_ds)} (expected 243)")
print(f"len(test_ds)  = {len(test_ds)} (expected 244)")
assert len(train_ds) == 1136 and len(val_ds) == 243 and len(test_ds) == 244, (
    "split counts drifted from Phase 1 — investigate splits.expected_counts() "
    "and the staged videos.zip BEFORE running training."
)

print(f"\npos_weight (train-derived): {train_ds.pos_weight.tolist()}")
print("                            expected approximately [≈6.10, ≈0.45]")
# Phase 2 D9 invariant — pos_weight identical across all three split instances
# (computed from train regardless of which split this is). Phase 3 D2 consumes
# the same train-derived weights for BCEWithLogitsLoss on every loader.
assert val_ds.pos_weight.tolist() == train_ds.pos_weight.tolist(), (
    "pos_weight differs across splits — Phase 2 D9 invariant broken"
)
assert test_ds.pos_weight.tolist() == train_ds.pos_weight.tolist(), (
    "pos_weight differs across splits — Phase 2 D9 invariant broken"
)

# D4: direct DataLoader construction with num_workers=4 + seed_worker. First
# `next(iter(...))` is slow on first call because the 4 workers fork-spawn.
loaders = _build_dataloaders(
    seed=42, config=config, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
)
clip, label = next(iter(loaders["train"]))
print(f"\nFirst train batch:")
print(f"  clip.shape  = {tuple(clip.shape)}  (expected ({config.batch_size}, 3, {config.num_frames}, {config.crop_size}, {config.crop_size}))")
print(f"  clip.dtype  = {clip.dtype}")
print(f"  label.shape = {tuple(label.shape)}  (expected ({config.batch_size}, 2))")
print(f"  label.dtype = {label.dtype}")
print(f"  label (KIE/KFE per clip):\n{label}")

import torch
assert tuple(clip.shape) == (
    config.batch_size, 3, config.num_frames, config.crop_size, config.crop_size,
), tuple(clip.shape)
assert tuple(label.shape) == (config.batch_size, 2), tuple(label.shape)
assert label.dtype == torch.float32, label.dtype


# %% [markdown]
# ## Step 2 — model construction smoke (Task 9)
#
# Build R(2+1)D-18 + `Linear(512, 2)` head, run one forward + backward pass on a
# real batch, measure peak VRAM. Confirms L4 headroom before Step 4 commits to
# multi-hour training.

# %%
import torch  # noqa: F811 — also imported in Step 1; harmless in notebook
from backend.training.aqa.harness.supervised_train import build_model

model = build_model().to("cuda")  # ~120 MB Kinetics-V1 download on first call
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total_params     = {total_params:_}")
print(f"trainable_params = {trainable_params:_}")
# D1: end-to-end fine-tune — every parameter must be trainable.
assert trainable_params == total_params, (
    f"Expected end-to-end fine-tune; got {trainable_params:_} trainable / "
    f"{total_params:_} total — D1 broken"
)
# Sanity range — R(2+1)D-18 with 2-output head sits around 31.3 M params. The
# ±2% tolerance absorbs torchvision minor-version drift in BN running-mean
# params across Kinetics-V1 weight pubs.
assert 30_000_000 < total_params < 33_000_000, (
    f"total_params={total_params:_} outside the 30–33 M sanity range — "
    "torchvision API may have changed (RESEARCH §3)"
)

# Forward smoke — Step 1's `loaders` is in scope.
clip, label = next(iter(loaders["train"]))
clip = clip.to("cuda")
label = label.to("cuda")
with torch.no_grad():
    logits = model(clip)
print(f"\nlogits.shape = {tuple(logits.shape)}  (expected (16, 2))")
print(f"logits.dtype = {logits.dtype}")
print(f"logits sample:\n{logits.detach().cpu()}")
assert tuple(logits.shape) == (16, 2), tuple(logits.shape)
assert logits.dtype == torch.float32, logits.dtype

# VRAM probe — forward only.
peak_vram_fwd_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"\nPeak VRAM after forward (batch 16, no backward): {peak_vram_fwd_gb:.2f} GB")
torch.cuda.reset_peak_memory_stats()

# Backward smoke — exposes optimizer/gradient VRAM. Uses train_ds.pos_weight per D2.
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight.to("cuda"))
logits = model(clip)
loss = criterion(logits, label)
loss.backward()
peak_vram_train_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Peak VRAM after backward (batch 16, fp32): {peak_vram_train_gb:.2f} GB")
assert peak_vram_train_gb < 20.0, (
    f"VRAM after backward = {peak_vram_train_gb:.2f} GB exceeds 20 GB headroom "
    "on L4 (22 GB usable). Reduce batch_size in SupervisedConfig or enable AMP "
    "(overriding RESEARCH §11)."
)

# Free GPU memory before Step 3 timing probe constructs its own model.
del model, logits, loss, criterion
torch.cuda.empty_cache()
print("\nGPU freed; ready for Step 3 timing probe.")


# %% [markdown]
# ## Step 3 — epoch 0 timing probe (Task 10) — **BLOCKING gate before full training**
#
# One-epoch fresh run via `run_supervised_epoch(max_epochs=1, resume=False,
# run_name="..._timing")`. Measures wall time + writes a ~360 MB checkpoint to
# Drive + round-trips `torch.load`. Paste back `epoch_0_time_s` and
# `estimated_total_h = epoch_0_time_s * 50 / 3600`. Pick option-a / -b / -c per
# Task 10 of the PLAN:
#
# - **option-a:** `estimated_total_h <= 8`  → proceed with locked defaults (50 epochs)
# - **option-b:** `estimated_total_h > 8`   → drop to 15 epochs + early_stop_patience=5
# - **option-c:** override RESEARCH §11 and enable AMP (NOT recommended — determinism conflict)

# %%
import time
import os

import torch  # noqa: F811 — already imported earlier; harmless in notebook
from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    run_supervised_epoch,
)

timing_config = SupervisedConfig()  # locked defaults (D6)

# T-03-02: write to a `_timing` run dir so this probe doesn't collide with the
# production training dir. The probe checkpoint can be pruned manually after.
TIMING_RUN_NAME = "r2plus1d18_squat_supervised_v1_timing"

t0 = time.perf_counter()
result = run_supervised_epoch(
    run_name=TIMING_RUN_NAME,
    drive_root=MYDRIVE,
    videos_root=VIDEOS_ROOT,
    seed=42,
    config=timing_config,
    resume=False,           # fresh run — don't pick up a stale timing checkpoint
    max_epochs=1,
)
wall_time_s = time.perf_counter() - t0

# Trainer's own per-epoch wall-time (includes train pass + val pass + atomic write).
epoch_0_time_s = result["metrics_history"][0]["epoch_wall_time_s"]
estimated_total_h = epoch_0_time_s * timing_config.max_epochs / 3600.0
val_macro_f1_after_1_epoch = result["metrics_history"][0]["val_macro_f1"]

print(f"epoch_0_time_s        = {epoch_0_time_s:.1f} s")
print(f"wall_time_s (outer)   = {wall_time_s:.1f} s")
print(f"max_epochs (config)   = {timing_config.max_epochs}")
print(f"estimated_total_h     = {estimated_total_h:.2f}  ({timing_config.max_epochs} × epoch_0 / 3600)")
print(f"val_macro_f1 (1 epoch)= {val_macro_f1_after_1_epoch:.4f}  (sanity: > 0.3 expected — pos_weight + Kinetics init give a strong KFE prior)")

# Per-batch wall-time printout (RESEARCH §13 Q2).
loss_per_batch = result["metrics_history"][0]["train_loss_per_batch"]
n_train_batches = len(loss_per_batch)
n_val_batches = (243 + timing_config.batch_size - 1) // timing_config.batch_size
expected_train_batches = (1136 + timing_config.batch_size - 1) // timing_config.batch_size
print(f"\nTrain batches      : {n_train_batches}  (expected ≈{expected_train_batches})")
print(f"Val batches        : {n_val_batches}")
print(f"Mean per-batch time: {epoch_0_time_s / (n_train_batches + n_val_batches) * 1000:.0f} ms  (incl. data load + I/O)")

# T-03-02: Drive checkpoint size audit (RESEARCH §7 — ~360 MB).
ckpt_path = result["checkpoint_path"]
ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
print(f"\nepoch_000.pt path = {ckpt_path}")
print(f"epoch_000.pt size = {ckpt_size_mb:.0f} MB  (expected 300–500 MB per RESEARCH §7)")
assert 300 < ckpt_size_mb < 500, (
    f"Checkpoint size {ckpt_size_mb:.0f} MB outside expected 300–500 MB range "
    "(T-03-02). Investigate before authorising the full run."
)

# T-03-02: Round-trip torch.load smoke — verify the checkpoint reads back and
# has every Phase 3 D8 key (SQUAT-03-e at runtime, not just unit-test).
payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
_required = {
    "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "rng_state", "metrics_history", "best_f1_val", "best_thresholds",
    "config_hash", "config_repr", "code_version",
}
_missing = _required - payload.keys()
assert not _missing, f"Checkpoint missing keys: {_missing} — D8 schema broken"
assert payload["code_version"] == "phase03-supervised-baseline", payload["code_version"]
print(f"\nCheckpoint schema OK (all 11 D8 keys present; code_version='{payload['code_version']}')")

print("\n" + "=" * 72)
print("DECISION GATE — paste back the numbers above and select option-a / -b / -c:")
print("  option-a: estimated_total_h ≤ 8  → proceed with locked defaults (50 epochs)")
print("  option-b: estimated_total_h > 8  → drop to 15 epochs + early_stop_patience=5")
print("  option-c: enable AMP (NOT recommended — RESEARCH §11 determinism conflict)")
print("=" * 72)


# %% [markdown]
# ## Step 4 — full supervised training run (Task 11)
#
# **Long-running cell — ~3-4 h on L4 with Step 3's measured 5.3 min/epoch and
# 8-epoch val-macro-F1 early-stop patience.** Per-epoch tqdm + metric log lines
# + atomic `epoch_NNN.pt` per epoch + `best.pt` on val-macro-F1 improvement
# (D13). Disconnect mid-run? Re-run Cell A → Step 0 → Step 4 — `resume=True`
# picks up from `latest.txt` automatically (Phase 2 atomic-write contract).

# %%
import os

from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    run_supervised_epoch,
)

config = SupervisedConfig()  # option-a authorised at Step 3 (4.44 h ≤ 8 h)
RUN_NAME = "r2plus1d18_squat_supervised_v1"  # PLAN.md specifics

print(f"Starting Phase 3 production run: {RUN_NAME}")
print(f"  max_epochs            = {config.max_epochs}")
print(f"  early_stop_patience   = {config.early_stop_patience}")
print(f"  batch_size            = {config.batch_size}")
print(f"  num_workers           = {config.num_workers}")
print(f"  learning_rate         = {config.learning_rate}")
print(f"  weight_decay          = {config.weight_decay}")
print(f"  scheduler             = {config.scheduler_name} (T_max={config.scheduler_t_max})")
print(f"\nDisconnect-safe: re-run Cell A → Step 0 → Step 4 to resume from latest.txt.\n")

result = run_supervised_epoch(
    run_name=RUN_NAME,
    drive_root=MYDRIVE,
    videos_root=VIDEOS_ROOT,
    seed=42,
    config=config,
    resume=True,                       # idempotent — picks up from latest.txt on rerun
    max_epochs=config.max_epochs,      # 50
)

run_dir = os.path.dirname(result["checkpoint_path"])
print("\n" + "=" * 72)
print("Phase 3 training complete (or stopped early on val-macro-F1 plateau).")
print(f"  Final epoch reached    : {result['epoch']}")
print(f"  best_f1_val (macro)    : {result['best_f1_val']:.4f}")
print(f"  Last checkpoint        : {result['checkpoint_path']}")
print(f"  Best checkpoint        : {result['best_checkpoint_path']}")
print(f"  Run dir                : {run_dir}")
print("=" * 72)

# SQUAT-03-f acceptance: best.pt exists at ~360 MB scale.
assert os.path.exists(result["best_checkpoint_path"]), result["best_checkpoint_path"]
best_size_mb = os.path.getsize(result["best_checkpoint_path"]) / (1024 ** 2)
print(f"\nbest.pt size: {best_size_mb:.0f} MB")
assert 300 < best_size_mb < 500, f"best.pt size {best_size_mb:.0f} MB out of range"
assert result["epoch"] >= 5, f"Final epoch {result['epoch']} < 5; training did not converge enough"
assert result["best_f1_val"] > 0.5, f"best_f1_val={result['best_f1_val']:.4f} ≤ 0.5; suspect training divergence"

# Per-epoch summary table (for paste-back).
print("\nPer-epoch metrics history:")
print(f"  {'ep':>3s} {'tr_loss':>8s} {'vl_loss':>8s} {'f1_kie':>7s} {'f1_kfe':>7s} {'macro':>7s} {'time_s':>7s}")
for entry in result["metrics_history"]:
    print(
        f"  {entry['epoch']:>3d} {entry['train_loss_mean']:>8.4f} "
        f"{entry['val_loss_mean']:>8.4f} {entry['val_f1_kie']:>7.4f} "
        f"{entry['val_f1_kfe']:>7.4f} {entry['val_macro_f1']:>7.4f} "
        f"{entry['epoch_wall_time_s']:>7.1f}"
    )


# %% [markdown]
# ## Step 5 — threshold sweep on val (Task 12)
#
# Per-error threshold tuning via `sklearn.precision_recall_curve` (RESEARCH §5;
# replaces the linspace at D7). Pickles `best_thresholds = {"kie", "kfe"}` into
# `best.pt`. The sweep operates on val sigmoid scores from the loaded `best.pt`
# model and picks the F1-maximizing threshold independently per error head.
#
# **`latest.txt` preservation:** `atomic_save_checkpoint` always updates
# `latest.txt` to point at the basename of its target — that's correct during
# training (each epoch's checkpoint becomes the latest) but WRONG when we
# update `best.pt` (a separate fixed-name file). The sweep saves the current
# `latest.txt` content, writes the updated `best.pt`, then restores
# `latest.txt` so resume from `latest.txt` still lands on `epoch_011.pt` (or
# whichever epoch was last completed).

# %%
import os

import numpy as np
import torch

from backend.training.aqa.eval.metrics import f1_per_error, threshold_sweep
from backend.training.aqa.harness.colab import atomic_save_checkpoint
from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    _build_dataloaders,
    _val_pass,
    build_model,
)

config = SupervisedConfig()
RUN_NAME = "r2plus1d18_squat_supervised_v1"
run_dir = os.path.join(MYDRIVE, "FitNova/checkpoints/phase03", RUN_NAME)
best_path = os.path.join(run_dir, "best.pt")
latest_path = os.path.join(run_dir, "latest.txt")

# 1. Load best.pt (CPU — model.load_state_dict handles device transfer; same
#    map_location='cpu' rule as the resume fix at a0841b4).
print(f"Loading best.pt from {best_path}")
payload = torch.load(best_path, map_location="cpu", weights_only=False)
print(f"  best epoch              : {payload['epoch']}")
print(f"  best_f1_val (macro@0.5) : {payload['best_f1_val']:.4f}")
print(f"  config_hash             : {payload['config_hash']}")
print(f"  current best_thresholds : {payload.get('best_thresholds')}")

# 2. Rebuild model + val loader. Reuse the trainer's dataloader builder so
#    workers + worker_init_fn + persistent_workers stay consistent with training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(payload["model_state_dict"])
model.eval()

loaders = _build_dataloaders(
    seed=42, config=config, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
)
val_loader = loaders["val"]
val_ds = val_loader.dataset

# 3. Full val pass — gathers all 243 clips' sigmoid scores + labels on CPU.
#    The criterion is only used for the loss-mean sanity check (should match
#    the metrics_history value of 1.2708 at epoch 3 — confirms we loaded the
#    same weights that produced the best_f1_val).
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_ds.pos_weight.to(device))
val_loss_mean, val_scores, val_labels = _val_pass(model, val_loader, criterion, device)
print(f"\nVal pass: scores.shape={val_scores.shape}, labels.shape={val_labels.shape}")
print(f"  val_loss_mean: {val_loss_mean:.4f}  (epoch 3 trained value: 1.2708)")

# 4. Per-error threshold sweep via sklearn.precision_recall_curve.
print("\nThreshold sweep on val (per error, F1-maximizing via sklearn.precision_recall_curve):")
threshold_kie, f1_kie_sweep = threshold_sweep(val_labels[:, 0], val_scores[:, 0])
threshold_kfe, f1_kfe_sweep = threshold_sweep(val_labels[:, 1], val_scores[:, 1])

# Compare to threshold=0.5 baseline (the training-time proxy used for best.pt selection).
f1_kie_at_0p5 = f1_per_error(val_labels[:, 0], (val_scores[:, 0] >= 0.5).astype(int))
f1_kfe_at_0p5 = f1_per_error(val_labels[:, 1], (val_scores[:, 1] >= 0.5).astype(int))

print(f"  KIE  best_threshold = {threshold_kie:.4f}   F1 = {f1_kie_sweep:.4f}   "
      f"(vs F1@0.5 = {f1_kie_at_0p5:.4f}, delta = {f1_kie_sweep - f1_kie_at_0p5:+.4f})")
print(f"  KFE  best_threshold = {threshold_kfe:.4f}   F1 = {f1_kfe_sweep:.4f}   "
      f"(vs F1@0.5 = {f1_kfe_at_0p5:.4f}, delta = {f1_kfe_sweep - f1_kfe_at_0p5:+.4f})")

macro_sweep = (f1_kie_sweep + f1_kfe_sweep) / 2.0
macro_at_0p5 = (f1_kie_at_0p5 + f1_kfe_at_0p5) / 2.0
print(f"\n  Macro F1 @ swept thresholds : {macro_sweep:.4f}")
print(f"  Macro F1 @ threshold = 0.5  : {macro_at_0p5:.4f}  (matches best_f1_val = {payload['best_f1_val']:.4f})")
print(f"  Delta from threshold tuning : {macro_sweep - macro_at_0p5:+.4f}")

# 5. Save latest.txt BEFORE atomic_save_checkpoint mutates it.
with open(latest_path, "r", encoding="utf-8") as f:
    latest_before = f.read()
print(f"\nlatest.txt before sweep write: {latest_before.strip()!r}")

# 6. Update best.pt with best_thresholds (D13 contract — completes the payload).
best_thresholds = {"kie": threshold_kie, "kfe": threshold_kfe}
payload["best_thresholds"] = best_thresholds
print(f"\nWriting best_thresholds = {best_thresholds} into best.pt ...")
atomic_save_checkpoint(payload, best_path)

# 7. Restore latest.txt — resume must still land on the last completed epoch
#    checkpoint, NOT best.pt (which is a separate fixed-name artifact).
with open(latest_path, "w", encoding="utf-8") as f:
    f.write(latest_before)
print(f"latest.txt restored to       : {latest_before.strip()!r}")

# 8. Round-trip verify: re-read best.pt and confirm best_thresholds landed.
reloaded = torch.load(best_path, map_location="cpu", weights_only=False)
assert reloaded["best_thresholds"] == best_thresholds, reloaded["best_thresholds"]
print(f"\nRound-trip verified: best.pt['best_thresholds'] = {reloaded['best_thresholds']}")

# 9. Free GPU before Step 6 test pass.
del model
torch.cuda.empty_cache()
print("\nGPU freed; ready for Step 6 test evaluation.")


# %% [markdown]
# ## Step 6 — test evaluation (Task 13)
#
# Apply val-tuned per-error thresholds to the 244-clip test split. Compute F1 +
# PR-AUC + confusion matrices per error. Save `results.pkl` to `figures/`
# BEFORE any `plt.show()` per `[[feedback_notebook_disconnect_safe]]`. Also
# back up `results.pkl` to Drive (the figures/ tree lives on `/content/` which
# gets wiped on runtime restart; Drive backup survives).
#
# This produces the **headline Phase 3 numbers** — test F1 per error at
# val-tuned thresholds, comparable to the paper's Table 2 row.

# %%
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
)
from backend.training.aqa.harness.supervised_train import (
    SupervisedConfig,
    _build_dataloaders,
    _val_pass,
    build_model,
)

config = SupervisedConfig()
RUN_NAME = "r2plus1d18_squat_supervised_v1"
run_dir = os.path.join(MYDRIVE, "FitNova/checkpoints/phase03", RUN_NAME)
best_path = os.path.join(run_dir, "best.pt")

# 1. Load best.pt (with best_thresholds from Step 5).
payload = torch.load(best_path, map_location="cpu", weights_only=False)
best_thresholds = payload["best_thresholds"]
assert best_thresholds is not None, "Step 5 must run first to populate best_thresholds"
print(f"Loaded best.pt: epoch={payload['epoch']}, best_thresholds={best_thresholds}")

# 2. Rebuild model + loaders.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(payload["model_state_dict"])
model.eval()

loaders = _build_dataloaders(
    seed=42, config=config, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
)

# 3. Val pass at SWEPT thresholds — these are the "internal" val numbers used
#    for best.pt selection; refresh them here so results.pkl is self-contained.
val_loader = loaders["val"]
val_ds = val_loader.dataset
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_ds.pos_weight.to(device))
val_loss_mean, val_scores, val_labels = _val_pass(model, val_loader, criterion, device)
val_pred_kie = (val_scores[:, 0] >= best_thresholds["kie"]).astype(int)
val_pred_kfe = (val_scores[:, 1] >= best_thresholds["kfe"]).astype(int)
val_f1_kie = f1_per_error(val_labels[:, 0], val_pred_kie)
val_f1_kfe = f1_per_error(val_labels[:, 1], val_pred_kfe)
val_macro_f1 = (val_f1_kie + val_f1_kfe) / 2.0
print(f"\nVal @ swept thresholds: kie={val_f1_kie:.4f} kfe={val_f1_kfe:.4f} macro={val_macro_f1:.4f}")

# 4. TEST pass at val-tuned thresholds — the HEADLINE PHASE 3 NUMBERS.
test_loader = loaders["test"]
test_ds = test_loader.dataset
test_loss_mean, test_scores, test_labels = _val_pass(model, test_loader, criterion, device)
test_pred_kie = (test_scores[:, 0] >= best_thresholds["kie"]).astype(int)
test_pred_kfe = (test_scores[:, 1] >= best_thresholds["kfe"]).astype(int)

test_f1_kie = f1_per_error(test_labels[:, 0], test_pred_kie)
test_f1_kfe = f1_per_error(test_labels[:, 1], test_pred_kfe)
test_macro_f1 = (test_f1_kie + test_f1_kfe) / 2.0
test_pr_auc_kie = pr_auc_per_error(test_labels[:, 0], test_scores[:, 0])
test_pr_auc_kfe = pr_auc_per_error(test_labels[:, 1], test_scores[:, 1])
test_confusion_kie = confusion_matrix_per_error(test_labels[:, 0], test_pred_kie)
test_confusion_kfe = confusion_matrix_per_error(test_labels[:, 1], test_pred_kfe)

# 5. Clip IDs (in dataset order — matches scores/labels indices since loader
#    has shuffle=False). Needed by Step 7's sample-prediction grid.
test_clip_ids = [rec.clip_id for rec in test_ds.records]
val_clip_ids = [rec.clip_id for rec in val_ds.records]

# 6. Headline table.
print("\n" + "=" * 72)
print("PHASE 3 TEST RESULTS (val-tuned thresholds, 244 test clips)")
print("=" * 72)
print(f"  test_f1_kie     : {test_f1_kie:.4f}")
print(f"  test_f1_kfe     : {test_f1_kfe:.4f}")
print(f"  test_macro_f1   : {test_macro_f1:.4f}")
print(f"  test_pr_auc_kie : {test_pr_auc_kie:.4f}")
print(f"  test_pr_auc_kfe : {test_pr_auc_kfe:.4f}")
print(f"  test_loss_mean  : {test_loss_mean:.4f}")
print()
print(f"  KIE confusion matrix (rows=true [0,1], cols=pred [0,1]):")
print(f"    {test_confusion_kie[0].tolist()}")
print(f"    {test_confusion_kie[1].tolist()}")
print(f"  KFE confusion matrix:")
print(f"    {test_confusion_kfe[0].tolist()}")
print(f"    {test_confusion_kfe[1].tolist()}")
print()
print("Paper Kinetics row : kie=0.297, kfe=0.818, macro≈0.557")
print(f"  Δ vs us           : kie={test_f1_kie - 0.297:+.4f}, kfe={test_f1_kfe - 0.818:+.4f}, macro={test_macro_f1 - 0.557:+.4f}")
print()
print("Paper MD row       : kie=0.419, kfe=0.834, macro≈0.626")
print(f"  Δ vs us           : kie={test_f1_kie - 0.419:+.4f}, kfe={test_f1_kfe - 0.834:+.4f}, macro={test_macro_f1 - 0.626:+.4f}")
print("=" * 72)

# 7. Save results.pkl to figures/ BEFORE any plt.show (CONTEXT D11 / D9).
#    `Path(out).parent.mkdir(parents=True, exist_ok=True)` defensive.
FIG_DIR = Path("/content/fitnova/.planning/phases/03-squat-supervised-baseline/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
results_path = FIG_DIR / "results.pkl"

results = {
    # Val numbers at swept thresholds (matches Step 5 sweep output).
    "val_f1_kie": val_f1_kie,
    "val_f1_kfe": val_f1_kfe,
    "val_macro_f1": val_macro_f1,
    "val_thresholds": best_thresholds,
    "val_scores": val_scores,
    "val_labels": val_labels,
    "val_clip_ids": val_clip_ids,
    "val_loss_mean": val_loss_mean,
    # Test numbers at val-tuned thresholds (HEADLINE).
    "test_f1_kie": test_f1_kie,
    "test_f1_kfe": test_f1_kfe,
    "test_macro_f1": test_macro_f1,
    "test_pr_auc_kie": test_pr_auc_kie,
    "test_pr_auc_kfe": test_pr_auc_kfe,
    "test_confusion_kie": test_confusion_kie,
    "test_confusion_kfe": test_confusion_kfe,
    "test_scores": test_scores,
    "test_labels": test_labels,
    "test_clip_ids": test_clip_ids,
    "test_loss_mean": test_loss_mean,
    # Run provenance.
    "run_name": RUN_NAME,
    "best_epoch": payload["epoch"],
    "config_hash": payload["config_hash"],
    "config_repr": payload["config_repr"],
    "code_version": payload["code_version"],
    "metrics_history": payload["metrics_history"],
}

with results_path.open("wb") as f:
    pickle.dump(results, f)
print(f"\nresults.pkl saved to git tree: {results_path}")
print(f"  size: {results_path.stat().st_size / 1024:.1f} KB")

# 8. Drive backup — /content/ gets wiped on runtime restart; Drive survives.
import shutil
drive_results_path = os.path.join(run_dir, "results.pkl")
shutil.copy2(str(results_path), drive_results_path)
print(f"results.pkl backed up to Drive: {drive_results_path}")

# 9. Round-trip verify.
with results_path.open("rb") as f:
    reloaded = pickle.load(f)
assert reloaded["test_f1_kie"] == test_f1_kie
assert reloaded["test_macro_f1"] == test_macro_f1
assert reloaded["test_confusion_kie"].tolist() == test_confusion_kie.tolist()
print("results.pkl round-trip verified.")

# 10. Free GPU before Step 7 visualization (which also constructs the model
#     for the sample-prediction grid).
del model
torch.cuda.empty_cache()
print("\nGPU freed; ready for Step 7 visualization production.")


# %% [markdown]
# ## Step 7 — visualization production (Task 14)
#
# Produce all six supervisor figures (training_curves, confusion_{kie,kfe},
# pr_{kie,kfe}, sample_predictions_{kie,kfe}) per CONTEXT D11 / D14 +
# [[project_supervisor_visualizations]]. Save PNG to `figures/` BEFORE
# `plt.show()` so a Colab disconnect mid-display still leaves the file.
