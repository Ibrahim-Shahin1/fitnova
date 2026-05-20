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


# %% [markdown]
# ## Step 2 — model construction smoke (Task 9)
#
# Build R(2+1)D-18 + `Linear(512, 2)` head, run one forward + backward pass on a
# real batch, measure peak VRAM. Confirms L4 headroom before Step 4 commits to
# multi-hour training.


# %% [markdown]
# ## Step 3 — epoch 0 timing probe (Task 10) — **BLOCKING gate before full training**
#
# One-epoch fresh run via `run_supervised_epoch(max_epochs=1, resume=False,
# run_name="..._timing")`. Measures wall time + writes a ~360 MB checkpoint to
# Drive + round-trips `torch.load`. Paste back `epoch_0_time_s` and
# `estimated_total_h = epoch_0_time_s * 50 / 3600`. Pick option-a / -b / -c per
# Task 10 of the PLAN.


# %% [markdown]
# ## Step 4 — full supervised training run (Task 11)
#
# Run `run_supervised_epoch(resume=True, max_epochs=config.max_epochs)` with the
# config authorised in Step 3. Per-epoch tqdm + per-epoch metric log lines +
# atomic checkpoint per epoch + `best.pt` on val-macro-F1 improvement (D13).
# Disconnect during? Re-run Step 0 + skip Steps 1–3 + re-run Step 4 — resume
# from `latest.txt`.


# %% [markdown]
# ## Step 5 — threshold sweep on val (Task 12)
#
# Per-error threshold tuning via `sklearn.precision_recall_curve` (RESEARCH §5;
# replaces the linspace at D7). Pickles `best_thresholds = {"kie", "kfe"}` into
# `best.pt`.


# %% [markdown]
# ## Step 6 — test evaluation (Task 13)
#
# Apply val-tuned per-error thresholds to the 244-clip test split. Compute F1 +
# PR-AUC + confusion matrices per error. Save `results.pkl` to `figures/`
# BEFORE any `plt.show()` per `[[feedback_notebook_disconnect_safe]]`.


# %% [markdown]
# ## Step 7 — visualization production (Task 14)
#
# Produce all six supervisor figures (training_curves, confusion_{kie,kfe},
# pr_{kie,kfe}, sample_predictions_{kie,kfe}) per CONTEXT D11 / D14 +
# [[project_supervisor_visualizations]]. Save PNG to `figures/` BEFORE
# `plt.show()` so a Colab disconnect mid-display still leaves the file.
