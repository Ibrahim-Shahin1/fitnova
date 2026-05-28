# Phase 6 — OHP MD-SSL Pretrain (R(2+1)D-18, Motion-Disentangling)
#
# Companion to .planning/phases/06-overhead-press/06-03-PLAN.md. The contribution: SSL
# pretrain on the 5,490 unlabeled OHP clips + barbell trajectories -> backbone.pt, which
# Plan 04 fine-tunes (3 seeds) + ensembles.
#
# Runs on L4 or A100. RESUMABLE across a runtime change or units running out: checkpoints
# every epoch to My Drive/FitNova/checkpoints/phase06/ohp_md_pretrain_v2/, resumes from
# latest.txt with map_location="cpu". config_hash is GPU-agnostic — a run can stop and
# resume on EITHER A100 or L4. Keep MDConfig (batch 8) FIXED across the switch so the
# config_hash stays stable; the A100 just runs the same recipe faster.
#
# Arc: Cell A -> Step 0 stage -> Step 1 VRAM probe -> Step 2 epoch-0 timing gate (BLOCKING)
#   -> Step 3 full pretrain (15-26h, multi-session, resume-safe).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)

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
    print(f"Repo at {REPO_DIR}; pulling latest from origin/{BRANCH} ...")
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
_head = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
print(f"\nRepo ready: {REPO_DIR} @ {_head}")


# %% [markdown]
# ## Step 0 — env + GPU check + Drive mount + stage labeled & unlabeled+traj
#
# Stages the labeled set (2367, for the linear-probe) and the unlabeled set (5490 + the
# bar trajectories). One-time per-runtime cost (~10-20 min on a fresh runtime); resume-safe.

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import shutil
import subprocess
import sys

try:
    import av; _pyav_status = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True); import av; _pyav_status = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav_status})")

_DEPS = ["torch", "torchvision", "scikit-learn", "matplotlib", "tqdm", "numpy", "scipy"]
print("\n[supply-chain] Dependency versions:")
for _pkg in _DEPS:
    print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")

import torch
import torchvision
print("\ntorch:", torch.__version__, "| torchvision:", torchvision.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU required — Runtime > Change runtime type > L4 or A100"
_gpu = torch.cuda.get_device_name(0); _vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"CUDA device : {_gpu} ({_vram:.1f} GB VRAM)")

_has_rv = hasattr(torchvision.io, "read_video") and hasattr(torchvision.io, "read_video_timestamps")
if not _has_rv:
    import cv2
    print(f"decode backend: cv2 fallback (tv {torchvision.__version__} removed read_video; OpenCV {cv2.__version__})")
else:
    print(f"decode backend: torchvision.read_video (tv {torchvision.__version__})")

from google.colab import drive
drive.mount('/content/drive')
MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
print("\nROOT :", ROOT, "| exists:", os.path.exists(ROOT))

from pathlib import Path
from backend.training.aqa.harness.colab import stage_ohp_videos, stage_unlabeled_ohp_videos

_uz = Path(ROOT) / "OHP/Unlabeled_Dataset/videos.zip"
if _uz.is_file(): print(f"[disk] unlabeled videos.zip: {_uz.stat().st_size / 1e9:.2f} GB")
print(f"[disk] free /content: {shutil.disk_usage('/content').free / 1e9:.1f} GB")

VIDEOS_ROOT = stage_ohp_videos(MYDRIVE)
print(f"LABELED VIDEOS_ROOT: {VIDEOS_ROOT} | mp4: {len(list(Path(VIDEOS_ROOT).glob('*.mp4')))} (expect 2367)")
assert len(list(Path(VIDEOS_ROOT).glob('*.mp4'))) == 2367

UNLABELED_VIDEOS_ROOT, TRAJ_ROOT = stage_unlabeled_ohp_videos(MYDRIVE, MYDRIVE)
print(f"UNLABELED_VIDEOS_ROOT: {UNLABELED_VIDEOS_ROOT} | mp4: {len(list(Path(UNLABELED_VIDEOS_ROOT).glob('*.mp4')))} (expect 5490)")
print(f"TRAJ_ROOT: {TRAJ_ROOT}")
assert len(list(Path(UNLABELED_VIDEOS_ROOT).glob('*.mp4'))) == 5490


# %% [markdown]
# ## Step 1 — VRAM probe: peak backward memory of the 3-branch triplet at batch 8
#
# The < 22 GB assert is the L4 budget — keeps the run resumable on L4 even when started on
# A100. On A100 this passes with large headroom.

# %%
import importlib

import backend.training.aqa.datasets.ohp_ssl as _sslds
import backend.training.aqa.harness.md_pretrain as _mdp
importlib.reload(_sslds)
importlib.reload(_mdp)
from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset, build_ssl_loader
from backend.training.aqa.harness.md_pretrain import MDConfig, build_md_model, md_triplet_loss

config = MDConfig()
_dev = torch.device("cuda")
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

_bb, _proj = build_md_model()
_bb, _proj = _bb.to(_dev), _proj.to(_dev)
_ds = OHPSSLDataset(videos_root=UNLABELED_VIDEOS_ROOT, trajectories_root=TRAJ_ROOT,
                    frames_per_half=config.frames_per_half, crop_size=config.crop_size)
print(f"OHPSSLDataset clips: {len(_ds)} (expect ~5490)")
_b = next(iter(build_ssl_loader(_ds, config, seed=42)))
_opt = torch.optim.AdamW(list(_bb.parameters()) + list(_proj.parameters()),
                         lr=config.learning_rate, weight_decay=config.weight_decay)
_bb.train(); _proj.train()
_pa = _proj(_bb(_b["anchor"].to(_dev)))
_pp = _proj(_bb(_b["positive"].to(_dev)))
_pn = _proj(_bb(_b["negative"].to(_dev)))
_loss = md_triplet_loss(_pa, _pp, _pn, squared=config.loss_squared, three_term=config.loss_three_term)
_opt.zero_grad(); _loss.backward(); _opt.step()
_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"batch_size={config.batch_size} | peak backward VRAM = {_peak:.2f} GB | loss = {_loss.item():.4f}")
assert _peak < 22.0, f"VRAM {_peak:.1f} GB exceeds the L4 budget — choose option-b (batch 5)"
print(f"VRAM fits L4 (< 22 GB) at batch {config.batch_size}")
del _bb, _proj, _opt, _pa, _pp, _pn, _loss, _b
torch.cuda.empty_cache()


# %% [markdown]
# ## Step 2 — epoch-0 timing gate ⚠ BLOCKING DECISION
#
# One fresh epoch under the production trainer with the OHP seams
# (ssl_dataset_cls=OHPSSLDataset, probe_dataset_cls=OHPElbowsKneesDataset,
# checkpoint_phase="phase06"). Confirms: epoch time, no collapse, and the linear-probe runs
# on OHP labels (NOT Squat). Decide option-a (proceed batch 8) / option-b (batch 5).
# Note: linear_probe_f1_kie/kfe key names are inherited from Phase 4 — for OHP they hold
# Elbows (index 0) / Knees (index 1).

# %%
import os
import time

from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset
from backend.training.aqa.harness.md_pretrain import MDConfig, run_md_pretrain_epoch

config = MDConfig()
_t = time.perf_counter()
res = run_md_pretrain_epoch(
    run_name="ohp_md_pretrain_v2_timing",
    drive_root=MYDRIVE,
    videos_root=UNLABELED_VIDEOS_ROOT,
    trajectories_root=TRAJ_ROOT,
    labeled_videos_root=VIDEOS_ROOT,
    seed=42, config=config, resume=False, max_epochs=1,
    ssl_dataset_cls=OHPSSLDataset,
    probe_dataset_cls=OHPElbowsKneesDataset,
    checkpoint_phase="phase06",
)
cell_wall = time.perf_counter() - _t
e0 = res["metrics_history"][0]
ssl_t = e0["epoch_wall_time_s"]
estimated_total_h = ssl_t * config.max_epochs / 3600.0
print(f"epoch-0 SSL pass = {ssl_t:.1f}s | full cell (incl. linear-probe) = {cell_wall:.1f}s")
print(f"ssl_loss = {e0['ssl_loss_mean']:.4f} | embedding_std = {e0['embedding_std']:.5f} (collapse -> 0) | effective_rank = {e0['effective_rank']:.1f}")
print(f"estimated SSL training time (x{config.max_epochs} epochs) = {estimated_total_h:.1f} h")
if res.get("linear_probe_history"):
    lp = res["linear_probe_history"][0]
    print(f"epoch-0 linear-probe macro-F1 = {lp['linear_probe_f1_macro']:.4f} "
          f"(Elbows={lp.get('linear_probe_f1_kie')} Knees={lp.get('linear_probe_f1_kfe')}) -- OHP labels, confirms probe_dataset_cls")

ckpt = res["checkpoint_path"]
print(f"\ncheckpoint: {ckpt} ({os.path.getsize(ckpt) / 1e6:.1f} MB)")
pl = torch.load(ckpt, map_location="cpu", weights_only=False)
assert pl["code_version"] == "phase04-md-pretrain", pl.get("code_version")
assert "phase06" in ckpt, f"checkpoint not under phase06: {ckpt}"
print("checkpoint round-trip OK; under phase06; payload keys:", sorted(pl.keys()))
print(f"\n=== DECISION: est_total≈{estimated_total_h:.1f}h, VRAM fits. option-a (batch 8) if reasonable; option-b (batch 5) if VRAM tight. NO full run until you choose. ===")
