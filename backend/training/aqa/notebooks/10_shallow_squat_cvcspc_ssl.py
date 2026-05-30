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


# %% [markdown]
# ## Probe (BLOCKING human-verify) — phase-matched triplet on real frames
#
# Instantiates the SSL dataset (with the val/test exclusion), confirms the post-exclusion clip count,
# checks _traj2phase spans ~0..360°, and plots one draw: the two clips' phase curves with the
# anchor/positive/negative phases marked + the decoded anchor/positive/negative frames. Confirm the
# anchor (v0) and positive (v1) are at a SIMILAR pose (same phase, different rep) and the negative
# (v1) is at a DIFFERENT pose.

# %%
import matplotlib.pyplot as plt
import numpy as np

from backend.training.aqa.datasets.cvcspc_ssl import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ShallowSquatSSLDataset,
)

ds = ShallowSquatSSLDataset(
    frames_root=FRAMES_ROOT, trajectories_root=TRAJ_ROOT,
    traj_nan_path=TRAJ_NAN_PATH, exclude_ids=EXCLUDE_IDS, ssl_contrastive_phase_gap=30.0,
)
print(f"SSL dataset clips after exclusion: {len(ds)}  (4970 - 179 holdout - degenerate)")

_p0 = ds._load_phase(ds._clip_ids[0])
print(f"sample phase span: {_p0.min():.1f}..{_p0.max():.1f} deg (n={len(_p0)})")

random.seed(0)
sel = ds._select_triplet(0)
v0, v1, p = sel["v0"], sel["v1"], sel["p_anchor"]
ai, pi, ni = sel["anchor_idx"], sel["positive_idx"], sel["negative_idx"]
ph0, ph1 = sel["phase0"], sel["phase1"]
print(f"\nanchor   clip {v0} @ phase {ph0[ai]:.1f}  (target {p:.1f})")
print(f"positive clip {v1} @ phase {ph1[pi]:.1f}")
print(f"negative clip {v1} @ phase {ph1[ni]:.1f}  (|gap|={abs(ph1[ni] - p):.1f} >= 30)")


def _denorm(t):
    m = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    s = np.array(IMAGENET_STD).reshape(3, 1, 1)
    return np.clip(t.numpy() * s + m, 0, 1).transpose(1, 2, 0)


a_img = _denorm(ds._load_frame(v0, ai))
p_img = _denorm(ds._load_frame(v1, pi))
n_img = _denorm(ds._load_frame(v1, ni))

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].plot(ph0, label=f"anchor clip {v0}")
axes[0].plot(ph1, label=f"pos/neg clip {v1}")
axes[0].axhline(p, color="k", ls="--", lw=0.8)
axes[0].scatter([ai], [ph0[ai]], c="g", s=70, zorder=5, label="anchor")
axes[0].scatter([pi], [ph1[pi]], c="b", s=70, zorder=5, label="positive")
axes[0].scatter([ni], [ph1[ni]], c="r", s=70, zorder=5, label="negative")
axes[0].set_title("bar-traj phase (deg)")
axes[0].set_xlabel("frame")
axes[0].legend(fontsize=7)
for ax, im, ttl, c in zip(
    axes[1:], [a_img, p_img, n_img],
    ["anchor (v0)", "positive (v1)", "negative (v1)"], ["g", "b", "r"],
):
    ax.imshow(im)
    ax.set_title(ttl, color=c)
    ax.axis("off")

_figdir = ".planning/phases/07-image-based-errors-cvcspc/figures"
os.makedirs(_figdir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{_figdir}/cvcspc_triplet_probe.png", dpi=110, bbox_inches="tight")
plt.show()
print(f"\nsaved {_figdir}/cvcspc_triplet_probe.png")


# %% [markdown]
# ## Step 1 + 2 — SSL dataset smoke + VRAM probe
#
# Builds the CVCSPC loader (reusing the probe's `ds`), confirms the triplet batch shapes
# (B,3,224,224) per branch, then builds the model + runs one 3-branch forward+backward to measure
# peak VRAM at batch 25, checks the projector output is unit-norm, and _triplet_accuracy returns [0,1].

# %%
from backend.training.aqa.datasets.cvcspc_ssl import build_cvcspc_loader
from backend.training.aqa.harness.cvcspc_pretrain import (
    CVCSPCConfig,
    _triplet_accuracy,
    build_cvcspc_model,
    cvcspc_triplet_loss,
)

_cfg = CVCSPCConfig()
print("CVCSPCConfig:", _cfg)

_loader = build_cvcspc_loader(ds, _cfg, seed=42)
_batch = next(iter(_loader))
for _k in ("anchor", "positive", "negative"):
    print(f"  {_k:9s} {tuple(_batch[_k].shape)} {_batch[_k].dtype}")
    assert _batch[_k].shape == (_cfg.batch_size, 3, 224, 224)

_device = torch.device("cuda")
_backbone, _projector = build_cvcspc_model(_cfg)
_backbone, _projector = _backbone.to(_device), _projector.to(_device)

torch.cuda.reset_peak_memory_stats()
_phi_a = _projector(_backbone(_batch["anchor"].to(_device)))
_phi_p = _projector(_backbone(_batch["positive"].to(_device)))
_phi_n = _projector(_backbone(_batch["negative"].to(_device)))
_loss = cvcspc_triplet_loss(_phi_a, _phi_p, _phi_n)
_loss.backward()
_peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\n3-branch fwd+bwd batch {_cfg.batch_size}: loss={_loss.item():.4f}, "
      f"peak VRAM={_peak_gb:.2f} GB / 23.7 GB")
_unit = torch.allclose(_phi_a.norm(dim=-1), torch.ones(_cfg.batch_size, device=_device), atol=1e-4)
print(f"projector out {tuple(_phi_a.shape)}, unit-norm={_unit}")

_ta = _triplet_accuracy(_backbone, _projector, [_batch], _device)
print(f"triplet-accuracy on the smoke batch: {_ta:.3f} (untrained -> ~0.5)")
assert 0.0 <= _ta <= 1.0

del _backbone, _projector, _loss
torch.cuda.empty_cache()


# %% [markdown]
# ## Step 3 — epoch-0 timing gate (BLOCKING decision)
#
# Runs ONE real SSL epoch under run_cvcspc_pretrain_epoch on the 4791 clips (a separate `_timing` dir,
# resume=False) to measure the per-epoch time + estimate the full 100-epoch run, sanity-check the
# ssl_loss + epoch-0 triplet-accuracy, and verify the checkpoint round-trip. epoch 0 is a triplet-acc
# cadence epoch, so its time is an UPPER bound (4/5 epochs skip the triplet-accuracy pass).

# %%
import os as _os
import time

from backend.training.aqa.harness.cvcspc_pretrain import run_cvcspc_pretrain_epoch

_t0 = time.perf_counter()
_res = run_cvcspc_pretrain_epoch(
    run_name="shallow_squat_cvcspc_v1_timing",
    drive_root=MYDRIVE,
    frames_root=FRAMES_ROOT, trajectories_root=TRAJ_ROOT,
    traj_nan_path=TRAJ_NAN_PATH, exclude_ids=EXCLUDE_IDS,
    seed=42, resume=False, max_epochs=1,
)
_wall = time.perf_counter() - _t0
_m0 = _res["metrics_history"][0]
_epoch_s = _m0["epoch_wall_time_s"]
print(f"\nepoch-0 time: {_epoch_s:.1f}s ({_epoch_s / 60:.1f} min)  |  cell wall incl setup: {_wall:.1f}s")
print(f"estimated full run (100 ep, UPPER bound): {_epoch_s * 100 / 3600:.2f} h")
print(f"epoch-0 ssl_loss={_m0['ssl_loss_mean']:.4f}")
if _res["triplet_acc_history"]:
    print(f"epoch-0 triplet_acc={_res['triplet_acc_history'][-1]['triplet_acc']:.4f}")

_ck = torch.load(_res["checkpoint_path"], map_location="cpu", weights_only=False)
print(f"\ncheckpoint: {_os.path.basename(_res['checkpoint_path'])}  code_version={_ck['code_version']}")
assert _ck["code_version"] == "phase07-cvcspc-pretrain"


# %% [markdown]
# ## Step 4 — full CVCSPC SSL pretrain (100 epochs, multi-session resume)
#
# The single cell that IS the whole run. Checkpoints every epoch + writes backbone.pt on
# triplet-accuracy improvement (update_latest=False) + anneals the phase-gap (a no-op at the default
# 30° start). resume=True -> a disconnect is recoverable (re-run Cell A -> Step 0 -> this cell; it
# resumes from latest.txt). The triplet-accuracy monitor logs every 5 epochs (the convergence signal).
# ~1.2-1.4h on L4 — backgroundable.

# %%
import logging

from backend.training.aqa.harness.cvcspc_pretrain import (
    CVCSPCConfig,
    run_cvcspc_pretrain_epoch,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
logging.getLogger("aqa.phase07").setLevel(logging.INFO)

RUN_NAME = "shallow_squat_cvcspc_v1"
_config = CVCSPCConfig()
_res = run_cvcspc_pretrain_epoch(
    run_name=RUN_NAME,
    drive_root=MYDRIVE,
    frames_root=FRAMES_ROOT, trajectories_root=TRAJ_ROOT,
    traj_nan_path=TRAJ_NAN_PATH, exclude_ids=EXCLUDE_IDS,
    seed=42, config=_config, resume=True, max_epochs=_config.max_epochs,
)
_best_ta = max((e["triplet_acc"] for e in _res["triplet_acc_history"]), default=float("nan"))
print(f"\nSSL done: final epoch {_res['epoch']}, best triplet_acc {_best_ta:.4f}")
print(f"backbone.pt: {_res['backbone_path']}")
print(f"final phase_gap: {_config.phase_gap_start}")


# %% [markdown]
# ## Step 5 — SSL convergence figure (defense visualization)
#
# Plots the SSL contrastive loss + the triplet-accuracy over the 100 epochs. Loaded from the
# `latest.txt`-pointed checkpoint (the full metrics_history — backbone.pt is frozen at the best epoch).

# %%
import matplotlib.pyplot as plt

_run_dir = f"{MYDRIVE}/FitNova/checkpoints/phase07/shallow_squat_cvcspc_v1"
with open(f"{_run_dir}/latest.txt") as _f:
    _latest = _f.read().strip()
_ck = torch.load(f"{_run_dir}/{_latest}", map_location="cpu", weights_only=False)
_mh, _tah = _ck["metrics_history"], _ck["triplet_acc_history"]

_ep = [m["epoch"] for m in _mh]
_loss = [m["ssl_loss_mean"] for m in _mh]
_ta_ep = [t["epoch"] for t in _tah]
_ta = [t["triplet_acc"] for t in _tah]
_best = max(_ta)
_best_ep = _ta_ep[_ta.index(_best)]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(_ep, _loss, color="tab:blue", lw=1.8)
ax1.set_xlabel("epoch")
ax1.set_ylabel("SSL contrastive loss (3-term)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(_ta_ep, _ta, color="tab:green", marker="o", lw=1.8, label="triplet acc (AP<AN)")
ax2.axhline(0.5, color="gray", ls=":", lw=1, label="chance 0.5")
ax2.scatter([_best_ep], [_best], color="red", s=80, zorder=5,
            label=f"best {_best:.3f} @ ep{_best_ep} -> backbone.pt")
ax2.set_ylabel("triplet accuracy", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.set_ylim(0.45, 1.0)
ax2.legend(loc="center right", fontsize=8)
ax1.set_title("CVCSPC SSL pretrain — Shallow-Squat backbone convergence (4791 unlabeled clips)")

_figdir = ".planning/phases/07-image-based-errors-cvcspc/figures"
os.makedirs(_figdir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{_figdir}/cvcspc_ssl_curves.png", dpi=130, bbox_inches="tight")
plt.show()
print(f"saved {_figdir}/cvcspc_ssl_curves.png  "
      f"(loss {_loss[0]:.3f}->{_loss[-1]:.3f}, triplet_acc {_ta[0]:.3f}->{_best:.3f})")
