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


# %% [markdown]
# ## Step 0 — env + dep probe + GPU/VRAM check + Drive mount + stage labeled & UNLABELED+traj
#
# **F8 import-order constraint:** the first executable line MUST be
# `from backend.training.aqa.harness import _envinit` (sets `CUBLAS_WORKSPACE_CONFIG`
# before any `import torch`). Do not reorder.
#
# Stages BOTH OHP sets from the **consolidated** Drive tree (your Drive has no
# -3-001/-3-002 split — videos.zip + bar_trajectories_raw.zip sit together under
# `OHP/Unlabeled_Dataset/`), so `stage_unlabeled_ohp_videos(MYDRIVE, MYDRIVE)` passes
# the same root for both args. One-time per-runtime cost: ~865 MB labeled + ~1765 MB
# unlabeled copied from Drive + extracted (a few minutes; resume-safe if it disconnects).

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import shutil
import subprocess
import sys

# Phase 2–5 carry-forward: PyAV before the first torch/torchvision import.
try:
    import av  # noqa: F401
    _pyav_status = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av  # noqa: F401
    _pyav_status = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav_status})")

# Dep version probe (supply-chain guard). Zero new pip installs beyond PyAV — scipy is
# Colab-provided / in requirements.txt (split_half_cycles uses scipy.ndimage; the probe
# uses scipy.signal.find_peaks).
_PHASE6_DEPS = ["torch", "torchvision", "scikit-learn", "matplotlib", "tqdm", "numpy", "scipy"]
print("\n[supply-chain] Dependency versions:")
for _pkg in _PHASE6_DEPS:
    try:
        print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"Phase 6 dependency '{_pkg}' is NOT installed. Phase 6 adds zero new pip "
            "dependencies beyond PyAV — every package was already available. If this "
            "fires, the Colab runtime is misconfigured."
        )

import torch
import torchvision

print("\npython     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

# GPU + VRAM check (CONTEXT D7 / [[feedback_heavy_training_new_notebook]]).
assert torch.cuda.is_available(), (
    "GPU required — Runtime > Change runtime type > Hardware accelerator: L4"
)
_gpu_name = torch.cuda.get_device_name(0)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"CUDA device : {_gpu_name} ({_vram_gb:.1f} GB VRAM)")
assert "L4" in _gpu_name or _vram_gb >= 22.0, (
    f"Phase 6 expects L4 (>=22 GB); got {_gpu_name} {_vram_gb:.1f} GB."
)

# Verify the PUBLIC torchvision.io decode API that decode_clip needs. tv 0.26 removed
# the internal `torchvision.io.video` submodule (the old PyAV-cache probe), so we check
# the public API instead. tv >= 0.27 removes read_video entirely (then switch decode_clip
# to the cv2 backend — opencv-python-headless is already a backend dep).
assert hasattr(torchvision.io, "read_video") and hasattr(torchvision.io, "read_video_timestamps"), (
    f"torchvision {torchvision.__version__} lacks read_video/read_video_timestamps — "
    "tv >= 0.27 removed them; pin torchvision <0.27 in Colab OR switch decode_clip to cv2."
)
print(f"torchvision.io decode API ready: read_video present (tv {torchvision.__version__}, PyAV {av.__version__})")

# Drive mount (idempotent).
from google.colab import drive
drive.mount('/content/drive')
MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
print("\nROOT :", ROOT, "| exists:", os.path.exists(ROOT))

from pathlib import Path
from backend.training.aqa.harness.colab import stage_ohp_videos, stage_unlabeled_ohp_videos

# Log the zip sizes + free /content disk BEFORE staging (~865 MB + ~1765 MB).
for _lbl, _rel in [
    ("labeled videos.zip", "OHP/Labeled_Dataset/videos.zip"),
    ("unlabeled videos.zip", "OHP/Unlabeled_Dataset/videos.zip"),
    ("trajectories.zip", "OHP/Unlabeled_Dataset/bar_trajectories_raw.zip"),
]:
    _z = Path(ROOT) / _rel
    if _z.is_file():
        print(f"[disk] {_lbl:<22}: {_z.stat().st_size / 1e9:.2f} GB")
print(f"[disk] free /content        : {shutil.disk_usage('/content').free / 1e9:.1f} GB")

# Stage LABELED videos (2367 mp4 — baseline train/val/test reads from here).
VIDEOS_ROOT = stage_ohp_videos(MYDRIVE)
_labeled_n = len(list(Path(VIDEOS_ROOT).glob('*.mp4')))
print(f"\nLABELED VIDEOS_ROOT: {VIDEOS_ROOT} | mp4 count: {_labeled_n} (expected 2367)")
assert _labeled_n == 2367, f"Expected 2367 labeled mp4s, got {_labeled_n}"

# Stage UNLABELED videos + barbell trajectories (5490 — the SSL pretrain set, Plan 03).
# Consolidated Drive: both release-folder roots collapse to MYDRIVE.
UNLABELED_VIDEOS_ROOT, TRAJ_ROOT = stage_unlabeled_ohp_videos(MYDRIVE, MYDRIVE)
_unlabeled_n = len(list(Path(UNLABELED_VIDEOS_ROOT).glob('*.mp4')))
print(f"\nUNLABELED_VIDEOS_ROOT: {UNLABELED_VIDEOS_ROOT} | mp4 count: {_unlabeled_n} (expected 5490)")
print(f"TRAJ_ROOT: {TRAJ_ROOT}")
assert _unlabeled_n == 5490, f"Expected 5490 unlabeled mp4s, got {_unlabeled_n}"


# %% [markdown]
# ## Step 1 — Trajectory-format + half-cycle-sign PROBE  ⚠ BLOCKING human-verify gate
#
# **STOP after this cell and paste back the report + `ohp_traj_probe.png` + the resolved sign.**
# This is the single highest-leverage sequencing decision (RESEARCH §3 / Open Question 1):
# a wrong argmax-vs-argmin sign or a wrong traj↔frame mapping silently wastes the 15–26 h
# SSL run in Plan 03.
#
# The probe is **exploratory — it does NOT assume the format**. It reports:
#   (1) the trajectory dir LAYOUT (flat vs nested under `bar_trajectories_raw/`) — this
#       decides whether `ohp_ssl.OHPSSLDataset` needs `glob` vs `rglob` (the Squat `len==0` trap);
#   (2) the live per-frame BBox structure of 2 sample files (`[region_0, region_1, region_2]`);
#   (3) y_center=(y1+y2)/2 from region 0 + the empty-region-0 frequency (NaN → interpolate);
#   (4) traj length vs `read_video_timestamps` frame count (the 1:1 mapping check);
#   (5) the half-cycle split overlaid for BOTH signs on sample clips + decoded frame grids;
#   (6) multi-rep frequency across all 5490 trajectories (`scipy.signal.find_peaks`).
#
# **Your job:** look at `ohp_traj_probe.png` — for each clip the split is drawn for both
# signs (argmin green, argmax red) with the two half-cycle frame strips. Reply with which
# sign (`bottom_is_argmax=False`=argmin or `True`=argmax) puts the split at the **overhead
# lockout** (barbell highest in frame = lowest y pixel) — the rep turning point. Expected:
# `bottom_is_argmax=False` (argmin). Also confirm the layout (flat/nested) + the mapping (1:1?).

# %%
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torchvision
from scipy.signal import find_peaks

from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset, split_half_cycles
from backend.training.aqa.datasets.transforms import decode_clip

FIG_DIR = Path(".planning/phases/06-overhead-press/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _ohp_y_center(data):
    """Extract the barbell y_center series from an OHP BBox trajectory JSON.

    Each frame = [region_0, region_1, region_2]; region_0 = list of 0/1 bboxes
    [x1,y1,x2,y2,conf]; region 0 = barbell. y_center=(y1+y2)/2. Empty region 0 -> NaN.
    Returns (y_raw_with_nan: np.ndarray, n_empty: int). Mirrors ohp_ssl._load_trajectory
    pre-interpolation so the probe REPORTS the empty frequency.
    """
    y = np.empty(len(data), dtype=float)
    n_empty = 0
    for i, frame in enumerate(data):
        region_0 = frame[0] if len(frame) > 0 else []
        if region_0:
            x1, y1, x2, y2, conf = region_0[0]
            y[i] = (y1 + y2) / 2.0
        else:
            y[i] = np.nan
            n_empty += 1
    return y, n_empty


def _interp_nan(y):
    nan = np.isnan(y)
    if nan.any():
        idx = np.arange(len(y))
        y = y.copy()
        y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y


# ── (1) Trajectory dir LAYOUT — flat vs nested (decides glob vs rglob) ─────────
all_traj = sorted(glob.glob(os.path.join(TRAJ_ROOT, "**", "*.json"), recursive=True))
flat_traj = sorted(glob.glob(os.path.join(TRAJ_ROOT, "*.json")))
print(f"[1 layout] {len(all_traj)} .json under {TRAJ_ROOT} (recursive)")
print(f"           {len(flat_traj)} .json FLAT at top level")
if all_traj:
    _rel = os.path.relpath(all_traj[0], TRAJ_ROOT)
    print(f"           sample rel path: {_rel}")
    _is_flat = (len(flat_traj) == len(all_traj))
    print(f"           LAYOUT = {'FLAT (glob ok)' if _is_flat else 'NESTED under a subdir -> ohp_ssl needs rglob OR trajectories_root=that subdir'}")
    _traj_dir_for_ds = TRAJ_ROOT if _is_flat else os.path.dirname(all_traj[0])
    print(f"           => OHPSSLDataset trajectories_root should be: {_traj_dir_for_ds}")

# ── (2) Live per-frame BBox structure of 2 sample files ────────────────────────
sample_files = all_traj[:5]
print(f"\n[2 BBox structure] inspecting {len(sample_files)} files:")
for p in sample_files[:2]:
    with open(p) as fh:
        data = json.load(fh)
    print(f"  {Path(p).name}: top-type={type(data).__name__}, n_frames={len(data)}")
    f0 = data[0]
    print(f"    frame[0] has {len(f0)} regions; region_0={f0[0] if len(f0) else '[]'}")
    # find a non-empty region-0 frame to show the bbox
    for fr in data:
        if len(fr) > 0 and fr[0]:
            print(f"    sample non-empty region_0[0] = {fr[0][0]}  -> y_center=(y1+y2)/2=({fr[0][0][1]}+{fr[0][0][3]})/2={(fr[0][0][1]+fr[0][0][3])/2:.1f}")
            break

# ── (3) y_center + empty-region-0 frequency across samples ─────────────────────
print(f"\n[3 y_center + empty freq]:")
parsed = []  # (clip_id, y_interp)
for p in sample_files:
    with open(p) as fh:
        data = json.load(fh)
    y_raw, n_empty = _ohp_y_center(data)
    y = _interp_nan(y_raw)
    parsed.append((Path(p).stem, y))
    print(f"  {Path(p).stem}: len={len(y)}, empty_region0={n_empty} ({100*n_empty/max(len(y),1):.0f}%), "
          f"y[:3]={np.round(y[:3],1).tolist()}, range=[{y.min():.0f},{y.max():.0f}]")

# Confirm the Plan-01 parser matches the inline extraction (bypass the glob via _traj_paths).
_ds = OHPSSLDataset.__new__(OHPSSLDataset)
_ds._traj_paths = {Path(p).stem: Path(p) for p in sample_files}
_cid0 = parsed[0][0]
_parser_y = _ds._load_trajectory(_cid0)
_match = np.allclose(_parser_y, parsed[0][1], equal_nan=False)
print(f"  ohp_ssl._load_trajectory({_cid0}) matches inline extraction: {_match} (len {len(_parser_y)})")

# ── (4) traj length vs video frame count (the 1:1 mapping check) ───────────────
print(f"\n[4 traj<->frame mapping]:")
for clip_id, y in parsed:
    vpath = os.path.join(UNLABELED_VIDEOS_ROOT, f"{clip_id}.mp4")
    if not os.path.isfile(vpath):
        print(f"  {clip_id}: video not found at {vpath}")
        continue
    try:
        pts, _fps = torchvision.io.read_video_timestamps(vpath, pts_unit="sec")
        n_frames = len(pts)
        ratio = n_frames / len(y) if len(y) else float("nan")
        print(f"  {clip_id}: traj_len={len(y)}  video_frames={n_frames}  frames/traj={ratio:.3f} "
              f"({'1:1' if abs(ratio - 1.0) < 0.05 else 'NOT 1:1 — Task 2 must rescale'})")
    except Exception as e:  # noqa: BLE001
        print(f"  {clip_id}: read_video_timestamps failed -> {e!r}")

# ── (5) SIGN figure: split overlay (both signs) + half-cycle frame grids ───────
n_show = min(3, sum(1 for _, y in parsed if len(y) >= 4))
fig = plt.figure(figsize=(16, 4 * max(n_show, 1)))
row = 0
for clip_id, y in parsed:
    if len(y) < 4 or row >= n_show:
        continue
    sm = scipy.ndimage.gaussian_filter1d(y, sigma=2.0)
    b_max, b_min = int(np.argmax(sm)), int(np.argmin(sm))

    ax = fig.add_subplot(n_show, 3, row * 3 + 1)
    ax.plot(y, color="0.6", lw=1, label="raw y_center")
    ax.plot(sm, color="C0", lw=2, label="smoothed")
    ax.axvline(b_min, color="C2", ls="-", lw=2, label=f"argmin @{b_min} (overhead?)")
    ax.axvline(b_max, color="C3", ls="--", label=f"argmax @{b_max}")
    ax.set_title(f"{clip_id} (len {len(y)})  — y grows DOWN, so overhead=min y")
    ax.legend(fontsize=7)
    ax.invert_yaxis()  # image coords: small y = top of frame = overhead

    vpath = os.path.join(UNLABELED_VIDEOS_ROOT, f"{clip_id}.mp4")
    try:
        # argmin split (the expected OHP sign): phase 1 = 0..overhead, phase 2 = overhead..end
        desc, asc = split_half_cycles(y, frames_per_half=8, bottom_is_argmax=False)
        for label, idxs, sub in [("phase1 0->argmin", desc, 2), ("phase2 argmin->end", asc, 3)]:
            clip = decode_clip(vpath, torch.as_tensor(np.asarray(idxs), dtype=torch.long))  # [T,3,H,W] uint8
            strip = torch.cat([clip[k] for k in range(min(8, clip.shape[0]))], dim=2)
            axg = fig.add_subplot(n_show, 3, row * 3 + sub)
            axg.imshow(strip.permute(1, 2, 0).numpy())
            axg.set_title(f"{label}  idx {np.asarray(idxs)[:4].tolist()}...", fontsize=8)
            axg.axis("off")
    except Exception as e:  # noqa: BLE001
        print(f"  [frame grid] {clip_id} failed: {e!r}")
    row += 1

fig.suptitle(
    "OHP trajectory probe — confirm argmin (green) lands at the OVERHEAD lockout (rep turning point)\n"
    "and that phase1 frames show the press going UP. Reply with the resolved sign.",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
_probe_png = FIG_DIR / "ohp_traj_probe.png"
fig.savefig(_probe_png, dpi=110, bbox_inches="tight")  # save BEFORE show (disconnect-safe)
print(f"\n[figure] saved {_probe_png}")
plt.show()

# ── (6) multi-rep frequency across ALL trajectories ────────────────────────────
multi_rep = 0
total_ok = 0
for p in all_traj:
    try:
        with open(p) as fh:
            y_raw, _ = _ohp_y_center(json.load(fh))
        if np.isnan(y_raw).all() or (~np.isnan(y_raw)).sum() < 4:
            continue
        y = _interp_nan(y_raw)
        sm = scipy.ndimage.gaussian_filter1d(y, sigma=2.0)
        rng = sm.max() - sm.min()
        if rng <= 0:
            continue
        peaks, _ = find_peaks(sm, prominence=0.3 * rng)
        valleys, _ = find_peaks(-sm, prominence=0.3 * rng)
        total_ok += 1
        if (len(peaks) + len(valleys)) > 1:
            multi_rep += 1
    except Exception:  # noqa: BLE001
        continue
print(f"\n[6 multi-rep] {multi_rep}/{total_ok} trajectories have >1 prominent extremum "
      f"({100 * multi_rep / max(total_ok, 1):.1f}% — single-rep argmin/argmax covers the rest)")

print("\n=== PROBE COMPLETE — paste back this report + ohp_traj_probe.png + the resolved sign. NO GPU yet. ===")
