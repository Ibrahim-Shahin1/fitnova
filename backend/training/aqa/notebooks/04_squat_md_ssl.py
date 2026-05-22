# Phase 4 — Squat Motion-Disentangling SSL (R(2+1)D-18, triplet distance-ratio pretext)
#
# Companion notebook to `.planning/phases/04-squat-motion-disentangling-ssl/04-02-PLAN.md`
# (and 04-03 / 04-04). Reference: `04-CONTEXT.md` (locked decisions D1–D9),
# `04-RESEARCH.md` (paper §3.2/§5 + the 8 resolved questions), `04-PATTERNS.md`,
# `04-VALIDATION.md` (Nyquist).
#
# Run in Colab on **L4 GPU** (24 GB VRAM). Mounts the Drive shortcut
# `My Drive/Fitness-AQA_dataset_release` (reused from Phase 2/3 — no new shortcut).
#
# **Step 1 (THE PROBE) is a BLOCKING human-verify gate** (Plan 02 Task 1): it confirms
# the unlabeled trajectory file format + the argmax-vs-argmin half-cycle SIGN on real
# clips BEFORE `squat_ssl._load_trajectory` is finalized (Task 2) and BEFORE any GPU
# pretraining. A wrong sign swaps descent/ascent and wastes the 12–24h run. Stop after
# Step 1 and paste back the probe report + traj_probe.png + the resolved sign.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Convert via
# `pip install jupytext && jupytext --to ipynb 04_squat_md_ssl.py`, or paste cells
# one at a time per the FitNova interactive working agreement.

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in a fresh Colab session.** Bakes the repo onto sys.path so
# `from backend...` resolves before Step 0 triggers the F8 `_envinit` import.
# Idempotent: clones if `/content/fitnova` is missing, else `git pull --ff-only`.
#
# NOTE ([[reference_colab_module_reload_after_git_pull]]): after a `git pull` that
# changes a trainer module already imported this session, `sys.modules` is STALE —
# restart the runtime OR `importlib.reload` the changed module before re-running its cell.

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
print(f"cwd: {os.getcwd()}")


# %% [markdown]
# ## Step 0 — env + dep probe + GPU/VRAM check + Drive mount + stage labeled & UNLABELED (Task 1)
#
# **F8 import-order constraint:** the first executable line MUST be
# `from backend.training.aqa.harness import _envinit` (sets `CUBLAS_WORKSPACE_CONFIG`
# before any `import torch`). Do not reorder.
#
# Phase 4 deltas vs Phase 3 Step 0:
# 1. Dep probe adds **scipy** (split_half_cycles uses `scipy.ndimage`; the probe uses
#    `scipy.signal.find_peaks`). Still zero new pip installs beyond PyAV (scipy is in
#    requirements.txt / Colab-provided).
# 2. `stage_unlabeled_squat_videos` — stages the 4,970 unlabeled clips + the barbell
#    trajectories (Plan 01). Logs the videos.zip size + free /content disk (T-04-08).

# %%
from backend.training.aqa.harness import _envinit  # F8: sets CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import shutil
import subprocess
import sys

# Phase 2/3 carry-forward: PyAV before the first torch/torchvision import.
try:
    import av  # noqa: F401
    _pyav_status = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av  # noqa: F401
    _pyav_status = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav_status})")

# Dep version probe (supply-chain guard) — Phase 4 adds scipy. Zero new pip installs.
_PHASE4_DEPS = ["torch", "torchvision", "scikit-learn", "matplotlib", "tqdm", "numpy", "scipy"]
print("\n[T-04-04] Dependency versions (supply-chain guard):")
for _pkg in _PHASE4_DEPS:
    try:
        print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"[T-04-04] Phase 4 dependency '{_pkg}' is NOT installed. Phase 4 introduces "
            "zero new pip dependencies beyond PyAV — every package was already in "
            "backend/requirements.txt. If this fires, the Colab runtime is misconfigured."
        )

import torch
import torchvision

print("\npython     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

# GPU + VRAM check (CONTEXT D2 / [[feedback_heavy_training_new_notebook]]).
assert torch.cuda.is_available(), (
    "GPU required — Runtime > Change runtime type > Hardware accelerator: L4"
)
_gpu_name = torch.cuda.get_device_name(0)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"CUDA device : {_gpu_name} ({_vram_gb:.1f} GB VRAM)")
assert "L4" in _gpu_name or _vram_gb >= 22.0, (
    f"Phase 4 expects L4 (>=22 GB); got {_gpu_name} {_vram_gb:.1f} GB. "
    "RESEARCH §5 VRAM sizing assumes 24 GB headroom."
)

# Verify torchvision.io PyAV backend (no real video needed).
from torchvision.io.video import av as _tv_av_ref
assert not isinstance(_tv_av_ref, Exception), (
    "torchvision.io cached `av` as an Exception — PyAV install ordered wrong."
)
print("torchvision.io PyAV backend: ready")

# Drive mount (idempotent).
from google.colab import drive
drive.mount('/content/drive')
MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')
print("\nROOT :", ROOT, "| exists:", os.path.exists(ROOT))

# Stage LABELED videos (1739 — reused for the linear-probe monitor in Plan 02 Wave 2).
from pathlib import Path
from backend.training.aqa.harness.colab import stage_squat_videos, stage_unlabeled_squat_videos

VIDEOS_ROOT = stage_squat_videos(MYDRIVE)
_labeled_n = len(list(Path(VIDEOS_ROOT).glob('*.mp4')))
print(f"\nLABELED VIDEOS_ROOT: {VIDEOS_ROOT} | mp4 count: {_labeled_n} (expected 1739)")
assert _labeled_n == 1739, f"Expected 1739 labeled mp4s, got {_labeled_n}"

# T-04-08: log the unlabeled videos.zip size + free /content disk BEFORE staging the
# 4,970-clip set (it is ~4x the labeled set — confirm there is room).
_unlabeled_zip = Path(ROOT) / "Squat/Unlabeled_Dataset/videos.zip"
if _unlabeled_zip.is_file():
    print(f"\n[T-04-08] unlabeled videos.zip: {_unlabeled_zip.stat().st_size / 1e9:.2f} GB")
_free_gb = shutil.disk_usage("/content").free / 1e9
print(f"[T-04-08] free /content disk: {_free_gb:.1f} GB")

# Stage UNLABELED videos + barbell trajectories (4,970 — the SSL pretrain set, Plan 01).
UNLABELED_VIDEOS_ROOT, TRAJ_ROOT = stage_unlabeled_squat_videos(MYDRIVE)
_unlabeled_n = len(list(Path(UNLABELED_VIDEOS_ROOT).glob('*.mp4')))
print(f"\nUNLABELED_VIDEOS_ROOT: {UNLABELED_VIDEOS_ROOT} | mp4 count: {_unlabeled_n} (expected 4970)")
print(f"TRAJ_ROOT: {TRAJ_ROOT}")
assert _unlabeled_n == 4970, f"Expected 4970 unlabeled mp4s, got {_unlabeled_n}"


# %% [markdown]
# ## Step 1 — Trajectory-format + half-cycle-sign PROBE  ⚠ BLOCKING human-verify gate (Task 1)
#
# **STOP after this cell and paste back the report + `traj_probe.png` + the resolved sign.**
# This is the single highest-leverage sequencing decision (RESEARCH Open Question 1):
# a wrong argmax-vs-argmin sign swaps descent/ascent and wastes the 12–24h SSL run.
#
# The probe is exploratory — it does NOT assume the format. It reports the 6 §8 items
# (archive structure / value type / raw-vs-smoothed / traj-length-vs-frame-count mapping /
# ID alignment / traj_nan), overlays `split_half_cycles` with BOTH sign values on 3 sample
# clips with decoded descent/ascent frame grids, and counts multi-rep frequency across all
# 4,970 trajectories via `scipy.signal.find_peaks`.
#
# **Your job:** look at `traj_probe.png` — for each sample clip the split point is drawn
# for both signs and the descent/ascent frames are shown. Reply with which sign
# (`bottom_is_argmax=True` or `False`) puts the split at the visual rep-bottom (deepest
# squat) AND makes the "descent" frames show the lifter going DOWN. Also confirm whether
# the trajectory format matches the [ASSUMED] per-clip JSON flat float list, or describe
# what it actually is.

# %%
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torchvision
from scipy.signal import find_peaks

from backend.training.aqa.datasets.squat_ssl import split_half_cycles
from backend.training.aqa.datasets.transforms import decode_clip

FIG_DIR = Path(".planning/phases/04-squat-motion-disentangling-ssl/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _extract_y_series(data):
    """Defensively pull a 1-D y-trajectory out of whatever the JSON holds.

    Returns (y_series: np.ndarray, how: str) so the probe REPORTS the path taken
    rather than silently assuming a format.
    """
    arr = np.asarray(data, dtype=object)
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return np.asarray(data, dtype=float), "flat list of floats (== [ASSUMED])"
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        cols = len(data[0])
        col = 1 if cols in (2, 4) else 0  # [x,y] or [x,y,w,h] -> y at index 1
        return np.asarray([row[col] for row in data], dtype=float), f"list of {cols}-tuples -> col {col}"
    if isinstance(data, dict):
        for key in ("y", "y_center", "cy", "centers", "trajectory", "traj"):
            if key in data:
                return _extract_y_series(data[key])[0], f"dict['{key}']"
        return np.asarray([], dtype=float), f"dict with keys {list(data.keys())[:8]} (UNHANDLED — describe below)"
    return np.asarray(arr, dtype=float).ravel(), "fallback ravel (UNHANDLED — describe below)"


# ── (1) Archive internal structure ───────────────────────────────────────────
all_traj = sorted(p for p in glob.glob(os.path.join(TRAJ_ROOT, "**", "*"), recursive=True) if os.path.isfile(p))
exts = {}
for p in all_traj:
    exts[Path(p).suffix.lower()] = exts.get(Path(p).suffix.lower(), 0) + 1
print(f"[§8.1 structure] {len(all_traj)} trajectory files under {TRAJ_ROOT}")
print(f"  extensions: {exts}")
print(f"  sample names: {[os.path.basename(p) for p in all_traj[:5]]}")

# ── (2)+(3) Value type + raw/smoothed (load up to 5) ──────────────────────────
json_files = [p for p in all_traj if p.lower().endswith(".json")]
sample_files = json_files[:5] if json_files else all_traj[:5]
print(f"\n[§8.2/3 value type] inspecting {len(sample_files)} files:")
parsed = []  # (clip_id, y_series, how)
for p in sample_files:
    try:
        with open(p) as fh:
            data = json.load(fh)
        y, how = _extract_y_series(data)
        parsed.append((Path(p).stem, y, how))
        print(f"  {Path(p).name}: top-type={type(data).__name__} -> y via {how}; "
              f"len={len(y)}, y[:3]={np.round(y[:3], 2).tolist()}, "
              f"range=[{y.min():.1f},{y.max():.1f}]" if len(y) else f"  {Path(p).name}: EMPTY/UNHANDLED ({how})")
    except Exception as e:  # noqa: BLE001
        print(f"  {Path(p).name}: NOT JSON or load error -> {e!r} (describe the real format below)")

# ── (5) ID alignment: do traj stems match clip IDs 1:1? ───────────────────────
traj_stems = {Path(p).stem for p in json_files} if json_files else {Path(p).stem for p in all_traj}
vid_stems = {p.stem for p in Path(UNLABELED_VIDEOS_ROOT).glob("*.mp4")}
print(f"\n[§8.5 ID alignment] traj stems={len(traj_stems)} vid stems={len(vid_stems)} "
      f"| intersect={len(traj_stems & vid_stems)} | traj-only={len(traj_stems - vid_stems)} "
      f"| vid-only={len(vid_stems - traj_stems)}")

# ── (6) traj_nan list present in the unlabeled set? ───────────────────────────
_nan_candidates = [p for p in all_traj if "nan" in os.path.basename(p).lower()]
print(f"[§8.6 traj_nan] files with 'nan' in name: {[os.path.basename(p) for p in _nan_candidates[:5]]} "
      f"(total {len(_nan_candidates)})")

# ── (4) traj length vs video frame count (mapping) for the sample clips ───────
import torchvision
print(f"\n[§8.4 traj->frame mapping] (traj_len vs video_frame_count per clip):")
for clip_id, y, _how in parsed:
    if len(y) == 0:
        continue
    vpath = os.path.join(UNLABELED_VIDEOS_ROOT, f"{clip_id}.mp4")
    if not os.path.isfile(vpath):
        print(f"  {clip_id}: video not found at {vpath}")
        continue
    try:
        pts, _fps = torchvision.io.read_video_timestamps(vpath, pts_unit="sec")
        n_frames = len(pts)
        ratio = n_frames / len(y) if len(y) else float("nan")
        print(f"  {clip_id}: traj_len={len(y)}  video_frames={n_frames}  frames/traj={ratio:.3f} "
              f"({'1:1' if abs(ratio - 1.0) < 0.05 else 'NOT 1:1 — Task 2 must rescale traj->frame'})")
    except Exception as e:  # noqa: BLE001
        print(f"  {clip_id}: read_video_timestamps failed -> {e!r}")

# ── SIGN DECISION figure: split overlay (both signs) + descent/ascent frame grids ──
n_show = min(3, sum(1 for _, y, _ in parsed if len(y) >= 4))
fig = plt.figure(figsize=(16, 4 * max(n_show, 1)))
row = 0
for clip_id, y, _how in parsed:
    if len(y) < 4:
        continue
    if row >= n_show:
        break
    sm = None
    try:
        import scipy.ndimage
        sm = scipy.ndimage.gaussian_filter1d(y, sigma=2.0)
    except Exception:
        sm = y
    b_max, b_min = int(np.argmax(sm)), int(np.argmin(sm))

    # Trajectory plot with BOTH candidate bottoms.
    ax = fig.add_subplot(n_show, 3, row * 3 + 1)
    ax.plot(y, color="0.6", lw=1, label="raw y")
    ax.plot(sm, color="C0", lw=2, label="smoothed")
    ax.axvline(b_max, color="C3", ls="--", label=f"argmax bottom @{b_max}")
    ax.axvline(b_min, color="C2", ls=":", label=f"argmin bottom @{b_min}")
    ax.set_title(f"{clip_id}  (len {len(y)})")
    ax.legend(fontsize=7)

    # Descent/ascent frame strips for the argmax sign (the [ASSUMED] default).
    vpath = os.path.join(UNLABELED_VIDEOS_ROOT, f"{clip_id}.mp4")
    try:
        desc, asc = split_half_cycles(y, frames_per_half=8, bottom_is_argmax=True)
        for j, (idxs, label, sub) in enumerate([(desc, "DESCENT (argmax)", 2), (asc, "ASCENT (argmax)", 3)]):
            clip = decode_clip(vpath, torch.as_tensor(np.asarray(idxs), dtype=torch.long))  # [T,3,H,W] uint8
            strip = torch.cat([clip[k] for k in range(min(8, clip.shape[0]))], dim=2)  # concat along W
            axg = fig.add_subplot(n_show, 3, row * 3 + sub)
            axg.imshow(strip.permute(1, 2, 0).numpy())
            axg.set_title(f"{label}  idx {np.asarray(idxs)[:4].tolist()}...", fontsize=8)
            axg.axis("off")
    except Exception as e:  # noqa: BLE001
        print(f"  [frame grid] {clip_id} failed: {e!r}")
    row += 1

fig.suptitle("Trajectory probe — confirm which sign puts the split at the rep-bottom (deepest squat)\n"
             "argmax (red --) vs argmin (green :) — and which makes DESCENT frames go DOWN", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
_probe_png = FIG_DIR / "traj_probe.png"
fig.savefig(_probe_png, dpi=110, bbox_inches="tight")  # save BEFORE show (disconnect-safe)
print(f"\n[figure] saved {_probe_png}")
plt.show()

# ── multi-rep frequency across ALL 4,970 trajectories (RESEARCH §1 pt.4) ──────
import scipy.ndimage
multi_rep = 0
total_ok = 0
for p in (json_files or all_traj):
    try:
        with open(p) as fh:
            y, _ = _extract_y_series(json.load(fh))
        if len(y) < 4:
            continue
        sm = scipy.ndimage.gaussian_filter1d(np.asarray(y, dtype=float), sigma=2.0)
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
print(f"\n[§1.4 multi-rep] {multi_rep}/{total_ok} trajectories have >1 prominent extremum "
      f"({100 * multi_rep / max(total_ok, 1):.1f}% — single-rep argmax/argmin covers the rest; "
      "Task 2 adds find_peaks splitting only if this fraction is large)")

print("\n=== PROBE COMPLETE — paste back this report + traj_probe.png + the resolved sign. NO GPU yet. ===")


# %% [markdown]
# ## Step 2 — SSL dataset smoke (triplet shapes + decoded-frame sanity grid) (Task 3)
#
# Confirms the triplet pipeline wires up on the staged unlabeled set BEFORE the trainer
# spins up. Instantiates `SquatSSLDataset` (finalized in Task 2 with the probe-confirmed
# argmin sign + NaN-interp), pulls one batch via `build_ssl_loader`, asserts the three
# `(B, 3, 16, 112, 112)` float32 shapes, and renders an 8-frame strip of anchor (descent)
# and negative (ascent) — a SECOND visual confirmation of the half-cycle split, now through
# the full real-data pipeline. **Re-run Cell A first (git pull)**; this cell `importlib.reload`s
# squat_ssl so the Task-2 changes take effect without a runtime restart.

# %%
import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch

import backend.training.aqa.datasets.squat_ssl as _squat_ssl
importlib.reload(_squat_ssl)  # pick up Task 2 changes after git pull (sys.modules is stale)
from backend.training.aqa.datasets.squat_ssl import SquatSSLDataset, build_ssl_loader
from backend.training.aqa.harness.md_pretrain import MDConfig

_FIG_DIR = Path(".planning/phases/04-squat-motion-disentangling-ssl/figures")
_FIG_DIR.mkdir(parents=True, exist_ok=True)

config = MDConfig()
ds = SquatSSLDataset(
    videos_root=UNLABELED_VIDEOS_ROOT,
    trajectories_root=TRAJ_ROOT,
    frames_per_half=config.frames_per_half,
    crop_size=config.crop_size,
)
print(f"len(ds) = {len(ds)} (expect ~4970)")

loader = build_ssl_loader(ds, config, seed=42)
batch = next(iter(loader))
_expected = (config.batch_size, 3, config.frames_per_half, config.crop_size, config.crop_size)
for _k in ("anchor", "positive", "negative"):
    _t = batch[_k]
    print(f"  {_k:>8}: shape={tuple(_t.shape)} dtype={_t.dtype}")
    assert tuple(_t.shape) == _expected, (_k, tuple(_t.shape), _expected)
    assert _t.dtype == torch.float32, (_k, _t.dtype)
print("triplet shapes ok")

# Sanity grid: decode the RAW descent/ascent for one clip (NO augs / NO temporal-reverse) so the
# half-cycle split is cleanly visible. The batch's anchor/negative are augmented AND randomly
# temporal-reversed (the §3 global-motion equalizer), which would obscure the down/up direction —
# so the shapes are asserted from the batch (above), but the SPLIT is shown from the raw frames.
from backend.training.aqa.datasets.squat_ssl import split_half_cycles
from backend.training.aqa.datasets.transforms import decode_clip

_cid = ds._clip_ids[0]
_traj = ds._load_trajectory(_cid)
_desc, _asc = split_half_cycles(_traj, frames_per_half=config.frames_per_half, bottom_is_argmax=False)
_vp = os.path.join(UNLABELED_VIDEOS_ROOT, f"{_cid}.mp4")
_desc_u8 = decode_clip(_vp, torch.as_tensor(_desc, dtype=torch.long))  # [16,3,H,W] uint8
_asc_u8 = decode_clip(_vp, torch.as_tensor(_asc, dtype=torch.long))


def _u8_strip(clip_tchw: torch.Tensor, n: int = 8) -> np.ndarray:
    """[T,3,H,W] uint8 -> [H, n*W, 3] uint8 (n evenly-spaced frames) for imshow."""
    sel = torch.linspace(0, clip_tchw.shape[0] - 1, n).round().long()
    return np.concatenate([clip_tchw[t].permute(1, 2, 0).numpy() for t in sel], axis=1)


fig, axes = plt.subplots(2, 1, figsize=(16, 5))
axes[0].imshow(_u8_strip(_desc_u8))
axes[0].set_title(f"{_cid} DESCENT (frames {_desc[:4].tolist()}...) — lifter should be going DOWN")
axes[0].axis("off")
axes[1].imshow(_u8_strip(_asc_u8))
axes[1].set_title(f"{_cid} ASCENT (frames {_asc[:4].tolist()}...) — lifter should be going UP")
axes[1].axis("off")
fig.suptitle("SSL triplet sanity — RAW descent/ascent (re-confirms argmin on real data; batch shapes asserted above)")
fig.tight_layout()
_sanity_png = _FIG_DIR / "ssl_triplet_sanity.png"
fig.savefig(_sanity_png, dpi=110, bbox_inches="tight")  # save BEFORE show (disconnect-safe)
print(f"saved {_sanity_png}")
plt.show()

print("\n=== STEP 2 COMPLETE — paste back len(ds), the 3 shapes, + ssl_triplet_sanity.png. Still NO GPU pretrain. ===")


# %% [markdown]
# ## Step 3 — VRAM probe: peak backward memory of the 3-branch triplet at batch 8 (Task 5)
#
# RESEARCH §5 estimates batch 8 ≈ 11.7 GB (extrapolated from Phase 3's MEASURED 15.22 GB) — but
# the Phase 3 estimate was off, so **measure, don't trust the table**. One forward+backward of the
# 3-branch triplet, then `max_memory_allocated`. If it exceeds ~20 GB / OOMs → option-b (batch 5).
# Re-run Cell A first (pulls Task 4's trainer + the squat_ssl recursive-glob fix).

# %%
import importlib
import torch

import backend.training.aqa.datasets.squat_ssl as _sslds
import backend.training.aqa.harness.md_pretrain as _mdp
importlib.reload(_sslds)  # pick up the recursive-glob fix after git pull
importlib.reload(_mdp)     # pick up the finalized trainer after git pull
from backend.training.aqa.datasets.squat_ssl import SquatSSLDataset, build_ssl_loader
from backend.training.aqa.harness.md_pretrain import MDConfig, build_md_model, md_triplet_loss

config = MDConfig()
_dev = torch.device("cuda")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

_bb, _proj = build_md_model()
_bb, _proj = _bb.to(_dev), _proj.to(_dev)
_ds = SquatSSLDataset(videos_root=UNLABELED_VIDEOS_ROOT, trajectories_root=TRAJ_ROOT,
                      frames_per_half=config.frames_per_half, crop_size=config.crop_size)
_b = next(iter(build_ssl_loader(_ds, config, seed=42)))
_opt = torch.optim.AdamW(list(_bb.parameters()) + list(_proj.parameters()),
                         lr=config.learning_rate, weight_decay=config.weight_decay)
_bb.train(); _proj.train()
_pa = _proj(_bb(_b["anchor"].to(_dev)))
_pp = _proj(_bb(_b["positive"].to(_dev)))
_pn = _proj(_bb(_b["negative"].to(_dev)))
_loss = md_triplet_loss(_pa, _pp, _pn, squared=config.loss_squared, three_term=config.loss_three_term)
_opt.zero_grad(); _loss.backward(); _opt.step()
_peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"batch_size={config.batch_size} | peak backward VRAM = {_peak_gb:.2f} GB | loss = {_loss.item():.4f}")
assert _peak_gb < 22.0, f"VRAM {_peak_gb:.1f} GB exceeds the L4 budget — choose option-b (batch 5)"
print(f"VRAM fits L4 (< 22 GB) at batch {config.batch_size}")
del _bb, _proj, _opt, _pa, _pp, _pn, _loss, _b
torch.cuda.empty_cache()


# %% [markdown]
# ## Step 4 — epoch-0 timing gate ⚠ BLOCKING DECISION (Task 5)
#
# Runs ONE fresh epoch under the production trainer (`max_epochs=1`, separate `_timing` run dir),
# then you authorize the full 12-24h run. `epoch_wall_time_s` is the pure SSL-pass time (measured
# before the linear-probe), so `estimated_total_h = ssl_epoch_time * max_epochs` is the training
# estimate; the linear-probe adds a smaller increment every 5 epochs (you'll see it in the cell's
# total time). **Decide:** option-a (proceed batch 8, est ≤ ~18h) / option-b (batch 5) / option-c
# (TorchCodec, only if GPU < 50% util = decode-bound).

# %%
import os
import time

import torch

from backend.training.aqa.harness.md_pretrain import MDConfig, run_md_pretrain_epoch

config = MDConfig()
_t_cell = time.perf_counter()
_res = run_md_pretrain_epoch(
    run_name="md_pretrain_v1_timing",
    drive_root=MYDRIVE,
    videos_root=UNLABELED_VIDEOS_ROOT,
    trajectories_root=TRAJ_ROOT,
    labeled_videos_root=VIDEOS_ROOT,
    seed=42, config=config, resume=False, max_epochs=1,
)
_cell_wall = time.perf_counter() - _t_cell
_e0 = _res["metrics_history"][0]
_ssl_t = _e0["epoch_wall_time_s"]
_est_h = _ssl_t * config.max_epochs / 3600.0
print(f"epoch-0 SSL pass = {_ssl_t:.1f}s | full cell (incl. linear-probe) = {_cell_wall:.1f}s")
print(f"ssl_loss = {_e0['ssl_loss_mean']:.4f} | embedding_std = {_e0['embedding_std']:.5f} "
      f"(healthy ~{1/512**0.5:.5f}; collapse -> 0) | effective_rank = {_e0['effective_rank']:.1f}")
print(f"estimated SSL training time (x{config.max_epochs} epochs) = {_est_h:.1f} h "
      f"(+ linear-probe every {config.linear_probe_cadence} epochs)")
if _res["linear_probe_history"]:
    _lp = _res["linear_probe_history"][0]
    print(f"epoch-0 linear-probe macro-F1 = {_lp['linear_probe_f1_macro']:.4f} "
          f"(kie={_lp['linear_probe_f1_kie']:.4f} kfe={_lp['linear_probe_f1_kfe']:.4f})")

# Checkpoint round-trip verify (T-04-06).
_ckpt = _res["checkpoint_path"]
print(f"\ncheckpoint: {_ckpt} ({os.path.getsize(_ckpt) / 1e6:.1f} MB)")
_pl = torch.load(_ckpt, map_location="cpu", weights_only=False)
assert _pl["code_version"] == "phase04-md-pretrain", _pl.get("code_version")
print("checkpoint round-trip OK; payload keys:", sorted(_pl.keys()))

print(f"\n=== DECISION: est_total≈{_est_h:.1f}h, VRAM fits. "
      "option-a (proceed batch 8) if ≤~18h; option-b (batch 5) if >24h or VRAM tight; "
      "option-c (TorchCodec) only if GPU<50% util (decode-bound). NO full run until you choose. ===")


# %% [markdown]
# ## Step 5 — full MD-SSL pretrain (multi-session, resume-safe) (Task 6)
#
# **This single cell IS the whole run** (batch 8, 60-epoch cosine, linear-probe every 5 — the
# user chose the full-extend cap). `resume=True`, so a Colab disconnect is recoverable: just
# re-run Step 0 (cache hit) + this cell — the first post-resume epoch logs `Resumed from
# epoch_NNN.pt`. The trainer checkpoints every epoch to Drive + writes `backbone.pt` on
# linear-probe-macro improvement (the fork point for Plan 03).
#
# **Paste back, staged:**
# - **~epoch 5** (≈1.5-3h in): SSL loss DECREASING (not flat/NaN), `embedding_std` above
#   0.1/√512≈0.0044 (no collapse, §12), first linear-probe macro above the random baseline.
#   If collapsed/flat → ABORT, raise SSL aug strength (add `translation`) or lower LR, restart.
# - **each session boundary**: the printed linear-probe + ssl_loss/emb_std/eff_rank trend — I'll
#   tell you when the linear-probe plateaus (the §6 convergence stop) so you can stop before 60.
# - **on resume**: confirm `Resumed from epoch_NNN.pt` appears (T-04-06).

# %%
import importlib

import backend.training.aqa.harness.md_pretrain as _mdp
importlib.reload(_mdp)  # pick up the finalized trainer + the collapse-metric fix after git pull
from backend.training.aqa.harness.md_pretrain import MDConfig, run_md_pretrain_epoch

config = MDConfig()  # batch 8, 60-epoch cosine, linear_probe_cadence=5
RUN_NAME = "md_pretrain_v1"
result = run_md_pretrain_epoch(
    run_name=RUN_NAME,
    drive_root=MYDRIVE,
    videos_root=UNLABELED_VIDEOS_ROOT,
    trajectories_root=TRAJ_ROOT,
    labeled_videos_root=VIDEOS_ROOT,
    seed=42, config=config, resume=True, max_epochs=config.max_epochs,
)

print(f"\nfinal epoch = {result['epoch']} | collapsed = {result['collapsed']}")
print(f"backbone.pt = {result['backbone_path']}")
print("\nlinear-probe history (watch macro rise then plateau — the §6 convergence stop):")
for _e in result["linear_probe_history"]:
    print(f"  epoch {_e['epoch']:>2}: macro={_e['linear_probe_f1_macro']:.4f} "
          f"(kie={_e['linear_probe_f1_kie']:.4f} kfe={_e['linear_probe_f1_kfe']:.4f})")
print("\nssl_loss + collapse trend (loss down; emb_std not -> 0; eff_rank not -> 1):")
for _m in result["metrics_history"]:
    print(f"  epoch {_m['epoch']:>2}: ssl_loss={_m['ssl_loss_mean']:.4f} "
          f"emb_std={_m['embedding_std']:.5f} eff_rank={_m['effective_rank']:.1f}")
