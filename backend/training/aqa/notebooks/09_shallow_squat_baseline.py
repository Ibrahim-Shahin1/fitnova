# Phase 7 — Shallow-Squat Supervised Baseline (ImageNet ResNet-18, IMAGENET1K_V1 init)
#
# Companion notebook to `.planning/phases/07-image-based-errors-cvcspc/07-02-PLAN.md`.
#
# Run in Colab on an L4 GPU (a T4 is acceptable — ResNet-18 on ~3.7k crops is light).
# Mounts the Drive shortcut holding `Fitness-AQA_dataset_release`. The Shallow-Squat data
# lives under the -3-001 release folder:
#   Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/
#     images.zip (3738 crops_unaligned/{id}.jpg) + labels_shallow_depth.json + splits/
#
# Plan 02 arc: Cell A bootstrap -> Step 0 stage (images.zip + labels + splits)
#   -> Step 1 split reconciliation (2542/529/540) + dataset smoke
#   -> Step 2 model + VRAM probe (single-logit head) -> Step 3 epoch-0 timing gate (BLOCKING)
#   -> Step 4 multi-seed baseline training -> Step 5 ensemble + threshold sweep -> Step 6 test eval.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell. Cells are delivered one at a
# time in chat per the FitNova interactive working agreement (paste-back gating).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# **Run this FIRST in the fresh Colab session.** Pulls `fresh-start` so Colab gets the Phase-7
# Plan-01 modules (`datasets/shallow_squat.py`, `datasets/cvcspc_ssl.py`,
# `harness/image_supervised_train.py`, `harness/colab.py` with `stage_shallow_squat_images`).
# Then `os.chdir` + `sys.path.insert` so `from backend.training.aqa...` resolves.
#
# Idempotent: clones if `/content/fitnova` is missing, otherwise fetch + checkout + ff-only pull.

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

for _m in [
    "backend/training/aqa/datasets/shallow_squat.py",
    "backend/training/aqa/datasets/cvcspc_ssl.py",
    "backend/training/aqa/harness/image_supervised_train.py",
    "backend/training/aqa/harness/cvcspc_pretrain.py",
]:
    print(f"  {'OK ' if os.path.isfile(_m) else 'MISSING'} {_m}")


# %% [markdown]
# ## Step 0 — env + dep probe + GPU check + Drive mount + stage Shallow-Squat images
#
# **F8 import-order constraint:** the first executable line MUST be
# `from backend.training.aqa.harness import _envinit` (sets `CUBLAS_WORKSPACE_CONFIG`
# before any `import torch`). Do not reorder.
#
# No PyAV — crops load via PIL (no video decode). Stages `images.zip` (3738 crops) +
# `labels_shallow_depth.json` + `splits/` from the consolidated Drive root to
# `/content/squat_shallow_images/` (~100 MB copy + extract; resume-safe on disconnect).

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS_WORKSPACE_CONFIG before torch

import importlib.metadata
import os
import shutil
import sys

for _pkg in ["torch", "torchvision", "scikit-learn", "numpy", "tqdm"]:
    print(f"  {_pkg:<14} {importlib.metadata.version(_pkg)}")

import torch
import torchvision

print("\npython     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

assert torch.cuda.is_available(), (
    "GPU required — Runtime > Change runtime type > Hardware accelerator: L4 (T4 also fine)"
)
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

from backend.training.aqa.harness.colab import mount_drive, stage_shallow_squat_images

MYDRIVE = mount_drive()
DRIVE_ROOT_3001 = MYDRIVE  # consolidated Drive root (no -3-001 split on Drive)

_expected = os.path.join(
    DRIVE_ROOT_3001,
    "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images.zip",
)
print(f"\nimages.zip on Drive: {'FOUND' if os.path.isfile(_expected) else 'NOT FOUND'}\n  {_expected}")

SHALLOW_ROOT = stage_shallow_squat_images(DRIVE_ROOT_3001, local_root="/content/squat_shallow_images")
IMAGES_ROOT = os.path.join(SHALLOW_ROOT, "crops_unaligned")
LABELS_PATH = os.path.join(SHALLOW_ROOT, "labels_shallow_depth.json")
SPLITS_ROOT = os.path.join(SHALLOW_ROOT, "splits")

_n_jpg = sum(1 for _r, _d, _fs in os.walk(IMAGES_ROOT) for _f in _fs if _f.endswith(".jpg"))
print(f"\nstaged: {_n_jpg} jpgs at {IMAGES_ROOT}")
print(f"labels: {'OK' if os.path.isfile(LABELS_PATH) else 'MISSING'}   "
      f"splits: {'OK' if os.path.isdir(SPLITS_ROOT) else 'MISSING'}")
print(f"free disk on /content: {shutil.disk_usage('/content').free / 1e9:.1f} GB")


# %% [markdown]
# ## Step 1 — split reconciliation (2542/529/540) + dataset smoke
#
# Builds the loaders against the staged crops, reconciles the official split sizes live, checks
# the single-head `pos_weight` (~1.28, shape (1,)), pulls one train batch (shape (B,3,224,224) +
# label batch (B,) — single binary head, NOT (B,2)), and shows a few denormalized crops so you
# can eyeball that the images load correctly.

# %%
import torch
import matplotlib.pyplot as plt

from backend.training.aqa.datasets.shallow_squat import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_loaders,
)

loaders = build_loaders(
    images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
    batch_size=32, num_workers=2,
)

_EXPECT = {"train": 2542, "val": 529, "test": 540}
for _split, _exp in _EXPECT.items():
    _ds = loaders[_split].dataset
    _pos = sum(_lbl for _, _lbl in _ds.records)
    print(f"{_split:5s}: {len(_ds):4d} crops (expected {_exp})  pos={_pos} "
          f"({100 * _pos / len(_ds):.1f}%)  {'OK' if len(_ds) == _exp else 'MISMATCH'}")

_pw = loaders["train"].dataset.pos_weight
print(f"\npos_weight {tuple(_pw.shape)}: {float(_pw[0]):.4f}")
assert tuple(_pw.shape) == (1,), "pos_weight must be single-head (1,)"

_imgs, _labels = next(iter(loaders["train"]))
print(f"train batch: imgs {tuple(_imgs.shape)} {_imgs.dtype}, "
      f"labels {tuple(_labels.shape)} {_labels.dtype}, "
      f"label values {sorted(set(int(_x) for _x in _labels.tolist()))}")
assert _imgs.shape[1:] == (3, 224, 224) and _imgs.dtype == torch.float32
assert _labels.ndim == 1, f"label batch must be (B,) single-head, got {tuple(_labels.shape)}"

_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for _i, _ax in enumerate(axes):
    _crop = (_imgs[_i] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
    _ax.imshow(_crop)
    _ax.set_title(f"y={int(_labels[_i])}")
    _ax.axis("off")
plt.suptitle("Shallow-Squat train crops (denormalized) — y=1 is shallow-depth error")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Step 2 — model + VRAM probe (single-logit head assert)
#
# Builds the ImageNet ResNet-18 + `Linear(512,1)` head, asserts the single-logit head shape
# (zeros[2,3,224,224] -> [2,1]), and runs one forward+backward at batch 32 to measure peak VRAM.
# First call downloads the IMAGENET1K_V1 weights (~45 MB).

# %%
import torch.nn as nn

from backend.training.aqa.harness.image_supervised_train import build_resnet18

_device = torch.device("cuda")
_model = build_resnet18().to(_device)
assert isinstance(_model.fc, nn.Linear) and _model.fc.out_features == 1, "head must be Linear(512, 1)"
with torch.no_grad():
    _probe = _model(torch.zeros(2, 3, 224, 224, device=_device))
print("head:", _model.fc, "| forward zeros[2,3,224,224] ->", tuple(_probe.shape))
assert tuple(_probe.shape) == (2, 1), f"expected [2,1], got {tuple(_probe.shape)}"

torch.cuda.reset_peak_memory_stats()
_imgs, _labels = next(iter(loaders["train"]))
_target = _labels.view(-1, 1).to(_device)
_crit = nn.BCEWithLogitsLoss(pos_weight=_pw.to(_device))
_logits = _model(_imgs.to(_device))
_loss = _crit(_logits, _target)
_loss.backward()
_peak_gb = torch.cuda.max_memory_allocated() / 1e9
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"forward+backward batch {_imgs.shape[0]}: loss={_loss.item():.4f}, "
      f"peak VRAM={_peak_gb:.2f} GB / {_total_gb:.1f} GB")
assert _peak_gb < _total_gb, "VRAM exceeded — drop batch_size"

del _model, _logits, _loss
torch.cuda.empty_cache()


# %% [markdown]
# ## Step 3 — epoch-0 timing gate (BLOCKING)
#
# Runs ONE real epoch under the production trainer on the 2542 train crops (a separate `_timing`
# run dir, resume=False) to measure minutes/epoch + estimate the full run, sanity-check the
# epoch-0 train loss / val F1, and verify the checkpoint round-trip. This authorizes the
# multi-seed run.

# %%
import os as _os
import time

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.harness.image_supervised_train import ImageConfig, run_image_epoch

_cfg = ImageConfig()
print("ImageConfig:", _cfg)

_t0 = time.perf_counter()
_res = run_image_epoch(
    run_name="shallow_squat_baseline_seed42_timing",
    drive_root=MYDRIVE,
    images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
    seed=42, config=_cfg, resume=False, max_epochs=1,
    dataset_cls=ShallowSquatDataset,
)
_wall = time.perf_counter() - _t0
_m0 = _res["metrics_history"][0]
_epoch_s = _m0["epoch_wall_time_s"]
print(f"\nepoch-0 time: {_epoch_s:.1f}s ({_epoch_s / 60:.1f} min)  |  cell wall incl setup: {_wall:.1f}s")
print(f"estimated full run ({_cfg.max_epochs} ep): {_epoch_s * _cfg.max_epochs / 3600:.2f} h")
print(f"epoch-0 train_loss={_m0['train_loss_mean']:.4f}  "
      f"val_f1={_m0['val_f1']:.4f}  val_pr_auc={_m0['val_pr_auc']:.4f}")

_ck = torch.load(_res["checkpoint_path"], map_location="cpu", weights_only=False)
print(f"\ncheckpoint: {_os.path.basename(_res['checkpoint_path'])}  "
      f"code_version={_ck['code_version']}")
print("payload keys:", sorted(_ck.keys()))
assert _ck["code_version"] == "phase07-image-baseline"


# %% [markdown]
# ## Step 4 — multi-seed baseline training (seed 42 first)
#
# Trains seed 42 to convergence (resume=True -> disconnect-safe; just re-run the cell to resume
# from latest.txt). ~5 min on L4. The INFO logging surfaces per-epoch val F1 so you can watch it
# climb (IMG-02 convergence); the tqdm bars show in-epoch progress. Seeds 1337 + 7 follow in the
# next cell.

# %%
import logging

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.harness.image_supervised_train import ImageConfig, run_image_epoch

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
logging.getLogger("aqa.phase07").setLevel(logging.INFO)

SEEDS = [42, 1337, 7]
_results = {}

_n = 42
_results[_n] = run_image_epoch(
    run_name=f"shallow_squat_baseline_seed{_n}",
    drive_root=MYDRIVE,
    images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
    seed=_n, config=ImageConfig(), resume=True, max_epochs=50,
    dataset_cls=ShallowSquatDataset,
)
_r = _results[_n]
print(f"\nseed {_n}: best_f1_val={_r['best_f1_val']:.4f}  (last epoch {_r['epoch']})")
print(f"best.pt: {_r['best_checkpoint_path']}")


# %% [markdown]
# ## Step 4 (cont.) — seeds 1337 + 7
#
# Trains the remaining seeds (resume=True each, ~5 min/seed), then prints the per-seed best val F1
# table (D4 variance evidence). Run after the seed-42 cell in the same session.

# %%
for _n in SEEDS[1:]:
    print(f"\n===== seed {_n} =====")
    _results[_n] = run_image_epoch(
        run_name=f"shallow_squat_baseline_seed{_n}",
        drive_root=MYDRIVE,
        images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
        seed=_n, config=ImageConfig(), resume=True, max_epochs=50,
        dataset_cls=ShallowSquatDataset,
    )

print("\n\nper-seed best val F1 (0.5 proxy):")
for _n in SEEDS:
    print(f"  seed {_n:5d}: best_f1_val={_results[_n]['best_f1_val']:.4f}  "
          f"(last epoch {_results[_n]['epoch']})")


# %% [markdown]
# ## Step 5 + 6 — mean-of-sigmoids ensemble + val-tuned threshold + test F1 (the SSL-lift control)
#
# Reloads each seed's best.pt, gathers per-crop sigmoid scores on the 529-crop val + 540-crop test
# splits (num_workers=0 — single pass, no worker teardown), ensembles by mean-of-sigmoids, tunes ONE
# decision threshold on the ensemble VAL scores, and reports the **test F1** on the official 540-crop
# split at that threshold — computed in code. This is the SSL-lift control (no paper supervised row).

# %%
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)
from backend.training.aqa.harness.image_supervised_train import build_resnet18

_device = torch.device("cuda")


def _split_loader(split):
    ds = ShallowSquatDataset(
        split=split, images_root=IMAGES_ROOT, labels_path=LABELS_PATH,
        splits_root=SPLITS_ROOT, train_aug=False,
    )
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)


@torch.no_grad()
def _gather(model, loader):
    model.eval()
    scs, lbs = [], []
    for _img, _lbl in loader:
        scs.append(torch.sigmoid(model(_img.to(_device))).squeeze(-1).cpu())
        lbs.append(_lbl)
    return torch.cat(scs).numpy(), torch.cat(lbs).numpy().astype(int)


_val_loader, _test_loader = _split_loader("val"), _split_loader("test")
_per_seed_val, _per_seed_test = {}, {}
_val_labels = _test_labels = None
for _n in SEEDS:
    _ck = torch.load(
        f"{MYDRIVE}/FitNova/checkpoints/phase07/shallow_squat_baseline_seed{_n}/best.pt",
        map_location=_device, weights_only=False,
    )
    _m = build_resnet18().to(_device)
    _m.load_state_dict(_ck["model_state_dict"])
    _per_seed_val[_n], _val_labels = _gather(_m, _val_loader)
    _per_seed_test[_n], _test_labels = _gather(_m, _test_loader)
    print(f"seed {_n:5d}: val F1@0.5={f1_per_error(_val_labels, (_per_seed_val[_n] >= 0.5).astype(int)):.4f}  "
          f"test F1@0.5={f1_per_error(_test_labels, (_per_seed_test[_n] >= 0.5).astype(int)):.4f}")
    del _m
    torch.cuda.empty_cache()

_ens_val = np.mean([_per_seed_val[_n] for _n in SEEDS], axis=0)
_ens_test = np.mean([_per_seed_test[_n] for _n in SEEDS], axis=0)
_t_star, _f1_val_at_t = threshold_sweep(_val_labels, _ens_val)

_test_pred = (_ens_test >= _t_star).astype(int)
_test_f1 = f1_per_error(_test_labels, _test_pred)
_test_pr_auc = pr_auc_per_error(_test_labels, _ens_test)
_test_cm = confusion_matrix_per_error(_test_labels, _test_pred)
_best_seed = max(SEEDS, key=lambda n: f1_per_error(_val_labels, (_per_seed_val[n] >= 0.5).astype(int)))
_best_seed_test_f1 = f1_per_error(_test_labels, (_per_seed_test[_best_seed] >= _t_star).astype(int))

print(f"\nensemble val F1@0.5={f1_per_error(_val_labels, (_ens_val >= 0.5).astype(int)):.4f}  "
      f"|  val-tuned t*={_t_star:.4f} (ensemble val F1@t*={_f1_val_at_t:.4f})")
print("=" * 60)
print("Shallow-Squat SUPERVISED BASELINE  (SSL-lift control)")
print("=" * 60)
print(f"  ensemble ({len(SEEDS)} seeds) test F1 @ t*={_t_star:.3f} : {_test_f1:.4f}")
print(f"  ensemble test PR-AUC (threshold-free)   : {_test_pr_auc:.4f}")
print(f"  single best-val seed ({_best_seed}) test F1     : {_best_seed_test_f1:.4f}")
print(f"  confusion [[TN,FP],[FN,TP]]:\n{_test_cm}")
print("-" * 60)
print("  paper rows (context — NO supervised-ImageNet row exists):")
print("    CVCSPC (Plan-04 SSL target) : 0.8694")
print("    SimSiam (image SSL)         : 0.8286")
print("    OpenPose-TDM (2D pose)      : 0.8340")
