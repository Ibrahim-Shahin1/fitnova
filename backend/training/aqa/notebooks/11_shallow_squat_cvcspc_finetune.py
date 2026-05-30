# Phase 7 — Shallow-Squat CVCSPC Fine-tune + Eval (ResNet-18 from the CVCSPC backbone)
#
# Companion notebook to `.planning/phases/07-image-based-errors-cvcspc/07-04-PLAN.md`.
#
# Fine-tunes the Plan-03 CVCSPC backbone (phase07/shallow_squat_cvcspc_v1/backbone.pt) on the labeled
# Shallow-Squat crops across multiple seeds, ensembles (mean-of-sigmoids) + a val-tuned threshold,
# evaluates on the official 540-crop test split, and writes results.pkl. The comparison: our baseline
# (Plan-02 control, 0.8750) -> our CVCSPC ensemble -> paper CVCSPC 0.8694.
#
# Light (~30-45 min). Can continue in the Plan-03 SSL Colab session (code already pulled, backbone on
# Drive) or run standalone via Cell A. Jupytext "percent" format; cells delivered one at a time.

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)
#
# Only needed for a standalone session. If continuing in the Plan-03 SSL session, the repo is already
# pulled — skip to Step 0+1.

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
print("cwd:", os.getcwd())


# %% [markdown]
# ## Step 0 + 1 — stage labeled crops + build_cvcspc_finetune_model + backbone smoke
#
# Stages the labeled Shallow-Squat crops (the fine-tune trains on these), asserts the Plan-03
# backbone.pt exists (the CVCSPC backbone, NOT a from-scratch path), and defines the model builder
# that loads backbone_state_dict (strict=False) into a fresh resnet18 + a new Linear(512,1) head (the
# SSL projector discarded). Smoke: the load is sane (only fc missing) and a forward gives [B,1].

# %%
from backend.training.aqa.harness import _envinit  # F8

import torch
import torch.nn as nn

from backend.training.aqa.harness.colab import mount_drive, stage_shallow_squat_images

MYDRIVE = mount_drive()
SHALLOW_ROOT = stage_shallow_squat_images(MYDRIVE, local_root="/content/squat_shallow_images")
IMAGES_ROOT = os.path.join(SHALLOW_ROOT, "crops_unaligned")
LABELS_PATH = os.path.join(SHALLOW_ROOT, "labels_shallow_depth.json")
SPLITS_ROOT = os.path.join(SHALLOW_ROOT, "splits")
print("labeled crops:", IMAGES_ROOT)

BACKBONE_PATH = os.path.join(
    MYDRIVE, "FitNova/checkpoints/phase07/shallow_squat_cvcspc_v1/backbone.pt"
)
assert os.path.exists(BACKBONE_PATH), f"Plan 03 backbone.pt missing: {BACKBONE_PATH}"
assert "shallow_squat_cvcspc_v1" in BACKBONE_PATH, "must be the CVCSPC backbone, not from-scratch"


def build_cvcspc_finetune_model(backbone_path=BACKBONE_PATH):
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    model.fc = nn.Linear(512, 1)
    return model


_ck = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=False)
from torchvision.models import resnet18 as _r18

_miss, _unexp = _r18(weights=None).load_state_dict(_ck["backbone_state_dict"], strict=False)
print(f"backbone code_version={_ck.get('code_version')} | strict=False load: "
      f"{len(_miss)} missing (expect fc.*), {len(_unexp)} unexpected")

_m = build_cvcspc_finetune_model().cuda()
with torch.no_grad():
    _out = _m(torch.zeros(2, 3, 224, 224, device="cuda"))
assert isinstance(_m.fc, nn.Linear) and _m.fc.out_features == 1
assert tuple(_out.shape) == (2, 1)
print(f"head {_m.fc} | forward zeros[2,3,224,224] -> {tuple(_out.shape)}")
del _m
torch.cuda.empty_cache()


# %% [markdown]
# ## Step 2 — multi-seed CVCSPC fine-tune (seed 42 first)
#
# Fine-tunes from the CVCSPC backbone via the model_builder seam (NOT ImageNet-from-scratch).
# Same recipe as the Plan-02 baseline (Adam 1e-4, 50ep/8-patience cosine, batch 32) so the only
# difference is the initialization — an apples-to-apples SSL-lift measurement. ~5 min/seed,
# resume-safe.

# %%
import logging

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.harness.image_supervised_train import ImageConfig, run_image_epoch

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
logging.getLogger("aqa.phase07").setLevel(logging.INFO)

SEEDS = [42, 1337, 7]
_ft_results = {}

_n = 42
_ft_results[_n] = run_image_epoch(
    run_name=f"shallow_squat_cvcspc_finetune_seed{_n}",
    drive_root=MYDRIVE,
    images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
    seed=_n, config=ImageConfig(), resume=True, max_epochs=50,
    dataset_cls=ShallowSquatDataset,
    model_builder=lambda: build_cvcspc_finetune_model(),
)
_r = _ft_results[_n]
print(f"\nseed {_n}: best_f1_val={_r['best_f1_val']:.4f}  (last epoch {_r['epoch']})")
print(f"best.pt: {_r['best_checkpoint_path']}")


# %% [markdown]
# ## Step 2 (cont.) — seeds 1337 + 7
#
# Trains the remaining seeds, then prints the per-seed best val F1 vs the Plan-02 baseline (loaded
# from Drive — not hardcoded). ~5 min/seed.

# %%
for _n in SEEDS[1:]:
    print(f"\n===== seed {_n} =====")
    _ft_results[_n] = run_image_epoch(
        run_name=f"shallow_squat_cvcspc_finetune_seed{_n}",
        drive_root=MYDRIVE,
        images_root=IMAGES_ROOT, labels_path=LABELS_PATH, splits_root=SPLITS_ROOT,
        seed=_n, config=ImageConfig(), resume=True, max_epochs=50,
        dataset_cls=ShallowSquatDataset,
        model_builder=lambda: build_cvcspc_finetune_model(),
    )

print("\n\nper-seed best val F1 — CVCSPC fine-tune vs Plan-02 baseline:")
for _n in SEEDS:
    _bl = torch.load(
        f"{MYDRIVE}/FitNova/checkpoints/phase07/shallow_squat_baseline_seed{_n}/best.pt",
        map_location="cpu", weights_only=False,
    )["best_f1_val"]
    _cv = _ft_results[_n]["best_f1_val"]
    print(f"  seed {_n:5d}: CVCSPC {_cv:.4f}  vs  baseline {_bl:.4f}  (delta {_cv - _bl:+.4f})")


# %% [markdown]
# ## Step 3 + 4 — ensemble + val-tuned threshold + test F1 + baseline->CVCSPC->paper + results.pkl
#
# Reloads the per-seed best.pt for BOTH families (CVCSPC fine-tune + Plan-02 baseline), gathers val+test
# sigmoid scores, ensembles by mean-of-sigmoids, tunes ONE threshold per family on its ensemble VAL
# scores, reports the official 540-crop test F1, the SSL lift, the val->test gap, and the paper rows.
# Writes results.pkl (source of truth for Plan 05) to Drive. Every number computed in code.

# %%
import pickle
from pathlib import Path

import numpy as np
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
_ckdir = f"{MYDRIVE}/FitNova/checkpoints/phase07"


def _split_loader(split):
    ds = ShallowSquatDataset(split=split, images_root=IMAGES_ROOT, labels_path=LABELS_PATH,
                             splits_root=SPLITS_ROOT, train_aug=False)
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


def _gather_family(prefix):
    pv, pt, vl, tl = {}, {}, None, None
    for _n in SEEDS:
        _ck = torch.load(f"{_ckdir}/{prefix}_seed{_n}/best.pt", map_location=_device, weights_only=False)
        _m = build_resnet18()
        _m.load_state_dict(_ck["model_state_dict"])
        _m = _m.to(_device)
        pv[_n], vl = _gather(_m, _val_loader)
        pt[_n], tl = _gather(_m, _test_loader)
        del _m
        torch.cuda.empty_cache()
    return pv, pt, vl, tl


_cv_pv, _cv_pt, _val_labels, _test_labels = _gather_family("shallow_squat_cvcspc_finetune")
_bl_pv, _bl_pt, _, _ = _gather_family("shallow_squat_baseline")


def _ensemble_eval(pv, pt):
    ev = np.mean([pv[n] for n in SEEDS], axis=0)
    et = np.mean([pt[n] for n in SEEDS], axis=0)
    t, _ = threshold_sweep(_val_labels, ev)
    vf1 = f1_per_error(_val_labels, (ev >= t).astype(int))
    tf1 = f1_per_error(_test_labels, (et >= t).astype(int))
    return ev, et, t, vf1, tf1


_cv_ev, _cv_et, _cv_t, _cv_vf1, _cv_tf1 = _ensemble_eval(_cv_pv, _cv_pt)
_bl_ev, _bl_et, _bl_t, _bl_vf1, _bl_tf1 = _ensemble_eval(_bl_pv, _bl_pt)

_cv_pr = pr_auc_per_error(_test_labels, _cv_et)
_cv_cm = confusion_matrix_per_error(_test_labels, (_cv_et >= _cv_t).astype(int))
_cv_per_seed_test = {n: f1_per_error(_test_labels, (_cv_pt[n] >= _cv_t).astype(int)) for n in SEEDS}
_best_seed = max(SEEDS, key=lambda n: f1_per_error(_val_labels, (_cv_pv[n] >= 0.5).astype(int)))
_cv_single = f1_per_error(_test_labels, (_cv_pt[_best_seed] >= _cv_t).astype(int))

_ssl_lift = _cv_tf1 - _bl_tf1
_val_test_gap = abs(_cv_vf1 - _cv_tf1)
PAPER = {"cvcspc": 0.8694, "simsiam": 0.8286, "openpose_tdm": 0.8340}

print("=" * 62)
print("SHALLOW-SQUAT — baseline -> CVCSPC -> paper (official 540-crop test F1)")
print("=" * 62)
print(f"  our supervised baseline (control) : {_bl_tf1:.4f}  @ t*={_bl_t:.3f}")
print(f"  our CVCSPC ensemble               : {_cv_tf1:.4f}  @ t*={_cv_t:.3f}  (PR-AUC {_cv_pr:.4f})")
print(f"  SSL lift (CVCSPC - baseline)      : {_ssl_lift:+.4f}")
print(f"  CVCSPC single best-seed ({_best_seed})        : {_cv_single:.4f}")
print(f"  per-seed CVCSPC test F1           : " + ", ".join(f"{n}:{_cv_per_seed_test[n]:.4f}" for n in SEEDS))
print(f"  val->test gap (CVCSPC)            : {_val_test_gap:.4f}"
      + ("   [>0.05 — check over-tuning]" if _val_test_gap > 0.05 else ""))
print("-" * 62)
print(f"  paper CVCSPC (SSL target)         : {PAPER['cvcspc']:.4f}")
print(f"  paper SimSiam (image SSL)         : {PAPER['simsiam']:.4f}")
print(f"  paper OpenPose-TDM (2D pose)      : {PAPER['openpose_tdm']:.4f}")
print(f"  confusion [[TN,FP],[FN,TP]]:\n{_cv_cm}")


def _curve(prefix, n):
    rd = f"{_ckdir}/{prefix}_seed{n}"
    with open(f"{rd}/latest.txt") as _f:
        return torch.load(f"{rd}/{_f.read().strip()}", map_location="cpu", weights_only=False)["metrics_history"]


_ssl_rd = f"{_ckdir}/shallow_squat_cvcspc_v1"
with open(f"{_ssl_rd}/latest.txt") as _f:
    _ssl_tah = torch.load(f"{_ssl_rd}/{_f.read().strip()}", map_location="cpu", weights_only=False)["triplet_acc_history"]

_results = {
    "ssl_triplet_acc": _ssl_tah,
    "finetune_seeds": {n: _curve("shallow_squat_cvcspc_finetune", n) for n in SEEDS},
    "baseline_seeds": {n: _curve("shallow_squat_baseline", n) for n in SEEDS},
    "ensemble_val_scores": _cv_ev, "ensemble_test_scores": _cv_et,
    "val_labels": _val_labels, "test_labels": _test_labels,
    "best_threshold": _cv_t,
    "test_f1": {"cvcspc_ensemble": _cv_tf1, "single_model": _cv_single, "per_seed": _cv_per_seed_test},
    "test_pr_auc": _cv_pr, "test_confusion": _cv_cm,
    "baseline_control": {"test_f1": _bl_tf1, "threshold": _bl_t, "ensemble_test_scores": _bl_et,
                         "ensemble_val_f1": _bl_vf1},
    "cvcspc_ensemble_val_f1": _cv_vf1,
    "ssl_lift": _ssl_lift, "val_test_gap": _val_test_gap,
    "paper_targets": PAPER,
    "loss_deviation_note": "SSL pretrained with the 3-term code loss (train_test.py:68); paper Eq.1 is 2-term.",
    "seeds": SEEDS,
}
_drive_pkl = f"{_ckdir}/results.pkl"
with open(_drive_pkl, "wb") as _f:
    pickle.dump(_results, _f)
_repo_pkl = Path(".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl")
_repo_pkl.parent.mkdir(parents=True, exist_ok=True)
with open(_repo_pkl, "wb") as _f:
    pickle.dump(_results, _f)
print(f"\nwrote results.pkl -> Drive ({_drive_pkl}) + repo ({_repo_pkl}), "
      f"{len(pickle.dumps(_results)) / 1e3:.0f} KB")
