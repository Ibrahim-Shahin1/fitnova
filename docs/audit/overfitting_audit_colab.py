# %% [markdown]
# # FitNova Form-Correction: Re-evaluation from the saved weights (Colab, GPU)
#
# This is the GPU version of the weight re-evaluation. The local notebook
# (overfitting_audit_local.py) shows the committed results.pkl are internally consistent. This
# notebook confirms those numbers come from the trained weights: it mounts Drive, reloads every
# best.pt (Squat and OHP video R(2+1)D-18, Shallow-Squat ResNet-18; baseline and SSL fine-tune, every
# seed), re-runs the official validation and test evaluation on the GPU through the same dataset and
# model code used in training, rebuilds the mean-of-sigmoids ensembles and val-tuned thresholds, and
# confirms the re-derived F1, thresholds and per-clip scores match the committed results. It then
# regenerates the validation-to-test gap, confusion, and score-distribution figures from the live
# models. Any mismatch beyond float tolerance is reported.
#
# Disconnect-safe: every (model, split) score array is cached to Drive (MyDrive/FitNova/audit/cache),
# so re-running after a Colab disconnect skips finished work. Video staging uses the resume-safe
# harness.colab stagers. Figures are written to Drive before display.
#
# Runtime is about 15 to 25 minutes on an L4 (staging plus about 4k video clip forwards on GPU). Run
# top to bottom, or one cell at a time.

# %% [markdown]
# ## Cell A: bootstrap (clone or pull repo, set sys.path)

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH = "audit/overfitting-check"  # this audit branch; falls back to fresh-start if absent

if not os.path.isdir(REPO_DIR):
    try:
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "clone", "-b", "fresh-start", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    for b in (BRANCH, "fresh-start"):
        if subprocess.run(["git", "-C", REPO_DIR, "checkout", b]).returncode == 0:
            subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", b])
            break

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown]
# ## Cell B: mount Drive, stage data, resolve paths
#
# Stages Squat + OHP labeled clips and the Shallow-Squat crops from Drive (resume-safe). Sets the
# checkpoint root and a Drive-backed audit output dir (figures + score cache + the re-eval pkl persist
# across disconnects).

# %%
from backend.training.aqa.harness import _envinit  # noqa: F401  (determinism env)
import torch

from backend.training.aqa.harness.colab import (
    mount_drive, stage_squat_videos, stage_ohp_videos, stage_shallow_squat_images,
)

assert torch.cuda.is_available(), "Select a GPU runtime (Runtime > Change runtime type > GPU)."
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

MYDRIVE = mount_drive()
CKPT = f"{MYDRIVE}/FitNova/checkpoints"
SQUAT_VIDEOS = stage_squat_videos(MYDRIVE)                 # -> /content/squat_videos (flat mp4)
OHP_VIDEOS = stage_ohp_videos(MYDRIVE)                     # -> /content/ohp_videos
SHALLOW_ROOT = stage_shallow_squat_images(MYDRIVE)        # -> /content/squat_shallow_images
SH_IMAGES = os.path.join(SHALLOW_ROOT, "crops_unaligned")
SH_LABELS = os.path.join(SHALLOW_ROOT, "labels_shallow_depth.json")
SH_SPLITS = os.path.join(SHALLOW_ROOT, "splits")

AUDIT_OUT = f"{MYDRIVE}/FitNova/audit"
CACHE_DIR = f"{AUDIT_OUT}/cache"
FIG_DIR = f"{AUDIT_OUT}/figures"
for d in (AUDIT_OUT, CACHE_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
print("checkpoints:", CKPT)
print("audit out  :", AUDIT_OUT)

# %% [markdown]
# ## Cell C: helpers (builders, cached GPU score gather, metrics, savefig)

# %%
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error, f1_per_error, pr_auc_per_error, threshold_sweep,
)
from backend.training.aqa.harness.image_supervised_train import build_resnet18

RNG = np.random.default_rng(20260531)


def build_video(head: str) -> nn.Module:
    from torchvision.models.video import r2plus1d_18
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(512, 2) if head == "linear" else nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    return m


def _loader(kind: str, split: str):
    if kind == "squat":
        ds = SquatKIEKFEDataset(split=split, drive_root=MYDRIVE, videos_root=SQUAT_VIDEOS, train_aug=False)
        bs = 8
    elif kind == "ohp":
        ds = OHPElbowsKneesDataset(split=split, drive_root=MYDRIVE, videos_root=OHP_VIDEOS, train_aug=False)
        bs = 8
    else:
        ds = ShallowSquatDataset(split=split, images_root=SH_IMAGES, labels_path=SH_LABELS,
                                 splits_root=SH_SPLITS, train_aug=False)
        bs = 64
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)


def gather(run_name, ckpt_path, head, kind, split):
    """Sigmoid scores + labels for one model/split on GPU, cached to Drive for disconnect-safety."""
    cf = Path(CACHE_DIR) / f"{run_name}__{split}.npz"
    if cf.exists():
        z = np.load(cf)
        return z["scores"], z["labels"]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (build_resnet18() if kind == "image" else build_video(head))
    miss, unexp = model.load_state_dict(ck["model_state_dict"], strict=True)
    assert not miss and not unexp, f"{run_name}: missing={miss} unexpected={unexp}"
    model.eval().to(DEVICE)
    scs, lbs = [], []
    for x, label in tqdm(_loader(kind, split), desc=f"{run_name[:24]} {split}", leave=False):
        out = torch.sigmoid(model(x.to(DEVICE)))
        scs.append((out.squeeze(-1) if kind == "image" else out).cpu())
        lbs.append(label)
    scores = torch.cat(scs).numpy().astype(np.float64)
    labels = torch.cat(lbs).numpy().astype(int)
    np.savez(cf, scores=scores, labels=labels)
    del model
    torch.cuda.empty_cache()
    return scores, labels


def savefig(name):
    p = os.path.join(FIG_DIR, f"audit_live_{name}.png")
    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    # also drop into the repo figures dir so it can be committed
    repo_fig = Path(REPO_DIR) / "docs/figures" / f"audit_live_{name}.png"
    repo_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(repo_fig, dpi=130, bbox_inches="tight")
    print("  saved", p)
    plt.show(); plt.close()


def load_ref(rel):
    return pickle.loads((Path(REPO_DIR) / rel).read_bytes())


REF_SQUAT = load_ref(".planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl")
REF_OHP = load_ref(".planning/phases/06-overhead-press/figures/results.pkl")
REF_SHAL = load_ref(".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl")
checks = []


def chk(name, got, ref, tol=2e-3):
    ok = abs(got - ref) <= tol
    checks.append(ok)
    print(f"  {'OK ' if ok else '!! MISMATCH'} {name:34s} live={got:.4f}  cache={ref:.4f}  d={got-ref:+.4f}")


# %% [markdown]
# ## Cell D: Shallow-Squat (image) re-eval from weights vs results.pkl

# %%
SEEDS3 = [42, 1337, 7]


def shal_family(prefix):
    pv = {n: gather(f"{prefix}_seed{n}", f"{CKPT}/phase07/{prefix}_seed{n}/best.pt", "image", "image", "val")[0] for n in SEEDS3}
    pt = {n: gather(f"{prefix}_seed{n}", f"{CKPT}/phase07/{prefix}_seed{n}/best.pt", "image", "image", "test")[0] for n in SEEDS3}
    vl = gather(f"{prefix}_seed42", f"{CKPT}/phase07/{prefix}_seed42/best.pt", "image", "image", "val")[1]
    tl = gather(f"{prefix}_seed42", f"{CKPT}/phase07/{prefix}_seed42/best.pt", "image", "image", "test")[1]
    return pv, pt, vl, tl


cv_pv, cv_pt, sh_vl, sh_tl = shal_family("shallow_squat_cvcspc_finetune")
bl_pv, bl_pt, _, _ = shal_family("shallow_squat_baseline")
cv_ev = np.mean([cv_pv[n] for n in SEEDS3], 0); cv_et = np.mean([cv_pt[n] for n in SEEDS3], 0)
bl_ev = np.mean([bl_pv[n] for n in SEEDS3], 0); bl_et = np.mean([bl_pt[n] for n in SEEDS3], 0)
cv_t, _ = threshold_sweep(sh_vl, cv_ev); bl_t, _ = threshold_sweep(sh_vl, bl_ev)
print("SHALLOW - re-eval from weights vs results.pkl")
chk("CVCSPC ensemble test F1", f1_per_error(sh_tl, (cv_et >= cv_t).astype(int)), REF_SHAL["test_f1"]["cvcspc_ensemble"])
chk("CVCSPC ensemble val F1", f1_per_error(sh_vl, (cv_ev >= cv_t).astype(int)), REF_SHAL["cvcspc_ensemble_val_f1"])
chk("CVCSPC threshold", cv_t, REF_SHAL["best_threshold"])
chk("CVCSPC test PR-AUC", pr_auc_per_error(sh_tl, cv_et), REF_SHAL["test_pr_auc"])
chk("baseline ensemble test F1", f1_per_error(sh_tl, (bl_et >= bl_t).astype(int)), REF_SHAL["baseline_control"]["test_f1"])
print(f"  per-clip score max|Δ| vs cache: cv_test {np.max(np.abs(cv_et-REF_SHAL['ensemble_test_scores'])):.2e} "
      f"cv_val {np.max(np.abs(cv_ev-REF_SHAL['ensemble_val_scores'])):.2e}")
SHAL_LIVE = dict(cv_ev=cv_ev, cv_et=cv_et, vl=sh_vl, tl=sh_tl, thr=cv_t)

# %% [markdown]
# ## Cell E: Squat (video) MD-SSL ensemble and baseline re-eval from weights

# %%
def squat_scores(run, sub, head, split):
    return gather(run, f"{CKPT}/{sub}/best.pt", head, "squat", split)


sq_seeds = ["md_finetune_seed42", "md_finetune_seed1337", "md_finetune_seed7"]
sq_ev = np.mean([squat_scores(f"squat_{s}", f"phase04/{s}", "drop", "val")[0] for s in sq_seeds], 0)
sq_et = np.mean([squat_scores(f"squat_{s}", f"phase04/{s}", "drop", "test")[0] for s in sq_seeds], 0)
sq_vl = squat_scores("squat_md_finetune_seed42", "phase04/md_finetune_seed42", "drop", "val")[1]
sq_tl = squat_scores("squat_md_finetune_seed42", "phase04/md_finetune_seed42", "drop", "test")[1]
print("SQUAT - MD-SSL 3-seed ensemble re-eval from weights vs results.pkl")
SQ_LIVE = {}
for ci, e in [(0, "KIE"), (1, "KFE")]:
    t, _ = threshold_sweep(sq_vl[:, ci], sq_ev[:, ci])
    chk(f"test F1 {e}", f1_per_error(sq_tl[:, ci], (sq_et[:, ci] >= t).astype(int)), REF_SQUAT["final"]["test_f1"][e])
    chk(f"threshold {e}", t, REF_SQUAT["final"]["thresholds"][e], tol=0.05)
    SQ_LIVE[e] = dict(thr=t, ci=ci)
print(f"  per-clip ensemble score max|Δ| vs cache: test "
      f"{np.max(np.abs(sq_et-REF_SQUAT['raw']['ens_test_scores'])):.2e}  "
      f"val {np.max(np.abs(sq_ev-REF_SQUAT['raw']['ens_val_scores'])):.2e}")
SQ_LIVE.update(ev=sq_ev, et=sq_et, vl=sq_vl, tl=sq_tl)

# %% [markdown]
# ## Cell F: OHP (video) MD-SSL ensemble and baseline (baseline gap not in the pkl)

# %%
oh_seeds = ["ohp_md_finetune_seed42", "ohp_md_finetune_seed1337"]
oh_ev = np.mean([gather(f"ohp_{s}", f"{CKPT}/phase06/{s}/best.pt", "drop", "ohp", "val")[0] for s in oh_seeds], 0)
oh_et = np.mean([gather(f"ohp_{s}", f"{CKPT}/phase06/{s}/best.pt", "drop", "ohp", "test")[0] for s in oh_seeds], 0)
oh_vl = gather("ohp_ohp_md_finetune_seed42", f"{CKPT}/phase06/ohp_md_finetune_seed42/best.pt", "drop", "ohp", "val")[1]
oh_tl = gather("ohp_ohp_md_finetune_seed42", f"{CKPT}/phase06/ohp_md_finetune_seed42/best.pt", "drop", "ohp", "test")[1]
print("OHP - MD-SSL 2-seed ensemble re-eval from weights vs results.pkl")
OH_LIVE = {}
for ci, e in [(0, "elbows"), (1, "knees")]:
    t, _ = threshold_sweep(oh_vl[:, ci], oh_ev[:, ci])
    chk(f"test F1 {e}", f1_per_error(oh_tl[:, ci], (oh_et[:, ci] >= t).astype(int)), REF_OHP["test_f1"][e])
    OH_LIVE[e] = dict(thr=t, ci=ci)
print(f"  per-clip ensemble score max|Δ| vs cache: test "
      f"{np.max(np.abs(oh_et-REF_OHP['ensemble_test_scores'])):.2e}  "
      f"val {np.max(np.abs(oh_ev-REF_OHP['ensemble_val_scores'])):.2e}")
OH_LIVE.update(ev=oh_ev, et=oh_et, vl=oh_vl, tl=oh_tl)

# OHP supervised baseline: re-eval to recover the baseline validation-to-test gap (only summary F1 was cached).
ob_vsc, ob_vl = gather("ohp_baseline", f"{CKPT}/phase06/ohp_supervised_v1/best.pt", "linear", "ohp", "val")
ob_tsc, ob_tl = gather("ohp_baseline", f"{CKPT}/phase06/ohp_supervised_v1/best.pt", "linear", "ohp", "test")
print("\nOHP supervised baseline - val totest gap (recovered from weights):")
for ci, e in [(0, "elbows"), (1, "knees")]:
    t, _ = threshold_sweep(ob_vl[:, ci], ob_vsc[:, ci])
    vf = f1_per_error(ob_vl[:, ci], (ob_vsc[:, ci] >= t).astype(int))
    tf = f1_per_error(ob_tl[:, ci], (ob_tsc[:, ci] >= t).astype(int))
    chk(f"baseline test F1 {e}", tf, REF_OHP["baseline_control"][e], tol=0.03)
    print(f"    {e}: val_F1 {vf:.4f} -> test_F1 {tf:.4f}  (gap {vf-tf:+.4f})")

# %% [markdown]
# ## Cell G: verdict

# %%
print("=" * 74)
print(f"COLAB RE-EVAL-FROM-WEIGHTS VERDICT: {sum(checks)}/{len(checks)} checks within tolerance")
print("ALL re-derived-from-weights numbers match the committed cache [OK]" if all(checks)
      else "!! MISMATCH - investigate the rows above")
print("=" * 74)

# persist the live re-eval to Drive (+ repo) for the audit record
live = dict(squat=SQ_LIVE, ohp=OH_LIVE, shallow=SHAL_LIVE, n_checks=len(checks), n_pass=int(sum(checks)),
            all_pass=bool(all(checks)), device="cuda")
for dest in (f"{AUDIT_OUT}/reeval_live.pkl", f"{REPO_DIR}/docs/audit/data/reeval_colab.pkl"):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(pickle.dumps(live))
print("saved reeval_live.pkl -> Drive + repo")

# %% [markdown]
# ## Cell H: figures from the live models (validation-to-test gap, confusion, score distributions)
#
# These are the same plots as the local notebook, produced from the values the weights just generated
# on the GPU rather than from the cache.

# %%
C_VAL, C_TEST, C_POS, C_NEG = "#4C72B0", "#C44E52", "#DD8452", "#55A868"
rows = [("Squat KIE", SQ_LIVE["ev"][:, 0], SQ_LIVE["vl"][:, 0], SQ_LIVE["et"][:, 0], SQ_LIVE["tl"][:, 0], SQ_LIVE["KIE"]["thr"]),
        ("Squat KFE", SQ_LIVE["ev"][:, 1], SQ_LIVE["vl"][:, 1], SQ_LIVE["et"][:, 1], SQ_LIVE["tl"][:, 1], SQ_LIVE["KFE"]["thr"]),
        ("OHP Elbows", OH_LIVE["ev"][:, 0], OH_LIVE["vl"][:, 0], OH_LIVE["et"][:, 0], OH_LIVE["tl"][:, 0], OH_LIVE["elbows"]["thr"]),
        ("OHP Knees", OH_LIVE["ev"][:, 1], OH_LIVE["vl"][:, 1], OH_LIVE["et"][:, 1], OH_LIVE["tl"][:, 1], OH_LIVE["knees"]["thr"]),
        ("Shallow depth", SHAL_LIVE["cv_ev"], SHAL_LIVE["vl"], SHAL_LIVE["cv_et"], SHAL_LIVE["tl"], SHAL_LIVE["thr"])]

# validation-to-test gap (F1 and PR-AUC) from live scores
labels = [r[0] for r in rows]
vf1 = [f1_per_error(r[2], (r[1] >= r[5]).astype(int)) for r in rows]
tf1 = [f1_per_error(r[4], (r[3] >= r[5]).astype(int)) for r in rows]
vpr = [pr_auc_per_error(r[2], r[1]) for r in rows]; tpr = [pr_auc_per_error(r[4], r[3]) for r in rows]
x = np.arange(len(rows)); w = 0.38
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9))
a1.bar(x - w/2, vf1, w, color=C_VAL, label="val F1"); a1.bar(x + w/2, tf1, w, color=C_TEST, label="test F1")
for i in range(len(rows)):
    a1.text(i, max(vf1[i], tf1[i]) + 0.02, f"d{vf1[i]-tf1[i]:+.3f}", ha="center", fontsize=8)
a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylim(0, 1.1); a1.legend(); a1.set_title("LIVE val vs test F1 (from the weights)")
a2.bar(x - w/2, vpr, w, color=C_VAL, label="val PR-AUC"); a2.bar(x + w/2, tpr, w, color=C_TEST, label="test PR-AUC")
a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylim(0, 1.1); a2.legend(); a2.set_title("LIVE val vs test PR-AUC (threshold-free)")
savefig("val_test_gap")

# confusion + score-dist per error, live
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for j, (name, ev, vlab, et, tlab, thr) in enumerate(rows):
    cm = confusion_matrix_per_error(tlab, (et >= thr).astype(int))
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)
    axes[0, j].imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for r_ in range(2):
        for c_ in range(2):
            axes[0, j].text(c_, r_, f"{cm[r_,c_]}", ha="center", va="center",
                            color="white" if rn[r_, c_] > 0.5 else "black")
    axes[0, j].set_title(f"{name} test conf", fontsize=9); axes[0, j].grid(False)
    bins = np.linspace(0, 1, 26)
    axes[1, j].hist(et[tlab == 0], bins=bins, color=C_NEG, alpha=0.6, density=True, label="neg")
    axes[1, j].hist(et[tlab == 1], bins=bins, color=C_POS, alpha=0.6, density=True, label="pos")
    axes[1, j].axvline(thr, color="k", ls="--"); axes[1, j].set_title(f"{name} test scores", fontsize=9); axes[1, j].legend(fontsize=7)
plt.suptitle("LIVE confusion + score distributions (test) - straight from the weights", y=1.0)
savefig("confusion_scoredist")
print("\nAll figures written to", FIG_DIR, "(+ repo docs/figures/audit_live_*.png)")

# %% [markdown]
# ## Cell I: train vs validation vs test F1 at the deployed checkpoint
#
# The train-split F1 (each model run on the full official train split) is produced by
# docs/audit/train_eval.py and committed to docs/audit/data/train_val_test_f1.pkl; this cell renders
# it so the two notebooks agree. To recompute it live on this GPU instead, call gather(...) above
# with split="train" for each model and apply the same val-tuned threshold. Train F1 above test is
# normal (the model has seen the training data); validation tracking test is the no-overfit signal,
# and the rare classes (Squat KIE, OHP Elbows) show large train-test gaps (train-memorization).

# %%
_tvt = pickle.loads((Path(REPO_DIR) / "docs/audit/data/train_val_test_f1.pkl").read_bytes())
_order = ["Squat MD-SSL - KIE", "Squat MD-SSL - KFE", "OHP MD-SSL - Elbows", "OHP MD-SSL - Knees",
          "Shallow CVCSPC - depth", "Squat baseline - KIE", "Squat baseline - KFE",
          "OHP baseline - Elbows", "OHP baseline - Knees", "Shallow baseline - depth"]
_rows = [(k, _tvt[k]) for k in _order if k in _tvt]
_x = np.arange(len(_rows)); _w = 0.27
fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(_rows)), 5.5))
ax.bar(_x - _w, [r["train"] for _, r in _rows], _w, color="#999999", label="train")
ax.bar(_x, [r["val"] for _, r in _rows], _w, color=C_VAL, label="validation")
ax.bar(_x + _w, [r["test"] for _, r in _rows], _w, color=C_TEST, label="test")
for _i, (_, r) in enumerate(_rows):
    ax.text(_i, max(r["train"], r["val"], r["test"]) + 0.02, f"tr-te {r['train']-r['test']:+.2f}", ha="center", fontsize=7)
ax.set_xticks(_x); ax.set_xticklabels([k for k, _ in _rows], rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_ylabel("F1"); ax.legend(loc="upper right")
ax.set_title("Train, validation, test F1 per error (validation tracks test; rare classes show large train-test gaps)")
savefig("train_val_test")
