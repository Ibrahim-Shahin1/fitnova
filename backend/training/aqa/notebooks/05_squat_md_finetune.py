# Phase 4 (Plans 03-04) — Squat MD-SSL fine-tune + ensemble eval (R(2+1)D-18)
#
# Companion to `.planning/phases/04-squat-motion-disentangling-ssl/04-03-PLAN.md` +
# `04-04-PLAN.md`. Fine-tunes the MD-pretrained backbone (`md_pretrain_v2/backbone.pt`,
# the strong-aug ep5-best from notebook 04) across 3 seeds, then ensembles + evaluates on
# the official test split and produces the figures + results.pkl.
#
# RESULT (this run): no-TTA 3-seed ensemble TEST macro-F1 = 0.6304 (KIE 0.4198 / KFE 0.8410)
# — matches paper MD (0.6262); +0.087 over Phase 3 (0.5429). TTA evaluated, not adopted.
#
# Run on Colab L4. Jupytext "percent": `pip install jupytext && jupytext --to ipynb 05_squat_md_finetune.py`,
# or paste cells one at a time per the FitNova interactive working agreement (one cell, paste back, next).
# Every long cell has a tqdm bar. All checkpoints are on Drive; runs are resume-safe (resume=True).

# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path). Run FIRST in a fresh session.
# After a git pull that changed an already-imported trainer module, restart the runtime OR
# importlib.reload it ([[reference_colab_module_reload_after_git_pull]]).

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH = "fresh-start"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
_head = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
print(f"Repo ready: {REPO_DIR} @ {_head}")

# %% [markdown]
# ## Step 0 — env + GPU/VRAM + Drive mount + stage LABELED videos + MD backbone path
# F8: `_envinit` MUST be the first import (sets CUBLAS_WORKSPACE_CONFIG before torch).
# Fine-tune + eval need only the labeled set (1739 mp4s) + the SSL backbone — NOT the unlabeled set.

# %%
from backend.training.aqa.harness import _envinit  # F8: CUBLAS before torch
import subprocess
import sys
from pathlib import Path

try:
    import av  # noqa: F401
    _pyav = "already installed"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av  # noqa: F401
    _pyav = "installed by Step 0"
print(f"PyAV {av.__version__} ({_pyav})")

import torch
import torchvision
assert torch.cuda.is_available(), "GPU required — Runtime > Change runtime type > L4"
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "|", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")
MYDRIVE = "/content/drive/MyDrive"

from backend.training.aqa.harness.colab import stage_squat_videos
VIDEOS_ROOT = stage_squat_videos(MYDRIVE)
print("LABELED VIDEOS_ROOT:", VIDEOS_ROOT, "| mp4s:",
      len(list(Path(VIDEOS_ROOT).glob("*.mp4"))), "(expected 1739)")

MD_BACKBONE = os.path.join(MYDRIVE, "FitNova/checkpoints/phase04/md_pretrain_v2/backbone.pt")
assert os.path.isfile(MD_BACKBONE), f"missing SSL backbone: {MD_BACKBONE}"
print("MD backbone (SSL deliverable):", MD_BACKBONE)

# %% [markdown]
# ## Step 1 — sanity: the v2-ep5 backbone loads into the fine-tune model + 1-batch forward

# %%
import importlib
from torch.utils.data import DataLoader
import backend.training.aqa.harness.supervised_train as _sup; importlib.reload(_sup)
import backend.training.aqa.harness.md_finetune as _mft; importlib.reload(_mft)
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset

device = torch.device("cuda")
_ck = torch.load(MD_BACKBONE, map_location="cpu", weights_only=False)
print(f"MD backbone epoch = {_ck['epoch']} (expect 5)")
model = _mft.build_finetune_model(MD_BACKBONE, dropout=0.2).to(device).eval()
print("fine-tune head:", model.fc)  # Sequential(Dropout(0.2), Linear(512,2))
ds = SquatKIEKFEDataset(split="val", train_aug=False, drive_root=MYDRIVE,
                        videos_root=VIDEOS_ROOT, num_frames=32, crop_size=112)
clip, label = next(iter(DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)))
with torch.no_grad():
    logits = model(clip.to(device))
print(f"forward OK: {tuple(clip.shape)} -> {tuple(logits.shape)} (expect (4,2)); "
      f"val={len(ds)} pos_weight={ds.pos_weight.tolist()}")

# %% [markdown]
# ## Step 2 — timing probe (seed 42, 1 real epoch; resume-safe). Confirms wall time + D6 + GPU loop.

# %%
config = _mft.FinetuneConfig(max_epochs=50)  # AdamW lr1e-4 wd1e-4 dropout0.2, 50ep/8-patience, cosine T_max=50
res = _mft.run_md_finetune_epoch(run_name="md_finetune_seed42", md_backbone_path=MD_BACKBONE,
                                 drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
                                 seed=42, config=config, resume=True, max_epochs=1)
m0 = res["metrics_history"][-1]
print(f"epoch {m0['epoch']}: wall={m0['epoch_wall_time_s']:.0f}s val_macro={m0['val_macro_f1']:.4f} "
      f"val/train={m0['val_train_loss_ratio']:.2f} aborted_overfit={res['aborted_overfit']}")

# %% [markdown]
# ## Step 3 — seed 42 full fine-tune (resume ep1 -> 50, self early-stops on 8-patience)

# %%
config = _mft.FinetuneConfig(max_epochs=50)
res42 = _mft.run_md_finetune_epoch(run_name="md_finetune_seed42", md_backbone_path=MD_BACKBONE,
                                   drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
                                   seed=42, config=config, resume=True, max_epochs=50)
print(f"seed 42: best_val_macro={res42['best_f1_val']:.4f} final_epoch={res42['epoch']} "
      f"aborted_overfit={res42['aborted_overfit']}")

# %% [markdown]
# ## Step 4 — seeds 1337 + 7 (LOCKED base recipe; seed 42 did not trip D6). Resume-safe per seed.

# %%
for SEED in [1337, 7]:
    config = _mft.FinetuneConfig(max_epochs=50)
    r = _mft.run_md_finetune_epoch(run_name=f"md_finetune_seed{SEED}", md_backbone_path=MD_BACKBONE,
                                   drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
                                   seed=SEED, config=config, resume=True, max_epochs=50)
    print(f"seed {SEED}: best_val_macro={r['best_f1_val']:.4f} final_epoch={r['epoch']} "
          f"aborted_overfit={r['aborted_overfit']}")

# %% [markdown]
# ## Step 5 — base ensemble TEST eval (mean-of-sigmoids, threshold tuned on ensemble VAL, applied to TEST)

# %%
import numpy as np
from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
from backend.training.aqa.eval.metrics import threshold_sweep, f1_per_error, pr_auc_per_error
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k): return x

CKPT = os.path.join(MYDRIVE, "FitNova/checkpoints/phase04")
SEEDS = [42, 1337, 7]

def _load_ft(seed):
    ck = torch.load(os.path.join(CKPT, f"md_finetune_seed{seed}/best.pt"), map_location="cpu", weights_only=False)
    m = _mft.build_finetune_model(MD_BACKBONE, dropout=0.2); m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval()

models = [_load_ft(s) for s in SEEDS]

def _fwd(split):
    d = SquatKIEKFEDataset(split=split, train_aug=False, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
                           num_frames=32, crop_size=112)
    loader = DataLoader(d, batch_size=16, num_workers=4, shuffle=False, drop_last=False)
    outs = [[] for _ in models]; labs = []
    with torch.no_grad():
        for clip, label in tqdm(loader, desc=f"{split} ({len(d)} x3)"):
            clip = clip.to(device, non_blocking=True); labs.append(label.numpy())
            for i, m in enumerate(models): outs[i].append(m(clip).cpu().numpy())
    return [np.concatenate(o) for o in outs], np.concatenate(labs).astype(int)

val_logits, val_y = _fwd("val"); test_logits, test_y = _fwd("test")
ens_val, ens_test = aggregate_sigmoid_mean(val_logits), aggregate_sigmoid_mean(test_logits)
heads = ["KIE", "KFE"]; thr = {}; test_f1 = {}
for h, name in enumerate(heads):
    t, _ = threshold_sweep(val_y[:, h], ens_val[:, h]); thr[name] = float(t)
    test_f1[name] = float(f1_per_error(test_y[:, h], (ens_test[:, h] >= t).astype(int)))
ens_test_macro = (test_f1["KIE"] + test_f1["KFE"]) / 2
print(f"thresholds {thr}\nENSEMBLE TEST macro = {ens_test_macro:.4f} "
      f"(KIE {test_f1['KIE']:.4f} / KFE {test_f1['KFE']:.4f}) | paper MD 0.6262 | Phase3 0.5429")

# %% [markdown]
# ## Step 6 — val-tuned TTA (select on ensemble VAL, apply to TEST). Result: flip OOD, temporal_jitter
# reversed on test -> TTA NOT adopted; the no-TTA ensemble (Step 5) is the final result.

# %%
import itertools
VIEWS = ["orig", "temporal_jitter", "flip"]  # 5-crop omitted (inline-upsample fidelity caveat, tta.py)
def _view(clip, v):
    if v == "orig": return clip
    if v == "temporal_jitter": return torch.roll(clip, shifts=1, dims=2)
    if v == "flip": return torch.flip(clip, dims=(-1,))
    raise ValueError(v)
def _cache(split):
    d = SquatKIEKFEDataset(split=split, train_aug=False, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT,
                           num_frames=32, crop_size=112)
    loader = DataLoader(d, batch_size=16, num_workers=4, shuffle=False, drop_last=False)
    cache = {i: {v: [] for v in VIEWS} for i in range(len(models))}; labs = []
    with torch.no_grad():
        for clip, label in tqdm(loader, desc=f"{split} TTA ({len(d)})"):
            clip = clip.to(device, non_blocking=True); labs.append(label.numpy())
            for v in VIEWS:
                cv = _view(clip, v)
                for i, m in enumerate(models): cache[i][v].append(m(cv).cpu().numpy())
    for i in cache:
        for v in VIEWS: cache[i][v] = np.concatenate(cache[i][v])
    return cache, np.concatenate(labs).astype(int)
val_cache, vy = _cache("val"); test_cache, ty2 = _cache("test")
def _ens_under(cache, recipe):
    return aggregate_sigmoid_mean([np.mean([cache[i][v] for v in recipe], axis=0) for i in range(len(models))])
extra = [v for v in VIEWS if v != "orig"]
cands = [("orig",)] + [("orig", *c) for r in range(1, len(extra)+1) for c in itertools.combinations(extra, r)]
best = None
for recipe in cands:
    e = _ens_under(val_cache, recipe); macro = float(np.mean([threshold_sweep(vy[:, h], e[:, h])[1] for h in range(2)]))
    print(f"  {recipe}: val macro={macro:.4f}")
    if best is None or macro > best[1]: best = (recipe, macro)
recipe = best[0]; et = _ens_under(test_cache, recipe); ev = _ens_under(val_cache, recipe)
tta_f1 = {n: float(f1_per_error(ty2[:, h], (et[:, h] >= threshold_sweep(vy[:, h], ev[:, h])[0]).astype(int)))
          for h, n in enumerate(heads)}
print(f"\nval-tuned recipe {recipe}; ensemble+TTA TEST macro = {(tta_f1['KIE']+tta_f1['KFE'])/2:.4f} "
      f"(no-TTA was {ens_test_macro:.4f})")

# %% [markdown]
# ## Step 7 — build results.pkl (source of truth: re-derives no-TTA ensemble + loads SSL/fine-tune curves)

# %%
import pickle
from backend.training.aqa.eval.metrics import confusion_matrix_per_error
REPO = os.path.abspath(os.path.join(os.path.dirname(_mft.__file__), "../../../.."))
FIGDIR = os.path.join(REPO, ".planning/phases/04-squat-motion-disentangling-ssl/figures")
DRIVE_FIG = os.path.join(MYDRIVE, "FitNova/phase04_figures")
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(DRIVE_FIG, exist_ok=True)

test_conf = {n: confusion_matrix_per_error(test_y[:, h], (ens_test[:, h] >= thr[n]).astype(int)).tolist()
             for h, n in enumerate(heads)}
test_prauc = {n: float(pr_auc_per_error(test_y[:, h], ens_test[:, h])) for h, n in enumerate(heads)}
per_seed = {}
for i, s in enumerate(SEEDS):
    sig = 1/(1+np.exp(-test_logits[i]))
    f = {n: float(f1_per_error(test_y[:, h], (sig[:, h] >= thr[n]).astype(int))) for h, n in enumerate(heads)}
    per_seed[s] = {**f, "macro": (f["KIE"]+f["KFE"])/2}
def _curves(run):
    try:
        dd = os.path.join(CKPT, run); nm = open(os.path.join(dd, "latest.txt")).read().strip()
        c = torch.load(os.path.join(dd, nm), map_location="cpu", weights_only=False)
        return {"metrics": c.get("metrics_history", []), "linear_probe": c.get("linear_probe_history", [])}
    except Exception as e:
        print(f"  (curve load failed {run}: {e})"); return {"metrics": [], "linear_probe": []}
results = {
    "final": {"method": "MD-SSL 3-seed ensemble, no TTA", "test_macro": ens_test_macro,
              "test_f1": test_f1, "test_prauc": test_prauc, "test_confusion": test_conf, "thresholds": thr},
    "per_seed": per_seed,
    "comparison": {"phase3": {"KIE": 0.2857, "KFE": 0.8000, "macro": 0.5429},
                   "paper_kinetics": {"KIE": 0.2970, "KFE": 0.8184, "macro": 0.5577},
                   "paper_md": {"KIE": 0.4186, "KFE": 0.8338, "macro": 0.6262},
                   "phase4_ensemble": {"KIE": test_f1["KIE"], "KFE": test_f1["KFE"], "macro": ens_test_macro}},
    "ssl_v1": _curves("md_pretrain_v1"), "ssl_v2": _curves("md_pretrain_v2"),
    "finetune_curves": {s: _curves(f"md_finetune_seed{s}")["metrics"] for s in SEEDS},
    "raw": {"ens_test_scores": ens_test, "test_labels": test_y, "ens_val_scores": ens_val, "val_labels": val_y},
}
for p in (os.path.join(FIGDIR, "results.pkl"), os.path.join(CKPT, "phase04_results.pkl")):
    pickle.dump(results, open(p, "wb"))
print(f"results.pkl written ({FIGDIR} + Drive). ensemble test macro {ens_test_macro:.4f}")

# %% [markdown]
# ## Step 8 — the 9 figures from results.pkl (pure plotting; savefig before show; Drive backup)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
R = pickle.load(open(os.path.join(FIGDIR, "results.pkl"), "rb"))
def _save(fig, name):
    for d in (FIGDIR, DRIVE_FIG): fig.savefig(os.path.join(d, name), dpi=130, bbox_inches="tight")
    plt.show(); plt.close(fig)
# 1 SSL loss; 2 linear-probe ablation
for ykey, title, fname in [("ssl_loss", "MD-SSL pretraining loss", "ssl_loss_curve.png")]:
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, k in [("v1 weak-aug", "ssl_v1"), ("v2 strong-aug", "ssl_v2")]:
        mh = R[k]["metrics"]
        if mh: ax.plot([m["epoch"] for m in mh], [m["ssl_loss_mean"] for m in mh], marker=".", label=tag)
    ax.set(xlabel="SSL epoch", ylabel="triplet loss", title=title); ax.legend(); ax.grid(alpha=.3); _save(fig, fname)
fig, ax = plt.subplots(figsize=(6, 4))
for tag, k in [("v1 weak-aug", "ssl_v1"), ("v2 strong-aug", "ssl_v2")]:
    lp = R[k]["linear_probe"]
    if lp: ax.plot([e["epoch"] for e in lp], [e["linear_probe_f1_macro"] for e in lp], marker="o", label=tag)
ax.axhline(0.4297, ls="--", c="gray", label="frozen Kinetics 0.43")
ax.set(xlabel="SSL epoch", ylabel="linear-probe macro-F1", title="SSL convergence: weak vs strong aug")
ax.legend(); ax.grid(alpha=.3); _save(fig, "linear_probe_curve.png")
# 3 per-seed fine-tune
fig, axx = plt.subplots(1, 2, figsize=(11, 4))
for s, mh in R["finetune_curves"].items():
    if not mh: continue
    ep = [m["epoch"] for m in mh]
    axx[0].plot(ep, [m["val_macro_f1"] for m in mh], marker=".", label=f"seed {s}")
    axx[1].plot(ep, [m["train_loss_mean"] for m in mh], label=f"{s} train")
    axx[1].plot(ep, [m["val_loss_mean"] for m in mh], ls="--", label=f"{s} val")
axx[0].set(xlabel="epoch", ylabel="val macro-F1", title="Fine-tune val macro-F1"); axx[0].legend(); axx[0].grid(alpha=.3)
axx[1].set(xlabel="epoch", ylabel="BCE loss", title="Fine-tune train/val loss (overfit view)"); axx[1].legend(fontsize=7); axx[1].grid(alpha=.3)
_save(fig, "finetune_curves_per_seed.png")
# 4 ensemble vs single
fig, ax = plt.subplots(figsize=(7, 4)); ps = R["per_seed"]; labs = [f"seed {s}" for s in ps] + ["ENSEMBLE"]
kie = [ps[s]["KIE"] for s in ps] + [R["final"]["test_f1"]["KIE"]]; kfe = [ps[s]["KFE"] for s in ps] + [R["final"]["test_f1"]["KFE"]]
mac = [ps[s]["macro"] for s in ps] + [R["final"]["test_macro"]]; x = np.arange(len(labs)); w = .25
ax.bar(x-w, kie, w, label="KIE"); ax.bar(x, kfe, w, label="KFE"); ax.bar(x+w, mac, w, label="macro")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set(ylabel="test F1", title="Ensemble vs single seed"); ax.legend(); ax.grid(alpha=.3, axis="y"); _save(fig, "ensemble_vs_single.png")
# 5 headline
fig, ax = plt.subplots(figsize=(8, 4.5)); C = R["comparison"]
rows = [("Phase 3\nKinetics", "phase3"), ("Paper\nKinetics", "paper_kinetics"), ("Paper\nMD", "paper_md"), ("Phase 4\nMD-SSL ours", "phase4_ensemble")]; x = np.arange(len(rows)); w = .27
for j, col in enumerate(["KIE", "KFE", "macro"]): ax.bar(x+(j-1)*w, [C[k][col] for _, k in rows], w, label=col)
ax.axhline(C["paper_md"]["macro"], ls=":", c="red", alpha=.6)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set(ylabel="test F1", title="Squat KIE/KFE — Phase 3 vs Phase 4 vs paper"); ax.legend(); ax.grid(alpha=.3, axis="y"); _save(fig, "phase3_vs_phase4_comparison.png")
# 6-7 confusion
for name in ["KIE", "KFE"]:
    cm = np.array(R["final"]["test_confusion"][name]); fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "err"], yticklabels=["good", "err"], xlabel="pred", ylabel="true", title=f"{name} test confusion (thr {R['final']['thresholds'][name]:.2f})"); _save(fig, f"confusion_{name.lower()}.png")
# 8-9 PR
ensr = R["raw"]["ens_test_scores"]; tyr = R["raw"]["test_labels"]
for h, name in enumerate(["KIE", "KFE"]):
    p, r, _ = precision_recall_curve(tyr[:, h], ensr[:, h]); base = float(tyr[:, h].mean()); fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, marker=".", ms=3); ax.axhline(base, ls="--", c="gray", label=f"random {base:.2f}")
    ax.set(xlabel="recall", ylabel="precision", title=f"{name} test PR (AUC {R['final']['test_prauc'][name]:.3f})"); ax.legend(); ax.grid(alpha=.3); _save(fig, f"pr_{name.lower()}.png")
print("9 figures saved to", FIGDIR)

# %% [markdown]
# ## Step 9 — sample-prediction grid (top TP / FP / FN per head) on the ensemble test scores

# %%
R = pickle.load(open(os.path.join(FIGDIR, "results.pkl"), "rb"))
ensr = R["raw"]["ens_test_scores"]; tyr = R["raw"]["test_labels"]; thr_ = R["final"]["thresholds"]
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
_ds = SquatKIEKFEDataset(split="test", train_aug=False, drive_root=MYDRIVE, videos_root=VIDEOS_ROOT, num_frames=32, crop_size=112)
mids = []
for clip, _ in tqdm(DataLoader(_ds, batch_size=16, num_workers=4, shuffle=False), desc="collect mid-frames"):
    mid = clip[:, :, clip.shape[2] // 2]                       # [B,3,H,W] center frame
    mid = (mid * _STD + _MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()  # de-norm -> [B,H,W,3]
    mids.extend(list(mid))
mids = np.stack(mids)
fig, axes = plt.subplots(2, 3, figsize=(8, 5.5)); fig.suptitle("Sample predictions (center frame; s = ensemble score)")
for rowi, name in enumerate(["KIE", "KFE"]):
    h = ["KIE", "KFE"].index(name); pred = (ensr[:, h] >= thr_[name]).astype(int); true = tyr[:, h]
    tp = np.where((pred == 1) & (true == 1))[0]; fp = np.where((pred == 1) & (true == 0))[0]; fn = np.where((pred == 0) & (true == 1))[0]
    cells = [("TP", tp[np.argsort(-ensr[tp, h])] if len(tp) else tp),
             ("FP", fp[np.argsort(-ensr[fp, h])] if len(fp) else fp),
             ("FN", fn[np.argsort(ensr[fn, h])] if len(fn) else fn)]
    for c, (lab, idxs) in enumerate(cells):
        ax = axes[rowi][c]; ax.axis("off")
        if len(idxs):
            i = idxs[0]; ax.imshow(mids[i]); ax.set_title(f"{name} {lab}  s={ensr[i, h]:.2f}", fontsize=9)
        else:
            ax.set_title(f"{name} {lab}: none", fontsize=9)
for d in (FIGDIR, DRIVE_FIG): fig.savefig(os.path.join(d, "sample_predictions.png"), dpi=130, bbox_inches="tight")
plt.show(); plt.close(fig); print("sample_predictions.png saved")

# %% [markdown]
# ## Step 10 — commit the figures + results.pkl to fresh-start (from the Colab repo)
# If push fails on auth, download figures/ from MyDrive/FitNova/phase04_figures and commit locally.

# %%
FIGREL = ".planning/phases/04-squat-motion-disentangling-ssl/figures"
def _sh(c):
    r = subprocess.run(c, shell=True, cwd=REPO_DIR, capture_output=True, text=True)
    print(f"$ {c}\n{(r.stdout + r.stderr).strip()[-1500:]}"); return r.returncode
_sh(f"git add {FIGREL}")
_sh('git -c user.email="colab@fitnova" -c user.name="Colab" commit -m "docs(04): Phase 4 figures + results.pkl"')
_sh("git pull --no-rebase --no-edit origin fresh-start")
_rc = _sh("git push origin fresh-start")
print("\nPUSH OK" if _rc == 0 else
      "\nPUSH FAILED (no push auth on this clone) — download figures/ from "
      "MyDrive/FitNova/phase04_figures and commit them locally.")
