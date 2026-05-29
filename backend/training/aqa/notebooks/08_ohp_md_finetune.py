# Phase 6 — OHP MD-SSL Fine-tune + Ensemble + Test Eval
#
# Companion to .planning/phases/06-overhead-press/06-04-PLAN.md. Fine-tunes the Plan-03
# backbone.pt (3 seeds) -> mean-of-sigmoids ensemble -> val-tuned threshold + TTA ->
# test F1 per error vs the paper (Elbow 0.4552 / Knees 0.8452) + the val->test gap ->
# results.pkl (source of truth for the Plan-05 visualization pack).
#
# Runs on L4 (the fine-tune train loader jitters frames per epoch -> uncached -> decode-bound,
# so the A100 buys nothing here). 3 seeds sequential; each is resume-safe.
#
# Arc: Cell A -> Step 0 stage labeled -> Step 1 backbone assert -> Step 2 3-seed fine-tune
#   (42 first + D6 gate, then 1337/7) -> Step 3 ensemble + val threshold -> Step 4 TTA
#   -> Step 5 test eval + results.pkl.

# %% [markdown]
# ## Cell A — bootstrap

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fitnova"
BRANCH = "fresh-start"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, "https://github.com/Ibrahim-Shahin1/fitnova.git", REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("Repo @", subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip())


# %% [markdown]
# ## Step 0 — env + stage labeled OHP (the fine-tune trains on the 2367 labeled clips)

# %%
from backend.training.aqa.harness import _envinit
import os
import shutil
import subprocess
import sys

try:
    import av
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    import av
import torch
import torchvision
print("tv", torchvision.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "GPU required (L4)"

from google.colab import drive
drive.mount('/content/drive')
MYDRIVE = '/content/drive/MyDrive'

from pathlib import Path
from backend.training.aqa.harness.colab import stage_ohp_videos
VIDEOS_ROOT = stage_ohp_videos(MYDRIVE)
assert len(list(Path(VIDEOS_ROOT).glob('*.mp4'))) == 2367
print("labeled staged:", VIDEOS_ROOT, "(2367) | free", round(shutil.disk_usage('/content').free / 1e9, 1), "GB")


# %% [markdown]
# ## Step 1 — assert the Plan-03 MD backbone exists (NOT a Kinetics path)

# %%
import os
import torch

BACKBONE_PATH = os.path.join(MYDRIVE, "FitNova/checkpoints/phase06/ohp_md_pretrain_v2/backbone.pt")
assert os.path.exists(BACKBONE_PATH), f"Plan 03 backbone.pt missing: {BACKBONE_PATH}"
_bb = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=False)
assert "backbone_state_dict" in _bb, "backbone.pt missing backbone_state_dict"
print("backbone.pt epoch:", _bb["epoch"], "| has backbone_state_dict:", True)
del _bb


# %% [markdown]
# ## Step 2a — fine-tune seed 42 (recipe-locking seed; D6 overfit gate)
#
# Watch `val/train` (the D6 overfit ratio; >10 before epoch 10 aborts) and the 8-epoch
# early-stop on val_macro_f1. val_f1_kie = Elbows, val_f1_kfe = Knees (Squat-inherited keys).
# If aborted_overfit comes back True, the recipe gets bumped (wd 5e-4 / dropout 0.3) and
# locked for the other seeds — but with OHP's balanced pos_weights it likely won't fire.

# %%
import os

from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.harness.md_finetune import FinetuneConfig, run_md_finetune_epoch

base_config = FinetuneConfig()
result42 = run_md_finetune_epoch(
    run_name="ohp_md_finetune_seed42", md_backbone_path=BACKBONE_PATH,
    drive_root=MYDRIVE, videos_root=VIDEOS_ROOT, seed=42,
    config=base_config, resume=True, dataset_cls=OHPElbowsKneesDataset, checkpoint_phase="phase06",
)
for m in result42["metrics_history"]:
    print(f"ep {m['epoch']:>2}: train={m['train_loss_mean']:.4f} val={m['val_loss_mean']:.4f} "
          f"val/train={m['val_train_loss_ratio']:.2f} | Elbows={m['val_f1_kie']:.4f} Knees={m['val_f1_kfe']:.4f} "
          f"macro={m['val_macro_f1']:.4f} ({m['epoch_wall_time_s']:.0f}s)")
print(f"\nseed42: best_f1_val={result42['best_f1_val']:.4f} | aborted_overfit={result42.get('aborted_overfit')} "
      f"| epochs_run={len(result42['metrics_history'])}")


# %% [markdown]
# ## Step 2b — fine-tune seed 1337 (locked recipe = base, seed 42 did not trip D6)
#
# 2-seed ensemble (42 + 1337) — a time-driven reduction from the 3-seed Squat protocol.
# The ensemble methodology is identical; documented in the SUMMARY.

# %%
result1337 = run_md_finetune_epoch(
    run_name="ohp_md_finetune_seed1337", md_backbone_path=BACKBONE_PATH,
    drive_root=MYDRIVE, videos_root=VIDEOS_ROOT, seed=1337,
    config=base_config, resume=True, dataset_cls=OHPElbowsKneesDataset, checkpoint_phase="phase06",
)
for m in result1337["metrics_history"]:
    print(f"ep {m['epoch']:>2}: train={m['train_loss_mean']:.4f} val={m['val_loss_mean']:.4f} "
          f"val/train={m['val_train_loss_ratio']:.2f} | Elbows={m['val_f1_kie']:.4f} Knees={m['val_f1_kfe']:.4f} "
          f"macro={m['val_macro_f1']:.4f} ({m['epoch_wall_time_s']:.0f}s)")
print(f"\nseed1337: best_f1_val={result1337['best_f1_val']:.4f} | aborted_overfit={result1337.get('aborted_overfit')} "
      f"| epochs_run={len(result1337['metrics_history'])}")


# %% [markdown]
# ## Step 3 — 2-seed ensemble + val-tuned threshold + test eval + results.pkl
#
# Ensemble = mean of per-seed sigmoid scores (D4). Threshold tuned on the ensemble VAL
# scores, applied to the held-out TEST split. TTA is NOT re-tuned: Phase 4 evaluated it
# and it reversed on test (not adopted) — we report the no-TTA ensemble, consistent with
# the Phase-4 headline protocol. val->test gap is the overfitting evidence.

# %%
import os
import pickle

import numpy as np
import torch

from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.harness.md_finetune import FinetuneConfig, build_finetune_model
from backend.training.aqa.harness.supervised_train import _build_dataloaders, _val_pass
from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)

device = torch.device("cuda")
cfg = FinetuneConfig()
P6 = os.path.join(MYDRIVE, "FitNova/checkpoints/phase06")
SEEDS = [42, 1337]

loaders = _build_dataloaders(42, cfg, MYDRIVE, VIDEOS_ROOT, dataset_cls=OHPElbowsKneesDataset)
val_loader, test_loader = loaders["val"], loaders["test"]
crit = torch.nn.BCEWithLogitsLoss()

val_scores, test_scores = {}, {}
val_labels = test_labels = None
for s in SEEDS:
    model = build_finetune_model(BACKBONE_PATH, dropout=0.2).to(device)
    ck = torch.load(os.path.join(P6, f"ohp_md_finetune_seed{s}/best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    _, val_scores[s], val_labels = _val_pass(model, val_loader, crit, device)
    _, test_scores[s], test_labels = _val_pass(model, test_loader, crit, device)
    del model
    torch.cuda.empty_cache()

ens_val = np.mean([val_scores[s] for s in SEEDS], axis=0)
ens_test = np.mean([test_scores[s] for s in SEEDS], axis=0)

t_elbows, _ = threshold_sweep(val_labels[:, 0], ens_val[:, 0])
t_knees, _ = threshold_sweep(val_labels[:, 1], ens_val[:, 1])

pe = (ens_test[:, 0] >= t_elbows).astype(int)
pk = (ens_test[:, 1] >= t_knees).astype(int)
f1_e = f1_per_error(test_labels[:, 0], pe)
f1_k = f1_per_error(test_labels[:, 1], pk)
macro = (f1_e + f1_k) / 2.0
ap_e = pr_auc_per_error(test_labels[:, 0], ens_test[:, 0])
ap_k = pr_auc_per_error(test_labels[:, 1], ens_test[:, 1])

ve = (ens_val[:, 0] >= t_elbows).astype(int)
vk = (ens_val[:, 1] >= t_knees).astype(int)
val_macro = (f1_per_error(val_labels[:, 0], ve) + f1_per_error(val_labels[:, 1], vk)) / 2.0
val_test_gap = abs(val_macro - macro)

per_seed_test = {}
for s in SEEDS:
    se = (test_scores[s][:, 0] >= t_elbows).astype(int)
    sk = (test_scores[s][:, 1] >= t_knees).astype(int)
    per_seed_test[s] = {"elbows": f1_per_error(test_labels[:, 0], se), "knees": f1_per_error(test_labels[:, 1], sk)}

baseline = {"elbows": 0.4167, "knees": 0.8069, "macro": 0.6118}
paper = {"elbows": 0.4552, "knees": 0.8452}
ssl_lift = macro - baseline["macro"]

ssl_run = os.path.join(P6, "ohp_md_pretrain_v2")
with open(os.path.join(ssl_run, "latest.txt")) as f:
    ssl_latest = f.read().strip()
ssl_ck = torch.load(os.path.join(ssl_run, ssl_latest), map_location="cpu", weights_only=False)

results = {
    "ssl_metrics_history": ssl_ck["metrics_history"],
    "ssl_linear_probe_history": ssl_ck["linear_probe_history"],
    "finetune_seeds": {42: result42["metrics_history"], 1337: result1337["metrics_history"]},
    "ensemble_val_scores": ens_val, "ensemble_test_scores": ens_test,
    "val_labels": val_labels, "test_labels": test_labels,
    "test_clip_ids": [r.clip_id for r in test_loader.dataset.records],
    "thresholds": {"elbows": t_elbows, "knees": t_knees},
    "test_f1": {"elbows": f1_e, "knees": f1_k, "macro": macro},
    "test_pr_auc": {"elbows": ap_e, "knees": ap_k},
    "test_confusion": {
        "elbows": confusion_matrix_per_error(test_labels[:, 0], pe),
        "knees": confusion_matrix_per_error(test_labels[:, 1], pk),
    },
    "per_seed_test_f1": per_seed_test,
    "per_seed_val_macro": {42: result42["best_f1_val"], 1337: result1337["best_f1_val"]},
    "baseline_control": baseline, "ssl_lift": ssl_lift, "val_test_gap": val_test_gap,
    "paper_targets": paper, "n_seeds": len(SEEDS), "tta": "none",
}
OUT = os.path.join(REPO_DIR, ".planning/phases/06-overhead-press/figures/results.pkl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(results, f)

print(f"thresholds: Elbows {t_elbows:.3f} | Knees {t_knees:.3f}")
print("\n=== OHP test F1 — official 339-clip split (2-seed MD-SSL ensemble, no TTA) ===")
print(f"{'':<22}{'Elbows':>9}{'Knees':>9}{'macro':>9}")
print(f"{'baseline (Plan 02)':<22}{baseline['elbows']:>9.4f}{baseline['knees']:>9.4f}{baseline['macro']:>9.4f}")
print(f"{'our MD-SSL ensemble':<22}{f1_e:>9.4f}{f1_k:>9.4f}{macro:>9.4f}")
print(f"{'paper Ours-MD':<22}{paper['elbows']:>9.4f}{paper['knees']:>9.4f}{(paper['elbows'] + paper['knees']) / 2:>9.4f}")
print(f"\nAP: Elbows {ap_e:.4f} | Knees {ap_k:.4f}")
print(f"SSL lift over baseline (macro): {ssl_lift:+.4f}")
print(f"val->test gap (macro): {val_test_gap:.4f}  ({'OK <0.05' if val_test_gap < 0.05 else 'FLAG >0.05'})")
print("per-seed test F1: " + " | ".join(f"s{s} E={per_seed_test[s]['elbows']:.3f}/K={per_seed_test[s]['knees']:.3f}" for s in SEEDS))
print(f"results.pkl -> {OUT}")
