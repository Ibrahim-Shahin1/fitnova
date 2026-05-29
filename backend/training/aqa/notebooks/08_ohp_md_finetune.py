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
