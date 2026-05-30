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
