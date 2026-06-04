"""Train vs validation vs test F1 at the deployed checkpoint.

Computes train, validation and test F1 (all at the val-tuned threshold) for each deployed model and
the supervised baselines, by running the saved weights on all three official splits. This is the
train-to-validation generalization view: train F1 close to validation and test means no overfitting;
train F1 far above test means overfitting (the OHP baseline Elbows case).

Validation and test F1 are read from the existing reeval_*.pkl; the train scores are computed fresh
here. Results merge into docs/audit/data/train_val_test_f1.pkl.

Usage: python train_eval.py [shallow|video|all]   (default: all)
Paths via env match the other re-eval scripts.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.aqa.eval.metrics import f1_per_error
from backend.training.aqa.harness.image_supervised_train import build_resnet18

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)
TARGET = sys.argv[1] if len(sys.argv) > 1 else "all"

CKPT = Path(os.environ.get("FITNOVA_CKPT", r"G:/My Drive/FitNova/checkpoints"))
_P1 = (r"C:/Users/tsh_x/Desktop/FitNova Application/Fitness-AQA"
       r"/Fitness-AQA_dataset_release-20260518T073812Z-3-001/Fitness-AQA_dataset_release")
SHALLOW = Path(os.environ.get("FITNOVA_SHALLOW_DATA", _P1 + "/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset"))
CROPS = Path(os.environ.get("FITNOVA_SHALLOW_CROPS", r"C:/Users/tsh_x/fitnova_audit_scratch/shallow_extract/crops_unaligned"))
SQUAT_VIDEOS = Path(os.environ.get("FITNOVA_SQUAT_VIDEOS", _P1 + "/Squat/Labeled_Dataset/videos_extracted/videos"))
OHP_VIDEOS = Path(os.environ.get("FITNOVA_OHP_VIDEOS", r"C:/Users/tsh_x/fitnova_audit_scratch/ohp_videos/videos"))
DECODE = Path(os.environ.get("FITNOVA_REEVAL_CACHE", r"C:/Users/tsh_x/fitnova_audit_scratch/reeval_video_cache")).parent / "decode_cache"
SEEDS = [42, 1337, 7]

OUT = ROOT / "docs/audit/data/train_val_test_f1.pkl"
results = pickle.loads(OUT.read_bytes()) if OUT.exists() else {}
REEVAL_SH = pickle.loads((ROOT / "docs/audit/data/reeval_shallow.pkl").read_bytes())
REEVAL_VID = pickle.loads((ROOT / "docs/audit/data/reeval_video.pkl").read_bytes())


def build_video(head: str) -> nn.Module:
    from torchvision.models.video import r2plus1d_18
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(512, 2) if head == "linear" else nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    return m


def gather(model, loader, image, desc):
    model.eval()
    scs, lbs = [], []
    for x, lbl in tqdm(loader, desc=desc, leave=False):
        out = torch.sigmoid(model(x.to(DEVICE)))
        scs.append((out.squeeze(-1) if image else out).cpu())
        lbs.append(lbl)
    return torch.cat(scs).numpy().astype(np.float64), torch.cat(lbs).numpy().astype(int)


def save():
    OUT.write_bytes(pickle.dumps(results))
    print(f"  saved {OUT.relative_to(ROOT)}")


# ── Shallow-Squat (image) ──────────────────────────────────────────────────────
if TARGET in ("shallow", "all"):
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset

    def sh_loader(split):
        ds = ShallowSquatDataset(split=split, images_root=str(CROPS), labels_path=str(SHALLOW / "labels_shallow_depth.json"),
                                 splits_root=str(SHALLOW / "splits"), train_aug=False)
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    tr_loader = sh_loader("train")

    def sh_train_ens(prefix):
        per = []
        for n in SEEDS:
            ck = torch.load(CKPT / f"phase07/{prefix}_seed{n}" / "best.pt", map_location=DEVICE, weights_only=False)
            m = build_resnet18()
            m.load_state_dict(ck["model_state_dict"])
            sc, lb = gather(m.to(DEVICE), tr_loader, True, f"{prefix} s{n} train")
            per.append(sc)
        return np.mean(per, 0), lb

    print("Shallow-Squat: computing TRAIN scores from weights ...")
    cv_tr, tr_lab = sh_train_ens("shallow_squat_cvcspc_finetune")
    bl_tr, _ = sh_train_ens("shallow_squat_baseline")
    results["Shallow CVCSPC - depth"] = dict(
        train=f1_per_error(tr_lab, (cv_tr >= REEVAL_SH["cv_threshold"]).astype(int)),
        val=REEVAL_SH["cv_val_f1"], test=REEVAL_SH["cv_test_f1"], deployed=True)
    results["Shallow baseline - depth"] = dict(
        train=f1_per_error(tr_lab, (bl_tr >= REEVAL_SH["bl_threshold"]).astype(int)),
        val=REEVAL_SH["bl_val_f1"], test=REEVAL_SH["bl_test_f1"], deployed=False)
    for k in ("Shallow CVCSPC - depth", "Shallow baseline - depth"):
        r = results[k]
        print(f"  {k:26s} train {r['train']:.4f}  val {r['val']:.4f}  test {r['test']:.4f}")
    save()


# ── Squat + OHP (video) ────────────────────────────────────────────────────────
if TARGET in ("video", "all"):
    from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
    from backend.training.aqa.datasets.squat import SquatKIEKFEDataset

    os.makedirs(DECODE, exist_ok=True)

    def vid_loader(kind):
        cache_dir = str(DECODE / kind)
        os.makedirs(cache_dir, exist_ok=True)
        if kind == "squat":
            ds = SquatKIEKFEDataset(split="train", drive_root=str(Path(_P1).parent), videos_root=str(SQUAT_VIDEOS),
                                    train_aug=False, cache_dir=cache_dir)
        else:
            ds = OHPElbowsKneesDataset(split="train", drive_root=str(Path(_P1).parent), videos_root=str(OHP_VIDEOS),
                                       train_aug=False, cache_dir=cache_dir)
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    def vid_train_scores(run_name, ckpt_sub, head, kind):
        cf = DECODE.parent / "train_scores" / f"{run_name}.npz"
        cf.parent.mkdir(parents=True, exist_ok=True)
        if cf.exists():
            z = np.load(cf); return z["scores"], z["labels"]
        ck = torch.load(CKPT / ckpt_sub / "best.pt", map_location=DEVICE, weights_only=False)
        m = build_video(head); m.load_state_dict(ck["model_state_dict"], strict=True)
        sc, lb = gather(m.to(DEVICE), vid_loader(kind), False, f"{run_name} train")
        np.savez(cf, scores=sc, labels=lb)
        return sc, lb

    def add_video(name, train_ens, tr_lab, ci, ref_per_error):
        thr = ref_per_error["thr"]
        results[name] = dict(train=f1_per_error(tr_lab[:, ci], (train_ens[:, ci] >= thr).astype(int)),
                             val=ref_per_error["val_f1"], test=ref_per_error["test_f1"],
                             deployed=("baseline" not in name))
        r = results[name]
        print(f"  {name:26s} train {r['train']:.4f}  val {r['val']:.4f}  test {r['test']:.4f}")

    print("\nSquat MD-SSL: TRAIN scores (3 seeds) ...")
    sq_tr = np.mean([vid_train_scores(f"squat_mdssl_{n}", f"phase04/md_finetune_seed{n}", "drop", "squat")[0] for n in SEEDS], 0)
    _, sq_lab = vid_train_scores("squat_mdssl_42", "phase04/md_finetune_seed42", "drop", "squat")
    for ci, e in [(0, "KIE"), (1, "KFE")]:
        add_video(f"Squat MD-SSL - {e}", sq_tr, sq_lab, ci, REEVAL_VID["squat_mdssl"]["per_error"][e]); save()

    print("\nSquat baseline: TRAIN scores ...")
    sqb_tr, sqb_lab = vid_train_scores("squat_baseline", "phase03/r2plus1d18_squat_supervised_v1", "linear", "squat")
    for ci, e in [(0, "KIE"), (1, "KFE")]:
        add_video(f"Squat baseline - {e}", sqb_tr, sqb_lab, ci, REEVAL_VID["squat_baseline"]["per_error"][e]); save()

    print("\nOHP MD-SSL: TRAIN scores (2 seeds) ...")
    oh_tr = np.mean([vid_train_scores(f"ohp_mdssl_{n}", f"phase06/ohp_md_finetune_seed{n}", "drop", "ohp")[0] for n in (42, 1337)], 0)
    _, oh_lab = vid_train_scores("ohp_mdssl_42", "phase06/ohp_md_finetune_seed42", "drop", "ohp")
    for ci, e in [(0, "elbows"), (1, "knees")]:
        add_video(f"OHP MD-SSL - {e.capitalize()}", oh_tr, oh_lab, ci, REEVAL_VID["ohp_mdssl"]["per_error"][e]); save()

    print("\nOHP baseline: TRAIN scores ...")
    ohb_tr, ohb_lab = vid_train_scores("ohp_baseline", "phase06/ohp_supervised_v1", "linear", "ohp")
    for ci, e in [(0, "elbows"), (1, "knees")]:
        nm = f"OHP baseline - {e.capitalize()}"
        add_video(nm, ohb_tr, ohb_lab, ci, REEVAL_VID["ohp_baseline"]["per_error"][e]); save()

print(f"\nDone. {len(results)} model/error rows in {OUT.relative_to(ROOT)}")
