"""Re-evaluation from weights for Shallow-Squat (image, ResNet-18).

Loads every best.pt for the 3 supervised-baseline seeds and the 3 CVCSPC fine-tune seeds from Drive,
re-runs the official validation and test evaluation through the same ShallowSquatDataset and
build_resnet18 used in training, rebuilds the mean-of-sigmoids ensembles and the val-tuned threshold,
and confirms the re-derived per-clip scores and F1 match the committed results.pkl. Any mismatch
beyond float tolerance is reported.

Runs on CPU (ResNet-18, about 1k crops) in a couple of minutes.

Paths (override via env): FITNOVA_CKPT_PHASE07, FITNOVA_SHALLOW_DATA, FITNOVA_SHALLOW_CROPS.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)
from backend.training.aqa.harness.image_supervised_train import build_resnet18

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x

CKPT = Path(os.environ.get("FITNOVA_CKPT_PHASE07", r"G:/My Drive/FitNova/checkpoints/phase07"))
DATA = Path(os.environ.get(
    "FITNOVA_SHALLOW_DATA",
    r"C:/Users/tsh_x/Desktop/FitNova Application/Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
    r"/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset"))
CROPS = Path(os.environ.get("FITNOVA_SHALLOW_CROPS",
                            r"C:/Users/tsh_x/fitnova_audit_scratch/shallow_extract/crops_unaligned"))
LABELS = DATA / "labels_shallow_depth.json"
SPLITS = DATA / "splits"
SEEDS = [42, 1337, 7]
DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

print(f"ckpt   : {CKPT}")
print(f"crops  : {CROPS}  (exists={CROPS.exists()})")
print(f"labels : {LABELS}  (exists={LABELS.exists()})")


def split_loader(split: str) -> DataLoader:
    ds = ShallowSquatDataset(split=split, images_root=str(CROPS), labels_path=str(LABELS),
                             splits_root=str(SPLITS), train_aug=False)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)


def gather(model, loader, desc):
    model.eval()
    scs, lbs = [], []
    for img, lbl in tqdm(loader, desc=desc, leave=False):
        scs.append(torch.sigmoid(model(img.to(DEVICE))).squeeze(-1).cpu())
        lbs.append(lbl)
    return torch.cat(scs).numpy(), torch.cat(lbs).numpy().astype(int)


val_loader, test_loader = split_loader("val"), split_loader("test")


def gather_family(prefix: str):
    pv, pt, vl, tl = {}, {}, None, None
    for n in SEEDS:
        ckpt_path = CKPT / f"{prefix}_seed{n}" / "best.pt"
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        m = build_resnet18()
        miss, unexp = m.load_state_dict(ck["model_state_dict"], strict=True)
        assert not miss and not unexp, f"{prefix} seed{n}: missing={miss} unexpected={unexp}"
        m = m.to(DEVICE)
        pv[n], vl = gather(m, val_loader, f"{prefix[:18]} s{n} val")
        pt[n], tl = gather(m, test_loader, f"{prefix[:18]} s{n} test")
        print(f"  loaded {prefix}_seed{n} (best_f1_val={ck.get('best_f1_val'):.4f}, "
              f"epoch={ck.get('epoch')})")
    return pv, pt, vl, tl


print("\nRe-evaluating CVCSPC fine-tune seeds from weights ...")
cv_pv, cv_pt, val_labels, test_labels = gather_family("shallow_squat_cvcspc_finetune")
print("Re-evaluating supervised baseline seeds from weights ...")
bl_pv, bl_pt, _, _ = gather_family("shallow_squat_baseline")


def ensemble_eval(pv, pt):
    ev = np.mean([pv[n] for n in SEEDS], axis=0)
    et = np.mean([pt[n] for n in SEEDS], axis=0)
    t, _ = threshold_sweep(val_labels, ev)
    return ev, et, t, f1_per_error(val_labels, (ev >= t).astype(int)), f1_per_error(test_labels, (et >= t).astype(int))


cv_ev, cv_et, cv_t, cv_vf1, cv_tf1 = ensemble_eval(cv_pv, cv_pt)
bl_ev, bl_et, bl_t, bl_vf1, bl_tf1 = ensemble_eval(bl_pv, bl_pt)
cv_pr = pr_auc_per_error(test_labels, cv_et)
cv_cm = confusion_matrix_per_error(test_labels, (cv_et >= cv_t).astype(int))
cv_per_seed = {n: f1_per_error(test_labels, (cv_pt[n] >= cv_t).astype(int)) for n in SEEDS}
best_seed = max(SEEDS, key=lambda n: f1_per_error(val_labels, (cv_pv[n] >= 0.5).astype(int)))
cv_single = f1_per_error(test_labels, (cv_pt[best_seed] >= cv_t).astype(int))

# ── compare against the committed results.pkl ──────────────────────────────────
REF = pickle.loads((ROOT / ".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl").read_bytes())
checks = []


def chk(name, got, ref, tol=2e-3):
    ok = abs(got - ref) <= tol
    checks.append(ok)
    print(f"  {'OK ' if ok else '!! MISMATCH'} {name:34s} re-eval={got:.4f}  results.pkl={ref:.4f}  d={got-ref:+.4f}")


print("\n" + "=" * 78)
print("CERTAINTY CROSS-CHECK - re-eval-from-weights vs committed results.pkl")
print("=" * 78)
chk("CVCSPC ensemble test F1", cv_tf1, REF["test_f1"]["cvcspc_ensemble"])
chk("CVCSPC ensemble val F1", cv_vf1, REF["cvcspc_ensemble_val_f1"])
chk("CVCSPC val-tuned threshold", cv_t, REF["best_threshold"])
chk("CVCSPC test PR-AUC", cv_pr, REF["test_pr_auc"])
chk("CVCSPC single best-seed F1", cv_single, REF["test_f1"]["single_model"])
for n in SEEDS:
    chk(f"CVCSPC per-seed test F1 [{n}]", cv_per_seed[n], REF["test_f1"]["per_seed"][n])
chk("baseline ensemble test F1", bl_tf1, REF["baseline_control"]["test_f1"])
chk("baseline ensemble val F1", bl_vf1, REF["baseline_control"]["ensemble_val_f1"])

# per-clip score fidelity - the tight check that the cached scores ARE these weights' outputs
d_cv_t = float(np.max(np.abs(cv_et - REF["ensemble_test_scores"])))
d_cv_v = float(np.max(np.abs(cv_ev - REF["ensemble_val_scores"])))
d_bl_t = float(np.max(np.abs(bl_et - REF["baseline_control"]["ensemble_test_scores"])))
print(f"\n  per-clip ensemble-score max|Δ| vs results.pkl:")
print(f"    CVCSPC test {d_cv_t:.2e} | CVCSPC val {d_cv_v:.2e} | baseline test {d_bl_t:.2e}")
score_ok = max(d_cv_t, d_cv_v, d_bl_t) < 5e-3
checks.append(score_ok)
print(f"    {'OK ' if score_ok else '!! '} per-clip scores reproduce the cache (max|Δ| < 5e-3)")

ref_cm = np.asarray(REF["test_confusion"])
cm_ok = np.array_equal(cv_cm, ref_cm)
checks.append(cm_ok)
print(f"\n  confusion (CVCSPC test): re-eval {cv_cm.tolist()} vs results.pkl {ref_cm.tolist()} "
      f"-> {'OK match' if cm_ok else '!! differ'}")

print("\n" + "=" * 78)
print(f"SHALLOW RE-EVAL VERDICT: {sum(checks)}/{len(checks)} checks pass - "
      + ("ALL re-derived from weights match the committed cache [OK]" if all(checks)
         else "!! MISMATCH(es) above - investigate"))
print("=" * 78)

out = {
    "cv_ensemble_val_scores": cv_ev, "cv_ensemble_test_scores": cv_et,
    "bl_ensemble_val_scores": bl_ev, "bl_ensemble_test_scores": bl_et,
    "val_labels": val_labels, "test_labels": test_labels,
    "cv_threshold": cv_t, "bl_threshold": bl_t,
    "cv_test_f1": cv_tf1, "cv_val_f1": cv_vf1, "cv_test_prauc": cv_pr,
    "cv_per_seed_test_f1": cv_per_seed, "cv_single_f1": cv_single, "cv_confusion": cv_cm,
    "bl_test_f1": bl_tf1, "bl_val_f1": bl_vf1,
    "score_maxabs_diff": {"cv_test": d_cv_t, "cv_val": d_cv_v, "bl_test": d_bl_t},
    "all_checks_pass": bool(all(checks)),
    "device": "cpu", "source": "reeval_shallow.py - weights reloaded from Drive",
}
outpath = ROOT / "docs/audit/data/reeval_shallow.pkl"
outpath.write_bytes(pickle.dumps(out))
print(f"\nsaved {outpath.relative_to(ROOT)}")
