"""Re-evaluation from weights for the Squat and OHP video models (R(2+1)D-18).

Loads every video best.pt from Drive, re-runs the official validation and test evaluation through the
same squat.py and ohp.py pipeline used in training (cv2 decode on torchvision >= 0.26, the same
decode path as the Colab environment that produced the committed results.pkl), rebuilds the
mean-of-sigmoids ensembles and the per-error val-tuned thresholds, and compares the re-derived F1 and
per-clip scores against the cache.

Runs on CPU. Resume-safe: each (run, split) score array is cached to scratch, so a disconnect or
re-run skips finished work. About 3 to 4k clip decodes, slow on CPU (tens of minutes); intended as a
background job. The companion overfitting_audit_colab.py is the GPU version.

Paths via env: FITNOVA_CKPT, FITNOVA_SQUAT_ROOT, FITNOVA_SQUAT_VIDEOS, FITNOVA_OHP_ROOT,
FITNOVA_OHP_VIDEOS, FITNOVA_REEVAL_CACHE.
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

from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
from backend.training.aqa.eval.metrics import (
    confusion_matrix_per_error,
    f1_per_error,
    pr_auc_per_error,
    threshold_sweep,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x

CKPT = Path(os.environ.get("FITNOVA_CKPT", r"G:/My Drive/FitNova/checkpoints"))
_P1 = (r"C:/Users/tsh_x/Desktop/FitNova Application/Fitness-AQA"
       r"/Fitness-AQA_dataset_release-20260518T073812Z-3-001")
SQUAT_ROOT = Path(os.environ.get("FITNOVA_SQUAT_ROOT", _P1))
SQUAT_VIDEOS = Path(os.environ.get(
    "FITNOVA_SQUAT_VIDEOS",
    _P1 + "/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos_extracted/videos"))
OHP_ROOT = Path(os.environ.get("FITNOVA_OHP_ROOT", _P1))
OHP_VIDEOS = Path(os.environ.get("FITNOVA_OHP_VIDEOS",
                                 r"C:/Users/tsh_x/fitnova_audit_scratch/ohp_videos/videos"))
CACHE = Path(os.environ.get("FITNOVA_REEVAL_CACHE", r"C:/Users/tsh_x/fitnova_audit_scratch/reeval_video_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)


def build_video(head: str) -> nn.Module:
    """R(2+1)D-18 architecture only (weights=None); the trained state_dict is loaded over it."""
    from torchvision.models.video import r2plus1d_18

    m = r2plus1d_18(weights=None)
    assert m.fc.in_features == 512
    m.fc = nn.Linear(512, 2) if head == "linear" else nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    return m


def make_loader(kind: str, split: str):
    # Decode-cache the short-side-resized frames so shared clips aren't re-decoded per seed.
    cache_dir = str(CACHE.parent / "decode_cache" / kind)
    os.makedirs(cache_dir, exist_ok=True)
    if kind == "squat":
        ds = SquatKIEKFEDataset(split=split, drive_root=str(SQUAT_ROOT), videos_root=str(SQUAT_VIDEOS),
                                train_aug=False, cache_dir=cache_dir)
    else:
        ds = OHPElbowsKneesDataset(split=split, drive_root=str(OHP_ROOT), videos_root=str(OHP_VIDEOS),
                                   train_aug=False, cache_dir=cache_dir)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)


def gather(run_name: str, head: str, kind: str, split: str):
    """Sigmoid scores (N,2) + labels for one model/split, cached to scratch for resume-safety."""
    cache_f = CACHE / f"{run_name}__{split}.npz"
    if cache_f.exists():
        z = np.load(cache_f)
        return z["scores"], z["labels"]
    ckpt_path = _RUNS[run_name]["ckpt"]
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_video(head)
    miss, unexp = model.load_state_dict(ck["model_state_dict"], strict=True)
    assert not miss and not unexp, f"{run_name}: missing={miss} unexpected={unexp}"
    model.eval().to(DEVICE)
    loader = make_loader(kind, split)
    scs, lbs = [], []
    for clip, label in tqdm(loader, desc=f"{run_name[:22]} {split}", leave=False):
        scs.append(torch.sigmoid(model(clip.to(DEVICE))).cpu())
        lbs.append(label)
    scores = torch.cat(scs).numpy().astype(np.float64)
    labels = torch.cat(lbs).numpy().astype(int)
    np.savez(cache_f, scores=scores, labels=labels)
    del model
    return scores, labels


_RUNS = {
    "squat_baseline": {"ckpt": CKPT / "phase03/r2plus1d18_squat_supervised_v1/best.pt", "head": "linear", "kind": "squat"},
    "squat_mdssl_42": {"ckpt": CKPT / "phase04/md_finetune_seed42/best.pt", "head": "drop", "kind": "squat"},
    "squat_mdssl_1337": {"ckpt": CKPT / "phase04/md_finetune_seed1337/best.pt", "head": "drop", "kind": "squat"},
    "squat_mdssl_7": {"ckpt": CKPT / "phase04/md_finetune_seed7/best.pt", "head": "drop", "kind": "squat"},
    "ohp_baseline": {"ckpt": CKPT / "phase06/ohp_supervised_v1/best.pt", "head": "linear", "kind": "ohp"},
    "ohp_mdssl_42": {"ckpt": CKPT / "phase06/ohp_md_finetune_seed42/best.pt", "head": "drop", "kind": "ohp"},
    "ohp_mdssl_1337": {"ckpt": CKPT / "phase06/ohp_md_finetune_seed1337/best.pt", "head": "drop", "kind": "ohp"},
}


def get(run, split):
    info = _RUNS[run]
    return gather(run, "linear" if info["head"] == "linear" else "drop", info["kind"], split)


def ensemble(runs, split):
    arrs = [get(r, split)[0] for r in runs]
    _, labels = get(runs[0], split)
    return np.mean(arrs, axis=0), labels


def per_error_eval(ev, vl, et, tl, errs):
    """Tune per-error threshold on val ensemble, return dict of f1/thr/prauc/gap per error."""
    out = {}
    for ci, name in errs:
        t, _ = threshold_sweep(vl[:, ci], ev[:, ci])
        vf = f1_per_error(vl[:, ci], (ev[:, ci] >= t).astype(int))
        tf = f1_per_error(tl[:, ci], (et[:, ci] >= t).astype(int))
        out[name] = dict(thr=t, val_f1=vf, test_f1=tf,
                         val_prauc=pr_auc_per_error(vl[:, ci], ev[:, ci]),
                         test_prauc=pr_auc_per_error(tl[:, ci], et[:, ci]),
                         test_cm=confusion_matrix_per_error(tl[:, ci], (et[:, ci] >= t).astype(int)))
    return out


def banner(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


REF_SQUAT = pickle.loads((ROOT / ".planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl").read_bytes())
REF_OHP = pickle.loads((ROOT / ".planning/phases/06-overhead-press/figures/results.pkl").read_bytes())
REF_SQUAT_BL = pickle.loads((ROOT / "docs/audit/data/squat_baseline_phase03_results.pkl").read_bytes())

results = {}
checks = []


def cmp(name, got, ref, tol=0.02):
    ok = abs(got - ref) <= tol
    checks.append(ok)
    print(f"  {'OK ' if ok else '!! MISMATCH'} {name:32s} re-eval={got:.4f}  cache={ref:.4f}  d={got-ref:+.4f}")


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ── SQUAT MD-SSL ensemble (3 seeds) ────────────────────────────────────────────
banner("SQUAT - MD-SSL 3-seed ensemble (re-eval from weights vs results.pkl)")
ev, vl = ensemble(["squat_mdssl_42", "squat_mdssl_1337", "squat_mdssl_7"], "val")
et, tl = ensemble(["squat_mdssl_42", "squat_mdssl_1337", "squat_mdssl_7"], "test")
sq = per_error_eval(ev, vl, et, tl, [(0, "KIE"), (1, "KFE")])
for e in ("KIE", "KFE"):
    cmp(f"test F1 {e}", sq[e]["test_f1"], REF_SQUAT["final"]["test_f1"][e])
    cmp(f"threshold {e}", sq[e]["thr"], REF_SQUAT["final"]["thresholds"][e], tol=0.05)
d_t = maxabs(et, REF_SQUAT["raw"]["ens_test_scores"]); d_v = maxabs(ev, REF_SQUAT["raw"]["ens_val_scores"])
print(f"  per-clip ensemble score max|Δ| vs cache: test {d_t:.3e}  val {d_v:.3e}")
results["squat_mdssl"] = dict(per_error=sq, score_maxabs=dict(test=d_t, val=d_v),
                              ens_val=ev, ens_test=et, val_labels=vl, test_labels=tl)

# ── SQUAT supervised baseline (single seed) ────────────────────────────────────
banner("SQUAT - supervised baseline (phase03)")
bvsc, bvl = get("squat_baseline", "val"); btsc, btl = get("squat_baseline", "test")
sqb = per_error_eval(bvsc, bvl, btsc, btl, [(0, "KIE"), (1, "KFE")])
for ci, e in [(0, "kie"), (1, "kfe")]:
    cmp(f"baseline test F1 {e.upper()}", sqb[e.upper()]["test_f1"], REF_SQUAT_BL[f"test_f1_{e}"])
print(f"  baseline per-clip score max|Δ| vs cache: test {maxabs(btsc, REF_SQUAT_BL['test_scores']):.3e}")
results["squat_baseline"] = dict(per_error=sqb)

# ── OHP MD-SSL ensemble (2 seeds) ──────────────────────────────────────────────
banner("OHP - MD-SSL 2-seed ensemble (re-eval from weights vs results.pkl)")
oev, ovl = ensemble(["ohp_mdssl_42", "ohp_mdssl_1337"], "val")
oet, otl = ensemble(["ohp_mdssl_42", "ohp_mdssl_1337"], "test")
oh = per_error_eval(oev, ovl, oet, otl, [(0, "elbows"), (1, "knees")])
for e in ("elbows", "knees"):
    cmp(f"test F1 {e}", oh[e]["test_f1"], REF_OHP["test_f1"][e])
    cmp(f"threshold {e}", oh[e]["thr"], REF_OHP["thresholds"][e], tol=0.05)
d_ot = maxabs(oet, REF_OHP["ensemble_test_scores"]); d_ov = maxabs(oev, REF_OHP["ensemble_val_scores"])
print(f"  per-clip ensemble score max|Δ| vs cache: test {d_ot:.3e}  val {d_ov:.3e}")
results["ohp_mdssl"] = dict(per_error=oh, score_maxabs=dict(test=d_ot, val=d_ov),
                            ens_val=oev, ens_test=oet, val_labels=ovl, test_labels=otl)

# ── OHP supervised baseline - fills the gap the committed pkl never stored ──────
banner("OHP - supervised baseline (phase06) - re-eval gives the baseline val totest gap")
obvsc, obvl = get("ohp_baseline", "val"); obtsc, obtl = get("ohp_baseline", "test")
ohb = per_error_eval(obvsc, obvl, obtsc, obtl, [(0, "elbows"), (1, "knees")])
for e in ("elbows", "knees"):
    cmp(f"baseline test F1 {e}", ohb[e]["test_f1"], REF_OHP["baseline_control"][e], tol=0.03)
    print(f"    {e}: val_F1 {ohb[e]['val_f1']:.4f} -> test_F1 {ohb[e]['test_f1']:.4f} "
          f"(gap {ohb[e]['val_f1']-ohb[e]['test_f1']:+.4f})")
results["ohp_baseline"] = dict(per_error=ohb)

banner(f"VIDEO RE-EVAL VERDICT - {sum(checks)}/{len(checks)} numeric checks within tolerance")
print("ALL within tolerance [OK]" if all(checks) else "!! see MISMATCH rows above")
results["all_checks_pass"] = bool(all(checks))
results["n_checks"] = len(checks)
results["n_pass"] = int(sum(checks))
out = ROOT / "docs/audit/data/reeval_video.pkl"
out.write_bytes(pickle.dumps(results))
print(f"\nsaved {out.relative_to(ROOT)}")
