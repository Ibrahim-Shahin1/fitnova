"""Certainty visualization - re-derived-from-weights vs committed-cache agreement.

Reads the local re-eval outputs (`reeval_shallow.pkl`, `reeval_video.pkl` when present) and the
committed `results.pkl`, then renders one figure proving the cache IS the weights' output:
  (left)  per-clip ensemble score - cached vs re-derived, on the y=x diagonal (all models pooled).
  (right) per-error test F1 - cache vs re-eval (identical bars).
Prints the per-model max|Δ| + F1 deltas. Saves docs/figures/audit_reeval_certainty.png.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/audit/data"
FIG = ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(p):
    return pickle.loads(Path(p).read_bytes())


REF_SQUAT = load(ROOT / ".planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl")
REF_OHP = load(ROOT / ".planning/phases/06-overhead-press/figures/results.pkl")
REF_SHAL = load(ROOT / ".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl")
SH = load(DATA / "reeval_shallow.pkl") if (DATA / "reeval_shallow.pkl").exists() else None
VID = load(DATA / "reeval_video.pkl") if (DATA / "reeval_video.pkl").exists() else None

# rows: (label, cached_scores, rederived_scores, cache_f1, reeval_f1, color)
rows = []
if SH is not None:
    rows.append(("Shallow CVCSPC", REF_SHAL["ensemble_test_scores"], SH["cv_ensemble_test_scores"],
                 REF_SHAL["test_f1"]["cvcspc_ensemble"], SH["cv_test_f1"], "#2CA02C"))
    rows.append(("Shallow baseline", REF_SHAL["baseline_control"]["ensemble_test_scores"],
                 SH["bl_ensemble_test_scores"], REF_SHAL["baseline_control"]["test_f1"], SH["bl_test_f1"], "#98DF8A"))
if VID is not None:
    sq, oh = VID["squat_mdssl"], VID["ohp_mdssl"]
    rows.append(("Squat KIE", REF_SQUAT["raw"]["ens_test_scores"][:, 0], sq["ens_test"][:, 0],
                 REF_SQUAT["final"]["test_f1"]["KIE"], sq["per_error"]["KIE"]["test_f1"], "#4C72B0"))
    rows.append(("Squat KFE", REF_SQUAT["raw"]["ens_test_scores"][:, 1], sq["ens_test"][:, 1],
                 REF_SQUAT["final"]["test_f1"]["KFE"], sq["per_error"]["KFE"]["test_f1"], "#6FA8DC"))
    rows.append(("OHP Elbows", REF_OHP["ensemble_test_scores"][:, 0], oh["ens_test"][:, 0],
                 REF_OHP["test_f1"]["elbows"], oh["per_error"]["elbows"]["test_f1"], "#C44E52"))
    rows.append(("OHP Knees", REF_OHP["ensemble_test_scores"][:, 1], oh["ens_test"][:, 1],
                 REF_OHP["test_f1"]["knees"], oh["per_error"]["knees"]["test_f1"], "#E59398"))

if not rows:
    print("No re-eval pkls found yet (need reeval_shallow.pkl and/or reeval_video.pkl).")
    sys.exit(0)

print(f"{'model · error':18s} {'cache F1':>9s} {'reeval F1':>10s} {'dF1':>8s} {'per-clip max|Δ|':>16s}")
print("-" * 70)
for lab, cs, rs, cf, rf, _c in rows:
    cs = np.asarray(cs, float); rs = np.asarray(rs, float)
    print(f"{lab:18s} {cf:9.4f} {rf:10.4f} {rf-cf:+8.4f} {float(np.max(np.abs(cs-rs))):16.2e}")

fig, (axS, axB) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})

# Left - cached vs re-derived per-clip scores
for lab, cs, rs, _cf, _rf, c in rows:
    axS.scatter(np.asarray(cs, float), np.asarray(rs, float), s=10, alpha=0.5, color=c, label=lab)
axS.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="y = x (exact)")
axS.set_xlabel("committed cache score (results.pkl)")
axS.set_ylabel("re-derived from weights")
axS.set_title("Per-clip ensemble scores: cache vs re-eval from weights\n(points on the diagonal mean the cache is the weights' output)")
axS.set_xlim(-0.02, 1.02); axS.set_ylim(-0.02, 1.02); axS.legend(fontsize=7, loc="upper left")
axS.grid(alpha=0.25)

# Right - cache vs re-eval test F1
labels = [r[0] for r in rows]
cf = [r[3] for r in rows]; rf = [r[4] for r in rows]
x = np.arange(len(rows)); w = 0.38
axB.bar(x - w / 2, cf, w, color="#999999", label="results.pkl F1")
axB.bar(x + w / 2, rf, w, color="#2CA02C", label="re-eval-from-weights F1")
for i in range(len(rows)):
    axB.text(i, max(cf[i], rf[i]) + 0.01, f"Δ{rf[i]-cf[i]:+.3f}", ha="center", fontsize=8)
axB.set_xticks(x); axB.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
axB.set_ylim(0, 1.05); axB.set_ylabel("test F1"); axB.legend()
axB.set_title("Test F1 - committed cache vs re-derived from the weights")
axB.grid(alpha=0.25, axis="y")

plt.tight_layout()
out = FIG / "audit_reeval_certainty.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved {out.relative_to(ROOT)}")
