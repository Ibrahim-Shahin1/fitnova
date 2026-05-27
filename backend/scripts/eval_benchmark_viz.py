"""Squat benchmark evaluation visualizations (Fitness-AQA official test split).

Reproducible: reads the per-clip scores produced by scoring the served model on the
244-clip official test set (docs/eval/squat_test_scores.csv) and regenerates every
figure into docs/figures/ — no re-scoring needed.

Figures (defense deliverable):
  1. squat_score_distributions.png — KFE/KIE score histograms, positives vs negatives.
  2. squat_pr_curves.png           — precision-recall curves + operating point + AP.
  3. squat_confusion_matrices.png  — confusion at the production thresholds.
  4. squat_f1_vs_paper.png         — our F1 vs Parmar (Kinetics baseline / MD-SSL).

Run:  python backend/scripts/eval_benchmark_viz.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

_CSV = "docs/eval/squat_test_scores.csv"
_OUT = "docs/figures"

# Production thresholds (val-tuned, Phase 4 / D-01).
KIE_THR, KFE_THR = 0.614, 0.385

# Published comparison points (Parmar et al., ECCV 2022 — from the approved plan).
PAPER = {
    "KFE": {"Kinetics baseline": 0.818, "MD-SSL (paper)": 0.834},
    "KIE": {"Kinetics baseline": 0.297, "MD-SSL (paper)": 0.419},
}
# Phase-4 official (Colab eval, torchvision decode) — the headline academic number.
OFFICIAL = {"KFE": 0.841, "KIE": 0.420, "macro": 0.6304}


def _load():
    kie_s, kfe_s, kie_l, kfe_l, ids = [], [], [], [], []
    with open(_CSV) as f:
        for row in csv.DictReader(f):
            ids.append(row["clip_id"])
            kie_s.append(float(row["kie_score"])); kfe_s.append(float(row["kfe_score"]))
            kie_l.append(int(row["kie_label"]));   kfe_l.append(int(row["kfe_label"]))
    return (np.array(ids), np.array(kie_s), np.array(kfe_s),
            np.array(kie_l), np.array(kfe_l))


def _f1(scores, labels, thr):
    return f1_score(labels, (scores >= thr).astype(int))


def fig_distributions(kie_s, kfe_s, kie_l, kfe_l):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, s, l, thr) in zip(axes, [
        ("KFE (knees forward)", kfe_s, kfe_l, KFE_THR),
        ("KIE (knees inward)",  kie_s, kie_l, KIE_THR),
    ]):
        bins = np.linspace(0, 1, 26)
        ax.hist(s[l == 0], bins=bins, alpha=0.6, label=f"negative (n={int((l==0).sum())})", color="#4C9F70")
        ax.hist(s[l == 1], bins=bins, alpha=0.6, label=f"positive (n={int((l==1).sum())})", color="#D7263D")
        ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold {thr}")
        ax.set_title(f"{name}  (F1={_f1(s,l,thr):.3f})")
        ax.set_xlabel("model confidence"); ax.set_ylabel("clip count"); ax.legend(fontsize=8)
    fig.suptitle("Squat error-score distributions on the Fitness-AQA test split (n=244)\n"
                 "Separation of positive vs negative clips per error head", fontsize=11)
    fig.tight_layout()
    p = os.path.join(_OUT, "squat_score_distributions.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_pr(kie_s, kfe_s, kie_l, kfe_l):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, s, l, thr) in zip(axes, [
        ("KFE", kfe_s, kfe_l, KFE_THR), ("KIE", kie_s, kie_l, KIE_THR),
    ]):
        prec, rec, _ = precision_recall_curve(l, s)
        ap = average_precision_score(l, s)
        ax.plot(rec, prec, color="#1B998B", lw=2)
        op_p = precision_score(l, (s >= thr).astype(int), zero_division=0)
        op_r = recall_score(l, (s >= thr).astype(int), zero_division=0)
        ax.scatter([op_r], [op_p], color="#D7263D", zorder=5,
                   label=f"operating point (thr={thr})\nP={op_p:.2f} R={op_r:.2f}")
        ax.axhline((l == 1).mean(), color="gray", ls=":", lw=1, label=f"baseline ({(l==1).mean():.2f})")
        ax.set_title(f"{name}  PR-curve  (AP={ap:.3f})")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Precision-Recall on the Fitness-AQA test split — threshold-free (AP) + operating point",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(_OUT, "squat_pr_curves.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_confusion(kie_s, kfe_s, kie_l, kfe_l):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, s, l, thr) in zip(axes, [
        ("KFE", kfe_s, kfe_l, KFE_THR), ("KIE", kie_s, kie_l, KIE_THR),
    ]):
        pred = (s >= thr).astype(int)
        cm = np.array([[int(((pred == 0) & (l == 0)).sum()), int(((pred == 1) & (l == 0)).sum())],
                       [int(((pred == 0) & (l == 1)).sum()), int(((pred == 1) & (l == 1)).sum())]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred -", "pred +"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true -", "true +"])
        ax.set_title(f"{name}  (F1={_f1(s,l,thr):.3f}, thr={thr})")
    fig.suptitle("Confusion matrices on the Fitness-AQA test split (production thresholds)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(_OUT, "squat_confusion_matrices.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_f1_vs_paper():
    groups = ["KFE", "KIE", "macro"]
    ours = [OFFICIAL["KFE"], OFFICIAL["KIE"], OFFICIAL["macro"]]
    md = [PAPER["KFE"]["MD-SSL (paper)"], PAPER["KIE"]["MD-SSL (paper)"],
          (PAPER["KFE"]["MD-SSL (paper)"] + PAPER["KIE"]["MD-SSL (paper)"]) / 2]
    base = [PAPER["KFE"]["Kinetics baseline"], PAPER["KIE"]["Kinetics baseline"],
            (PAPER["KFE"]["Kinetics baseline"] + PAPER["KIE"]["Kinetics baseline"]) / 2]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w, ours, w, label="Ours (MD-SSL, Phase 4)", color="#1B998B")
    b2 = ax.bar(x,     md,   w, label="Parmar MD-SSL (paper)", color="#2E86AB")
    b3 = ax.bar(x + w, base, w, label="Parmar Kinetics baseline", color="#BDBDBD")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01,
                    f"{r.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("F1-score (official test split)"); ax.set_ylim(0, 1.0)
    ax.set_title("Squat form-error detection: ours vs Parmar et al. (identical metric / split)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "squat_f1_vs_paper.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    os.makedirs(_OUT, exist_ok=True)
    ids, kie_s, kfe_s, kie_l, kfe_l = _load()
    print(f"loaded {len(ids)} clips from {_CSV}")
    for fn in (
        fig_distributions(kie_s, kfe_s, kie_l, kfe_l),
        fig_pr(kie_s, kfe_s, kie_l, kfe_l),
        fig_confusion(kie_s, kfe_s, kie_l, kfe_l),
        fig_f1_vs_paper(),
    ):
        print("wrote", fn)


if __name__ == "__main__":
    main()
