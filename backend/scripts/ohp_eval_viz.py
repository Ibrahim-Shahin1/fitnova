"""OHP evaluation + training figure pack (Phase 6 Plan 05), built from results.pkl.

SSL convergence (loss + effective_rank + linear-probe), PR curves, confusion matrices,
ensemble score distributions — all from the Plan-04 results.pkl (real test scores on the
official 339-clip split). Run: python backend/scripts/ohp_eval_viz.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

PKL = Path(".planning/phases/06-overhead-press/figures/results.pkl")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)
ERRORS = [("Elbows", 0, "elbows"), ("Knees", 1, "knees")]


def ssl_curves(d):
    mh = d["ssl_metrics_history"]
    ep = [m["epoch"] for m in mh]
    loss = [m["ssl_loss_mean"] for m in mh]
    erank = [m["effective_rank"] for m in mh]
    lp = d["ssl_linear_probe_history"]
    lep = [m["epoch"] for m in lp]
    lmac = [m["linear_probe_f1_macro"] for m in lp]
    lelb = [m["linear_probe_f1_kie"] for m in lp]
    lkne = [m["linear_probe_f1_kfe"] for m in lp]
    best = int(np.argmax(lmac))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    a0 = ax[0]
    a0.plot(ep, loss, "C0-", label="SSL loss")
    a0.set_xlabel("epoch"); a0.set_ylabel("SSL triplet loss", color="C0")
    a2 = a0.twinx()
    a2.plot(ep, erank, "C2-", alpha=0.8, label="effective_rank")
    a2.axhline(3, color="C3", ls=":", alpha=0.6, label="collapse floor (~3)")
    a2.set_ylabel("effective_rank", color="C2"); a2.set_ylim(0, max(erank) + 2)
    a0.set_title("MD-SSL pretrain — loss falls, effective_rank stays healthy (no collapse)")
    a0.legend(loc="upper right", fontsize=8); a2.legend(loc="center right", fontsize=8)

    a1 = ax[1]
    a1.plot(lep, lmac, "k-o", ms=4, label="linear-probe macro")
    a1.plot(lep, lelb, "C0-o", ms=3, label="Elbows")
    a1.plot(lep, lkne, "C2-o", ms=3, label="Knees")
    a1.axvline(lep[best], color="k", ls="--", alpha=0.5, label=f"best ep{lep[best]} -> backbone.pt")
    a1.set_xlabel("epoch"); a1.set_ylabel("linear-probe F1 (val)"); a1.set_ylim(0, 1)
    a1.set_title(f"Linear-probe convergence (frozen backbone) — peak macro {lmac[best]:.4f} @ ep{lep[best]}")
    a1.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ohp_ssl_curves.png", dpi=120, bbox_inches="tight")
    print("saved ohp_ssl_curves.png")


def pr_curves(d):
    s = np.asarray(d["ensemble_test_scores"]); y = np.asarray(d["test_labels"]).astype(int)
    th = d["thresholds"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (name, col, key) in enumerate(ERRORS):
        p, r, _ = precision_recall_curve(y[:, col], s[:, col])
        ap = average_precision_score(y[:, col], s[:, col])
        base = y[:, col].mean()
        ax[i].plot(r, p, "C0-")
        ax[i].axhline(base, color="grey", ls=":", label=f"base rate {base:.2f}")
        pred = (s[:, col] >= th[key]).astype(int)
        tp = int(((pred == 1) & (y[:, col] == 1)).sum()); fp = int(((pred == 1) & (y[:, col] == 0)).sum())
        fn = int(((pred == 0) & (y[:, col] == 1)).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        ax[i].plot(rec, prec, "C3o", ms=9, label=f"operating pt (thr {th[key]:.2f})")
        ax[i].set_xlabel("recall"); ax[i].set_ylabel("precision"); ax[i].set_xlim(0, 1); ax[i].set_ylim(0, 1.02)
        ax[i].set_title(f"{name} — PR (AP {ap:.3f})"); ax[i].legend(fontsize=8)
    fig.suptitle("OHP ensemble — precision-recall on the official test split", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "ohp_pr_curves.png", dpi=120, bbox_inches="tight")
    print("saved ohp_pr_curves.png")


def confusion(d):
    cm = d["test_confusion"]; f1 = d["test_f1"]; th = d["thresholds"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for i, (name, col, key) in enumerate(ERRORS):
        m = np.asarray(cm[key])
        ax[i].imshow(m, cmap="Blues")
        for (rr, cc), v in np.ndenumerate(m):
            ax[i].text(cc, rr, str(v), ha="center", va="center",
                       color="white" if v > m.max() / 2 else "black", fontsize=13)
        ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(["pred -", "pred +"])
        ax[i].set_yticks([0, 1]); ax[i].set_yticklabels(["true -", "true +"])
        ax[i].set_title(f"{name} — F1 {f1[key]:.3f} (thr {th[key]:.2f})")
    fig.suptitle("OHP ensemble — confusion at val-tuned thresholds (official test split)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "ohp_confusion.png", dpi=120, bbox_inches="tight")
    print("saved ohp_confusion.png")


def score_dists(d):
    s = np.asarray(d["ensemble_test_scores"]); y = np.asarray(d["test_labels"]).astype(int)
    th = d["thresholds"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (name, col, key) in enumerate(ERRORS):
        ax[i].hist(s[y[:, col] == 0, col], bins=22, alpha=0.6, color="C0", label="negative")
        ax[i].hist(s[y[:, col] == 1, col], bins=22, alpha=0.6, color="C1", label="positive")
        ax[i].axvline(th[key], color="k", ls="--", label=f"threshold {th[key]:.2f}")
        ax[i].set_xlabel("ensemble score"); ax[i].set_ylabel("clips"); ax[i].set_title(f"{name} — score separation")
        ax[i].legend(fontsize=8)
    fig.suptitle("OHP ensemble — score distributions (positives vs negatives), official test split", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "ohp_score_distributions.png", dpi=120, bbox_inches="tight")
    print("saved ohp_score_distributions.png")


if __name__ == "__main__":
    d = pickle.load(open(PKL, "rb"))
    ssl_curves(d)
    pr_curves(d)
    confusion(d)
    score_dists(d)
