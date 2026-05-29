"""Per-seed OHP fine-tune training curves (Phase 6 Plan 04/05).

Reads the per-seed metrics_history (from a {seed: run_md_finetune_epoch-result} pickle, or
from the full results.pkl's finetune_seeds). Shows the val/train overfit ratio + val F1 per
epoch with the best-epoch marker — the overfit-control story (early-stop keeps the pre-overfit
best epoch).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_seed_histories(path: str) -> dict[int, list[dict]]:
    d = pickle.load(open(path, "rb"))
    if "finetune_seeds" in d:
        seeds = d["finetune_seeds"]
    else:
        seeds = {k: (v["metrics_history"] if isinstance(v, dict) and "metrics_history" in v else v) for k, v in d.items()}
    return {int(k): v for k, v in seeds.items()}


def main(pkl_path: str) -> None:
    hist = _load_seed_histories(pkl_path)
    seeds = sorted(hist)
    fig, axes = plt.subplots(len(seeds), 2, figsize=(13, 4.2 * len(seeds)), squeeze=False)
    for r, s in enumerate(seeds):
        mh = hist[s]
        ep = [m["epoch"] for m in mh]
        tr = [m["train_loss_mean"] for m in mh]
        vl = [m["val_loss_mean"] for m in mh]
        ratio = [m["val_train_loss_ratio"] for m in mh]
        macro = [m["val_macro_f1"] for m in mh]
        elb = [m["val_f1_kie"] for m in mh]
        kne = [m["val_f1_kfe"] for m in mh]
        best = int(np.argmax(macro))

        ax = axes[r][0]
        ax.plot(ep, tr, "C0-o", ms=3, label="train loss")
        ax.plot(ep, vl, "C1-o", ms=3, label="val loss")
        ax.axvline(ep[best], color="k", ls="--", alpha=0.5, label=f"best ep{ep[best]} (kept)")
        ax2 = ax.twinx()
        ax2.plot(ep, ratio, "C3-", alpha=0.7, label="val/train ratio")
        ax2.axhline(10, color="C3", ls=":", alpha=0.5)
        ax.set_title(f"seed {s} — loss + overfit ratio")
        ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax2.set_ylabel("val/train ratio")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

        ax = axes[r][1]
        ax.plot(ep, elb, "C0-o", ms=3, label="Elbows F1")
        ax.plot(ep, kne, "C2-o", ms=3, label="Knees F1")
        ax.plot(ep, macro, "k-o", ms=3, label="macro F1")
        ax.axvline(ep[best], color="k", ls="--", alpha=0.5)
        ax.set_title(f"seed {s} — val F1 (best macro {macro[best]:.4f} @ ep{ep[best]})")
        ax.set_xlabel("epoch"); ax.set_ylabel("val F1"); ax.set_ylim(0, 1); ax.legend(fontsize=8)

    fig.suptitle("OHP fine-tune — overfit control: val/train climbs after the best epoch; "
                 "8-epoch early-stop keeps the pre-overfit best.pt", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path("docs/figures/ohp_finetune_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ohp_results.pkl")
