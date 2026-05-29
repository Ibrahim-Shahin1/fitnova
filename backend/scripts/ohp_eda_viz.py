"""OHP dataset EDA figure (Phase 6 Plan 05) — from the local Fitness-AQA OHP archive.

Class balance per split + the Elbows/Knees co-occurrence (the two errors are largely
independent — the joint head is justified by the shared MD-SSL representation, not
co-occurrence). Reads the offline archive (research-use only; not committed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAB = Path("Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/"
           "Fitness-AQA_dataset_release/OHP/Labeled_Dataset")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)


def _pos(label_map, k):
    return bool(label_map.get(k))


def main():
    elb = json.load(open(LAB / "Labels/error_elbows.json"))
    kne = json.load(open(LAB / "Labels/error_knees.json"))
    splits = {s: json.load(open(LAB / f"Splits/{s}_keys.json")) for s in ("train", "val", "test")}

    rates = {s: {"Elbows": np.mean([_pos(elb, k) for k in ks]),
                 "Knees": np.mean([_pos(kne, k) for k in ks])} for s, ks in splits.items()}
    tr = splits["train"]
    both = sum(_pos(elb, k) and _pos(kne, k) for k in tr)
    e_only = sum(_pos(elb, k) and not _pos(kne, k) for k in tr)
    k_only = sum(not _pos(elb, k) and _pos(kne, k) for k in tr)
    neither = len(tr) - both - e_only - k_only

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    sp = ["train", "val", "test"]; x = np.arange(3); w = 0.36
    ax[0].bar(x - w / 2, [rates[s]["Elbows"] for s in sp], w, color="C0", label="Elbows+")
    ax[0].bar(x + w / 2, [rates[s]["Knees"] for s in sp], w, color="C1", label="Knees+")
    for i, s in enumerate(sp):
        ax[0].text(i - w / 2, rates[s]["Elbows"] + 0.005, f"{rates[s]['Elbows']:.2f}", ha="center", fontsize=8)
        ax[0].text(i + w / 2, rates[s]["Knees"] + 0.005, f"{rates[s]['Knees']:.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{s}\n(n={len(splits[s])})" for s in sp])
    ax[0].set_ylabel("positive rate"); ax[0].set_ylim(0, 0.5)
    ax[0].set_title("OHP class balance per official split"); ax[0].legend()

    cats = ["both", "Elbows\nonly", "Knees\nonly", "neither"]
    vals = [both, e_only, k_only, neither]
    bars = ax[1].bar(cats, vals, color=["C3", "C0", "C1", "0.7"])
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=9)
    ax[1].set_ylabel("train clips")
    ax[1].set_title(f"Train co-occurrence (n={len(tr)}) — errors near-independent\n"
                    f"only {both}/{e_only + both} Elbows+ are also Knees+")
    fig.suptitle("Overhead Press — Fitness-AQA labeled set (official splits)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "ohp_eda_balance.png", dpi=120, bbox_inches="tight")
    print("saved ohp_eda_balance.png")
    print(f"counts: train {len(tr)} / val {len(splits['val'])} / test {len(splits['test'])}")
    print(f"co-occurrence: both {both} | elbows-only {e_only} | knees-only {k_only} | neither {neither}")


if __name__ == "__main__":
    main()
