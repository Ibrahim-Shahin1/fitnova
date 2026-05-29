"""Headline OHP F1-vs-paper comparison chart (Phase 6 Plan 04).

Numbers from the Plan-04 test eval on the official 339-clip OHP test split
(2-seed MD-SSL ensemble, val-tuned thresholds, no TTA). The full figure pack
(PR / confusion / training curves) is regenerated from results.pkl in Plan 05.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROWS = {
    "Supervised baseline": {"Elbows": 0.4167, "Knees": 0.8069, "macro": 0.6118},
    "Our MD-SSL ensemble": {"Elbows": 0.4474, "Knees": 0.8770, "macro": 0.6622},
    "Paper Ours-MD": {"Elbows": 0.4552, "Knees": 0.8452, "macro": 0.6502},
}
COLORS = {"Supervised baseline": "0.65", "Our MD-SSL ensemble": "C0", "Paper Ours-MD": "C1"}
ERRORS = ["Elbows", "Knees", "macro"]

x = np.arange(len(ERRORS))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 5.2))
for i, (name, vals) in enumerate(ROWS.items()):
    bars = ax.bar(x + (i - 1) * w, [vals[e] for e in ERRORS], w, label=name, color=COLORS[name])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(["Elbows", "Knees", "macro"])
ax.set_ylabel("F1 (official OHP test split, 339 clips)")
ax.set_ylim(0, 1.0)
ax.set_title("Overhead Press form-error detection — F1 vs published (Parmar et al., ECCV 2022)\n"
             "MD-SSL: +0.050 macro over the supervised baseline; matches/edges the paper; val→test gap 0.011")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

out = Path("docs/figures/ohp_f1_vs_paper.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
