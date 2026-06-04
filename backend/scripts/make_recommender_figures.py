"""Generate the two recommender figures for the report.

Outputs to docs/figures/recommender/:
  recommender_hr_ndcg.png       - NeuMF vs component models, leave-one-out HR@10 / NDCG@10
  recommender_architecture.png  - the three-layer plan pipeline (content filter -> NeuMF -> LLM crew)

Numbers are the verified evaluation results from backend/models/neumf_metadata.pkl + README:
  GMF  HR@10 0.9075  NDCG@10 0.7251
  MLP  HR@10 0.9065  NDCG@10 0.7208
  NeuMF HR@10 0.9270 NDCG@10 0.7469
Run: python backend/scripts/make_recommender_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "recommender"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Figure 1
models = ["GMF", "MLP", "NeuMF"]
hr = [0.9075, 0.9065, 0.9270]
ndcg = [0.7251, 0.7208, 0.7469]
x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.6))
b1 = ax.bar(x - w/2, hr, w, label="HR@10", color="#3b6ea5")
b2 = ax.bar(x + w/2, ndcg, w, label="NDCG@10", color="#e08a3c")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.02); ax.set_ylabel("Score")
ax.set_title("Recommender ranking quality: leave-one-out HR@10 and NDCG@10\n"
             "(973 users, 1 held-out positive vs 99 sampled negatives)", fontsize=10.5)
ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "recommender_hr_ndcg.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(figsize=(9.5, 8)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

def box(cx, cy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec="#2f2f2f", lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9)

def varrow(cy_top, cy_bot, cx=3.4):
    ax.add_patch(FancyArrowPatch((cx, cy_top), (cx, cy_bot),
                 arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#444"))

CX = 3.4; W = 5.6
box(CX, 9.2, W, 0.95, "User profile + chat intake\n(GPT-4o-mini extracts experience, session duration, frequency)", "#eaeff5")
box(CX, 7.6, W, 0.95, "Layer 1: content-based filter\nweighted 4-rule score over 3,048 programs, keep top 50", "#dbe7d6")
box(CX, 6.0, W, 0.85, "Cold-start: cosine match to the nearest of 973 trained users", "#dbe7d6")
box(CX, 4.4, W, 0.95, "Layer 2: NeuMF re-rank (GMF + MLP)\nHR@10 0.927, NDCG@10 0.747", "#f3e4c6")
box(CX, 2.7, W, 0.95, "Layer 3: LLM plan crew (GPT-4o-mini)\nProfiler, Generator, Critic, Optimizer + deterministic validator", "#f0dcd2")
box(CX, 1.0, W, 0.9, "Grounded 7-day workout plan\n(dataset exercises only, hard-gated, no fabrication)", "#e3d7ea")
for top, bot in [(8.72, 8.08), (7.12, 6.43), (5.57, 4.88), (3.92, 3.18), (2.22, 1.45)]:
    varrow(top, bot)

# data-sources side panel
ax.add_patch(FancyBboxPatch((6.7, 5.4), 3.0, 2.4,
             boxstyle="round,pad=0.04,rounding_size=0.08", fc="#f7f7f7", ec="#999", lw=1.0, linestyle="--"))
ax.text(8.2, 7.45, "Datasets", ha="center", va="center", fontsize=9.5, fontweight="bold")
ax.text(8.2, 6.55, "Programs: 2,598 real\n+ 450 synthetic = 3,048\nUsers: 973\nInteractions: 809,439\n(synthetic)",
        ha="center", va="center", fontsize=8.2)
ax.add_patch(FancyArrowPatch((6.7, 6.0), (6.2, 5.0), arrowstyle="-|>", mutation_scale=12,
             lw=1.0, color="#999", linestyle="--"))

ax.set_title("FitNova workout-plan recommender pipeline", fontsize=12, y=0.99)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "recommender_architecture.png"), dpi=150); plt.close()

print("wrote:", os.path.join(OUT, "recommender_hr_ndcg.png"))
print("wrote:", os.path.join(OUT, "recommender_architecture.png"))
