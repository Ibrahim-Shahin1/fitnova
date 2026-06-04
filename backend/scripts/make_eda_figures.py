"""Generate dataset EDA figures for the dissertation.

Re-runnable. Reads committed local data only (no Google Drive) and writes four
PNGs under docs/figures/. The Fitness-AQA panel uses the per-error counts that
were reconciled to the published benchmark.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data"
FIG = ROOT / "docs" / "figures"
REC = FIG / "recommender"

LEVELS = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
BLUE, GREEN, RED, PURPLE, BROWN, ORANGE = (
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DD8452",
)


def programs_eda() -> Path:
    with open(DATA / "program_catalog.pkl", "rb") as f:
        catalog = pickle.load(f)
    progs = list(catalog.values())
    goals = Counter(p.get("goal") for p in progs)
    levels = Counter(LEVELS.get(p.get("level_encoded"), "?") for p in progs)
    ptypes = Counter(p.get("primary_type") for p in progs)
    times = [p.get("time_per_workout_minutes") for p in progs]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    g_sorted = sorted(goals.items(), key=lambda kv: kv[1])
    ax[0, 0].barh([k for k, _ in g_sorted], [v for _, v in g_sorted], color=BLUE)
    ax[0, 0].set_title("Training goal")
    ax[0, 0].set_xlabel("programs")

    order = ["Beginner", "Intermediate", "Advanced"]
    ax[0, 1].bar(order, [levels.get(o, 0) for o in order], color=GREEN)
    ax[0, 1].set_title("Experience level")
    ax[0, 1].set_ylabel("programs")

    t_sorted = sorted(ptypes.items(), key=lambda kv: kv[1], reverse=True)
    ax[1, 0].bar([k for k, _ in t_sorted], [v for _, v in t_sorted], color=RED)
    ax[1, 0].set_title("Primary type")
    ax[1, 0].set_ylabel("programs")

    ax[1, 1].hist(times, bins=20, color=PURPLE, edgecolor="white")
    ax[1, 1].set_title("Time per workout")
    ax[1, 1].set_xlabel("minutes")
    ax[1, 1].set_ylabel("programs")

    fig.suptitle(
        "Programs catalogue EDA  (3,048 programs = 2,598 real + 450 synthetic)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REC / "recommender_programs_eda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def users_eda() -> Path:
    # Raw tracking CSV keeps human-readable, un-normalised values; user_features.csv is min-max scaled.
    df = pd.read_csv(DATA / "datasets" / "gym_members_exercise_tracking.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(df["Age"], bins=20, color=BLUE, edgecolor="white")
    ax[0, 0].set_title("Age")
    ax[0, 0].set_xlabel("years")
    ax[0, 1].hist(df["BMI"], bins=25, color=PURPLE, edgecolor="white")
    ax[0, 1].set_title("BMI")
    ax[0, 2].hist(df["Session_Duration (hours)"], bins=15, color=BROWN, edgecolor="white")
    ax[0, 2].set_title("Session duration")
    ax[0, 2].set_xlabel("hours")

    gender = df["Gender"].value_counts()
    ax[1, 0].bar(gender.index, gender.values, color=GREEN)
    ax[1, 0].set_title("Gender")
    wt = df["Workout_Type"].value_counts()
    ax[1, 1].bar(wt.index, wt.values, color=RED)
    ax[1, 1].set_title("Workout type")
    ax[1, 1].tick_params(axis="x", rotation=20)
    exp = df["Experience_Level"].value_counts().sort_index()
    ax[1, 2].bar([str(i) for i in exp.index], exp.values, color=ORANGE)
    ax[1, 2].set_title("Experience level")
    ax[1, 2].set_xlabel("1 = beginner, 3 = advanced")

    fig.suptitle("Gym-members dataset EDA  (973 users)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REC / "recommender_users_eda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def interactions_eda() -> Path:
    df = pd.read_csv(DATA / "interactions.csv")
    total = len(df)
    pos = int((df["interaction"] == 1).sum())
    neg = total - pos
    n_users = df["user_id"].nunique()
    n_items = df["program_id"].nunique()
    sparsity = 100.0 * (1 - pos / (n_users * n_items))
    pos_per_user = df[df["interaction"] == 1].groupby("user_id").size()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(pos_per_user.values, bins=30, color=BLUE, edgecolor="white")
    ax[0].set_title("Positive interactions per user")
    ax[0].set_xlabel("positive programs per user")
    ax[0].set_ylabel("users")

    bars = ax[1].bar(["positive", "negative"], [pos, neg], color=[GREEN, RED])
    ax[1].set_title("Positive vs negative labels")
    ax[1].set_ylabel("interactions")
    for b, v in zip(bars, [pos, neg]):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")

    fig.suptitle(
        f"SYNTHETIC user-program interactions  "
        f"({total:,} total, {pos:,} positive, {sparsity:.1f}% sparse)",
        fontsize=14,
        fontweight="bold",
        color="#8B0000",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REC / "recommender_interactions_eda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fitnessaqa_eda() -> Path:
    errors = ["Squat\nKIE", "Squat\nKFE", "Shallow\nSquat", "OHP\nElbows", "OHP\nKnees"]
    pct = [14.29, 68.33, 43.87, 25.49, 34.38]
    # Squat KIE x KFE joint counts (n=1,623): rows KIE No/Yes, cols KFE No/Yes.
    co = np.array([[474, 917], [40, 192]])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax[0].bar(errors, pct, color=BLUE)
    ax[0].set_title("Percent erroneous per error")
    ax[0].set_ylabel("% positive (erroneous)")
    ax[0].set_ylim(0, 100)
    for b, v in zip(bars, pct):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom")

    ax[1].imshow(co, cmap="Blues")
    ax[1].set_xticks([0, 1], ["KFE No", "KFE Yes"])
    ax[1].set_yticks([0, 1], ["KIE No", "KIE Yes"])
    ax[1].set_title("Squat KIE x KFE co-occurrence (n=1,623)")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{co[i, j]:,}", ha="center", va="center", fontweight="bold")

    fig.suptitle(
        "Fitness-AQA error class balance and Squat KIE/KFE co-occurrence",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "fitnessaqa_class_balance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    REC.mkdir(parents=True, exist_ok=True)
    for builder in (programs_eda, users_eda, interactions_eda, fitnessaqa_eda):
        out = builder()
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
