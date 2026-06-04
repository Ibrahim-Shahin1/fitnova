"""Best-accuracy qualitative examples on the Fitness-AQA squat test split.

Selects the model's most-confident CORRECT detections (from the scored test set)
and renders an annotated frame grid + a findings list — the qualitative evidence
to include in the report alongside the F1 numbers.

Categories:
  - KFE true positives  (label=error, highest KFE score)   — confidently caught
  - KFE true negatives  (label=clean, lowest  KFE score)   — confidently clean
  - KIE true positives  (label=error, highest KIE score)   — the weak head's best hits

Run:  python backend/scripts/eval_best_examples.py
"""
from __future__ import annotations

import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CSV = "docs/eval/squat_test_scores.csv"
_OUT = "docs/figures"
_FINDINGS = "docs/eval/best_examples.md"
_VIDS = ("Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/"
         "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos_extracted/videos")
KIE_THR, KFE_THR = 0.614, 0.385
N = 4  # examples per category


def _load():
    rows = []
    with open(_CSV) as f:
        for r in csv.DictReader(f):
            rows.append((r["clip_id"], float(r["kie_score"]), float(r["kfe_score"]),
                         int(r["kie_label"]), int(r["kfe_label"])))
    return rows


def _mid_frame(clip_id):
    p = os.path.join(_VIDS, f"{clip_id}.mp4")
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(_OUT, exist_ok=True)
    rows = _load()
    kfe_tp = sorted([r for r in rows if r[4] == 1], key=lambda r: -r[2])[:N]
    kfe_tn = sorted([r for r in rows if r[4] == 0], key=lambda r: r[2])[:N]
    kie_tp = sorted([r for r in rows if r[3] == 1], key=lambda r: -r[1])[:N]

    cats = [
        ("KFE correctly DETECTED (knees forward)", kfe_tp, "kfe"),
        ("KFE correctly CLEAR (no knees-forward)", kfe_tn, "kfe"),
        ("KIE correctly DETECTED (knees inward)", kie_tp, "kie"),
    ]

    fig, axes = plt.subplots(len(cats), N, figsize=(4 * N, 4 * len(cats)))
    md_lines = ["# Best-accuracy qualitative examples (Fitness-AQA squat test split)\n",
                "Model's most-confident CORRECT detections — for the findings/report.\n"]
    for row_i, (title, items, head) in enumerate(cats):
        md_lines.append(f"\n## {title}\n")
        for col_i in range(N):
            ax = axes[row_i][col_i]
            ax.axis("off")
            if col_i >= len(items):
                continue
            cid, kie_sc, kfe_sc, kie_l, kfe_l = items[col_i]
            score = kfe_sc if head == "kfe" else kie_sc
            thr = KFE_THR if head == "kfe" else KIE_THR
            gt = (kfe_l if head == "kfe" else kie_l)
            frame = _mid_frame(cid)
            if frame is not None:
                ax.imshow(frame)
            verdict = "DETECTED" if score >= thr else "CLEAR"
            ax.set_title(f"{cid}\n{head.upper()}={score:.3f} → {verdict}\nGT: {'error' if gt else 'clean'} ✓",
                         fontsize=10)
            md_lines.append(f"- `{cid}`: {head.upper()}={score:.3f} (GT {'error' if gt else 'clean'})")
    fig.suptitle("Most-confident correct detections on the Fitness-AQA test split", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(_OUT, "squat_best_examples.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)

    with open(_FINDINGS, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("wrote", _FINDINGS)


if __name__ == "__main__":
    main()
