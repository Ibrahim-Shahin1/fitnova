# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FitNova — Squat Form-Error Detection: Benchmark Evaluation
#
# **Model:** R(2+1)D-18 + Motion-Disentangling SSL (pretrain → fine-tune), 3-seed ensemble
# (mean-of-sigmoids), production thresholds **KIE 0.614 / KFE 0.385**.
#
# **Test set:** Fitness-AQA **official Squat test split — 244 clips**. Identical metric
# (F1 per error) and split as *Parmar et al., ECCV 2022* (arXiv:2202.14019).
#
# This notebook reproduces the test-set metrics, the visualizations, and the
# best-accuracy qualitative examples from the committed per-clip scores
# (`docs/eval/squat_test_scores.csv`).

# %%
import os, sys

# Locate the project root (dir containing docs/eval/squat_test_scores.csv).
_d = os.getcwd()
while _d != os.path.dirname(_d):
    if os.path.exists(os.path.join(_d, "docs", "eval", "squat_test_scores.csv")):
        break
    _d = os.path.dirname(_d)
os.chdir(_d); sys.path.insert(0, _d)
print("project root:", _d)

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from IPython.display import Image, display

KIE_THR, KFE_THR = 0.614, 0.385
df = pd.read_csv("docs/eval/squat_test_scores.csv")
print(f"{len(df)} test clips  |  KFE positives {int(df.kfe_label.sum())}  |  KIE positives {int(df.kie_label.sum())}")
df.head()

# %% [markdown]
# ## 1. Test-set metrics (served model, local re-score)

# %%
def _metrics(score, label, thr):
    pred = (score >= thr).astype(int)
    return {
        "F1": round(f1_score(label, pred), 3),
        "Precision": round(precision_score(label, pred, zero_division=0), 3),
        "Recall": round(recall_score(label, pred, zero_division=0), 3),
        "AP": round(average_precision_score(label, score), 3),
        "positives": int(label.sum()), "n": int(len(label)), "threshold": thr,
    }

metrics = pd.DataFrame({
    "KFE": _metrics(df.kfe_score, df.kfe_label, KFE_THR),
    "KIE": _metrics(df.kie_score, df.kie_label, KIE_THR),
}).T
macro = round((metrics.loc["KFE", "F1"] + metrics.loc["KIE", "F1"]) / 2, 4)
print("macro-F1 (local serving re-score):", macro)
metrics

# %% [markdown]
# ### Comparison to Parmar et al. (identical metric & split)
#
# | Error | **Ours (official, Phase 4)** | Parmar MD-SSL (paper) | Kinetics baseline |
# |-------|------------------------------|------------------------|-------------------|
# | KFE   | **0.841** | 0.834 | 0.818 |
# | KIE   | **0.420** | 0.419 | 0.297 |
# | macro | **0.6304** | 0.6262 | 0.557 |
#
# Our MD-SSL result **matches the paper's method** and beats the supervised Kinetics
# baseline. The local re-score above is ~0.02 lower because the serving path uses a cv2
# decoder (documented parity-approx); the official Colab eval (torchvision decode) is the
# headline academic number.

# %% [markdown]
# ## 2. Visualizations
# Regenerated from the per-clip CSV.

# %%
import backend.scripts.eval_benchmark_viz as ev
ev.main()  # writes docs/figures/*.png from the CSV

for name, cap in [
    ("squat_score_distributions.png", "Score distributions — positives vs negatives per error head (+ threshold)."),
    ("squat_pr_curves.png",           "Precision-recall curves + operating point + average precision."),
    ("squat_confusion_matrices.png",  "Confusion matrices at the production thresholds."),
    ("squat_f1_vs_paper.png",         "F1: ours vs Parmar MD-SSL vs Kinetics baseline."),
]:
    print(cap)
    display(Image(filename=os.path.join("docs", "figures", name)))

# %% [markdown]
# ## 3. Best-accuracy examples (most-confident correct detections)

# %%
try:
    import backend.scripts.eval_best_examples as be
    be.main()
except Exception as e:
    print("(best-examples regeneration skipped — needs extracted test videos:", e, ")")
display(Image(filename=os.path.join("docs", "figures", "squat_best_examples.png")))

# %% [markdown]
# ## 4. Live model test on one clip
# Loads the ensemble and runs the full inference path on a single benchmark clip,
# printing the D-05 output. Skipped automatically if weights / clips are unavailable.

# %%
try:
    from backend.services.squat_form_service import SquatFormService
    from backend.services.clip_decode import decode_clip_cv2, get_frame_count_and_fps
    from backend.training.aqa.datasets.transforms import uniform_sample_indices

    svc = SquatFormService(model_dir="backend/models/form_model_squat_md")
    cand = ("Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/"
            "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos_extracted/videos/33301_1.mp4")
    if svc.model_ready and os.path.exists(cand):
        print("onnx_enabled:", svc.onnx_enabled)
        n, fps = get_frame_count_and_fps(cand)
        idx = uniform_sample_indices(n, 32, jitter=0).clamp(0, max(n - 1, 0))
        out = svc.classify_clip(decode_clip_cv2(cand, idx).numpy())
        print("clip 33301_1 (ground truth: KFE error)  ->")
        for e in out["errors"]:
            print(f"   {e['type']}: confidence={e['confidence']:.3f}  detected={e['detected']}  ({e['severity_word']})")
    else:
        print("skipped — weights or extracted clip not present (load the CSV-based sections above instead).")
except Exception as e:
    print("live test skipped:", e)

# %% [markdown]
# ## Findings (honest scope)
#
# The model reproduces the paper on the **benchmark** — KFE strong (0.84), KIE weak (0.42,
# the rare/hard class, also weak in the paper). It does **not** transfer to home /
# portrait phone clips (domain shift); the live + upload-on-own-video frontend was
# cancelled for that reason. The defensible deliverable is this benchmark evaluation.
# Full write-up: `docs/eval/FINDINGS.md`.
