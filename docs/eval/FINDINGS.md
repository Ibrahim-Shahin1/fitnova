# Squat Form-Error Detection — Benchmark Evaluation & Findings

**Model:** R(2+1)D-18, Motion-Disentangling SSL pretrain + fine-tune, 3-seed ensemble
(mean-of-sigmoids), production thresholds KIE 0.614 / KFE 0.385.
**Evaluation set:** Fitness-AQA **official Squat test split**, 244 clips (KFE 169 pos / 75 neg;
KIE 36 pos / 208 neg). Identical metric (F1 per error) and split as Parmar et al. (ECCV 2022).

## Headline results

| Error | Ours (official) | Parmar MD-SSL (paper) | Parmar Kinetics baseline |
|-------|--------------------------|------------------------|--------------------------|
| **KFE** (knees forward) | **0.841** | 0.834 | 0.818 |
| **KIE** (knees inward)  | **0.420** | 0.419 | 0.297 |
| **macro** | **0.6304** | 0.6262 | 0.557 |

**Our result matches the paper's MD-SSL on identical metric and split, and beats the
supervised Kinetics baseline.** KFE is strong (F1 0.84, AP 0.85); KIE is intrinsically
weak (F1 0.42) — the rare, hard class in the paper too (only 36 positives in 244).

*Reproduced locally via the serving pipeline (cv2 decode):* KFE 0.831 / KIE 0.382 /
macro 0.607 — within ~0.02 of the official Colab eval; the small gap is the documented
cv2-vs-torchvision decode parity-approx. The figures below are generated from this
local re-score (`docs/eval/squat_test_scores.csv`, per-clip).

## Figures (`docs/figures/`)

| File | What it shows |
|------|---------------|
| `squat_score_distributions.png` | Per-error score histograms, positives vs negatives + threshold. KFE separates clearly; KIE overlaps (honest). |
| `squat_pr_curves.png` | Precision-recall curves + operating point + AP (threshold-free quality). |
| `squat_confusion_matrices.png` | Confusion at the production thresholds. |
| `squat_f1_vs_paper.png` | Ours vs Parmar MD-SSL vs Kinetics baseline (the comparison bar). |
| `squat_best_examples.png` | Most-confident **correct** detections (qualitative findings). |

## Best-accuracy qualitative examples

See `squat_best_examples.png` + `best_examples.md`. The model's most confident correct
calls — e.g. KFE detected at 0.84–0.92 on genuine knees-forward squats, KIE detected at
0.92–0.99 on knees-inward squats, and confident "clean" calls on good form. All are
**side-on, in-the-wild gym clips** — the domain the model was trained on.

## Generalization (overfitting check)

Validation tracks test for both errors (KFE: test 0.841 ≥ val 0.813; KIE: val 0.500 vs test 0.420,
within sampling noise on 36 test positives), so the deployed operating point shows no
validation-to-test overfitting; AUC-ROC val↔test is essentially identical (KIE 0.790 vs 0.784). The
train→test view is honest about a limitation: on the rare **KIE** class the model fits training
examples much better than test (**train 0.93 vs test 0.42, gap +0.51**) — train-memorization,
intrinsic to the few-hundred training positives; KFE generalizes tightly (train→test +0.01). The
deployed test KIE F1 (0.420) equals the published paper (0.419), so this is the field's difficulty
ceiling, not a degraded checkpoint. Full analysis: `docs/eval/OVERFITTING_AUDIT.md` (Pillars 4 and 7).

## Honest scope note (domain shift)

The model achieves the above on the **benchmark** (in-the-wild gym video, side-on,
landscape). It does **not** transfer to home, front-facing, portrait phone clips — good
and bad score nearly identically there (verified). This is the known in-the-wild
generalization limit (the paper's own thesis). A live, record-yourself form-correction
frontend (live + upload-on-own-video) is therefore not deployed, because the models do not
transfer to home / phone-camera video (a known domain-shift limitation); instead
the models are surfaced through a benchmark inspector that runs them on the dataset's own
held-out test clips beside the expert ground truth. Making it work on a user's own camera
would require either filming to match the training domain (side-on/landscape) or
domain-adaptation fine-tuning.

## Reproduce

```
# 1. score the official test split (writes docs/eval/squat_test_scores.csv)
#    (requires the staged weights + extracted Fitness-AQA squat test videos)
# 2. regenerate all figures from the CSV:
python backend/scripts/eval_benchmark_viz.py
python backend/scripts/eval_best_examples.py
```
