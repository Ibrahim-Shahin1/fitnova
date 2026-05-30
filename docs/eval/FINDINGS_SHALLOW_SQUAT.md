# Shallow-Squat Form-Error Detection — Benchmark Evaluation & Findings

**Model:** ResNet-18, faithful CVCSPC pose-contrastive SSL pretrain → end-to-end fine-tune,
3-seed ensemble (mean-of-sigmoids), val-tuned decision threshold 0.395.
**Evaluation set:** Fitness-AQA **official Shallow-Squat test split**, 540 crops (236 positive /
304 negative). Identical metric (F1) and split as Parmar et al. (ECCV 2022).

## Headline results

| Method | Modality | Official test F1 |
|--------|----------|------------------|
| **Ours — CVCSPC (3-seed ensemble)** | Image | **0.8902** (PR-AUC 0.967) |
| Ours — single best seed (42) | Image | 0.8961 |
| Ours — supervised baseline (control) | Image | 0.8750 |
| Paper — CVCSPC (the target) | Image | 0.8694 |
| Paper — SimSiam | Image | 0.8286 |
| Paper — OpenPose-TDM | 2D pose | 0.8340 |

**Our faithful CVCSPC ensemble (0.8902) beats the published CVCSPC (0.8694) by +0.021** on the
identical metric and split, and beats SimSiam / OpenPose-TDM by ~0.06. The CVCSPC self-supervised
pretraining adds a **+0.0152 lift** over our own supervised baseline (0.8750) — consistent across
all three seeds (per-seed test F1: seed 42 → 0.8961, 1337 → 0.8902, 7 → 0.8825; each above its
baseline counterpart).

**Generalization (overfitting check):** the ensemble's **val→test F1 gap is 0.0217** (val 0.9119,
test 0.8902) — small, no overfitting. Confusion at the threshold: 223 TP / 13 FN / 42 FP / 262 TN
(precision 0.842, recall 0.945 — recall-leaning, catching ~94% of shallow reps).

## The honest framing — no paper supervised-ImageNet baseline

The paper's Table 3 has **no plain-supervised-ImageNet row** for Shallow-Squat (only the SSL methods
— SimSiam 0.829, CVCSPC 0.869 — and the 2D-pose baseline 0.834). So **our supervised baseline
(0.8750) is the only supervised control**, and the SSL contribution is measured as (CVCSPC ensemble
− our baseline) = +0.0152. Notably the supervised baseline alone already matches/edges the published
CVCSPC: squat depth in a static crop is visually discriminative and the data is near-balanced (43.9%
positive), so a well-tuned ImageNet ResNet-18 is a strong control — and the domain-knowledge SSL
still adds a real, consistent lift on top of it.

## Loss deviation (faithful to the code, documented — not reconciled)

The CVCSPC backbone was pretrained with the **3-term** distance-ratio loss the official code
implements (`train_test.py:68`: `-log(e^-d_ap / (e^-d_ap + e^-d_an + e^-d_pn))`), which adds a
positive-negative repulsion term **not** present in the paper's 2-term Eq.1. We implemented the
as-shipped code version (more numerically stable; it forces positive-negative separation) and
document the deviation here rather than silently reconciling it. The triplet-accuracy monitor
converged 0.48 (chance) → **0.959** over 100 epochs.

## Leakage-free SSL set

The CVCSPC SSL pretrained on **4,791** unlabeled Back-Squat clips — the 4,970-clip unlabeled set
minus the **179** clips whose frames appear in the labeled val/test crops (the labeled crops come
from clips that ~97–99% overlap the unlabeled set; the official `dataloader.py:70-74` excludes the
val/test clips from SSL). No eval-clip frames were seen during pretraining.

## Figures (`docs/figures/`)

| File | What it shows |
|------|---------------|
| `shallow_squat_f1_vs_paper.png` | Headline bar — our baseline → our CVCSPC → paper CVCSPC / SimSiam / OpenPose-TDM. |
| `shallow_squat_class_balance.png` | Near-balanced class distribution (43.9% pos), overall + per split. |
| `shallow_squat_samples.png` | Sample shallow (positive) vs deep (negative) crops. |
| `shallow_squat_training_curves.png` | CVCSPC SSL triplet-accuracy convergence (0.48→0.959); per-seed val-F1 (CVCSPC > baseline); baseline→CVCSPC test lift +0.0152. |
| `shallow_squat_pr_curve.png` | Precision-recall + operating point (PR-AUC 0.967). |
| `shallow_squat_confusion_matrix.png` | Confusion at the val-tuned threshold. |

## Method (faithful to the published CVCSPC)

ImageNet ResNet-18 supervised baseline established the control; then faithful CVCSPC pose-contrastive
SSL pretraining on the unlabeled Back-Squat set (anchor + positive = phase-matched frames across two
*different* reps at the same bar-trajectory phase; negative = a frame a phase-gap away on the second
rep; masking-only augmentation; the 3-term distance-ratio loss on L2-normalized features), then
end-to-end fine-tune with a fresh single-logit head (the SSL projector discarded). F1 on the official
split, threshold tuned on validation, 3-seed mean-of-sigmoids ensemble.

## Honest scope notes

- **Image modality + 2nd method:** Shallow-Squat adds the image error type + the CVCSPC method to the
  thesis, alongside the video MD-SSL work (Squat, OHP).
- **BarbellRow (IMG-03) cancelled** for compute/time — the milestone form-correction scope is
  Squat + OHP (video) + Shallow-Squat (image).
- **Serving descoped:** the form-correction frontend is cancelled; IMG-01/IMG-02 are met by the
  trained/evaluated models + the paper comparison + this visualization pack (same stance as OHP).
- Like Squat/OHP, these results are on the **benchmark** (in-the-wild gym video). Every number here
  is from `.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl` or the local Fitness-AQA
  archive — no fabrication.

## Reproduce

```
# results.pkl (per-crop ensemble scores + curves) is committed at
#   .planning/phases/07-image-based-errors-cvcspc/figures/results.pkl
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/11_shallow_squat_evaluation.ipynb
```
