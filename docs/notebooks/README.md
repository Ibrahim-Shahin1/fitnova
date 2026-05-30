# Form-Error Analysis Notebooks

Reproducible Jupyter notebooks documenting the form-error detection pipeline (Fitness-AQA),
from raw data to final evaluation — **Squat** (01–04), **Overhead Press** (05–08), and
**Shallow-Squat** (09–11, the image modality). Open in VS Code or `jupyter notebook docs/notebooks/`.
All are **pre-executed** (figures/tables render on open).

## Squat (KIE / KFE)

| Notebook | Covers |
|----------|--------|
| **01_squat_eda** | Official splits, class imbalance (KFE ~68% / KIE ~14%), KIE↔KFE co-occurrence (192/232), clip properties, sample frames |
| **02_squat_data_pipeline** | Verbatim splits; `pos_weight` (KIE 6.10× / KFE 0.45×); 32-frame → 112² crop → Kinetics norm; augmentation |
| **03_squat_training** | SSL contrastive loss + effective-rank collapse diagnostic; supervised fine-tune overfit + early-stop; baseline→MD-SSL lift |
| **04_squat_evaluation** | Score distributions, PR + ROC/AUC, confusion, per-seed vs ensemble, TTA ablation, paper comparison, best examples |

**Squat headline:** macro-F1 **0.6304** (KFE 0.841, KIE 0.420) — matches Parmar et al. MD-SSL
(0.626), beats the Kinetics baseline (0.558). Write-up: [`../eval/FINDINGS.md`](../eval/FINDINGS.md).

## Overhead Press (Elbows / Knees)

| Notebook | Covers |
|----------|--------|
| **05_ohp_eda** | Official splits, class balance (Elbows ~26% / Knees ~34%), Elbows↔Knees near-independence (85/407) |
| **06_ohp_data_pipeline** | `index_ohp`, train-derived `pos_weight` (Elbows 2.89× / Knees 1.92×), 32-frame/112² transform, SSL aug set, decoded sample frames |
| **07_ohp_training** | MD-SSL loss + effective-rank (no collapse), linear-probe backbone selection, fine-tune overfit control, baseline→SSL lift |
| **08_ohp_evaluation** | PR + ROC/AUC, confusion, score distributions, per-seed vs ensemble, paper comparison, val→test gap |

**OHP headline:** macro-F1 **0.6622** (Elbows 0.447, Knees 0.877) — matches/edges Parmar et al.
MD-SSL (0.650), +0.050 over the supervised baseline, val→test gap 0.011 (no overfitting).
Write-up: [`../eval/FINDINGS_OHP.md`](../eval/FINDINGS_OHP.md).

## Shallow-Squat (squat-depth, image error · CVCSPC)

| Notebook | Covers |
|----------|--------|
| **09_shallow_squat_eda** | Official splits (2542/529/540), near-balanced class distribution (43.9% pos / 1584-of-3611), sample shallow vs deep crops, uniform 299² crop size, the image pipeline (224² + ImageNet norm, single binary head) |
| **10_shallow_squat_training** | CVCSPC pose-contrastive SSL convergence (triplet-accuracy 0.48→0.959), per-seed baseline vs CVCSPC fine-tune val-F1, baseline→CVCSPC lift |
| **11_shallow_squat_evaluation** | Official-test-split F1, score distribution, PR curve, confusion, per-seed vs ensemble, F1-vs-paper (baseline → CVCSPC → paper CVCSPC 0.8694) |

**Shallow-Squat headline:** F1 **0.8902** (PR-AUC 0.967) on the official 540-crop test split —
**beats the published CVCSPC 0.8694** (+0.021), and SimSiam 0.829 / OpenPose-TDM 0.834; the faithful
CVCSPC SSL adds **+0.0152** over our supervised baseline (0.8750), consistent across all 3 seeds,
val→test gap 0.0217 (no overfitting). Adds the **image modality** + the **2nd method (CVCSPC)** to
the thesis. Write-up: [`../eval/FINDINGS_SHALLOW_SQUAT.md`](../eval/FINDINGS_SHALLOW_SQUAT.md).

## Full Evaluation (all 3 exercises)

| Notebook | Covers |
|----------|--------|
| **12_full_evaluation** | Cross-exercise master comparison (all 5 errors vs Parmar), EVAL-02 MD+CVCSPC ensemble finding (not-applicable), GYMetricPose + LMM related-work context, 4 master figures (all_errors_vs_paper, cross_exercise_macro, methodology_diagram, highlights_panel) |

**Full-evaluation headline:** 5-error spine -- Squat KIE/KFE (0.420 / 0.841, macro 0.6304),
OHP Elbows/Knees (0.447 / 0.877, macro 0.6622), Shallow-Squat depth (0.8902) -- all measured on
the official Fitness-AQA test splits, identical metric (F1) and split as Parmar et al. (ECCV 2022).
Write-up: [`../eval/FINDINGS_FULL.md`](../eval/FINDINGS_FULL.md).

## Data sources (real, no fabrication)

- Fitness-AQA labeled sets (local: labels, splits, clips).
- Recorded training artifacts: `.planning/phases/04-…/figures/results.pkl` (Squat),
  `.planning/phases/06-…/figures/results.pkl` (OHP), and
  `.planning/phases/07-…/figures/results.pkl` (Shallow-Squat) — per-epoch SSL + fine-tune curves,
  official per-clip/per-crop ensemble scores, paper comparison.

## Reproduce

Each notebook is jupytext-paired (`.py` source + `.ipynb`). To re-run:
`jupyter nbconvert --to notebook --execute --inplace docs/notebooks/<name>.ipynb`
