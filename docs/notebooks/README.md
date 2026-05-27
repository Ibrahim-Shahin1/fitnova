# Squat — Analysis Notebooks

Reproducible Jupyter notebooks documenting the **Squat** form-error pipeline (Fitness-AQA),
from raw data to final evaluation. Open in VS Code or `jupyter notebook docs/notebooks/`.
All are **pre-executed** (open and the figures/tables are already rendered).

| Notebook | Covers |
|----------|--------|
| **01_squat_eda** | Dataset EDA — official splits, class imbalance (KFE ~68% / KIE ~14%), KIE↔KFE co-occurrence (192/232), clip properties (duration/fps/resolution/orientation), label-interval distributions, sample frames per class |
| **02_squat_data_pipeline** | Splits used verbatim (no resampling); imbalance handled via loss `pos_weight` (KIE 6.10× / KFE 0.45×); 32-frame uniform sampling → 112×112 center crop → Kinetics norm; train-time random-crop augmentation vs the deterministic inference crop |
| **03_squat_training** | SSL pretraining (contrastive loss, **effective-rank collapse** diagnostic for weak vs strong augs, frozen linear-probe vs Kinetics); supervised fine-tune **train-vs-val loss** (overfit onset) + val-F1 early-stop; baseline→MD-SSL lift |
| **04_squat_evaluation** | Official test-split metrics — score distributions, PR + **ROC/AUC**, confusion matrices, per-seed vs ensemble, TTA ablation, comparison to Parmar et al., best-accuracy examples |

## Headline result
macro-F1 **0.6304** (KFE **0.841**, KIE **0.420**) on the Fitness-AQA official Squat test
split — matches Parmar et al. MD-SSL (0.626) and beats the Kinetics baseline (0.558).
Full write-up: [`../eval/FINDINGS.md`](../eval/FINDINGS.md).

## Data sources (real, no fabrication)
- Fitness-AQA Squat labeled set (local: labels, splits, extracted clips).
- Recorded Phase-4 training artifacts: `.planning/phases/04-…/figures/results.pkl`
  (per-epoch SSL + fine-tune curves, official per-clip ensemble scores, paper comparison).

## Reproduce
Each notebook is jupytext-paired (`.py` source + `.ipynb`). To re-run:
`jupyter nbconvert --to notebook --execute --inplace docs/notebooks/<name>.ipynb`

## Other exercises
The same notebook set will be produced for **Overhead Press** and **Barbell Row** once
those phases are complete.
