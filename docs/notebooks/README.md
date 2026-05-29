# Form-Error Analysis Notebooks

Reproducible Jupyter notebooks documenting the form-error detection pipeline (Fitness-AQA),
from raw data to final evaluation — **Squat** (01–04) and **Overhead Press** (05–08). Open in
VS Code or `jupyter notebook docs/notebooks/`. All are **pre-executed** (figures/tables render
on open).

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

## Data sources (real, no fabrication)

- Fitness-AQA labeled sets (local: labels, splits, clips).
- Recorded training artifacts: `.planning/phases/04-…/figures/results.pkl` (Squat) and
  `.planning/phases/06-…/figures/results.pkl` (OHP) — per-epoch SSL + fine-tune curves,
  official per-clip ensemble scores, paper comparison.

## Reproduce

Each notebook is jupytext-paired (`.py` source + `.ipynb`). To re-run:
`jupyter nbconvert --to notebook --execute --inplace docs/notebooks/<name>.ipynb`
