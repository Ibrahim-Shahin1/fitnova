# Report Index - Form-Correction Evaluation and Overfitting Audit

A single map of the evaluation evidence for the project report: the verdict, where each document
and notebook lives, and what every committed figure shows. All numbers are on the official
Fitness-AQA train/val/test splits, F1 per error, identical metric and splits to Parmar et al.
(ECCV 2022).

## Verdict (one line, not softened)

No deployed model shows validation-to-test (deployment) overfitting: validation tracks test at F1,
PR-AUC, and AUC-ROC, and re-evaluating from the actual `best.pt` weights reproduces the committed
results at delta 0. The honest limitation, disclosed in Pillar 7: the two rarest classes
(Squat Knees-Inward deployed, OHP-baseline Elbows) show large train-to-test gaps (up to +0.53), i.e.
train-memorization on classes with only a few hundred training positives. This caps their absolute
test F1 at the published ceiling but does not break deployment generalization, and SSL measurably
reduces the gap where it can (OHP Elbows +0.53 to +0.13).

## Headline results (deployed models)

| Exercise | Modality | Errors (test F1) | Macro | vs Parmar et al. |
|----------|----------|------------------|-------|------------------|
| Squat | video R(2+1)D-18 + MD-SSL | KIE 0.420, KFE 0.841 | 0.630 | matches MD-SSL 0.626; beats Kinetics 0.558 |
| OHP | video R(2+1)D-18 + MD-SSL | Elbows 0.447, Knees 0.877 | 0.662 | matches/edges MD-SSL 0.650 |
| Shallow-Squat | image ResNet-18 + CVCSPC | depth 0.890 | n/a | beats CVCSPC 0.869, SimSiam 0.829, pose-TDM 0.834 |

## Anchor documents

| Document | Role |
|----------|------|
| [`OVERFITTING_AUDIT.md`](OVERFITTING_AUDIT.md) | Full 7-pillar verdict (cache integrity, re-eval from weights, learning curves, val-to-test gap, confusion/score consistency, per-seed + SSL health, train-to-test view). |
| [`OVERFITTING_AND_SSL_FIX.md`](OVERFITTING_AND_SSL_FIX.md) | Plain-language note: the overfitting in the supervised baselines and how the deployed SSL models reduce it. |
| [`FINDINGS.md`](FINDINGS.md) / [`FINDINGS_FULL.md`](FINDINGS_FULL.md) / [`FINDINGS_OHP.md`](FINDINGS_OHP.md) / [`FINDINGS_SHALLOW_SQUAT.md`](FINDINGS_SHALLOW_SQUAT.md) | Per-exercise benchmark write-ups; each now carries a Generalization (overfitting check) paragraph. |
| [`best_examples.md`](best_examples.md) | Curated best-example clips per error (defense visuals). |

## Notebooks

| Path | Role |
|------|------|
| [`../audit/overfitting_summary.ipynb`](../audit/overfitting_summary.ipynb) | Report-ready summary notebook (7 figures): deployed models generalize, baselines overfit, the SSL fix. |
| [`../audit/overfitting_audit_local.ipynb`](../audit/overfitting_audit_local.ipynb) | Comprehensive committed-data audit (the 15 `audit_*` figures). |
| [`../audit/overfitting_audit_reeval.ipynb`](../audit/overfitting_audit_reeval.ipynb) | Re-eval from the saved weights (image live, video summarized). |
| [`../notebooks/`](../notebooks/) | Curated pipeline notebooks `01..12` (EDA, data, training, evaluation per exercise) plus README. |
| [`../notebooks/colab_originals/`](../notebooks/colab_originals/) | The raw executed Colab originals (provenance) behind the curated notebooks. |

## Audit scripts and data

- Scripts (`docs/audit/`): `reeval_shallow.py`, `reeval_video.py`, `reeval_certainty_figure.py`,
  `train_eval.py` (re-eval from weights, train-to-test sweep, certainty figure).
- Committed result data (`docs/audit/data/`): `squat_baseline_phase03_results.pkl`,
  `reeval_shallow.pkl`, `reeval_video.pkl`, `train_val_test_f1.pkl`, `ohp_baseline_curve.pkl`.
- For the report you do not need to re-run anything: the committed pkls and figures are the evidence.

## Figures (`docs/figures/`)

### Report-ready set (`summary_*`, from `overfitting_summary.ipynb`)

| Figure | What it shows |
|--------|---------------|
| `summary_deployed_learning_curves.png` | Deployed models: best.pt taken before any validation-loss rise (early-stop discipline). |
| `summary_deployed_train_val_test.png` | Deployed models: train vs val vs test F1 per error (val tracks test). |
| `summary_deployed_confusion.png` | Deployed models: confusion at the production threshold, val and test fail the same way. |
| `summary_deployed_val_test_gap.png` | Deployed models: val-to-test F1 gap per error, small or negative. |
| `summary_baseline_learning_curves.png` | Baselines (control): validation loss climbs while train loss falls, the overfitting signature. |
| `summary_baseline_train_val_test.png` | Baselines (control): large train-to-test gaps on the rare classes (OHP Elbows +0.53). |
| `summary_ssl_fix.png` | The SSL effect: train-to-test and val-to-test gaps shrink, OHP Elbows +0.53 to +0.13. |

### Comprehensive set (`audit_*`, from `overfitting_audit_local.ipynb`)

| Figure | What it shows |
|--------|---------------|
| `audit_squat_learning_curves.png` | Squat per-seed train/val loss + val-F1; early-stop checkpoint marked. |
| `audit_ohp_learning_curves.png` | OHP per-seed train/val loss + val-F1; early-stop checkpoint marked. |
| `audit_shallow_learning_curves.png` | Shallow-Squat per-seed curves (baseline and CVCSPC). |
| `audit_val_test_gap_summary.png` | Pillar 4: val-to-test F1 and PR-AUC gap per error with bootstrap CIs. |
| `audit_roc_curves.png` | ROC + AUC-ROC, validation vs test per error (stable on the rare KIE class). |
| `audit_train_val_test_f1.png` | Pillar 7: train vs val vs test F1 per error (train-memorization on rare classes). |
| `audit_squat_ohp_confusion.png` | Squat + OHP confusion matrices, val vs test. |
| `audit_shallow_confusion.png` | Shallow-Squat confusion at the val-tuned threshold, val vs test. |
| `audit_squat_ohp_score_dist.png` | Squat + OHP positive/negative score histograms, val vs test. |
| `audit_shallow_score_dist.png` | Shallow-Squat score distribution, val vs test. |
| `audit_per_seed_variance.png` | Per-seed test F1 spread; the ensemble exceeds every single seed. |
| `audit_ssl_health.png` | SSL backbone health: effective rank and embedding std stay above the collapse floor. |
| `audit_ssl_linear_probe.png` | Frozen-backbone linear probe macro-F1 (well above chance), the opposite of collapse. |
| `audit_headline_vs_paper.png` | Our deployed F1 per error vs the published Parmar et al. numbers. |
| `audit_reeval_certainty.png` | Pillar 2: re-eval from the actual weights reproduces the cache (delta 0). |

Note: the Colab GPU twin additionally emits `audit_live_val_test_gap.png` and
`audit_live_confusion_scoredist.png` when run on Drive; those are not committed (re-run the twin to
regenerate).
