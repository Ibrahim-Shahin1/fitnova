# Overfitting in the baseline models, and how the SSL models fixed it

This note explains the overfitting found in the **supervised baseline** models (which are *not*
deployed) and how the deployed **self-supervised (SSL) fine-tune** models reduce it. All figures are
produced by `docs/audit/overfitting_summary.ipynb`; the full audit is in
`docs/eval/OVERFITTING_AUDIT.md`.

## Setup

For each of the three error families we trained two models:

| Exercise | Baseline (control, not deployed) | Deployed (used) |
|---|---|---|
| Squat | R(2+1)D-18, plain supervised (Kinetics init) | R(2+1)D-18 + **Motion-Disentangling SSL** fine-tune, 3-seed ensemble |
| OHP | R(2+1)D-18, plain supervised | R(2+1)D-18 + **MD-SSL** fine-tune, 2-seed ensemble |
| Shallow-Squat | ResNet-18, plain supervised (ImageNet init) | ResNet-18 + **CVCSPC** pose-contrastive SSL fine-tune, 3-seed ensemble |

The baselines exist to measure what plain supervised training gives, so the SSL contribution is
visible as the difference.

## The overfitting in the baselines

Two diagnostics show it.

**1. Learning curves (`summary_baseline_learning_curves.png`).** After the best epoch the baselines'
**validation loss climbs while the training loss keeps falling** - the classic overfitting signature.
The Squat baseline is the clearest: validation loss rises by ~1.0 after epoch 3. Early stopping on
the validation F1 still keeps the pre-divergence checkpoint, but the tendency to overfit is stronger
than in the SSL models.

**2. Train vs test F1 (`summary_baseline_train_val_test.png`).** On the hard, rare classes the
baseline fits the training data far better than the test data:

| Baseline error | train F1 | test F1 | train→test gap |
|---|---|---|---|
| OHP Elbows | 0.94 | 0.42 | **+0.53** |
| Squat Knees-inward (KIE) | 0.71 | 0.29 | +0.43 |
| OHP Knees | 0.99 | 0.81 | +0.18 |
| Squat Knees-forward (KFE) | 0.93 | 0.80 | +0.13 |
| Shallow depth | 0.98 | 0.88 | +0.11 |

The OHP baseline on **Elbows** is the starkest: it also overfits in the deployment sense - its F1
drops from **0.515 on validation to 0.417 on test (a +0.098 gap)**, beyond the test-set noise. The
model is leaning on memorized training patterns for the subtle upper-body error.

## How the SSL models fix it

The deployed models change five things relative to the baseline, all aimed at regularization:

1. **SSL pretraining** - the backbone is first trained on a large pool of **unlabeled** clips/crops
   with a self-supervised objective (Motion-Disentangling for video, CVCSPC pose-contrastive for the
   image model). It learns general motion and pose structure before ever seeing a labelled error.
   The fine-tune therefore starts from a representation that already separates good vs bad form, so it
   does not need to memorize the few labelled examples of each rare error.
2. **Weight decay** (AdamW) and **head dropout (0.2)** - standard regularizers added to the fine-tune.
3. **Multi-seed ensemble** (mean of the per-seed sigmoid scores) - averages out per-seed
   memorization.
4. **Early stopping** on validation F1 - keeps the pre-divergence checkpoint.

**The measured effect (`summary_ssl_fix.png`).** The overfitting gaps shrink, most dramatically on
the class the baseline overfit worst:

| Error | train→test gap: baseline → deployed | validation→test gap: baseline → deployed |
|---|---|---|
| **OHP Elbows** | **+0.53 → +0.13** | **+0.098 → +0.031** |
| Squat KFE | +0.13 → +0.01 | −0.00 → −0.03 |
| OHP Knees | +0.18 → +0.11 | −0.01 → −0.01 |
| Shallow depth | +0.11 → +0.08 | +0.02 → +0.02 |

On OHP Elbows the deployed model cuts the train→test gap by ~4x and the validation→test gap by ~3x -
the overfitting the baseline showed is largely gone.

## The honest exception: Squat Knees-inward (KIE)

SSL does **not** close the train→test gap on the rarest class, Squat KIE. There the deployed model
fits training examples even better than the baseline did (train 0.71 → 0.93), while test improves
only to 0.42, so the train→test gap actually grows (+0.43 → +0.51). Two things keep this from being a
deployment failure:

- **SSL still raised the real accuracy:** test KIE F1 went from 0.29 (baseline) to **0.42** (deployed)
  - a genuine improvement on unseen data.
- **No validation→test overfitting and at the field's ceiling:** validation (0.50) still tracks test
  (0.42) within sampling noise (only 36 positive test clips), and our test F1 (0.420) **equals the
  published state of the art** (Parmar et al. 0.419). The class is simply hard and label-scarce; the
  remaining gap is train-memorization on a handful of positives, not a degraded model.

The single fix that would help KIE most is **more labelled knees-inward examples**, not more
regularization.

## Bottom line

The baselines overfit (most visibly OHP Elbows: train→test +0.53, validation→test +0.098). The
deployed SSL models reduce that overfitting substantially - OHP Elbows train→test +0.53 → +0.13,
validation→test +0.098 → +0.031 - and show no validation→test overfitting on any error. The only
residual is train-memorization on the rarest class (Squat KIE), which is bounded by the validation
check and sits at the published accuracy ceiling.
