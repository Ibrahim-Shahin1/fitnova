# Overhead Press Form-Error Detection — Benchmark Evaluation & Findings

**Model:** R(2+1)D-18, Motion-Disentangling SSL pretrain → fine-tune, 2-seed ensemble
(mean-of-sigmoids), production thresholds Elbows 0.357 / Knees 0.476.
**Evaluation set:** Fitness-AQA **official OHP test split**, 339 clips (Elbows 86 pos /
253 neg; Knees 128 pos / 211 neg). Identical metric (F1 per error) and split as
Parmar et al. (ECCV 2022).

## Headline results

| Error | Ours (official, Phase 6) | Parmar Ours-MD (paper, Table 4) | Our supervised baseline |
|-------|--------------------------|----------------------------------|-------------------------|
| **Elbows** | **0.4474** (AP 0.369) | 0.4552 | 0.4167 |
| **Knees**  | **0.8770** (AP 0.937) | 0.8452 | 0.8069 |
| **macro**  | **0.6622** | 0.6502 | 0.6118 |

**Our MD-SSL ensemble matches/edges the published paper on the identical metric and
split** (macro 0.6622 vs 0.6502; **beats it on Knees**, 0.877 vs 0.845), and beats our
supervised baseline by **+0.050 macro** (the measured SSL contribution). Elbows is the
intrinsically hard error in the dataset — even the paper's best is 0.455 — and our 0.447
sits a hair under it (within test-set noise on 86 positives).

**Generalization (overfitting check):** the reported model's **val→test macro gap is
0.011** — the held-out test F1 essentially matches validation. The un-regularized baseline
showed an Elbows val→test gap of ~0.10; the MD-SSL backbone + weight decay + dropout +
2-seed ensemble + early-stop close it. No overfitting in the reported model.

## Figures (`docs/figures/`)

| File | What it shows |
|------|---------------|
| `ohp_f1_vs_paper.png` | Headline bar — baseline → our MD-SSL → paper Ours-MD, per error + macro. |
| `ohp_eda_balance.png` | Class balance per split + Elbows/Knees co-occurrence (the errors are largely independent — only 85/407 Elbows+ are also Knees+). |
| `ohp_ssl_curves.png` | MD-SSL pretrain — loss falls, `effective_rank` stays healthy (no representation collapse), linear-probe converges (peak → `backbone.pt`). |
| `ohp_finetune_curves.png` | Per-seed fine-tune — `val/train` ratio climbs after the best epoch; the 8-epoch early-stop keeps the pre-overfit `best.pt`. |
| `ohp_pr_curves.png` | Precision-recall + operating point + AP (Knees AP 0.94; Elbows 0.37, the hard class). |
| `ohp_confusion.png` | Confusion at the production thresholds. |
| `ohp_score_distributions.png` | Per-error score separation, positives vs negatives + threshold. |

## Method (identical to the published approach)

Supervised baseline (Kinetics-init R(2+1)D-18, class-weighted) established the control,
then Motion-Disentangling self-supervised pretraining on the 5,089 usable unlabeled OHP
clips (half-cycle contrast on the barbell trajectory), fine-tuned for joint Elbows/Knees
detection. F1 per error on the official splits, thresholds tuned on validation. The joint
two-output head is justified by the shared MD-SSL motion representation + efficiency, **not**
error co-occurrence (Elbows is upper-body, Knees lower-body — they rarely co-occur).

## Honest scope notes

- **2-seed ensemble** (seeds 42, 1337), a time-driven reduction from the 3-seed Squat
  protocol — identical mean-of-sigmoids methodology, both seeds clean.
- **TTA not adopted** — evaluated in the Squat phase and found to reverse on the test split;
  the no-TTA ensemble is the reported headline.
- **501 unlabeled clips excluded** from SSL (401 with <50% barbell-detection coverage + the
  detector-failure tail) — the half-cycle split needs a reliable trajectory.
- Like Squat, these results are on the **benchmark** (in-the-wild gym video, side-on). The
  form-correction frontend is cancelled (the model does not transfer to home/phone camera —
  the known in-the-wild limit); the defensible deliverable is this benchmark evaluation.

## Reproduce

```
# results.pkl (per-clip scores + curves) is committed at
#   .planning/phases/06-overhead-press/figures/results.pkl
python backend/scripts/ohp_eval_viz.py        # SSL / PR / confusion / score-dist figures
python backend/scripts/ohp_finetune_curves.py .planning/phases/06-overhead-press/figures/results.pkl
python backend/scripts/ohp_headline_chart.py  # the F1-vs-paper bar
python backend/scripts/ohp_eda_viz.py         # class balance + co-occurrence (needs the local archive)
```
