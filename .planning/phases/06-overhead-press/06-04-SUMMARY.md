---
phase: 06-overhead-press
plan: 04
status: complete
completed: 2026-05-29
requirements: [OHP-01]
---

# Phase 6 · Plan 04 (fine-tune + ensemble + test eval) — Summary

Fine-tuned the Plan-03 MD backbone for joint Elbows/Knees, ensembled, evaluated on the
official OHP test split, wrote `results.pkl`. **This is the headline OHP result.**

## Headline (official 339-clip OHP test split, 2-seed MD-SSL ensemble, val-tuned thresholds, no TTA)

| | Elbows | Knees | macro |
|---|---|---|---|
| baseline (Plan 02 control) | 0.4167 | 0.8069 | 0.6118 |
| **our MD-SSL ensemble** | **0.4474** | **0.8770** | **0.6622** |
| paper Ours-MD (Table 4) | 0.4552 | 0.8452 | 0.6502 |

- **SSL lift over the baseline: +0.0504 macro** (both errors rose).
- **Matches/edges the paper:** macro 0.6622 vs 0.6502; **beats the paper on Knees**
  (0.877 vs 0.845, AP 0.937); Elbows 0.447 a hair under the paper's 0.455 (the dataset's
  hardest error; AP 0.369).
- **val→test gap = 0.011** — the generalization is tight (Squat's final was 0.026); no overfitting.
- Ensemble beats both single seeds (s42 0.442/0.859, s1337 0.416/0.855 → 0.447/0.877).
- Production thresholds: Elbows 0.357 / Knees 0.476.

## Deviations from PLAN (defensible)

- **2-seed ensemble (42, 1337), not 3** — time-driven; seed 7 skipped. Identical
  mean-of-sigmoids methodology; both seeds clean (`aborted_overfit=False`, early-stop
  selected a pre-overfit epoch — 42@ep6, 1337@ep8).
- **TTA not re-tuned** — Phase 4 evaluated it and it reversed on test (not adopted); the
  no-TTA ensemble is the headline, consistent with the Phase-4 protocol.
- **`md_finetune` seams** — added `checkpoint_phase` (it hardcoded phase04 → OHP checkpoints
  would miss phase06) and fixed the `best.pt` write to `update_latest=False` (a disconnect
  after a best epoch was resuming from `best.pt`, not the latest epoch).

## Overfitting handling (the explicit guard)

The fine-tune showed clean overfit control: `val/train` stayed ≤~2 through each seed's best
epoch, then climbed (42 → 9.6 by ep14, 1337 → 12.6 by ep16); the 8-epoch early-stop selected
the pre-overfit best epoch in both, so `best.pt` is from before overfitting set in. The
reported model's **val→test gap (0.011)** is the generalization proof.

## Artifacts

- Drive: `phase06/ohp_md_finetune_seed{42,1337}/best.pt`.
- `figures/results.pkl` (written on Colab — source of truth for Plan 05; **to be copied to
  Drive so Plan 05 builds the pack locally**).
- `harness/md_finetune.py` (seams), notebook `08_ohp_md_finetune.{py,ipynb}`,
  `docs/figures/ohp_f1_vs_paper.png` (headline chart), `backend/scripts/ohp_headline_chart.py`.

## Next

Plan 05 — the 4-notebook OHP visualization pack (EDA / pipeline / training / evaluation) +
figures + `FINDINGS.md`, built from `results.pkl`; then the phase SUMMARY.
