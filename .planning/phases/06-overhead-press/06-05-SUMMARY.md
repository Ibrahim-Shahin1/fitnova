---
phase: 06-overhead-press
plan: 05
status: complete
completed: 2026-05-30
requirements: [OHP-01]
---

# Phase 6 · Plan 05 (visualization pack + phase closeout) — Summary

The deliverable pack for OHP, built locally from `results.pkl` (Plan 04) + the local archive.

## What was done

- **7 figures** (`docs/figures/ohp_*.png`): headline F1-vs-paper, EDA class-balance +
  co-occurrence, SSL curves (loss + effective_rank no-collapse + linear-probe), per-seed
  fine-tune overfit control, PR curves, confusion, score distributions.
- **4 executed notebooks** (`docs/notebooks/05-08_ohp_*`): EDA / data-pipeline / training /
  evaluation — mirror the Squat pack (01-04). All executed cleanly (12 embedded figures, 0
  errors); regenerate from `results.pkl` without GPU.
- **`docs/eval/FINDINGS_OHP.md`**: headline, +0.050 SSL lift, val→test gap 0.011, honest scope.
- **Viz scripts** (`backend/scripts/ohp_{eval,eda,finetune_curves,headline_chart}.py`) +
  `results.pkl` committed (source of truth).

## Phase 6 headline (OHP-01)

OHP Elbows/Knees, official 339-clip test split, 2-seed MD-SSL ensemble:

| | Elbows | Knees | macro |
|---|---|---|---|
| baseline | 0.4167 | 0.8069 | 0.6118 |
| **MD-SSL ensemble** | **0.4474** | **0.8770** | **0.6622** |
| paper Ours-MD | 0.4552 | 0.8452 | 0.6502 |

Matches/edges the published paper on the identical metric + split (beats it on Knees);
+0.050 SSL lift over the baseline; val→test gap 0.011 (no overfitting).

## Deviations / honest scope

- 2-seed ensemble (time-driven, vs Squat's 3); TTA not adopted (Phase 4 settled it reverses
  on test); 501 unlabeled clips excluded from SSL (<50% barbell coverage).
- **OHP-01 serving** (API integration) was descoped in CONTEXT D1 (the form-correction
  frontend is cancelled; benchmark eval is the deliverable). OHP-01's model + F1-per-error +
  paper comparison are complete.

## Phase 6 close

Plans 01-05 complete. OHP-01 satisfied (model trained, F1 per error on the official splits,
compared to the published numbers, full visualization pack). Backbone/fine-tune checkpoints
on Drive (`phase06/`); all code + figures + notebooks on `fresh-start`.

## Next

Phase 7 — image-based errors (Shallow-Squat + BarbellRow) via CVCSPC; or milestone review.
