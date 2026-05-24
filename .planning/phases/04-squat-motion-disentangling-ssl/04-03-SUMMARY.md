---
phase: 04-squat-motion-disentangling-ssl
plan: 03
status: complete
completed: 2026-05-24
---

# Phase 4 · Plan 03 (3-seed fine-tune) — Summary

> Reconciled retroactively (2026-05-25): code tasks verified locally; the 3-seed fine-tune ran
> interactively on Colab with paste-back. This SUMMARY records the outcome.

## What was planned

Under `04-03-PLAN.md`: finalize `md_finetune.run_md_finetune_epoch` (parallel Phase 3's
`run_supervised_epoch` + the 4 D3 deltas + the D6 runtime overfit monitor) (T1); extend
`test_md_finetune.py` with `test_d6_overfit_monitor` (T2); then a 3-seed fine-tune (42/1337/7, the
seed-42-first reg-bump gate, then 1337/7 at the locked recipe) producing 3 `best.pt` for Plan 04.

## What was done

- **`run_md_finetune_epoch` finalized (T1):** MD-backbone init (`build_finetune_model`, NOT Kinetics),
  AdamW (wd 1e-4), `Dropout(0.2)+Linear(512,2)` head, `BCEWithLogitsLoss(pos_weight)`, cosine, 50-ep /
  8-patience early-stop, the D6 monitor + `aborted_overfit` early-return. Reuses Phase 3's
  `_build_dataloaders`/`_val_pass`/`_set_global_seed` via import (D9 — `supervised_train.py` untouched).
  `best.pt` payload mirrors the Phase 3 schema (`code_version "phase04-md-finetune"`). Verified locally:
  AST + signature + the 3 acceptance greps + `test_d6_overfit_monitor` green.
- **3-seed fine-tune (Colab, paste-back):** all from `md_pretrain_v2/backbone.pt`, base recipe (no D6
  trip). Timing probe: ~6.1 min/epoch. Best **val macro-F1**: seed42 = **0.6062** (KIE 0.411 / KFE 0.801,
  ep8), seed1337 = **0.6055** (0.431 / 0.780, ep2), seed7 = **0.6169** (KIE **0.493** / KFE 0.741, ep8).
  Mean **~0.609 ± 0.006** — low seed variance (trustworthy ensemble); diverse KIE/KFE balance across
  seeds (good ensemble inputs). All 3 beat Phase 3 (0.543) and the paper's Kinetics (0.558); approaching
  the paper's MD (0.626 test vs ours val). 3 `best.pt` on Drive at `phase04/md_finetune_seed{42,1337,7}/`.

## Key verifications

- Per-seed val macro recorded (D4 variance evidence): tight 0.6055–0.6169 band across 3 seeds.
- The val-macro-F1 early-stop is the effective overfit guard — val loss diverged above train in every
  seed; early-stop selected the val-macro-best epoch (ep8/ep2/ep8) before macro degraded.

## Deviations from PLAN

- **D6 monitor was inverted — FIXED (commit `6f0587b`):** the Plan 03 spec defined the overfit ratio as
  `train/val > 10`, which flags *under*-fitting and can never catch the Phase-3 overfit it exists for
  (overfit = val loss far ABOVE train; Phase 3's "~32×" is val/train). Corrected `_d6_overfit_abort` to
  `val/train`; updated the per-epoch metric key (`val_train_loss_ratio`), the log lines, and the unit
  test (now also asserts the inverted `train>>val` case does NOT abort — guards the regression). No effect
  on the 3 seeds (none exceeded the ratio before epoch 10 in either direction); validated by seed 7's real
  late overfit (val/train **13×** by ep16, after the `epoch<10` guard window — the early-stop handled it).
- **D6 reg-bump never triggered:** the base recipe (wd 1e-4, dropout 0.2) kept all seeds healthy, so the
  recipe stayed locked across all 3 (homogeneous ensemble), zero wasted compute — the common case the
  policy was designed for.

## Artifacts (committed on `fresh-start`)

- `harness/md_finetune.py` (finalized `run_md_finetune_epoch` + `_d6_overfit_abort`),
  `harness/test_md_finetune.py` (`test_d6_overfit_monitor`). 3 `best.pt` on Drive.
- Commits: `e83242a` (Task 1), `f1d9613` (Task 2), `6f0587b` (D6 fix).
- NOTE: the Plan 03 fine-tune cells were delivered as Colab chat-blocks (paste-back); their consolidation
  into a committed `05_squat_md_finetune.{py,ipynb}` is folded into Plan 04 (which extends the same
  notebook with the eval steps).

## Next

Plan 04 — mean-of-sigmoids ensemble + val-tuned TTA + test eval at val-tuned thresholds + 9 figures +
the Phase 3-vs-Phase 4 comparison chart + the phase SUMMARY / Phase 5 handoff.
