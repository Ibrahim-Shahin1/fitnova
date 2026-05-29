# Validation Strategy: Phase 7 - image-based-errors-cvcspc

**Created:** 2026-05-30
**Status:** Active

## Purpose

This document defines HOW each requirement and component will be validated. The planner reads this to embed validation requirements (Dimension 8) into each task.

## Validation Architecture

Phase 7 is Shallow-Squat (image, ResNet-18) — supervised ImageNet baseline first, then faithful
CVCSPC pose-contrastive SSL on the unlabeled Back-Squat set, fine-tune, measure the lift. The
genuinely-new logic (CVCSPC phase-matched triplet + 3-term loss) is unit-tested on synthetic data
BEFORE any GPU run (the Phase-4/6 Wave-0 precedent). Integration validation is via Colab
paste-back of real metrics (no fabricated numbers). IMG-03 (BarbellRow) is descoped — not validated.

## Per-Requirement Validation

| Requirement | Validation Type | Test Approach | Success Signal |
|-------------|-----------------|---------------|----------------|
| IMG-01 (CVCSPC ResNet-18 pipeline: baseline + SSL) | Integration | CVCSPC SSL pretrain on 4,970 unlabeled Squat clips; triplet-accuracy (AP<AN) monitor | triplet-accuracy rises above 50% chance on val; `backbone.pt` produced; fine-tune from it runs |
| IMG-02 (Shallow-Squat detector F1 on official split) | Integration | Train (baseline + CVCSPC-init) on 2542 train crops; eval on 540 test crops at val-tuned threshold | F1 reported on the official 540-crop test split; compared to paper CVCSPC 0.869 (computed in code) |
| Unit: `shallow_squat` dataset | Unit | construct dataset on synthetic crops + mock split/label JSON | returns `(3,224,224)` float32 + binary label; splits reconcile 2542/529/540; ImageNet norm applied |
| Unit: CVCSPC 3-term loss | Unit | hand-computed value on synthetic feature triplets | matches `train_test.py:64-68` closed form within 1e-6; AP≈P & far-from-N ⇒ lower loss |
| Unit: phase-matching triplet | Unit | synthetic bar-trajectories | anchor/positive at shared phase; negative ≥ phase-gap away; indices in range |
| Integration: frame-extraction (Step 0) | Integration | decode 4,970 mp4s → per-clip frame dirs on Colab | frame dirs created; skip-existing idempotent; disconnect-safe |
| Integration: SSL convergence + lift | Integration | triplet-accuracy curve over epochs; baseline vs CVCSPC-init test F1 | triplet-accuracy improves; the baseline→CVCSPC F1 delta reported honestly (lift or no-lift) |

## Validation Levels

### Unit Tests
- `shallow_squat` dataset shape + label + split reconciliation + ImageNet norm (CPU, synthetic).
- CVCSPC 3-term loss closed-form + directionality (CPU, synthetic) — the highest-leverage pre-GPU check.
- Phase-matching triplet construction on synthetic bar-trajectories (CPU).
- No-regression: the existing Phase-3/4/6 `backend/training/aqa/` suite stays green.

### Integration Tests
- Frame-extraction Step 0 (idempotent, skip-existing) on real unlabeled clips.
- CVCSPC SSL pretrain triplet-accuracy monitor (Colab paste-back).
- Supervised baseline + CVCSPC-init fine-tune training curves (Colab paste-back).
- Final test eval at val-tuned threshold + ensemble (Colab paste-back).

### End-to-End / Manual Validation
- The deliverable notebook pack executes end-to-end from `results.pkl` (no GPU) — figures render.
- Human reads the baseline→CVCSPC→paper(0.869) comparison + val→test gap for honesty/no-overfitting.

## Coverage Targets

- Requirements with validation: 2/2 in scope (IMG-01, IMG-02). IMG-03 descoped (0 — intentional).
- Critical paths covered: CVCSPC loss/triplet (pre-GPU unit), SSL convergence, supervised F1, the lift.

## Notes

- Wave-0 CPU unit gate before any GPU (Phase-4/6 precedent): a wrong CVCSPC loss/triplet would waste
  the frame-extraction + SSL compute.
- Code-vs-paper loss deviation (3-term vs 2-term) is implemented per code and documented in FINDINGS,
  not silently reconciled.
- All headline numbers computed in code from real scores ([[feedback_ai_correctness]]).
