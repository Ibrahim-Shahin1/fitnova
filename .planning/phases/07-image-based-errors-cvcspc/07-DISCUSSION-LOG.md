# Phase 7 — Discussion Log

**Date:** 2026-05-30
**Mode:** /gsd:discuss-phase (interactive)

## Context going in
- User cancelled BarbellRow (compute/time) before this discussion.
- Phases 1–6 complete; Squat + OHP (video) match the paper. Phase 7 is the image modality (new method).
- Data grounded before discussion: Shallow-Squat 3,611 crops, splits 2542/529/540, 43.9% pos, flat {id:0/1} labels.

## Questions & decisions

**Q1 — IMG-03 (BarbellRow) recording** → **Descope (cut from milestone).**
Marked out-of-scope (compute/time). Milestone form-correction scope = Squat + OHP + Shallow-Squat.

**Q2 — CVCSPC SSL risk handling** → **Research-gated, baseline ships first.**
ImageNet ResNet-18 baseline first (guaranteed IMG-02); researcher then resolves CVCSPC-for-
Shallow-Squat-alone and attempts the lift. Baseline + honest write-up is the acceptable floor.

**Q3 — Image input pipeline** → **New `datasets/shallow_squat.py` + ImageNet norm.**
JPEG crop → 224² → ImageNet mean/std (ResNet-18 native). No video decode/temporal axis.

**Q4 — Rigor** → **Match the others (multi-seed ensemble).**
3-seed (or 2-seed) ResNet-18 ensemble + val-tuned threshold; consistent with Squat/OHP; cheap.

## Deferred / out
- BarbellRow (IMG-03), CVCSPC cross-exercise transfer — cancelled this milestone.
- API/serving — descoped (D1, same as OHP).

## Claude's discretion (delegated)
- Exact CVCSPC reconstruction mechanics → researcher (from `CVCSPC.py` + paper).
- Final notebook count (2 vs 4) → planner, based on whether CVCSPC SSL lands.
