---
phase: 08
plan: 02
subsystem: evaluation
tags: [evaluation, findings, phase-close, milestone-complete, no-fabrication]
dependency_graph:
  requires:
    - .planning/phases/08-ensemble-evaluation-visualization-pack/08-01-SUMMARY.md
  provides:
    - docs/eval/FINDINGS_FULL.md
    - docs/notebooks/README.md (Full Evaluation section)
  affects: []
tech_stack:
  added: []
  patterns:
    - EVAL-02 not-applicable pattern: each error assigned to exactly one method -- no cross-method ensemble possible
    - GYMetricPose+LMM as labeled context-only paragraphs with comparability caveats -- not table columns
    - Per-phase detail links connecting consolidated FINDINGS_FULL to exercise-level FINDINGS docs
key_files:
  created:
    - docs/eval/FINDINGS_FULL.md
  modified:
    - docs/notebooks/README.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
key_decisions:
  - "EVAL-02: evaluated -> not-applicable; paper applies MD+CVCSPC ensemble only to BackSquat KIE/KFE (0.5263/0.8468); our scope has one method per error so no error has both scores"
  - "GYMetricPose + LMM as labeled context paragraphs in FINDINGS_FULL -- not table columns (D3); split-protocol comparability unconfirmed for GYMetricPose; LMM trains on train+val (not directly comparable)"
  - "D7 honest framing carried into consolidated write-up: no OHP/Shallow paper supervised baseline; 3-term vs 2-term loss; TTA not adopted; OHP 2-seed; benchmark-only / domain-shift caveat"
requirements_completed: [EVAL-01, EVAL-02, EVAL-03]
metrics:
  duration_minutes: 15
  completed_date: "2026-05-31"
  tasks_completed: 2
  files_changed: 4
---

# Phase 8 Plan 02: FINDINGS_FULL + Phase Close Summary

**FINDINGS_FULL.md consolidating the 5-error EVAL-01 master table, the EVAL-02 not-applicable finding,
GYMetricPose/LMM related-work context, and D7 honest-framing notes; Phase 8 and v1.0 milestone closed.**

## What Was Built

### Task 1: `docs/eval/FINDINGS_FULL.md` + `docs/notebooks/README.md` update

**FINDINGS_FULL.md (9 sections):**

1. **EVAL-01 master comparison table** -- the 5-row spine (Exercise/Error, Modality, Ours, Parmar SSL,
   Parmar baseline), sourced verbatim from results.pkl via the notebook assertions.
   Macro results: Squat 0.6304 vs 0.6262; OHP 0.6622 vs 0.6502; Shallow 0.8902 vs 0.8694.
   SimSiam 0.8286 / OpenPose-TDM 0.8340 noted in prose for Shallow context.

2. **EVAL-02 -- evaluated, not-applicable** -- the paper does apply MD+CVCSPC to BackSquat KIE/KFE
   (Table 2: KIE 0.5263 / KFE 0.8468) but exclusively to those two errors. Our 5-error scope
   assigns exactly one method per error; no error has both MD and CVCSPC scores; forming the
   ensemble would require either (a) out-of-scope CVCSPC re-run or (b) architecturally invalid MD
   on static crops. Conclusion: evaluated -> not-applicable.

3. **Related-work context (GYMetricPose + LMM)** -- labeled context paragraphs only (no columns).
   GYMetricPose: Squat KIE 0.4398 / KFE 0.8219 / OHP Elbows 0.4175 / Knees 0.8160 (from
   Dibenedetto et al. Table 4; direct split verification not possible -- caveat stated).
   LMM Two-Step: Squat KIE 0.1955 / KFE 0.6266 / OHP Elbows 0.4575 / Knees 0.7611 (train+val
   used -- not directly comparable; caveat stated + citation: UMAP Adjunct '25,
   DOI 10.1145/3708319.3733684).

4. **Cross-exercise synthesis** -- SSL lift over baselines (+0.087 Squat, +0.050 OHP, +0.0152 Shallow);
   val->test gaps (OHP 0.011, Shallow 0.0217); reference to 4 master figures.

5. **Figures table** -- all_errors_vs_paper.png, cross_exercise_macro.png, methodology_diagram.png,
   highlights_panel.png with one-line descriptions.

6. **Honest framing (D7)** -- 5 explicit bullets: no OHP/Shallow paper supervised baseline (0.6118
   and 0.8750 are the only supervised controls); 3-term vs 2-term CVCSPC loss deviation; TTA not
   adopted (reverses on test, delta -0.0073); OHP 2-seed vs Squat/Shallow 3-seed; benchmark-only /
   domain-shift caveat.

7. **Per-phase detail links** -- FINDINGS.md (Squat), FINDINGS_OHP.md (OHP), FINDINGS_SHALLOW_SQUAT.md.

8. **Reproduce block** -- nbconvert command + three results.pkl paths.

**README.md update:** Added "Full Evaluation (all 3 exercises)" section after the Shallow-Squat block,
with a one-row table (12_full_evaluation | cross-exercise master comparison + EVAL-02 + 4 figures),
a headline sentence on the 5-error spine, and a link to FINDINGS_FULL.md.

### Task 2: Phase close -- ROADMAP + REQUIREMENTS + STATE + suite

- **ROADMAP.md** Phase 8 checkbox flipped to [x] (complete 2026-05-31); Phase 8 success criteria
  marked satisfied; 08-02-PLAN.md entry flipped to [x]; Progress table row updated 2/2 / Complete.
- **REQUIREMENTS.md** EVAL-01/EVAL-02/EVAL-03 flipped to [x] with completion notes; Traceability
  row EVAL-02 flipped Pending -> Complete.
- **STATE** updated via gsd-sdk mutations (state.record-session, roadmap.update-plan-progress,
  requirements.mark-complete). Status: ready_for_verification.
- **AQA suite**: `python -m pytest backend/training/aqa/ -m "not slow" -q` -> **45 passed, 4 deselected**.

## v1.0 Milestone Status

**This plan (08-02) is the final plan of the v1.0 form-correction milestone.** Phase 8 is complete
on this plan close. The formal milestone-close step (orchestrator-level reconciliation) is run
separately by the orchestrator after verification -- it is NOT run here.

The v1.0 milestone delivered:
- Phases 1-4: Squat vertical slice (dataset -> pipeline -> supervised baseline -> MD-SSL);
  test macro 0.6304 matching paper MD-SSL 0.6262 on the official split.
- Phase 5: Backend inference integration (Squat live + upload API).
- Phase 6: OHP video pipeline; test macro 0.6622 vs paper 0.6502; +0.050 SSL lift.
- Phase 7: Shallow-Squat CVCSPC image pipeline; test F1 0.8902 beating paper 0.8694 (+0.021).
- Phase 8: Consolidated evaluation (notebook 12 + FINDINGS_FULL + 4 master figures); EVAL-02
  settled as not-applicable; GYMetricPose + LMM context documented honestly.

## EVAL-01 Spine (source of truth: results.pkl)

| Exercise/Error | Modality | Ours | Parmar SSL | Parmar baseline |
|---|---|---|---|---|
| Squat KIE | video (MD) | 0.4198 | 0.4186 | 0.2970 |
| Squat KFE | video (MD) | 0.8410 | 0.8338 | 0.8184 |
| OHP Elbows | video (MD) | 0.4474 | 0.4552 | -- |
| OHP Knees | video (MD) | 0.8770 | 0.8452 | -- |
| Shallow-Squat | image (CVCSPC) | 0.8902 | 0.8694 | -- |

All values loaded from committed results.pkl in notebook 12_full_evaluation.ipynb; asserted <=1e-6 vs stored test_f1.

## Task Commits

1. **Task 1: FINDINGS_FULL.md + README nb-12 row** -- da843c7 (docs)
2. **Task 2 + plan metadata** -- see final docs(08) commit (docs)

## Deviations from Plan

None -- plan executed exactly as written. All numbers sourced from RESEARCH.md / results.pkl verbatim;
no fabrication; GYMetricPose + LMM treated as context-only per D3; honest framing carried per D7.

## Known Stubs

None -- FINDINGS_FULL.md is fully wired (master table from pickles, comparator context from research).

## Threat Flags

None -- local documentation writes from committed artifacts; no new network surface.

## Self-Check

- `docs/eval/FINDINGS_FULL.md` -- FOUND (commit da843c7)
- `docs/notebooks/README.md` -- FOUND (commit da843c7)
- Grep 0.5263 in FINDINGS_FULL -- FOUND (EVAL-02 KIE)
- Grep 0.4398 in FINDINGS_FULL -- FOUND (GYMetricPose)
- Grep 0.1955 in FINDINGS_FULL -- FOUND (LMM)
- Grep FINDINGS_FULL in README -- FOUND
- Grep 12_full_evaluation in README -- FOUND
- AQA suite: 45 passed, 4 deselected
- ROADMAP Phase 8: [x] (complete 2026-05-31)
- REQUIREMENTS EVAL-02: [x] (complete)

## Self-Check: PASSED

---
*Phase: 08-ensemble-evaluation-visualization-pack*
*Completed: 2026-05-31*
