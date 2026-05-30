# Phase 8: Ensemble, Evaluation & Visualization Pack - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 8-ensemble-evaluation-visualization-pack
**Areas discussed:** Consolidated viz-pack ambition, External-comparator presentation (GYMetricPose/LMM)

> Most Phase-8 gray areas were resolved without a user prompt — settled by the master plan, the continuation handoff, the user's prompt, and Phase 4/6/7 precedent (per [[feedback_run_gsd_autonomously]]): LOCAL-only venue (no Colab/re-runs), EVAL-01 ours-vs-Parmar master table, EVAL-02 research-confirm-then-report-N/A (no invented ensemble), deliverable medium (notebook + FINDINGS + figures), no-fabrication. The two decisions below were the genuinely-open, defense-shaping forks the handoff flagged — put to the user.

---

## Consolidated viz-pack ambition (EVAL-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Comprehensive defense pack | Notebook 12 + FINDINGS_FULL + master figures: all-5-errors grouped bar, cross-exercise macro summary, two-method methodology/architecture diagram, per-exercise highlights panel. | ✓ |
| Focused comparison pack | Master table + one all-errors headline bar + consolidated notebook/FINDINGS; no methods diagram or extra panels. | |

**User's choice:** Comprehensive defense pack.
**Notes:** Aligns with [[project_supervisor_visualizations]] (viz heavily weighted at the 2026-06-03 defense; this is the FINAL consolidated viz deliverable). The methodology diagram is the one genuinely-new (non-results.pkl) asset and a high-value defense figure. All figures derive from the three committed `results.pkl` — fully local, no re-runs.

---

## External-comparator presentation — GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025)

| Option | Description | Selected |
|--------|-------------|----------|
| Columns where identical, else context | Add as comparison columns for errors on the identical Fitness-AQA metric/split; context note otherwise; omit + say so if unsourceable. | |
| Context note only — no columns | Keep ours-vs-Parmar as the only comparison columns; report GYMetricPose/LMM as a sourced related-work narrative paragraph, no columns. | ✓ |

**User's choice:** Context note only — no columns.
**Notes:** The more defensible stance. A table column implies identical-metric comparability; GYMetricPose/LMM may use different metrics/splits/error definitions, so a column would be apples-to-oranges. The EVAL-01 master table stays a clean ours-vs-Parmar comparison; the external points are credited in a cited narrative. LMM paper is local (`Fitness-AQA/Fine-Tuning Large Multimodal Models...pdf`); GYMetricPose needs a web pass; unsourceable numbers are omitted-and-noted, never fabricated ([[feedback_ai_correctness]]).

---

## Claude's Discretion

- Exact final figure count + filenames within the comprehensive set; inline-in-notebook vs a small `backend/scripts/full_eval_viz.py`.
- Whether FINDINGS_FULL restates the key per-phase tables or links them (lean: self-contained consolidated table + links).
- Plan/wave shape (likely one local plan) — planner decides.

## Deferred Ideas

- BarbellRow / IMG-03 (cancelled this milestone).
- Backend serving / unified 3-exercise API (descoped — frontend cancelled).
- A true MD+CVCSPC cross-method ensemble (N/A to single-method-per-error scope).
- Live/phone-camera domain adaptation (in-the-wild limit, out of milestone scope).
