# Phase 6: Overhead Press - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 6-overhead-press
**Areas discussed:** API integration scope, Detector structure

Phase 6 is a faithful re-run of the proven Squat pipeline (Phases 2–4) on OHP data, so most gray areas were settled by prior-phase precedent and carried forward unchanged (preprocessing, loss, fine-tune/SSL recipes, ensemble, threshold, eval, deliverable pattern, working agreement) rather than re-asked. Only the two genuinely OHP-specific decisions below were put to the user. The OHP dataset facts were verified against the local extracted archives during this discussion.

---

## API integration scope

| Option | Description | Selected |
|--------|-------------|----------|
| Model + eval + notebooks only | Train OHP baseline→MD-SSL, report F1 vs paper, ship the 4-notebook pack + FINDINGS; skip backend API wiring (frontend cancelled); descope ROADMAP SC3. | ✓ |
| Full backend parity with Squat | Also extend the form service to load OHP weights + serve OHP endpoints + tests, matching Phase 5. | |
| Minimal — load + /health only | Extend the service to load OHP weights + report in /health, no new endpoint logic/tests. | |

**User's choice:** Model + eval + notebooks only.
**Notes:** Consistent with the user's framing that the form-correction tab is cancelled and "all our work on the model will be paper-dependent — all the details, visualizations, results, comparison with the main paper." ROADMAP Phase-6 SC3 ("OHP selectable through the inference API") is descoped with that reason; OHP-01 is satisfied by trained/evaluated models + paper comparison + the visualization pack.

---

## Detector structure

| Option | Description | Selected |
|--------|-------------|----------|
| Shared backbone, joint 2-output head | Mirror Squat: one MD-SSL backbone → Dropout+Linear(512,2); BCEWithLogitsLoss treats Elbows/Knees independently; one SSL pretrain + one 3-seed fine-tune. | ✓ |
| Two separate single-error models | Independent Elbow and Knees detectors; doubles fine-tune compute. | |

**User's choice:** Shared backbone, joint 2-output head.
**Notes:** Data finding surfaced during discussion — OHP Elbows (upper-body) and Knees (lower-body) are nearly independent (train: only 85/407 Elbow+ clips also Knees+), unlike Squat's 83% KIE/KFE co-occurrence that originally justified a joint head. The joint head is retained, now justified by the shared MD-SSL representation + multi-label BCE handling independent labels + ~½ the compute — not by co-occurrence. No co-occurrence-driven lift should be expected.

## Claude's Discretion

- Factoring of shared split/staging/dataloader code for OHP (parameterize `splits.py`/staging/trainers vs add OHP variants) — deferred to planner/researcher.
- SSL/fine-tune batch sizing re-estimate on L4.
- Whether `ohp.py`'s `build_loaders` is on the training path or a notebook-smoke convenience.

## Deferred Ideas

- OHP backend inference API integration → a future UI milestone if form-correction serving is revived.
- Image-based errors (Shallow-Squat, BarbellRow) → Phase 7.
- Cross-method ensemble + full 3-exercise comparison pack → Phase 8.
- TTA as a shipped default; 5-seed ensemble; weak-aug ablation re-run — all settled by Phase-4 precedent, not re-run.
