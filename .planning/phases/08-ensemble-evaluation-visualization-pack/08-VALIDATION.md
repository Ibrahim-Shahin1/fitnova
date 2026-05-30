---
phase: 8
slug: ensemble-evaluation-visualization-pack
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-30
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Phase 8 adds NO production code — it assembles a consolidated eval/viz pack from three
> committed `results.pkl`. Validation = (a) the existing AQA suite stays green, (b) the
> notebook pre-executes cleanly, (c) every headline F1 is recomputed-in-code from the pickle
> raw scores and asserted equal to the stored value (structural no-fabrication enforcement),
> (d) every figure file is written.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (existing) + in-notebook assertions |
| **Config file** | none — invoked directly |
| **Quick run command** | `python -m pytest backend/training/aqa/ -m "not slow" -q` |
| **Full suite command** | `python -m pytest backend/training/aqa/ -m "not slow" -q` |
| **Estimated runtime** | ~8 seconds (suite) + notebook execute |
| **Notebook execute** | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` |

---

## Sampling Rate

- **After every task commit:** `python -m pytest backend/training/aqa/ -m "not slow" -q` (≈8 s; confirms no regression — expect 45 passed / 4 deselected)
- **After the notebook is built:** `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` (the in-notebook validation cell + figure-write asserts run here)
- **Before phase close:** suite green + notebook executes cleanly + all master figures exist
- **Max feedback latency:** < 60 seconds

---

## Per-Requirement Verification Map

Task IDs assigned by the planner; the requirement-level contract is fixed here.

| Requirement | Verifiable behavior | Test Type | Automated Command | File |
|-------------|--------------------|-----------|-------------------|------|
| EVAL-01 | Each of the 5 headline F1 values is recomputed from the pickle's raw scores + stored threshold via `metrics.f1_per_error` and equals the stored `test_f1` within 1e-6 (no hand-typed literal) | In-notebook assertion | runs on `nbconvert --execute` | `docs/notebooks/12_full_evaluation.ipynb` |
| EVAL-01 | Master comparison table + headline bar render ours-vs-Parmar for all 5 errors | In-notebook assertion + figure exists | `assert Path("docs/figures/all_errors_vs_paper.png").exists()` | same |
| EVAL-02 | The not-applicable finding is documented with the cited paper numbers (MD+CVCSPC Table 2: KIE 0.5263 / KFE 0.8468) | Manual (narrative) | N/A — `docs/eval/FINDINGS_FULL.md` | FINDINGS_FULL |
| EVAL-03 | The notebook pre-executes cleanly start-to-finish; all 4 master figures are written | Integration | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` | notebook + 4 PNGs |
| (regression) | AQA suite still green | Unit | `python -m pytest backend/training/aqa/ -m "not slow" -q` → 45 passed / 4 deselected | existing suite |

---

## Wave 0 Requirements

- [ ] `docs/notebooks/12_full_evaluation.py` — jupytext source (created during the phase)
- [ ] In-notebook validation cell: recompute F1 from raw arrays + assert ≤1e-6 deviation vs stored `test_f1` for all 5 errors (Squat KIE/KFE, OHP Elbows/Knees, Shallow-Squat) — uses `metrics.f1_per_error` + the stored thresholds (do NOT re-sweep on test)
- [ ] In-notebook validation cell: `assert` each of the 4 master figure PNGs exists after the figure cells

*The AQA test suite already exists and covers all production code. Phase 8 adds no production code, so no new unit-test files are required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| EVAL-02 not-applicable narrative is correct + honestly framed | EVAL-02 | Prose judgment, not automatable | Confirm FINDINGS_FULL states the paper ensembles MD+CVCSPC only for BackSquat KIE/KFE (Table 2: 0.5263/0.8468) and that our single-method-per-error scope cannot reproduce it — no invented ensemble |
| GYMetricPose/LMM context paragraph cites real numbers with comparability caveats, no columns | EVAL-01/D3 | Source-attribution judgment | Confirm GYMetricPose (KIE 0.440/KFE 0.822/Elbows 0.418/Knees 0.816, via Dibenedetto Table 4, split unconfirmed) and LMM (KIE 0.196/KFE 0.627/Elbows 0.458/Knees 0.761, trains on train+val → not comparable) appear only as labeled context, never as table columns |
| Figures are visually correct + defense-grade | EVAL-03 | Visual eye-check ([[project_supervisor_visualizations]]) | User reviews the 4 master figures at the end-of-phase checkpoint |

---

## Validation Sign-Off

- [ ] EVAL-01/02/03 each map to an automated assertion or a documented manual check
- [ ] No headline F1 is a hand-typed literal — all recomputed-and-asserted from the pickles
- [ ] Notebook pre-executes cleanly (figures + tables render on open)
- [ ] AQA suite green (45 passed / 4 deselected)
- [ ] `nyquist_compliant: true` set once the validation cells exist and pass

**Approval:** pending
