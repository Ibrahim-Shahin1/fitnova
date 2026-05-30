---
phase: 08
plan: 01
subsystem: evaluation
tags: [evaluation, visualization, cross-exercise, defense-deliverable, no-fabrication]
dependency_graph:
  requires:
    - .planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl
    - .planning/phases/06-overhead-press/figures/results.pkl
    - .planning/phases/07-image-based-errors-cvcspc/figures/results.pkl
  provides:
    - docs/notebooks/12_full_evaluation.py
    - docs/notebooks/12_full_evaluation.ipynb
    - docs/figures/all_errors_vs_paper.png
    - docs/figures/cross_exercise_macro.png
    - docs/figures/methodology_diagram.png
    - docs/figures/highlights_panel.png
  affects: []
tech_stack:
  added: []
  patterns:
    - jupytext percent-format source paired to .ipynb via nbconvert --execute
    - dual-path _CANDS pickle resolution (runs from repo root or docs/notebooks/)
    - no-fabrication validation: recompute-and-assert all 5 headline F1 values <=1e-6 vs stored
    - sklearn f1_score(pos_label=1, zero_division=0) as the recompute primitive
    - arithmetic-mean macro (matches paper convention; NOT sklearn average='macro')
key_files:
  created:
    - docs/notebooks/12_full_evaluation.py
    - docs/notebooks/12_full_evaluation.ipynb
    - docs/figures/all_errors_vs_paper.png
    - docs/figures/cross_exercise_macro.png
    - docs/figures/methodology_diagram.png
    - docs/figures/highlights_panel.png
  modified: []
decisions:
  - sklearn f1_score used directly in notebook (not backend.training.aqa.eval.metrics import) — nbconvert kernel cwd is docs/notebooks/ and repo root is not on sys.path; sklearn f1_score(pos_label=1, zero_division=0) is exactly equivalent to f1_per_error, verified by inspection of metrics.py
  - Shallow-Squat ensemble_test_scores applied as 1-D (540,) array directly; no [:,0] column slice
  - Macro computed as arithmetic mean of per-error F1s in all exercises; matches paper convention and stored test_macro values
  - methodology_diagram uses pure matplotlib boxes + annotate arrows — no fabricated numbers, no plt.imread of any image
  - highlights_panel Squat val->test gap computed from mean of per-seed val_best values (squat pkl has no direct val_test_gap key unlike OHP/Shallow)
metrics:
  duration_minutes: 30
  completed_date: "2026-05-31"
  tasks_completed: 2
  files_changed: 6
---

# Phase 8 Plan 01: Full Evaluation Notebook + 4 Master Figures Summary

Jupytext notebook `12_full_evaluation.{py,ipynb}` assembled locally from three committed results.pkl — 5-error no-fabrication validation + EVAL-01 master table + 4 defense figures.

## What Was Built

### Task 1: `docs/notebooks/12_full_evaluation.py`

Jupytext percent-format source (9 cells):

1. Title + framing markdown (5-error headline spine in prose)
2. Setup: dual-path `_CANDS` pickle load for all three exercises; FIG/EVAL dual-path + mkdir
3. No-fabrication validation: all 5 headline F1 recomputed from raw score arrays + stored thresholds, asserted <=1e-6 vs stored `test_f1`. Macro asserted as arithmetic mean.
4. EVAL-01 master table: 5-row fixed-width print; dashes for OHP/Shallow baseline cells (no paper supervised row exists for those exercises)
5. Figure: `all_errors_vs_paper.png` — 5 error groups, 3 series (Ours / Parmar SSL / Parmar baseline — baseline bars only for Squat KIE+KFE)
6. Figure: `cross_exercise_macro.png` — per-exercise macro vs Parmar SSL
7. Figure: `methodology_diagram.png` — pure matplotlib schematic, two-method architecture
8. Figure: `highlights_panel.png` — 3-panel per-exercise F1/SSL-lift/val-test-gap
9. Figure-existence assertion: loops over 4 filenames, asserts each exists, prints "4/4 master figures written"

### Task 2: Pre-execution

- `jupytext --to notebook docs/notebooks/12_full_evaluation.py` → `.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` → exit 0
- All in-notebook asserts pass (5-error F1 validation + 4 figure-exists checks)
- AQA suite: `python -m pytest backend/training/aqa/ -m "not slow" -q` → **45 passed, 4 deselected**

## Headline Results (from pickles, recomputed in-notebook)

| Exercise/Error | Modality | Ours | Parmar SSL | Parmar baseline |
|---|---|---|---|---|
| Squat KIE | video (MD) | 0.4198 | 0.4186 | 0.2970 |
| Squat KFE | video (MD) | 0.8410 | 0.8338 | 0.8184 |
| OHP Elbows | video (MD) | 0.4474 | 0.4552 | — |
| OHP Knees | video (MD) | 0.8770 | 0.8452 | — |
| Shallow-Squat | image (CVCSPC) | 0.8902 | 0.8694 | — |

## Deviations from Plan

### Auto-resolved Before Execution (Documented in Plan)

**[Already resolved in plan — not a runtime deviation] sklearn import instead of backend import**
- The plan itself (commit ca6c4de) already specifies `from sklearn.metrics import f1_score` rather than `from backend.training.aqa.eval.metrics import f1_per_error`
- Reason: `nbconvert --execute` sets the kernel cwd to `docs/notebooks/` and does not add the repo root to `sys.path`; a `from backend...` import would raise `ModuleNotFoundError`
- `f1_score(y_true, y_pred, pos_label=1, zero_division=0)` is verified identical to `f1_per_error` by inspection of `backend/training/aqa/eval/metrics.py`
- No impact on the no-fabrication guarantee: the recompute-and-assert pattern is fully preserved

### Runtime Fixes Applied

**[Rule 1 - Bug] Walrus operator in function-call argument**
- Found during: Task 1 authoring
- Issue: First draft used `BOX_BLUE := dict(...)` as a default-argument walrus expression inside a positional function call — syntactically invalid at the call site
- Fix: Defined `BOX_BLUE`, `BOX_GRN`, `BOX_GOLD` as module-level dict constants before the figure cell; helper functions `_box()` / `_arrow()` defined before the figure code
- Files modified: `docs/notebooks/12_full_evaluation.py`
- Commit: 2d24b7d

**[Rule 2 - Missing critical functionality] Squat val->test gap**
- Found during: Task 1 authoring
- Issue: Squat pkl has no direct `val_test_gap` key (unlike OHP and Shallow-Squat pickles); the highlights panel needed a consistent gap metric
- Fix: Computed `sq_val_mean = mean([per_seed[s]['val_best'] for s in [42,1337,7]])` and derived `sq_val_test_gap = final/test_macro - sq_val_mean`. Noted in panel as `val→test: +{val_test_gap:.4f}` for honesty
- Files modified: `docs/notebooks/12_full_evaluation.py`
- Commit: 2d24b7d

## Known Stubs

None — all values read from pickle keys. The master table and figures are fully wired.

## Threat Flags

None — local read of committed pkl files + write of docs/figures. No new attack surface.

## Self-Check

### Files exist

- `docs/notebooks/12_full_evaluation.py` — FOUND (commit 2d24b7d)
- `docs/notebooks/12_full_evaluation.ipynb` — FOUND (commit 64f107c)
- `docs/figures/all_errors_vs_paper.png` — FOUND (commit 64f107c)
- `docs/figures/cross_exercise_macro.png` — FOUND (commit 64f107c)
- `docs/figures/methodology_diagram.png` — FOUND (commit 64f107c)
- `docs/figures/highlights_panel.png` — FOUND (commit 64f107c)

### Commits exist

- 2d24b7d: `feat(08): author 12_full_evaluation.py — 3-pickle load, 5-error no-fabrication validation, EVAL-01 table, 4 master figures`
- 64f107c: `feat(08): pre-execute 12_full_evaluation.ipynb + 4 master figures`

### AQA suite

- `python -m pytest backend/training/aqa/ -m "not slow" -q` → `45 passed, 4 deselected`

## Self-Check: PASSED
