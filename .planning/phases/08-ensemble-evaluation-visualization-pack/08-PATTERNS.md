# Phase 8: Ensemble, Evaluation & Visualization Pack — Pattern Map

**Mapped:** 2026-05-30
**Files analyzed:** 7 (5 new, 1 modified, 1 optional)
**Analogs found:** 6 / 7 (methodology_diagram.png has no close analog — net-new schematic)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `docs/notebooks/12_full_evaluation.py` | notebook (jupytext source) | batch transform (pkl → figures + prints) | `docs/notebooks/11_shallow_squat_evaluation.py` | exact |
| `docs/notebooks/12_full_evaluation.ipynb` | notebook (pre-executed) | batch transform | `docs/notebooks/08_ohp_evaluation.ipynb` | exact |
| `docs/eval/FINDINGS_FULL.md` | documentation | — | `docs/eval/FINDINGS_OHP.md` + `docs/eval/FINDINGS_SHALLOW_SQUAT.md` | exact |
| `docs/figures/all_errors_vs_paper.png` | figure output | batch transform | `backend/scripts/ohp_headline_chart.py` + `backend/scripts/eval_benchmark_viz.py:fig_f1_vs_paper` | exact |
| `docs/figures/cross_exercise_macro.png` | figure output | batch transform | `backend/scripts/ohp_headline_chart.py` | role-match |
| `docs/figures/highlights_panel.png` | figure output | batch transform | `backend/scripts/ohp_eval_viz.py:score_dists` + `ohp_headline_chart.py` | role-match |
| `docs/figures/methodology_diagram.png` | figure output | transform (no data) | `backend/scripts/render_squat_result.py` (uses `matplotlib.patches`) | partial-match (no close analog — net-new schematic) |
| `docs/notebooks/README.md` | documentation (modified) | — | `docs/notebooks/README.md` itself (extend pattern) | exact |
| `backend/scripts/full_eval_viz.py` (optional) | utility script | batch transform | `backend/scripts/ohp_eval_viz.py` | exact |

---

## Pattern Assignments

### `docs/notebooks/12_full_evaluation.py` (jupytext notebook, batch transform)

**Analog:** `docs/notebooks/11_shallow_squat_evaluation.py` (primary) and `docs/notebooks/08_ohp_evaluation.py` (secondary)

**Jupytext cell marker pattern** (`11_shallow_squat_evaluation.py`, lines 1–15):
```python
# %% [markdown]
# # Shallow-Squat — Evaluation
#
# Benchmark evaluation ... All numbers + figures regenerate from the frozen `results.pkl`
# — no GPU. Identical metric (F1) and split as Parmar et al. (ECCV 2022).
#
# **Headline:** ...

# %% [markdown]
# ## Setup

# %%
import pickle
from pathlib import Path
```

**Pickle-loading pattern with dual-path fallback** (`11_shallow_squat_evaluation.py`, lines 24–35):
```python
_CANDS = [Path(".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl"),
          Path("../../.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl")]
R = pickle.load(open(next(p for p in _CANDS if p.exists()), "rb"))
S = np.asarray(R["ensemble_test_scores"], dtype=float)
Y = np.asarray(R["test_labels"]).astype(int)
TH = float(R["best_threshold"])

FIG = Path("docs/figures") if Path("docs/figures").exists() else Path("../../docs/figures")
EVAL = Path("docs/eval") if Path("docs/eval").exists() else Path("../../docs/eval")
FIG.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)
```

Apply the same dual-path fallback for all three pickles. Notebook 12 loads squat, ohp, and ss in a single Setup cell, then asserts each loaded correctly before proceeding.

**Headline print table pattern** (`08_ohp_evaluation.py`, lines 34–38):
```python
f1 = R["test_f1"]; base = R["baseline_control"]; paper = R["paper_targets"]
print(f"{'':<22}{'Elbows':>9}{'Knees':>9}{'macro':>9}")
print(f"{'baseline':<22}{base['elbows']:>9.4f}{base['knees']:>9.4f}{base['macro']:>9.4f}")
print(f"{'our MD-SSL ensemble':<22}{f1['elbows']:>9.4f}{f1['knees']:>9.4f}{f1['macro']:>9.4f}")
print(f"{'paper Ours-MD':<22}{paper['elbows']:>9.4f}{paper['knees']:>9.4f}{(paper['elbows']+paper['knees'])/2:>9.4f}")
```

**`savefig`-before-`show` pattern** (`11_shallow_squat_evaluation.py`, lines 86–88):
```python
plt.savefig(FIG / "shallow_squat_pr_curve.png", dpi=130, bbox_inches="tight")
plt.show()
```

Always call `savefig` before `plt.show()`. The pre-execution (nbconvert) will not see the figure after `show()` closes it.

**Honest-reporting cell pattern** (`11_shallow_squat_evaluation.py`, lines 179–195):
```python
# %% [markdown]
# ## 6. Honest reporting
#
# - **No paper supervised-ImageNet baseline row exists** for this task, so OUR supervised
#   baseline (0.875) is the only supervised control. The SSL lift is therefore measured
#   against our own control: CVCSPC − baseline = **+0.0152**.
# - **val → test gap = 0.0217** ...

# %%
print("baseline (supervised) test-F1 :", R["baseline_control"]["test_f1"])
print("CVCSPC ensemble test-F1       :", R["test_f1"]["cvcspc_ensemble"])
print("SSL lift (CVCSPC − baseline)  :", round(R["ssl_lift"], 4))
print("val→test gap                  :", round(R["val_test_gap"], 4))
print("loss deviation note           :", R["loss_deviation_note"])
```

**Critical pitfalls to carry into notebook 12:**

- **Pitfall 1 (Squat threshold):** Always use `squat["final"]["thresholds"]["KIE"]` when applying to test scores. Do NOT re-sweep on test data. (RESEARCH.md Pitfall 1)
- **Pitfall 2 (Shallow shape):** `ss["ensemble_test_scores"]` is shape `(540,)` not `(540,2)`. Apply threshold as `(ens >= thr).astype(int)`, never `ens[:, 0]`. (RESEARCH.md Pitfall 2)
- **Pitfall 3 (macro):** Compute macro as arithmetic mean of two per-error F1s: `(f1_a + f1_b) / 2`. Do NOT use `sklearn.f1_score(..., average='macro')` on the multi-label arrays. (RESEARCH.md Pitfall 3)
- **Pitfall 4 (Squat confusion):** `squat["final"]["test_confusion"]["KIE"]` is a list of lists, not an ndarray. Always wrap: `np.array(squat["final"]["test_confusion"]["KIE"])`. (RESEARCH.md Pitfall 4)
- **Pitfall 5 (GYMetricPose):** GYMetricPose numbers must NOT appear in the EVAL-01 table — context narrative only. (RESEARCH.md Pitfall 5)

---

### `docs/figures/all_errors_vs_paper.png` (grouped bar, 5 errors)

**Primary analog:** `backend/scripts/ohp_headline_chart.py` (the complete grouped-bar pattern)

**Bar-chart offset math** (`ohp_headline_chart.py`, lines 23–30):
```python
x = np.arange(len(ERRORS))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 5.2))
for i, (name, vals) in enumerate(ROWS.items()):
    bars = ax.bar(x + (i - 1) * w, [vals[e] for e in ERRORS], w, label=name, color=COLORS[name])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
```

`(i - 1) * w` centers the group: for 3 methods i=0 → -w, i=1 → 0, i=2 → +w. For the 5-error all-errors chart use the same offset formula. Widen `figsize` to `(13, 5.5)` to accommodate 5 error groups.

**Value annotation** (same file, line 29):
```python
ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
        ha="center", va="bottom", fontsize=8)
```

**`savefig` + `print` pattern** (`ohp_headline_chart.py`, lines 43–45):
```python
out = Path("docs/figures/ohp_f1_vs_paper.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
```

**Handling None baselines for OHP/Shallow** (RESEARCH.md Pattern — EVAL-01 grouped bar code example):
```python
base_x = [x[i] + w for i in range(2)]   # only Squat has a paper baseline
base_v = [paper_base[i] for i in range(2)]
ax.bar(base_x, base_v, w, label="Parmar baseline (paper)", color="0.65")
```

Only draw the baseline bars for indices 0 (Squat KIE) and 1 (Squat KFE) where `paper_base[i]` is not None. Skip OHP and Shallow baseline bars entirely.

**No-fabrication pickle read** (RESEARCH.md Pattern 2):
```python
ERRORS_DATA = [
    ("Squat KIE",     "video (MD)",    squat["final"]["test_f1"]["KIE"],
     squat["comparison"]["paper_md"]["KIE"],  squat["comparison"]["paper_kinetics"]["KIE"]),
    ("Squat KFE",     "video (MD)",    squat["final"]["test_f1"]["KFE"],
     squat["comparison"]["paper_md"]["KFE"],  squat["comparison"]["paper_kinetics"]["KFE"]),
    ("OHP Elbows",    "video (MD)",    ohp["test_f1"]["elbows"],
     ohp["paper_targets"]["elbows"],          None),
    ("OHP Knees",     "video (MD)",    ohp["test_f1"]["knees"],
     ohp["paper_targets"]["knees"],           None),
    ("Shallow-Squat", "image (CVCSPC)",ss["test_f1"]["cvcspc_ensemble"],
     ss["paper_targets"]["cvcspc"],           None),
]
```

Every F1 value comes from a pickle key, not a literal.

---

### `docs/figures/cross_exercise_macro.png` (per-exercise macro bar)

**Analog:** `backend/scripts/ohp_headline_chart.py` (same grouped-bar skeleton, 3 groups instead of 5)

**Pattern:** Three groups — "Squat", "OHP", "Shallow-Squat" — with two bars each: "Ours" vs "Parmar SSL". Use the same `x = np.arange(3); w = 0.26; ax.bar(x + (i-1)*w, ...)` offset formula. Macro values: Squat `squat["final"]["test_macro"]`, OHP `ohp["test_f1"]["macro"]`, Shallow `ss["test_f1"]["cvcspc_ensemble"]` (single error = macro). Paper: Squat `(squat["comparison"]["paper_md"]["KIE"] + squat["comparison"]["paper_md"]["KFE"]) / 2`, OHP `(ohp["paper_targets"]["elbows"] + ohp["paper_targets"]["knees"]) / 2`, Shallow `ss["paper_targets"]["cvcspc"]`.

Palette: mirror `ohp_headline_chart.py` — `color="C0"` for ours, `color="C1"` for paper. Same dpi=120, bbox_inches="tight", save before show.

---

### `docs/figures/highlights_panel.png` (per-exercise headline summary panel)

**Analog:** `backend/scripts/ohp_eval_viz.py` (multi-subplot layout) + `ohp_headline_chart.py` (compact text display)

**Multi-subplot skeleton** (`ohp_eval_viz.py`, lines 35–37):
```python
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
```

For the highlights panel use `fig, ax = plt.subplots(1, 3, figsize=(14, 5))` — one column per exercise. Each panel contains a compact text summary (headline F1, SSL lift, val→test gap). Use `ax[i].text(0.5, y, text, transform=ax[i].transAxes, ha="center")` for text-only summary panels, or a small horizontal bar per exercise. No per-phase figures (PR curves, confusion) — those already exist in `docs/figures/`.

**`fig.tight_layout()` pattern** (`ohp_eval_viz.py`, line 54):
```python
fig.tight_layout(); fig.savefig(FIG / "ohp_ssl_curves.png", dpi=120, bbox_inches="tight")
print("saved ohp_ssl_curves.png")
```

---

### `docs/figures/methodology_diagram.png` (net-new architecture schematic)

**No close analog.** The only matplotlib-patches usage found in the repo is `backend/scripts/render_squat_result.py` which uses `matplotlib.patches.mpatches` for a legend, not for drawn boxes. There is no existing two-branch architecture diagram to copy from.

**Closest partial-match:** `backend/scripts/render_squat_result.py` (lines 1–33 — imports `matplotlib.patches as mpatches`, uses `matplotlib.use("Agg")` for headless rendering).

**Net-new spec from RESEARCH.md Pattern 3:**

Two-branch schematic drawn with `ax.text`, `ax.annotate(arrowprops=dict(...))` or `matplotlib.patches.FancyArrowPatch`:
- Left branch: "Video errors\n(Squat KIE/KFE,\nOHP Elbows/Knees)" → "Motion-Disentangling SSL\n(barbell half-cycle contrast)" → "R(2+1)D-18\n(32-frame, 112², Kinetics-init)" → "multi-label head"
- Right branch: "Image errors\n(Shallow-Squat depth)" → "CVCSPC pose-contrastive SSL\n(phase-anchor/positive/negative)" → "ResNet-18\n(224², ImageNet-init)" → "single binary head"
- Both branches merge at: "Official train/val/test splits\n→ F1-per-error vs Parmar (ECCV 2022)"

No fabricated numbers anywhere in this figure. Pure text boxes and arrows.

**Mandatory headless pattern** (from `render_squat_result.py` and `eval_benchmark_viz.py`):
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
```

Use `matplotlib.use("Agg")` when the script runs as a standalone (not in-notebook). Inside the notebook, omit it — the notebook backend is already set.

---

### `docs/eval/FINDINGS_FULL.md` (consolidated write-up)

**Primary analog:** `docs/eval/FINDINGS_OHP.md` and `docs/eval/FINDINGS_SHALLOW_SQUAT.md`

**Section header structure** (from both FINDINGS files):
```markdown
# <Exercise/Scope> Form-Error Detection — Benchmark Evaluation & Findings

**Model:** ...
**Evaluation set:** ...

## Headline results

| Error | Ours ... | Parmar SSL (paper) | Parmar baseline (paper) |
...

## Figures (`docs/figures/`)

| File | What it shows |
...

## Method (identical to the published approach)
...

## Honest scope notes
...

## Reproduce
```

FINDINGS_FULL uses the same structure with scope = "All 3 Exercises (5 Errors)" and these additional sections:
- `## EVAL-01 — Master comparison table` (the 5-row table from CONTEXT D2)
- `## EVAL-02 — MD+CVCSPC ensemble: evaluated, not-applicable` (RESEARCH.md finding)
- `## Related-work context (GYMetricPose + LMM)` (D3 narrative, not a table column)
- `## Cross-exercise synthesis` (macro comparison, SSL lift summary)
- `## Honest framing (carried from per-phase FINDINGS)` (D7 items)
- `## Per-phase detail` (links to the three FINDINGS files)
- `## Reproduce`

**"Honest scope notes" prose pattern** (`FINDINGS_OHP.md`, lines 48–59):
```markdown
## Honest scope notes

- **2-seed ensemble** (seeds 42, 1337), a time-driven reduction from the 3-seed Squat
  protocol — identical mean-of-sigmoids methodology, both seeds clean.
- **TTA not adopted** — evaluated in the Squat phase and found to reverse on the test split;
  the no-TTA ensemble is the reported headline.
- Like Squat, these results are on the **benchmark** (in-the-wild gym video, side-on).
```

**"Figures" table pattern** (`FINDINGS_SHALLOW_SQUAT.md`, lines 55–62):
```markdown
## Figures (`docs/figures/`)

| File | What it shows |
|------|---------------|
| `shallow_squat_f1_vs_paper.png` | Headline bar — our baseline → our CVCSPC → paper CVCSPC / SimSiam / OpenPose-TDM. |
```

**"Reproduce" block pattern** (`FINDINGS_SHALLOW_SQUAT.md`, lines 89–92):
```markdown
## Reproduce

```
# results.pkl (per-crop ensemble scores + curves) is committed at
#   .planning/phases/07-image-based-errors-cvcspc/figures/results.pkl
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/11_shallow_squat_evaluation.ipynb
```
```

---

### `docs/notebooks/README.md` (modified — add notebook-12 row)

**Analog:** The file itself at lines 1–59 — extend the existing table pattern.

**Existing section pattern to mirror** (`README.md`, lines 32–45):
```markdown
## Shallow-Squat (squat-depth, image error · CVCSPC)

| Notebook | Covers |
|----------|--------|
| **09_shallow_squat_eda** | ... |
| **10_shallow_squat_training** | ... |
| **11_shallow_squat_evaluation** | ... |

**Shallow-Squat headline:** F1 **0.8902** ...
Write-up: [`../eval/FINDINGS_SHALLOW_SQUAT.md`](../eval/FINDINGS_SHALLOW_SQUAT.md).
```

Add a new `## Full Evaluation (all 3 exercises)` section after the Shallow-Squat block following the same format: heading, one-row table, headline sentence, link to `FINDINGS_FULL.md`.

**"Reproduce" block pattern** (`README.md`, lines 56–58):
```markdown
Each notebook is jupytext-paired (`.py` source + `.ipynb`). To re-run:
`jupyter nbconvert --to notebook --execute --inplace docs/notebooks/<name>.ipynb`
```

---

### `backend/scripts/full_eval_viz.py` (optional — planner's discretion)

**Analog:** `backend/scripts/ohp_eval_viz.py` (exact role match — standalone figure script)

**File header pattern** (`ohp_eval_viz.py`, lines 1–20):
```python
"""OHP evaluation + training figure pack (Phase 6 Plan 05), built from results.pkl.

SSL convergence (loss + effective_rank + linear-probe), PR curves, confusion matrices,
ensemble score distributions — all from the Plan-04 results.pkl (real test scores on the
official 339-clip split). Run: python backend/scripts/ohp_eval_viz.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

PKL = Path(".planning/phases/06-overhead-press/figures/results.pkl")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)
```

If `full_eval_viz.py` is created, define three PKL paths (squat, ohp, shallow), load all three at module level, and implement each figure as a named function. Call all functions in `if __name__ == "__main__":`. Same dpi=120, defensive `FIG.mkdir(parents=True, exist_ok=True)`, savefig before print.

---

## Shared Patterns

### Defensive `mkdir` (apply to all figure-creating code)
**Source:** `backend/scripts/ohp_eval_viz.py` line 19 + `ohp_headline_chart.py` line 43
```python
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)
```
Always the first thing before any `savefig`. In-notebook: use dual-path resolution first, then `mkdir`.

### `savefig`-before-`show` (apply to every figure cell)
**Source:** `docs/notebooks/11_shallow_squat_evaluation.py` lines 86–88
```python
plt.savefig(FIG / "name.png", dpi=130, bbox_inches="tight")
plt.show()
```
`plt.show()` clears the figure; `savefig` must come first. dpi=120 in scripts, dpi=130 in notebooks — both are acceptable; be consistent within the file.

### `from __future__ import annotations` (all Python files)
**Source:** `backend/scripts/ohp_eval_viz.py` line 8, `ohp_headline_chart.py` line 8, `backend/training/aqa/eval/metrics.py` line 19
Always the first import in every new `.py` file (project-wide convention per CLAUDE.md).

### `metrics.f1_per_error` import (apply to all F1 recomputation)
**Source:** `backend/training/aqa/eval/metrics.py` line 33
```python
from backend.training.aqa.eval.metrics import f1_per_error, threshold_sweep
```
Import path is absolute (`from backend.training...`). Reused UNCHANGED — do not redefine.

### Value annotation on bars (apply to all bar charts)
**Source:** `backend/scripts/ohp_headline_chart.py` lines 29–30
```python
ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
        ha="center", va="bottom", fontsize=8)
```

### `fig.tight_layout()` before `fig.savefig()` (apply to all multi-subplot figures)
**Source:** `backend/scripts/ohp_eval_viz.py` line 54
```python
fig.tight_layout(); fig.savefig(FIG / "name.png", dpi=120, bbox_inches="tight")
```

### Validation assertion pattern (apply in notebook 12 Setup cell)
**Source:** RESEARCH.md Pattern 1 — self-check that recomputed F1 matches stored value.
```python
assert abs(computed_f1 - stored_f1) < 1e-6, f"drift: computed {computed_f1}, stored {stored_f1}"
```
One assertion per error per exercise. Catches hand-typed literal drift.

### No hard-coded headline literals in data cells
**Source:** RESEARCH.md Anti-patterns (the Phase-3 bug)
Every F1 value in the notebook comes from `pickle[key]` or is computed via `f1_per_error`. No `OFFICIAL = {"KFE": 0.841, ...}` dicts with hand-typed values (contrast with the Squat FINDINGS pattern that embeds literals — that was pre-Phase-8 and is acceptable only in prose/markdown, never in code cells).

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `docs/figures/methodology_diagram.png` | figure output | transform (no ML data) | No two-branch architecture schematic exists in the repo. `render_squat_result.py` uses `matplotlib.patches` for a legend panel, not for box-and-arrow schematics. Use RESEARCH.md Pattern 3 as the spec; `FancyArrowPatch` or `ax.annotate(arrowprops=...)` for arrows. |

---

## Metadata

**Analog search scope:** `docs/notebooks/`, `docs/eval/`, `backend/scripts/`, `backend/training/aqa/eval/`
**Files scanned:** 11 (notebooks 08, 11; scripts ohp_headline_chart, ohp_eval_viz, eval_benchmark_viz, render_squat_result; FINDINGS.md, FINDINGS_OHP.md, FINDINGS_SHALLOW_SQUAT.md; metrics.py; README.md)
**Pattern extraction date:** 2026-05-30

---

## PATTERN MAPPING COMPLETE

**Phase:** 08 — Ensemble, Evaluation & Visualization Pack
**Files classified:** 7 (+ 1 optional)
**Analogs found:** 6 / 7 (methodology_diagram.png has no close analog)

### Coverage
- Files with exact analog: 4 (notebook .py, FINDINGS_FULL, all_errors_vs_paper, README.md row)
- Files with role-match analog: 2 (cross_exercise_macro, highlights_panel, full_eval_viz.py)
- Files with no analog: 1 (methodology_diagram.png — net-new schematic)

### Key Patterns Identified
- All figure scripts use `Path("docs/figures").mkdir(parents=True, exist_ok=True)` as first operation, `fig.savefig(..., dpi=120, bbox_inches="tight")` before `plt.show()`, then `print("saved", filename)`
- Bar charts follow the `x = np.arange(n); w = 0.26; ax.bar(x + (i-1)*w, ...)` offset formula with per-bar `ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f"{v:.3f}", ...)` annotation
- Notebooks use `# %% [markdown]` / `# %%` jupytext markers, dual-path `_CANDS` list for pkl resolution, and `plt.savefig()` before `plt.show()`
- FINDINGS files follow a fixed section order: model header → headline table → figures table → method → honest scope notes → reproduce
- All F1 values read from pickle keys; macro computed as arithmetic mean of two per-error F1s; thresholds always taken from stored `thresholds` dict, never re-swept on test

### File Created
`.planning/phases/08-ensemble-evaluation-visualization-pack/08-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
