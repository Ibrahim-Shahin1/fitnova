# Phase 8: Ensemble, Evaluation & Visualization Pack - Context

**Gathered:** 2026-05-30
**Status:** Ready for planning
**Source:** `/gsd:discuss-phase 8` (autonomous per [[feedback_run_gsd_autonomously]]). The phase scope is fixed by the master plan + ROADMAP Phase-8 success criteria + REQUIREMENTS EVAL-01/02/03; almost every gray area is resolved by the master plan, the user's continuation prompt, and Phase 4/6/7 precedent. Two genuinely-open, defense-shaping decisions (the consolidated viz-pack ambition; how the external comparators appear) were put to the user and resolved below (D5, D3). The three input `results.pkl` were loaded and the EVAL-01 spine cross-checked against them during this discussion — every headline number traces to a committed pickle (no fabrication).

<domain>
## Phase Boundary

**This phase delivers** (satisfies **EVAL-01, EVAL-02, EVAL-03**) — the FINAL phase of the v1.0 form-correction milestone:

1. **EVAL-01 — the full identical-metric comparison across every error.** A master comparison table covering all 5 detected errors (Squat KIE/KFE, OHP Elbows/Knees → video MD-SSL; Shallow-Squat → image CVCSPC) with our official-test F1 vs the published Parmar et al. (ECCV 2022) numbers (paper baseline where it exists + paper SSL/MD/CVCSPC). The independent comparison points **GYMetricPose (Gallardo 2024)** and **LMM (Dibenedetto 2025)** are reported as a **sourced related-work context narrative** (their real numbers, cited), NOT as table columns (D3). Plus the headline cross-exercise figure.
2. **EVAL-02 — the MD+CVCSPC ensemble where the paper applies it.** Research-confirm against the Parmar PDF whether the paper ensembles MD+CVCSPC for any single shared error; **report honestly**. Expected outcome: **NOT-APPLICABLE** to our scope — each of our 5 errors is detected by exactly ONE method, so no error has both an MD score and a CVCSPC score to ensemble (D4). Do NOT invent an ensemble the data doesn't support.
3. **EVAL-03 — the consolidated visualization/results pack.** A **comprehensive defense pack** (user decision D5): one consolidated pre-executed notebook (`docs/notebooks/12_full_evaluation*`) + a consolidated `docs/eval/FINDINGS_FULL.md` + master figures under `docs/figures/` — assembled from the three committed `results.pkl`. Fully local.
4. End-of-phase SUMMARY + milestone close (this is the last phase → v1.0 milestone DONE).

**This phase does NOT deliver:**
- **No model training / re-runs / Colab.** All three exercises are trained and evaluated; the committed `results.pkl` already hold per-clip/per-crop ensemble scores + per-epoch curves. Phase 8 is LOCAL assembly + a bounded research pass only.
- **No backend/serving/API work** — serving is descoped milestone-wide (CONTEXT D1, Phases 5–7; the form-correction frontend was cancelled). The orphaned form backend (`/ws/form-session`, `SquatLiveSession`, dead Flutter form screens) is intentionally KEPT for fallback — do NOT strip it.
- **No BarbellRow / IMG-03** — descoped (cancelled by the user for compute/time). The milestone form-correction scope is Squat + OHP (video) + Shallow-Squat (image) = 5 errors.
- **No invented ensemble** (EVAL-02) and **no fabricated comparator numbers** — source or omit-and-say-so.

**Requirements satisfied:** EVAL-01, EVAL-02, EVAL-03.

</domain>

<decisions>
## Implementation Decisions (Locked)

### D1 — Venue: LOCAL assembly from the three committed results.pkl; NO Colab, NO re-runs [LOCKED from precedent + verified]
**Choice:** Phase 8 is built entirely on the local Windows machine from the committed artifacts:
- Squat: `.planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl`
- OHP: `.planning/phases/06-overhead-press/figures/results.pkl`
- Shallow-Squat: `.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl`
**Why:** Each pickle holds the official per-clip/per-crop ensemble scores + per-epoch SSL/fine-tune curves + the computed F1/PR-AUC/thresholds + the paper-target dicts. The cross-exercise comparison + all master figures are derivable offline. No GPU, no Drive copy, no cell-by-cell paste-back (the Colab/L4/disconnect-safe machinery is training-phase-only, mirroring Phase 5's D-12). The Phase-8 notebook is built + pre-executed LOCALLY (`jupyter nbconvert --to notebook --execute --inplace`).

### D2 — EVAL-01 master table = ours vs Parmar (baseline + SSL) across all 5 errors [LOCKED; verified against the pickles]
**Choice:** The master comparison table's COLUMNS are: Exercise/Error · Modality · **Ours (official test F1)** · **Parmar SSL (paper)** · **Parmar baseline (paper, where it exists)**. All numbers computed-in-code from the pickles / cited from the paper. The verified spine (see `<specifics>`): Squat KIE 0.420/KFE 0.841 (macro 0.6304); OHP Elbows 0.447/Knees 0.877 (macro 0.6622); Shallow-Squat 0.8902. Paper: Squat KIE 0.297→0.419 / KFE 0.818→0.834; OHP Elbows 0.455 / Knees 0.845 (no paper supervised baseline); Shallow CVCSPC 0.8694 (no paper supervised baseline).
**Why:** This is the identical-metric, identical-split apples-to-apples comparison the milestone requires. It is the rigorous spine; GYMetricPose/LMM are kept OUT of these columns (D3) precisely to keep the columns defensible.

### D3 — GYMetricPose + LMM = sourced related-work CONTEXT narrative, NOT table columns [LOCKED via discussion — user decision]
**Choice (user, 2026-05-30):** "Context note only — no columns." Source GYMetricPose (Gallardo 2024) and LMM (Dibenedetto 2025) from the actual papers, report their numbers in a clearly-labeled **related-work / context paragraph** in `FINDINGS_FULL.md` (with citations), but do NOT add them as columns to the EVAL-01 master table.
**Why:** They may report different metrics / splits / error definitions than the Fitness-AQA official F1-per-error. A table column implies identical-metric comparability; presenting them as columns would be apples-to-oranges and undermine defensibility. A cited narrative is honest and still credits the independent comparison points. (The user explicitly chose the more conservative, more defensible option over forcing columns.)
**Sourcing:** LMM paper is LOCAL — `Fitness-AQA/Fine-Tuning Large Multimodal Models for Fitness Action Quality Assessment.pdf`. GYMetricPose (Gallardo 2024) is NOT local → web research pass. **If a number cannot be sourced, omit it and SAY SO** ([[feedback_ai_correctness]] — never fabricate). The researcher extracts/cites; the FINDINGS narrates.

### D4 — EVAL-02 (MD+CVCSPC ensemble): research-confirm, report as NOT-APPLICABLE; do NOT invent one [LOCKED approach]
**Choice:** The researcher confirms against the Parmar PDF whether the paper applies an MD+CVCSPC ensemble to any SINGLE shared error (i.e., an error scored by BOTH methods). Then report honestly. Expected: **N/A** — our 5 errors each use exactly one method (KIE/KFE/Elbows/Knees → MD; Shallow-Squat → CVCSPC), so no error has both an MD and a CVCSPC score to ensemble. EVAL-02 is recorded as "evaluated → not-applicable to our single-method-per-error 3-exercise scope," with the paper-check result cited.
**Why:** The milestone's two methods cover disjoint errors. A constructed "ensemble" across disjoint errors is not the paper's MD+CVCSPC ensemble — it would be an invented metric. [[feedback_ai_correctness]] + the user's explicit instruction forbid inventing one. Honest "evaluated/not-applicable" is the correct, defensible outcome. (If the paper-check surprises us and the paper DOES ensemble MD+CVCSPC for a shared error, report what the paper did + why our single-method-per-error scope doesn't reproduce it.)

### D5 — EVAL-03 deliverable = COMPREHENSIVE defense pack [LOCKED via discussion — user decision]
**Choice (user, 2026-05-30):** "Comprehensive defense pack." Deliver:
- **`docs/notebooks/12_full_evaluation.{py,ipynb}`** — a consolidated, pre-executed notebook that loads the three `results.pkl` and renders the cross-exercise synthesis (jupytext-paired `.py` + executed `.ipynb`, mirroring notebooks 01–11).
- **`docs/eval/FINDINGS_FULL.md`** — the consolidated write-up: the EVAL-01 master table, the cross-exercise narrative, the EVAL-02 not-applicable finding, the GYMetricPose/LMM context paragraph (D3), the honest-framing notes (D7), and links to the per-phase FINDINGS.
- **Master figures under `docs/figures/`** (the NEW synthesis — do NOT duplicate the per-phase PR/confusion/score-dist/training-curve figures that already exist):
  1. **All-5-errors grouped bar** — ours vs Parmar baseline vs Parmar SSL, per error (the headline; e.g. `all_errors_vs_paper.png`).
  2. **Cross-exercise macro-F1 summary vs paper** (per-exercise macro: Squat 0.6304 vs 0.6262; OHP 0.6622 vs 0.6502; Shallow 0.8902 vs 0.8694).
  3. **Two-method methodology / architecture diagram** — the thesis at a glance: video errors → Motion-Disentangling SSL → R(2+1)D-18; image errors → CVCSPC pose-contrastive SSL → ResNet-18; shared eval (official splits, F1-per-error). A schematic (matplotlib/drawn), no fabricated data.
  4. **Per-exercise highlights panel** — a compact consolidated view (e.g. each exercise's headline F1 + SSL-lift + val→test gap) pulled from the pickles.
**Why:** The supervisor weights visualizations heavily and this is the final consolidated viz deliverable ([[project_supervisor_visualizations]]). The methodology diagram is the one genuinely-new (non-results.pkl) asset and is a high-value defense figure (a named bucket in the supervisor mandate). Final figure count/exact filenames: planner's discretion within this set.

### D6 — Reuse the established eval/viz code + deliverable pattern; do NOT redefine [LOCKED from precedent]
**Choice:** `backend/training/aqa/eval/metrics.py` (f1/pr_auc/threshold_sweep/confusion) reused UNCHANGED. The new cross-exercise figures follow the established figure-script pattern (`backend/scripts/ohp_eval_viz.py`, `ohp_headline_chart.py`, `ohp_finetune_curves.py`, `ohp_eda_viz.py`, `eval_benchmark_viz.py` / `eval_best_examples.py` for Squat) — a reproducible script/notebook that reads a `results.pkl` and saves PNGs (defensive `mkdir`, save BEFORE `plt.show()`, mirror the Squat/OHP/Shallow palette + layout). The notebook is jupytext-paired and pre-executed; `docs/notebooks/README.md` is extended with the notebook-12 row. FINDINGS_FULL mirrors the FINDINGS / FINDINGS_OHP / FINDINGS_SHALLOW_SQUAT style.
**Why:** Proven, consistent for the supervisor, low risk. Whether to add a small consolidated figure script under `backend/scripts/` (e.g. `full_eval_viz.py`) vs build figures inline in the notebook is the planner's call (both match precedent).

### D7 — Honest framing carried into the consolidated write-up [LOCKED]
**Choice:** `FINDINGS_FULL.md` carries forward, not silently reconciles, the per-phase honesty notes:
- **No paper supervised baseline** exists for OHP or Shallow-Squat — our supervised baselines (OHP macro 0.6118; Shallow 0.8750) are the only supervised controls; the SSL lift is measured against them (OHP +0.050; Shallow +0.0152). Shallow's supervised baseline (0.8750) already edges paper CVCSPC (0.8694) — keep that honest framing.
- **The 3-term-vs-2-term CVCSPC loss deviation** (Shallow-Squat backbone, `train_test.py:68` vs paper Eq.1) — documented in FINDINGS_SHALLOW_SQUAT; carry it into the consolidated write-up.
- **TTA not adopted** (Squat/OHP: val gain reversed on test) — the reported headlines are no-TTA ensembles.
- **OHP 2-seed** ensemble (time-driven) vs Squat/Shallow 3-seed — note the methodology variance.
- **Benchmark-only / domain-shift caveat** — every result is on the in-the-wild Fitness-AQA test split; live/phone-camera transfer is the known in-the-wild limit (the paper's own thesis), which is why serving was descoped.

### D8 — No fabrication; every number computed-in-code or cited [LOCKED]
**Choice:** Every metric/curve in the pack comes from a real `results.pkl` (loaded + computed in code) or is a cited paper number. No hand-typed headline literals (the Phase-3 precedent the user caught). The EVAL-01 spine in `<specifics>` was already cross-checked against the three pickles this discussion; the notebook recomputes from the pickles rather than hardcoding.
**Why:** [[feedback_ai_correctness]], [[feedback_dont_agree]].

### Claude's Discretion (researcher/planner resolve)
- Exact final figure count + filenames within the D5 set; whether master figures are built inline in notebook 12 or via a small `backend/scripts/full_eval_viz.py` (both match precedent).
- Whether FINDINGS_FULL extends/links the three per-phase FINDINGS or restates the key tables (lean: a self-contained consolidated table + links).
- The exact GYMetricPose/LMM numbers + which errors they cover — depends on what the papers actually report (researcher sources; honesty rule D3/D8 governs).
- Plan/wave shape (likely a single local plan, or scaffold→research→assemble→FINDINGS→close) — planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, plan-checker, executor) MUST read these before producing their artefacts.**

### Master plan & milestone authority
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master rebuild plan; **Phase 8 paragraph (line 110-111)**; the **comparison-targets table (lines 56-65)**; the GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025) independent comparison points (line 65); the working agreement.
- `.planning/ROADMAP.md` — Phase 8 goal + the 3 success criteria (lines 197-208).
- `.planning/REQUIREMENTS.md` — **EVAL-01 / EVAL-02 / EVAL-03** (active); IMG-03 descoped; serving (API-*) complete-but-UI-less.
- `.planning/PROJECT.md` — milestone framing; constraints (PyTorch, official splits, F1-per-error, identical metrics).
- `~/.claude/plans/phase_8.md` — the continuation handoff (orienting; verified against files this session — the EVAL-01 spine + arrival checks confirmed).

### The EVAL inputs — the three results.pkl + the three FINDINGS (verify the pickles; read the FINDINGS)
- Squat: `.planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl` + `docs/eval/FINDINGS.md` + `docs/figures/squat_*.png` + `docs/notebooks/01-04_squat_*`.
- OHP: `.planning/phases/06-overhead-press/figures/results.pkl` + `docs/eval/FINDINGS_OHP.md` + `docs/figures/ohp_*.png` + `docs/notebooks/05-08_ohp_*`.
- Shallow-Squat: `.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl` + `docs/eval/FINDINGS_SHALLOW_SQUAT.md` + `docs/figures/shallow_squat_*.png` + `docs/notebooks/09-11_shallow_squat_*` + `.planning/phases/07-image-based-errors-cvcspc/07-05-SUMMARY.md`.
- `docs/notebooks/README.md` — the notebook index to extend with notebook 12.

### The reusable eval/viz CODE to mirror (do NOT redefine)
- `backend/training/aqa/eval/metrics.py` — f1 / pr_auc / threshold_sweep / confusion (reused UNCHANGED).
- `backend/scripts/ohp_eval_viz.py`, `backend/scripts/ohp_headline_chart.py`, `backend/scripts/ohp_finetune_curves.py`, `backend/scripts/ohp_eda_viz.py` — the OHP figure-pack pattern (results.pkl → PNGs).
- `backend/scripts/eval_benchmark_viz.py`, `backend/scripts/eval_best_examples.py` — the Squat figure-pack pattern.
- `backend/training/aqa/eval/ensemble.py` (`aggregate_sigmoid_mean`) — score aggregation, if any re-aggregation is needed (likely not — pickles hold ensemble scores).

### Comparator papers (researcher sources; D3 honesty rule governs)
- `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — Parmar et al. (ECCV 2022, arXiv:2202.14019): the paper baseline/SSL/MD/CVCSPC numbers (cross-check the spine) + the EVAL-02 ensemble question (does the paper ensemble MD+CVCSPC for a shared error?). Read tool can't render — extract via `fitz`/`pypdf`.
- `Fitness-AQA/Fine-Tuning Large Multimodal Models for Fitness Action Quality Assessment.pdf` — the **LMM (Dibenedetto 2025)** paper (LOCAL) — source its Fitness-AQA per-error numbers for the context narrative.
- **GYMetricPose (Gallardo 2024)** — NOT local → web research pass; source its per-error numbers if reported on Fitness-AQA; omit-and-say-so if unsourceable.

### Prior-phase context (the deliverable + descoping precedent)
- `.planning/phases/07-image-based-errors-cvcspc/07-CONTEXT.md` — CVCSPC scope, loss deviation, descoping (IMG-03, serving).
- `.planning/phases/06-overhead-press/06-CONTEXT.md` — OHP deliverable pattern; D1 serving descope; GYMetricPose/LMM flagged for Phase 8 (line 142, 194).
- `.planning/phases/04-squat-motion-disentangling-ssl/04-04-SUMMARY.md` — the comparison-chart pattern + production-protocol declaration.

### Working-agreement memories (load all)
[[feedback_ai_correctness]], [[feedback_dont_agree]], [[feedback_working_style]], [[feedback_no_ai_comments]], [[feedback_run_gsd_autonomously]], [[feedback_dont_ask_to_push]], [[feedback_deliver_colab_as_ipynb]], [[project_supervisor_visualizations]], [[project_form_correction_status]], [[project_realtime_demo_expectation]], [[reference_fitness_aqa]].

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`eval/metrics.py`** — binary F1 / PR-AUC / threshold-sweep / confusion; reused as-is for any recomputation.
- **`backend/scripts/ohp_*.py` + `eval_benchmark_viz.py`** — the established `results.pkl` → matplotlib → `docs/figures/*.png` pattern (palette, layout, defensive mkdir, save-before-show). The cross-exercise figures mirror these.
- **The three `results.pkl`** — verified schemas this discussion:
  - Squat: `final{test_macro,test_f1{KIE,KFE},test_prauc,thresholds}`, `comparison{phase3,paper_kinetics,paper_md,phase4_ensemble}`, `per_seed`, `ssl_v1/v2`, `finetune_curves`, `raw{ens_test_scores(244,2),test_labels,...}`.
  - OHP: `test_f1{elbows,knees,macro}`, `test_pr_auc`, `thresholds`, `baseline_control`, `paper_targets{elbows,knees}`, `ssl_lift`, `val_test_gap`, `ensemble_test_scores(339,2)`, `per_seed_*`, `n_seeds=2`.
  - Shallow: `test_f1{cvcspc_ensemble,single_model,per_seed}`, `test_pr_auc`, `baseline_control{test_f1=0.875}`, `paper_targets{cvcspc,simsiam,openpose_tdm}`, `ssl_lift`, `val_test_gap`, `ssl_triplet_acc`, `ensemble_test_scores(540,)`, `loss_deviation_note`.
- **`docs/notebooks/` + `docs/eval/` + `docs/figures/`** — the deliverable home; notebook 12 + FINDINGS_FULL + master figures land alongside the per-phase artifacts.

### Established Patterns
- jupytext-paired `.py` source + pre-executed `.ipynb`; figures saved to `docs/figures/` with captions; FINDINGS prose carries the comparison narrative + honest framing.
- macro/F1 computed in code from real scores; paper numbers cited; no hand-typed headline literals.

### Integration Points
- **Nothing touches the FastAPI runtime** (serving descoped). Phase 8 reads committed pickles + paper PDFs and writes docs/notebooks/figures only. Optionally a small `backend/scripts/full_eval_viz.py` (planner's call).
- The AQA test suite must stay green (`python -m pytest backend/training/aqa/ -m "not slow" -q` → 45 passed / 4 deselected) — Phase 8 adds no production code that should change it; if a viz script is added, a light smoke test is optional (planner's call).

</code_context>

<specifics>
## Specific Targets, Numbers, and Patterns

**The verified EVAL-01 spine (cross-checked against the three `results.pkl` this discussion — every value traced):**

| Exercise / Error | Modality | Ours (official test F1) | Parmar SSL (paper) | Parmar baseline (paper) |
|---|---|---|---|---|
| Squat — Knees-Inward (KIE) | video (MD) | 0.420 | 0.419 | 0.297 |
| Squat — Knees-Forward (KFE) | video (MD) | 0.841 | 0.834 | 0.818 |
| OHP — Elbows | video (MD) | 0.447 | 0.455 | — (no paper supervised row) |
| OHP — Knees | video (MD) | 0.877 | 0.845 | — (no paper supervised row) |
| Shallow-Squat — depth | image (CVCSPC) | 0.8902 | 0.8694 | — (no paper supervised row) |

- **Macros:** Squat 0.6304 (paper MD 0.6262); OHP 0.6622 (paper MD 0.6502); Shallow 0.8902 (paper CVCSPC 0.8694).
- **SSL lifts (over our own baselines):** Squat macro 0.6304 vs Phase-3 baseline 0.5429 (+0.087); OHP +0.050 (baseline macro 0.6118); Shallow +0.0152 (baseline 0.8750).
- **val→test gaps (no overfit):** OHP 0.011; Shallow 0.0217. (Squat: see results.pkl.)
- **Extra paper rows for the Shallow context:** SimSiam 0.8286, OpenPose-TDM 0.8340 (already in the Shallow results.pkl `paper_targets`).
- **GYMetricPose / LMM:** TO RESEARCH (LMM PDF local; GYMetricPose web) → **context narrative only, no columns** (D3). Cite or omit-and-say-so.

**Methodology diagram content (D5 fig 3):** video errors (Squat KIE/KFE, OHP Elbows/Knees) → Motion-Disentangling half-cycle SSL on barbell trajectories → R(2+1)D-18 (32-frame, 112², Kinetics-init) → joint multi-label head; image error (Shallow-Squat) → CVCSPC phase-contrastive SSL → ResNet-18 (224², ImageNet-init) → single binary head; both → official train/val/test splits → F1-per-error vs Parmar. Schematic only — no invented numbers.

**BarbellRow:** descoped (IMG-03). The milestone covers 5 errors across 3 exercises + 2 methods.

</specifics>

<deferred>
## Deferred Ideas

- **BarbellRow Lumbar/Torso (IMG-03)** — cancelled this milestone (compute/time). If a future milestone revives it, the CVCSPC image pipeline already exists.
- **Backend serving / inference API for OHP + Shallow-Squat + the unified 3-exercise API** — descoped (form-correction frontend cancelled). If a future UI milestone revives serving, integration goes there (parity with the kept Squat `SquatFormService`).
- **A true MD+CVCSPC cross-method ensemble** — only meaningful if a future error is scored by both methods; N/A to the current single-method-per-error scope (D4).
- **Live/phone-camera domain adaptation** — the in-the-wild generalization limit; out of milestone scope ([[project_realtime_demo_expectation]]).

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-ensemble-evaluation-visualization-pack*
*Context gathered: 2026-05-30 via /gsd:discuss-phase 8 (autonomous; 2 defense-shaping decisions put to the user → D3 context-only comparators, D5 comprehensive viz pack)*
