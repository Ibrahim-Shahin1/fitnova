# Phase 8: Ensemble, Evaluation & Visualization Pack — Research

**Researched:** 2026-05-30
**Domain:** Local assembly from three committed results.pkl + external comparator sourcing
**Confidence:** HIGH (all core inputs are verified from local files; only GYMetricPose numbers sourced via the LMM paper's Table 4 rather than from the GYMetricPose paper directly)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D1** — LOCAL assembly from three committed results.pkl; NO Colab, NO re-runs.
- **D2** — EVAL-01 master table = ours vs Parmar (baseline + SSL) across all 5 errors. Columns: Exercise/Error · Modality · Ours (official test F1) · Parmar SSL (paper) · Parmar baseline (paper, where it exists).
- **D3** — GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025) = sourced related-work CONTEXT NARRATIVE, NOT table columns. Source their real numbers + cite; if a number can't be sourced or isn't on the identical metric/split, present it as labeled context. NEVER fabricate.
- **D4** — EVAL-02 (MD+CVCSPC ensemble): research-confirm, report as NOT-APPLICABLE; do NOT invent one.
- **D5** — EVAL-03 deliverable = comprehensive defense pack: `docs/notebooks/12_full_evaluation.{py,ipynb}` (pre-executed) + `docs/eval/FINDINGS_FULL.md` + master figures under `docs/figures/`.
- **D6** — Reuse established eval/viz code and deliverable pattern; do NOT redefine.
- **D7** — Honest framing carried into the consolidated write-up (no paper supervised baseline for OHP/Shallow, 3-term vs 2-term CVCSPC loss deviation, TTA not adopted, OHP 2-seed vs Squat/Shallow 3-seed, benchmark-only caveat).
- **D8** — No fabrication; every number computed-in-code or cited.

### Claude's Discretion

- Exact final figure count + filenames within the D5 set.
- Whether master figures are built inline in notebook 12 or via a small `backend/scripts/full_eval_viz.py`.
- Whether FINDINGS_FULL extends/links the three per-phase FINDINGS or restates key tables (lean: consolidated table + links).
- Exact GYMetricPose/LMM numbers + which errors they cover — depends on what the papers actually report.
- Plan/wave shape.

### Deferred Ideas (OUT OF SCOPE)

- BarbellRow Lumbar/Torso (IMG-03) — cancelled this milestone.
- Backend serving / inference API for OHP + Shallow-Squat.
- A true MD+CVCSPC cross-method ensemble — only meaningful if a future error is scored by both methods.
- Live/phone-camera domain adaptation.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Identical-metric comparison (ours vs Parmar / GYMetricPose / LMM) across all errors | Parmar numbers verified from paper Tables 2/3/4; GYMetricPose numbers sourced from LMM paper Table 4 (with comparability caveat documented); LMM numbers sourced from local PDF Table 4. All numbers have provenance. |
| EVAL-02 | MD + CVCSPC ensemble evaluated where the paper applies it | Confirmed via Parmar Table 2: the paper does apply MD+CVCSPC ensemble to BackSquat KIE/KFE — but only those two errors. Our 5-error scope already implements each error with exactly one method. Finding: EVALUATED / NOT-APPLICABLE to our single-method-per-error scope (see Section EVAL-02 Analysis below). |
| EVAL-03 | Consolidated visualization / results pack | Notebook 12 + FINDINGS_FULL + 4 master figures — assembly pattern verified against notebooks 01–11 and the three results.pkl schemas. |

</phase_requirements>

---

## Summary

Phase 8 is local documentation/visualization assembly, not training. All three exercises are evaluated; the committed `results.pkl` files hold every number the notebook and figures need. The research task is: (a) settle the EVAL-02 ensemble question against the paper, (b) source external comparator numbers, and (c) confirm the code reuse pattern so the planner can write correct notebook code.

**EVAL-02 finding (confirmed):** Parmar et al. Table 2 does include a "Ours MD + CVCSPC" ensemble row for BackSquat KIE/KFE (KIE 0.5263, KFE 0.8468) — but that ensemble combines scores from two methods on the *same* error. In our implementation every error is assigned to exactly one method (KIE/KFE/Elbows/Knees → MD; Shallow-Squat → CVCSPC), so no error has both an MD score and a CVCSPC score. EVAL-02 is correctly reported as "evaluated → not-applicable to our single-method-per-error scope."

**GYMetricPose (Gallardo 2024):** Numbers found in LMM paper Table 4 — the only citable secondary source available locally. Comparability verdict: GYMetricPose was evaluated on Fitness-AQA Squat and OHP but the LMM paper does not confirm whether GYMetricPose used the identical official val/test split and threshold-tuning protocol. Present as labeled context, not a direct column comparison.

**LMM (Dibenedetto 2025):** Full paper read locally. Metric is F1-score on Squat and OHP errors only (Shallow-Squat not evaluated). The paper explicitly uses validation data in training ("incorporating validation data into training"), which deviates from the official train/val/test protocol used by us and Parmar. Not directly comparable on the identical split — present as context-only.

**Primary recommendation:** Write notebook 12 to load the three pickles, compute every headline F1 in code via `metrics.f1_per_error`, assert computed values match stored `test_f1` (self-check), then produce the four master figures and FINDINGS_FULL. GYMetricPose and LMM appear in a dedicated context paragraph in FINDINGS_FULL, cited, with explicit comparability caveats.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Load results.pkl and compute headline F1 | Python script / notebook | — | Pure offline computation from committed files |
| Master comparison figures | matplotlib (local) | — | No ML inference; pure plotting from pickle data |
| FINDINGS_FULL write-up | Markdown doc | — | Prose synthesis; no code |
| Validation assertions | notebook cell | — | In-notebook asserts catch drift before commit |
| AQA test suite | pytest | — | Must stay green; Phase 8 adds no production code |

---

## EVAL-02 Analysis: Does the paper apply MD+CVCSPC to a single shared error?

**Source:** Parmar et al. (ECCV 2022), Table 2 (page 12 of the local PDF at `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf`).

**Exact table row (cited verbatim):**
> Ours MD + CVCSPC | Image, Video | KIE: 0.5263 | KFE: 0.8468

**What the paper does:** It runs MD on the BackSquat labeled data and CVCSPC on the same BackSquat labeled data, then ensembles the two sets of scores for the *same* errors (KIE and KFE). Both methods produce a score for every clip; the ensemble is over those two score sets.

**Preceding paragraph (page 12):** "Furthermore, ensemble of our contrastive approaches outperformed all the models." — The note on the next page: "Note that in all the subsequent experiments, we selected only the best performing methods for further evaluation." The MD+CVCSPC ensemble does NOT appear in Table 3 (Shallow-Squat), Table 4 (OHP), or Table 5 (cross-exercise transfer). It is exclusive to BackSquat KIE/KFE.

**Why it is not applicable to our scope:**

In our implementation each error is assigned to a single method:

| Error | Our method |
|-------|-----------|
| Squat KIE | MD only |
| Squat KFE | MD only |
| OHP Elbows | MD only |
| OHP Knees | MD only |
| Shallow-Squat depth | CVCSPC only |

No error has both an MD score and a CVCSPC score. We cannot form the paper's ensemble without re-running CVCSPC on the BackSquat labeled data (which would require Colab training — out of scope per D1) and re-running MD on the Shallow-Squat data (which makes no architectural sense — MD requires video half-cycles and barbell trajectories; Shallow-Squat uses static image crops). Forming such an ensemble would be an invented metric, not the paper's ensemble.

**EVAL-02 conclusion:** EVALUATED. The paper does apply MD+CVCSPC ensemble but only to BackSquat KIE/KFE. This is not applicable to our 3-exercise, single-method-per-error scope. Report honestly as "evaluated → not-applicable."

**CONTEXT.md D2 spine cross-check against paper tables:**

| Value in D2 spine | Paper table | Match? |
|---|---|---|
| Squat KIE paper SSL 0.419 | Table 2, Ours MD row: 0.4186 | 0.4186 rounds to 0.419 — matches to 3 sig figs ✓ |
| Squat KFE paper SSL 0.834 | Table 2, Ours MD row: 0.8338 | 0.8338 rounds to 0.834 ✓ |
| Squat KIE paper baseline 0.297 | Table 2, Kinetics row: 0.2970 | Matches exactly ✓ |
| Squat KFE paper baseline 0.818 | Table 2, Kinetics row: 0.8184 | 0.8184 rounds to 0.818 ✓ |
| OHP Elbows paper SSL 0.455 | Table 4, Ours MD row: 0.4552 | 0.4552 rounds to 0.455 ✓ |
| OHP Knees paper SSL 0.845 | Table 4, Ours MD row: 0.8452 | 0.8452 rounds to 0.845 ✓ |
| Shallow CVCSPC paper SSL 0.869 | Table 3, Ours CVCSPC row: 0.8694 | 0.8694 rounds to 0.869 ✓ |
| Shallow SimSiam 0.829 | Table 3, SimSiam row: 0.8286 | 0.8286 rounds to 0.829 ✓ |
| Shallow OpenPose-TDM 0.834 | Table 3, OpenPose-TDM row: 0.8340 | 0.8340 rounds to 0.834 ✓ |

**No discrepancy found.** The D2 spine numbers are correct to the rounding used.

[VERIFIED: local PDF `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf`, Tables 2, 3, 4]

---

## LMM (Dibenedetto 2025) — Full Extraction

**Full citation:** Gaetano Dibenedetto, Elio Musacchio, Marco Polignano, Pasquale Lops. "Fine-Tuning Large Multimodal Models for Fitness Action Quality Assessment." In *Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization (UMAP Adjunct '25)*, June 16–19, 2025, New York City, NY, USA. ACM. DOI: 10.1145/3708319.3733684.

**Source:** Local PDF at `Fitness-AQA/Fine-Tuning Large Multimodal Models forFitness Action Quality Assessment.pdf`, Table 4.

**Which Fitness-AQA errors are reported:** BackSquat KIE + KFE, OHP Elbows + Knees. Shallow-Squat is NOT evaluated.

**Metric:** F1-score (per error body part) and mAP (for temporal action localization). The F1-score is the same metric as Parmar and as ours.

**Split deviation (critical):** Section 2.1 states: "We focus only on the labeled videos, incorporating validation data into training." This means Dibenedetto et al. trained on train+val and evaluated on the test split. This deviates from the official protocol (train only → val threshold tuning → test eval) used by Parmar and by us. Their test-set F1 numbers therefore cannot be compared column-for-column with ours or with Parmar — the larger training set gives an unfair advantage on the test split.

**LMM results from Table 4 (their best-performing variant = "Two-Step"):**

| Error | F1 (Two-Step) | F1 (Dynamic-Step) |
|-------|--------------|------------------|
| Squat KIE | 0.1955 | 0.0000 |
| Squat KFE | 0.6266 | 0.8155 |
| OHP Elbows | 0.4575 | 0.3959 |
| OHP Knees | 0.7611 | 0.7866 |

The paper's abstract acknowledges: "our model achieves results slightly lower than the baseline." The baselines in their Table 4 are the Parmar paper numbers (CVCSPC, MD, MD+CVCSPC) and GYMetricPose.

**Comparability verdict:** NOT directly comparable. The LMM paper uses train+val for training, which inflates the effective training set relative to the official protocol. Present in FINDINGS_FULL as: "Dibenedetto et al. (2025) fine-tune LLaVA-Video-7B-Qwen2 on Fitness-AQA Squat and OHP errors. They report F1 on the test split (train+val used for training, deviating from the official protocol), achieving KIE 0.20 / KFE 0.63 / Elbows 0.46 / Knees 0.76 (Two-Step variant). Despite the larger effective training set, results fall below the Parmar SSL baselines on most errors — demonstrating the difficulty of subtle form-error detection for large multimodal models without domain-specific SSL pretraining."

[VERIFIED: local PDF `Fitness-AQA/Fine-Tuning Large Multimodal Models forFitness Action Quality Assessment.pdf`, Section 2.1 + Table 4]

---

## GYMetricPose (Gallardo 2024) — Sourced Numbers + Comparability

**Citation:** Ulises Gallardo, Fernando Caro, Eluney Hernández, Ricardo Espinosa, Gilberto Ochoa-Ruiz. "GYMetricPose: A light-weight angle-based graph adaptation for action quality assessment." In *37th IEEE International Symposium on Computer-Based Medical Systems (CBMS 2024)*, Guadalajara, Mexico, June 26–28, 2024, pp. 43–50. IEEE. DOI: 10.1109/CBMS61543.2024.00016.

**Source of numbers:** LMM paper (Dibenedetto et al. 2025) Table 4, which uses GYMetricPose as one of its baselines. [CITED: LMM local PDF, Table 4, referencing Gallardo et al. [10]]. The GYMetricPose paper itself is not available locally; the specific numbers are sourced from the LMM paper's table.

**Numbers reported (GYMetricPose, F1-score on Fitness-AQA):**

| Error | GYMetricPose F1 |
|-------|----------------|
| Squat KIE | 0.4398 |
| Squat KFE | 0.8219 |
| OHP Elbows | 0.4175 |
| OHP Knees | 0.8160 |

Shallow-Squat: NOT reported (the LMM paper does not cover Shallow-Squat).

**Comparability verdict:** UNCERTAIN. The LMM paper cites these numbers as "GYMetricPose [10]" in its Table 4, presented alongside Parmar's published numbers and its own results. The LMM paper does not state which split GYMetricPose used. The GYMetricPose paper itself (IEEE CBMS 2024) is behind a paywall; we cannot verify directly whether the official val/test split and threshold-tuning protocol were followed.

**Plan for FINDINGS_FULL:** Present GYMetricPose numbers in a clearly-labeled context paragraph: "Gallardo et al. (2024) propose GYMetricPose, a lightweight angle-based graph adaptation using 3D pose estimation, reported on Fitness-AQA Squat and OHP (KIE 0.440 / KFE 0.822 / Elbows 0.418 / Knees 0.816 — numbers from Dibenedetto et al. Table 4; direct split verification not possible without access to the CBMS 2024 paper). The approach underperforms both CVCSPC and MD on all four errors reported." Include the caveat that direct split comparability is unconfirmed.

[CITED: LMM local PDF Table 4 re-reporting Gallardo et al. [10]; the GYMetricPose paper itself is IEEE CBMS 2024 DOI 10.1109/CBMS61543.2024.00016]

---

## The Three results.pkl — Exact Key Schemas (Verified)

All three files confirmed loadable. Schemas documented for notebook code correctness.

### Squat pkl: `.planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl`

```
final/
  method                => 'MD-SSL 3-seed ensemble, no TTA'
  test_macro            => 0.630389363722697       # float
  test_f1/
    KIE                 => 0.41975308641975306     # float
    KFE                 => 0.841025641025641       # float
  test_prauc/
    KIE                 => 0.38753950601104514
    KFE                 => 0.8811542736914009
  test_confusion/
    KIE                 => 2-element list (rows)   # use np.array(...)
    KFE                 => 2-element list
  thresholds/
    KIE                 => 0.61448742486136
    KFE                 => 0.38465035725739627
per_seed/
  42/   KIE, KFE, macro, val_best  (floats)
  1337/ KIE, KFE, macro, val_best
  7/    KIE, KFE, macro, val_best
tta_finding/
  selected_recipe       => list
  tta_test_macro        => 0.6231
  delta_vs_no_tta       => -0.0073
  decision              => str (TTA not adopted)
comparison/
  phase3/               => KIE 0.2857, KFE 0.8, macro 0.5429
  paper_kinetics/       => KIE 0.297, KFE 0.8184, macro 0.5577
  paper_md/             => KIE 0.4186, KFE 0.8338, macro 0.6262
  phase4_ensemble/      => KIE 0.41975..., KFE 0.841..., macro 0.6304
ssl_v1/metrics, ssl_v2/metrics, ssl_v1/linear_probe, ssl_v2/linear_probe
finetune_curves/        => {42: [...], 1337: [...], 7: [...]}
raw/
  ens_test_scores       => shape (244, 2)   # col 0 = KIE, col 1 = KFE
  test_labels           => shape (244, 2)
  ens_val_scores        => shape (243, 2)
  val_labels            => shape (243, 2)
```

### OHP pkl: `.planning/phases/06-overhead-press/figures/results.pkl`

```
ssl_metrics_history     => list[dict] len=60
ssl_linear_probe_history=> list[dict] len=12
finetune_seeds/
  42                    => list[dict] len=15
  1337                  => list[dict] len=17
ensemble_val_scores     => shape (339, 2)   # col 0 = Elbows, col 1 = Knees
ensemble_test_scores    => shape (339, 2)
val_labels              => shape (339, 2)
test_labels             => shape (339, 2)
test_clip_ids           => list len=339
thresholds/
  elbows                => 0.3573554754257202
  knees                 => 0.4764750003814697
test_f1/
  elbows                => 0.4473684210526316
  knees                 => 0.8770491803278688
  macro                 => 0.6622088006902502
test_pr_auc/
  elbows                => 0.3694151734885147
  knees                 => 0.9372072145369958
test_confusion/
  elbows                => shape (2, 2) ndarray
  knees                 => shape (2, 2) ndarray
per_seed_test_f1/
  42/ elbows, knees     (floats)
  1337/ elbows, knees
per_seed_val_macro/
  42                    => 0.6563745...
  1337                  => 0.6270473...
baseline_control/
  elbows                => 0.4167
  knees                 => 0.8069
  macro                 => 0.6118
ssl_lift                => 0.05040880069025022
val_test_gap            => 0.0109312514649883
paper_targets/
  elbows                => 0.4552
  knees                 => 0.8452
n_seeds                 => 2
tta                     => 'none'
```

### Shallow-Squat pkl: `.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl`

```
ssl_triplet_acc         => list len=20   # epoch-wise triplet accuracy
finetune_seeds/
  42                    => list[dict] len=21
  1337                  => list[dict] len=22
  7                     => list[dict] len=14
baseline_seeds/
  42,1337,7             => list[dict] (per-epoch val F1 for supervised baseline)
ensemble_val_scores     => shape (529,)   # single output (binary)
ensemble_test_scores    => shape (540,)   # single output
val_labels              => shape (529,)
test_labels             => shape (540,)
best_threshold          => 0.3953738510608673
test_f1/
  cvcspc_ensemble       => 0.8902195608782435   # THE headline
  single_model          => 0.8961303462321792   # best single seed (42)
  per_seed/
    42                  => 0.8961303462321792
    1337                => 0.8902195608782435
    7                   => 0.8824742268041237
test_pr_auc             => 0.9667833542870422
test_confusion          => shape (2, 2)
baseline_control/
  test_f1               => 0.875
  threshold             => 0.4733802080154419
  ensemble_test_scores  => shape (540,)
  ensemble_val_f1       => 0.8942115768463074
cvcspc_ensemble_val_f1  => 0.9119373776908023
ssl_lift                => 0.015219560878243499
val_test_gap            => 0.0217178168125588
paper_targets/
  cvcspc                => 0.8694
  simsiam               => 0.8286
  openpose_tdm          => 0.834
loss_deviation_note     => str  # the 3-term vs 2-term deviation note
seeds                   => list len=3
```

[VERIFIED: `python -c "import pickle; d=pickle.load(open(...)); ..."` — all three files opened and schemas confirmed]

---

## Standard Stack

No new packages are installed for Phase 8. All dependencies are already in `backend/requirements.txt`.

### Core (already installed)
| Library | Purpose | How used in Phase 8 |
|---------|---------|---------------------|
| `pickle` (stdlib) | Load results.pkl | `pickle.load(open(path, "rb"))` — all three pickles |
| `matplotlib` | Figures | All four master figures |
| `numpy` | Array ops | Score slicing, label arrays, assertion tolerance |
| `sklearn.metrics` | F1/PR recompute | `f1_score`, `precision_recall_curve`, `average_precision_score` |
| `jupytext` | Notebook pairing | `jupytext --to notebook 12_full_evaluation.py` then `jupyter nbconvert --to notebook --execute --inplace` |
| `jupyter` + `nbconvert` | Pre-execute notebook | Already present from prior phases |

### Reusable Code — Exact Signatures

**`backend/training/aqa/eval/metrics.py`** — reused UNCHANGED:
- `f1_per_error(y_true: np.ndarray, y_pred: np.ndarray) -> float`
- `pr_auc_per_error(y_true: np.ndarray, y_score: np.ndarray) -> float`
- `threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]`
- `confusion_matrix_per_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray`

Import path: `from backend.training.aqa.eval.metrics import f1_per_error, threshold_sweep`

**Figure script pattern** (from `backend/scripts/ohp_eval_viz.py`):
```python
PKL = Path(".planning/phases/06-overhead-press/figures/results.pkl")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)   # defensive mkdir
# ... build figure ...
fig.savefig(FIG / "name.png", dpi=120, bbox_inches="tight")  # save BEFORE plt.show()
print("saved name.png")
```

**Headline chart pattern** (from `backend/scripts/ohp_headline_chart.py`):
- `ROWS` dict of method → {error: f1}
- `x = np.arange(len(ERRORS)); w = 0.26; bars per method offset by (i-1)*w`
- Value annotations: `ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f"{v:.3f}", ...)`

## Package Legitimacy Audit

No new packages installed. N/A.

---

## Architecture Patterns

### System Architecture Diagram

```
Three committed results.pkl (local)
        |
        v
  12_full_evaluation.py (jupytext source)
        |
        ├── Load each pkl, extract arrays (raw scores + labels)
        |       Squat:  raw/ens_test_scores(244,2), raw/test_labels(244,2)
        |       OHP:    ensemble_test_scores(339,2), test_labels(339,2)
        |       Shallow:ensemble_test_scores(540,), test_labels(540,)
        |
        ├── Recompute F1 in code via metrics.f1_per_error
        |       (asserts match stored test_f1 values — self-check)
        |
        ├── Build master figures (4 PNGs → docs/figures/)
        |       fig1: all_errors_vs_paper.png (grouped bar: all 5 errors)
        |       fig2: cross_exercise_macro.png (per-exercise macro bar)
        |       fig3: methodology_diagram.png (architecture schematic)
        |       fig4: highlights_panel.png (per-exercise headline summary)
        |
        └── Export FINDINGS_FULL.md prose + links
                (narrative synthesis; GYMetricPose+LMM context paragraph)

  jupytext → 12_full_evaluation.ipynb → nbconvert --execute → pre-executed .ipynb
```

### Recommended Project Structure (additions only)

```
docs/
├── notebooks/
│   ├── 12_full_evaluation.py      # jupytext source
│   └── 12_full_evaluation.ipynb   # pre-executed
├── eval/
│   └── FINDINGS_FULL.md           # consolidated write-up
└── figures/
    ├── all_errors_vs_paper.png    # EVAL-01 headline bar
    ├── cross_exercise_macro.png   # per-exercise macro summary
    ├── methodology_diagram.png    # two-method architecture schematic
    └── highlights_panel.png       # per-exercise headline panel

backend/scripts/
└── full_eval_viz.py  (optional — planner's call; can be inline in nb 12)
```

### Pattern 1: Validation Assertion (No-Fabrication Structural Enforcement)

Compute F1 from raw arrays, then assert equality against the stored test_f1. Catches any drift or hand-typed literal.

```python
import numpy as np
from backend.training.aqa.eval.metrics import f1_per_error

squat = pickle.load(open(".planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl", "rb"))

ens   = np.asarray(squat["raw"]["ens_test_scores"])   # (244, 2)
labs  = np.asarray(squat["raw"]["test_labels"])        # (244, 2)
thr   = squat["final"]["thresholds"]

kie_pred = (ens[:, 0] >= thr["KIE"]).astype(int)
kfe_pred = (ens[:, 1] >= thr["KFE"]).astype(int)

kie_f1 = f1_per_error(labs[:, 0], kie_pred)
kfe_f1 = f1_per_error(labs[:, 1], kfe_pred)

assert abs(kie_f1 - squat["final"]["test_f1"]["KIE"]) < 1e-6, \
    f"KIE F1 drift: computed {kie_f1}, stored {squat['final']['test_f1']['KIE']}"
assert abs(kfe_f1 - squat["final"]["test_f1"]["KFE"]) < 1e-6, \
    f"KFE F1 drift: computed {kfe_f1}, stored {squat['final']['test_f1']['KFE']}"
```

For OHP (shape (339,2), keys `test_f1/elbows` and `test_f1/knees`, arrays `ensemble_test_scores` + `test_labels`):
```python
ohp = pickle.load(open(".planning/phases/06-overhead-press/figures/results.pkl", "rb"))
ens  = np.asarray(ohp["ensemble_test_scores"])   # (339, 2)
labs = np.asarray(ohp["test_labels"]).astype(int)
thr  = ohp["thresholds"]

elbow_pred = (ens[:, 0] >= thr["elbows"]).astype(int)
knee_pred  = (ens[:, 1] >= thr["knees"]).astype(int)

elbow_f1 = f1_per_error(labs[:, 0], elbow_pred)
knee_f1  = f1_per_error(labs[:, 1], knee_pred)

assert abs(elbow_f1 - ohp["test_f1"]["elbows"]) < 1e-6
assert abs(knee_f1  - ohp["test_f1"]["knees"])  < 1e-6
```

For Shallow-Squat (shape (540,) single-output):
```python
ss = pickle.load(open(".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl", "rb"))
ens  = np.asarray(ss["ensemble_test_scores"])   # (540,)
labs = np.asarray(ss["test_labels"]).astype(int)
thr  = ss["best_threshold"]

pred = (ens >= thr).astype(int)
ss_f1 = f1_per_error(labs, pred)

assert abs(ss_f1 - ss["test_f1"]["cvcspc_ensemble"]) < 1e-6
```

### Pattern 2: Master Comparison Table (EVAL-01)

The table header and all numeric cells are constructed from the pickle values in code — no hard-coded literals in the data cells. Paper comparison numbers are loaded from the `comparison` / `paper_targets` sub-dicts within each pickle (already stored there from Phase 4/6/7).

```python
ERRORS = [
    ("Squat KIE",        "video (MD)",   squat["final"]["test_f1"]["KIE"],
     squat["comparison"]["paper_md"]["KIE"],   squat["comparison"]["paper_kinetics"]["KIE"]),
    ("Squat KFE",        "video (MD)",   squat["final"]["test_f1"]["KFE"],
     squat["comparison"]["paper_md"]["KFE"],   squat["comparison"]["paper_kinetics"]["KFE"]),
    ("OHP Elbows",       "video (MD)",   ohp["test_f1"]["elbows"],
     ohp["paper_targets"]["elbows"],     None),
    ("OHP Knees",        "video (MD)",   ohp["test_f1"]["knees"],
     ohp["paper_targets"]["knees"],      None),
    ("Shallow-Squat",    "image (CVCSPC)", ss["test_f1"]["cvcspc_ensemble"],
     ss["paper_targets"]["cvcspc"],      None),
]
```

### Pattern 3: Methodology Diagram (matplotlib, no ML data)

A pure matplotlib schematic; no scores or curves. Two branches:
- Left: "Video errors (Squat KIE/KFE, OHP Elbows/Knees)" → "Motion-Disentangling SSL on barbell trajectories" → "R(2+1)D-18 (32-frame, 112², Kinetics-init)" → "multi-label head"
- Right: "Image errors (Shallow-Squat depth)" → "CVCSPC pose-contrastive SSL" → "ResNet-18 (224², ImageNet-init)" → "single binary head"
- Both → "Official train/val/test splits" → "F1-per-error vs Parmar (ECCV 2022)"

Use `ax.annotate`, `ax.text`, `FancyArrowPatch` or simple `ax.arrow`. No `plt.imread` of any image file.

### Anti-Patterns to Avoid

- **Hard-coded headline literals in notebook code:** the Phase-3 bug (user caught wrong hand-typed F1 values). Every F1 shown in the notebook must come from `pickle[key]` or computed via `f1_per_error`.
- **`plt.show()` before `fig.savefig()`:** figure disappears in non-interactive execution. Always `savefig` first.
- **Missing `mkdir(parents=True, exist_ok=True)`:** notebooks pre-execute from scratch; the figures directory must be created defensively.
- **AI narration comments in code cells:** comments only for non-obvious WHY; no "per D8", "mirrors Phase 6", "as per CONTEXT" in code.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| F1 computation | Custom F1 loop | `metrics.f1_per_error` (existing) | Already tested, zero_division handled, consistent with all prior phases |
| PR-AUC computation | Custom area-under-PR | `metrics.pr_auc_per_error` (existing) | Consistent |
| Threshold sweep | Custom grid search | `metrics.threshold_sweep` (existing) | Uses `precision_recall_curve` (all unique thresholds) |
| Confusion matrix | Manual counting | `metrics.confusion_matrix_per_error` | Fixed 2×2 shape guaranteed even for single-class predictions |
| Figure layout | New plot style | Mirror `ohp_headline_chart.py` palette + bar layout | Consistency across 11 existing notebooks |
| Notebook pairing | Custom serializer | `jupytext --to notebook` → `nbconvert --execute` | Proven pattern in notebooks 01–11 |

---

## Common Pitfalls

### Pitfall 1: Squat test_f1 threshold mismatch

**What goes wrong:** Notebook uses `comparison/phase4_ensemble/KIE` (which equals `final/test_f1/KIE`) — both are correct. But if `raw/ens_test_scores` is used to recompute, the threshold must come from `final/thresholds/KIE` (0.614), not from a fresh `threshold_sweep` (which would give a slightly different value from the val set). The official Colab eval used the val-tuned threshold; the assertion must use the same stored threshold.

**Prevention:** Always use `squat["final"]["thresholds"]["KIE"]` / `["KFE"]` when re-applying to test scores. Do not re-sweep the threshold on test data.

### Pitfall 2: Shallow-Squat score array shape

**What goes wrong:** Shallow-Squat `ensemble_test_scores` is shape `(540,)` (single sigmoid output), not `(540, 2)` like the video models. Attempting `ens[:, 0]` will raise IndexError.

**Prevention:** Use `ens = np.asarray(ss["ensemble_test_scores"])` directly (already 1-D); apply threshold as `(ens >= thr).astype(int)`.

### Pitfall 3: Macro-F1 computation divergence

**What goes wrong:** `(KIE_F1 + KFE_F1) / 2` gives 0.630389... which matches the stored `final/test_macro`. If `sklearn.metrics.f1_score(..., average='macro')` is called directly on the multi-label arrays it may give a different result due to micro/macro averaging semantics. The paper and our prior phases all use the simple arithmetic mean of the two per-error F1s.

**Prevention:** Compute macro as `(kie_f1 + kfe_f1) / 2` for Squat, `(elbow_f1 + knee_f1) / 2` for OHP. Assert against `squat["final"]["test_macro"]` and `ohp["test_f1"]["macro"]`.

### Pitfall 4: OHP confusion matrix is ndarray, Squat is list

**What goes wrong:** `squat["final"]["test_confusion"]["KIE"]` is a 2-element list of lists; `ohp["test_confusion"]["elbows"]` is already a `numpy.ndarray` shape `(2, 2)`. Code that calls `.shape` on the Squat confusion will fail.

**Prevention:** Always wrap with `np.array(squat["final"]["test_confusion"]["KIE"])` before using `.shape` or array indexing.

### Pitfall 5: GYMetricPose numbers used as direct column

**What goes wrong:** Planner adds GYMetricPose F1 as a fourth column in the EVAL-01 master table, making it look like a directly comparable number.

**Prevention:** GYMetricPose appears ONLY in the context narrative paragraph in FINDINGS_FULL, labeled explicitly as "numbers from Dibenedetto et al. Table 4; direct split verification not confirmed." No table column.

---

## Code Examples

### Loading all three pickles and extracting headline numbers

```python
import pickle
import numpy as np
from pathlib import Path

SQUAT_PKL  = Path(".planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl")
OHP_PKL    = Path(".planning/phases/06-overhead-press/figures/results.pkl")
SHALLOW_PKL= Path(".planning/phases/07-image-based-errors-cvcspc/figures/results.pkl")

squat  = pickle.load(SQUAT_PKL.open("rb"))
ohp    = pickle.load(OHP_PKL.open("rb"))
ss     = pickle.load(SHALLOW_PKL.open("rb"))

# Headline numbers (read from pickle, not hard-coded)
kie_f1  = squat["final"]["test_f1"]["KIE"]    # 0.41975...
kfe_f1  = squat["final"]["test_f1"]["KFE"]    # 0.84102...
s_macro = squat["final"]["test_macro"]          # 0.63038...

elb_f1  = ohp["test_f1"]["elbows"]            # 0.44736...
kne_f1  = ohp["test_f1"]["knees"]             # 0.87704...
o_macro = ohp["test_f1"]["macro"]              # 0.66220...

ss_f1   = ss["test_f1"]["cvcspc_ensemble"]    # 0.89021...
```

### EVAL-01 grouped bar chart (master figure 1)

```python
fig, ax = plt.subplots(figsize=(13, 5.5))
errors = ["Squat\nKIE", "Squat\nKFE", "OHP\nElbows", "OHP\nKnees", "Shallow-\nSquat"]
ours        = [kie_f1, kfe_f1, elb_f1, kne_f1, ss_f1]
paper_ssl   = [
    squat["comparison"]["paper_md"]["KIE"],
    squat["comparison"]["paper_md"]["KFE"],
    ohp["paper_targets"]["elbows"],
    ohp["paper_targets"]["knees"],
    ss["paper_targets"]["cvcspc"],
]
paper_base  = [
    squat["comparison"]["paper_kinetics"]["KIE"],
    squat["comparison"]["paper_kinetics"]["KFE"],
    None, None, None,   # no paper supervised baseline for OHP/Shallow
]

x = np.arange(len(errors))
w = 0.26
bars_ours = ax.bar(x - w, ours, w, label="Ours", color="C0")
bars_ssl  = ax.bar(x,     paper_ssl, w, label="Parmar SSL (paper)", color="C1")
# paper_base: skip None entries
base_x = [x[i] + w for i in range(2)]   # only Squat has baseline
base_v = [paper_base[i] for i in range(2)]
ax.bar(base_x, base_v, w, label="Parmar baseline (paper)", color="0.65")

for b in list(bars_ours) + list(bars_ssl):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(errors)
ax.set_ylabel("F1 (official test split)")
ax.set_ylim(0, 1.05)
ax.set_title("All 5 errors — FitNova vs Parmar et al. (ECCV 2022)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "all_errors_vs_paper.png", dpi=120, bbox_inches="tight")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pose estimator → hand-crafted features (OpenPose-TDM) | Raw-pixel CNN with domain-knowledge SSL | Parmar 2022 | SSL outperforms pose on in-the-wild gym video |
| Single general SSL (SimSiam) | Exercise-specific MD (half-cycle contrast) and CVCSPC (phase-contrastive) | Parmar 2022 | Domain-specific SSL consistently beats general SSL |
| Per-exercise single model | Multi-exercise LMM fine-tuning | Dibenedetto 2025 | LMM still underperforms the domain-specific CNN methods |

**Deprecated/outdated:**
- OpenPose-TDM for Fitness-AQA: superseded by raw-pixel SSL methods (paper 2022).
- LLaVA-Video without per-error SSL pretraining: results below MD/CVCSPC on most errors.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GYMetricPose numbers in LMM Table 4 match the numbers in the GYMetricPose CBMS 2024 paper | GYMetricPose section | If the LMM paper misreports or rounds differently, the cited numbers would be incorrect. Risk: low (Table 4 explicitly cites Gallardo [10] and uses same errors). Mitigation: explicitly label as "from Dibenedetto et al. Table 4" not "from GYMetricPose paper". |
| A2 | GYMetricPose used the identical official train/val/test split | GYMetricPose comparability verdict | If it used a different split, its F1 numbers are not directly comparable. Risk: medium (we cannot verify without access to the CBMS paper). Mitigation: the comparability caveat is already in the verdict. |

---

## Open Questions

1. **GYMetricPose paper access**
   - What we know: Numbers for Squat KIE/KFE and OHP Elbows/Knees sourced from LMM Table 4.
   - What's unclear: Whether GYMetricPose used the official split and val-tuned thresholds.
   - Recommendation: Present in FINDINGS_FULL with explicit caveat; do not add to the master table (D3 locked).

2. **Figure 3 (methodology diagram) tool choice**
   - What we know: Must be a matplotlib-drawn schematic, no fabricated data.
   - What's unclear: Whether to use basic `ax.text`/`ax.arrow` or `matplotlib.patches.FancyArrowPatch`.
   - Recommendation: Planner's discretion; `FancyArrowPatch` gives cleaner arrows.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.10.x | — |
| pickle (stdlib) | pkl loading | ✓ | stdlib | — |
| matplotlib | Figures | ✓ | (installed) | — |
| numpy | Array ops | ✓ | (installed) | — |
| sklearn | F1/PR recompute | ✓ | (installed) | — |
| jupytext | Notebook pairing | verify on run | (assumed from prior phases) | manual nbconvert |
| jupyter + nbconvert | Pre-execute nb | verify on run | (assumed from prior phases) | deliver .py only |
| Three results.pkl | Everything | ✓ | committed in repo | — |
| Local PDFs | Comparator sourcing | ✓ | read this session | — |

All hard dependencies are satisfied. Jupytext/nbconvert need to be present; if missing, the planner should add an install step.

---

## Validation Architecture

`nyquist_validation: true` in `.planning/config.json` — section required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | none (invoked as `python -m pytest backend/training/aqa/ -m "not slow" -q`) |
| Quick run command | `python -m pytest backend/training/aqa/ -m "not slow" -q` |
| Full suite command | `python -m pytest backend/training/aqa/ -m "not slow" -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Master comparison table correct (all 5 F1 values from pickles, not literals) | In-notebook assertion | Assertions execute on `nbconvert --execute` | ❌ Wave 0: in-notebook validation cell |
| EVAL-01 | All figure files written | In-notebook assertion | `assert Path("docs/figures/all_errors_vs_paper.png").exists()` etc. | ❌ Wave 0: in-notebook validation cell |
| EVAL-02 | Not-applicable finding documented | manual-only | N/A — narrative in FINDINGS_FULL | — |
| EVAL-03 | Notebook pre-executes cleanly | Integration | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` | ❌ Wave 0: notebook to be created |
| Suite green | AQA test suite still passes | Unit | `python -m pytest backend/training/aqa/ -m "not slow" -q` → 45 passed / 4 deselected | ✅ Exists |

### Sampling Rate

- **Per task commit:** `python -m pytest backend/training/aqa/ -m "not slow" -q` (6 s; verifies Phase 8 added no regression)
- **Per wave merge:** same + `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/12_full_evaluation.ipynb` (confirms pre-execution still clean)
- **Phase gate:** Full suite green + notebook executes cleanly + all 4 figure files exist

### Wave 0 Gaps

- [ ] `docs/notebooks/12_full_evaluation.py` — the jupytext source to create (covers EVAL-01 + EVAL-03)
- [ ] In-notebook validation cell: recompute F1 from raw arrays + assert ≤1e-6 deviation from stored test_f1 for all 5 errors
- [ ] In-notebook validation cell: assert all 4 figure PNGs exist after the figure-building cells

*(The AQA test suite itself already exists and covers all production code. Phase 8 adds no production code, so no new unit-test files are needed.)*

---

## Security Domain

`security_enforcement: true` in config. However, Phase 8 adds no authentication, session management, input validation of user-supplied data, or cryptographic operations. It reads committed local files and writes docs/figures. The ASVS categories V2/V3/V4/V6 are not applicable. V5 (input validation) applies trivially: the only "inputs" are the three committed pkl files and the local PDFs — no user-controlled input.

**Applicable ASVS categories for Phase 8:**

| ASVS Category | Applies | Note |
|---------------|---------|------|
| V2 Authentication | No | No auth in scope |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | Trivially | Pickle loads from committed repo files — no user input |
| V6 Cryptography | No | No crypto |

No security controls required for Phase 8.

---

## Sources

### Primary (HIGH confidence)
- Local PDF: `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — Parmar et al. ECCV 2022 (arXiv:2202.14019); Tables 2, 3, 4 read directly for all paper baseline/SSL numbers and the EVAL-02 ensemble question
- Local PDF: `Fitness-AQA/Fine-Tuning Large Multimodal Models forFitness Action Quality Assessment.pdf` — Dibenedetto et al. UMAP 2025; Section 2.1 (split protocol) + Table 4 (LMM F1 numbers + GYMetricPose F1 numbers)
- Three local `results.pkl` files — loaded and schemas verified this session

### Secondary (MEDIUM confidence)
- IEEE Xplore listing for GYMetricPose (Gallardo et al. 2024), DOI 10.1109/CBMS61543.2024.00016 — confirmed paper exists at CBMS 2024; numbers sourced via LMM Table 4 (not directly from the GYMetricPose paper)

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all existing
- Architecture: HIGH — all input files verified, key paths confirmed, code signatures read from source
- EVAL-02 finding: HIGH — read directly from Parmar Table 2
- LMM numbers: HIGH — read directly from local PDF Table 4
- GYMetricPose numbers: MEDIUM — sourced via LMM Table 4 (secondary citation); split protocol unconfirmed

**Research date:** 2026-05-30
**Valid until:** 2026-07-01 (stable; results.pkl are committed, paper PDFs are local)

---

## RESEARCH COMPLETE

**Phase:** 08 — Ensemble, Evaluation & Visualization Pack
**Confidence:** HIGH

### Key Findings

1. **EVAL-02 settled (HIGH confidence):** The Parmar paper does apply MD+CVCSPC ensemble but only to BackSquat KIE+KFE (Table 2, row "Ours MD + CVCSPC": KIE 0.5263, KFE 0.8468). It does NOT apply it to Shallow-Squat or OHP. Because our 5-error scope uses exactly one method per error, EVAL-02 is correctly reported as "evaluated → not-applicable to our single-method-per-error scope." No invented ensemble.

2. **CONTEXT.md D2 spine verified (HIGH confidence):** All 9 paper numbers in the EVAL-01 spine cross-check against Parmar Tables 2/3/4 — every value matches to the rounding used. No discrepancy.

3. **LMM (Dibenedetto 2025) numbers extracted (HIGH confidence):** Two-Step variant F1: Squat KIE 0.1955 / KFE 0.6266 / OHP Elbows 0.4575 / Knees 0.7611. **Comparability verdict: NOT directly comparable** — LMM paper trains on train+val (not official protocol). Shallow-Squat not evaluated. Present as context-only with explicit caveat.

4. **GYMetricPose (Gallardo 2024) numbers sourced (MEDIUM confidence):** F1 from LMM Table 4: Squat KIE 0.4398 / KFE 0.8219 / OHP Elbows 0.4175 / Knees 0.8160. **Comparability: uncertain** (split protocol not confirmable without paywalled paper). Present as context-only with explicit caveat. Shallow-Squat not covered.

5. **Pickle schemas confirmed:** All three files load cleanly. Key shapes: Squat `raw/ens_test_scores (244,2)`, OHP `ensemble_test_scores (339,2)`, Shallow `ensemble_test_scores (540,)`. The validation assertion pattern (recompute from raw arrays → assert ≤1e-6 vs stored `test_f1`) is directly implementable.

6. **No new packages needed.** Existing matplotlib/numpy/sklearn/metrics.py covers everything.

### File Created
`.planning/phases/08-ensemble-evaluation-visualization-pack/08-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| EVAL-02 finding | HIGH | Read from Parmar Table 2 in local PDF |
| LMM numbers | HIGH | Read from local PDF Table 4 |
| GYMetricPose numbers | MEDIUM | Secondary citation via LMM Table 4; direct paper paywalled |
| Pickle schemas | HIGH | Verified by direct Python load this session |
| Code reuse patterns | HIGH | Read actual script source |

### Open Questions

- GYMetricPose official split confirmation — not resolvable without IEEE access. Handled by the comparability caveat in FINDINGS_FULL.
- Methodology diagram implementation detail (arrow style) — planner's discretion.

### Ready for Planning
Research complete. Planner can now create PLAN.md.
