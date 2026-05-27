# Phase 06: Overhead Press — Context

**Gathered:** 2026-05-27
**Status:** Ready for planning
**Source:** Interactive `/gsd:discuss-phase 6`. Phase 6 is a faithful re-run of the proven Squat pipeline (Phases 2–4) on OHP data, so most gray areas are settled by prior-phase precedent and carried forward unchanged (per [[feedback_run_gsd_autonomously]] — decide settled gates from precedent, don't re-ask). Two genuinely OHP-specific decisions were put to the user (API scope; detector structure) and resolved below. The OHP dataset facts in `<specifics>` were verified against the local extracted archives during this discussion (counts, label format, class balance, co-occurrence, trajectory layout) — not assumed from the master plan.

<domain>
## Phase Boundary

**This phase delivers** (satisfies **OHP-01**):

1. A **supervised baseline** detector for OHP **Elbows + Knees** — R(2+1)D-18 (Kinetics-V1 init), joint multi-label 2-output head, class-weighted loss — trained on the official OHP labeled split. The SSL-lift control + a notebook deliverable. (No published OHP Kinetics baseline exists in the paper — see D3/`<specifics>`.)
2. **Motion-Disentangling (MD) SSL pretraining** of an R(2+1)D-18 backbone on the **5,490 unlabeled OHP clips** + shipped barbell trajectories (the paper's half-cycle contrastive pretext), then **fine-tune** on the labeled set — mirroring Phase 4.
3. **3-seed ensemble** (42/1337/7), mean-of-sigmoids, per-head decision thresholds tuned on val.
4. **F1 per error on the official OHP test split** (339 clips), compared to the paper's published MD numbers (**Knees 0.845 / Elbow 0.455**) and to our own supervised baseline (the measured SSL lift).
5. **Visualization deliverable — the OHP analog of the Squat pack** (`[[project_supervisor_visualizations]]`): a `results.pkl`-style artifact (per-epoch SSL + fine-tune curves, official per-clip ensemble scores) + the **4-notebook OHP pack** (EDA / data-pipeline / training / evaluation) + figures + a `FINDINGS.md`, replicating `docs/notebooks/` + `docs/eval/FINDINGS.md` for OHP.
6. End-of-phase SUMMARY declaring the production checkpoint(s), per-error thresholds, and the headline comparison.

**This phase does NOT deliver:**

- **Backend inference API integration for OHP** — **DESCOPED this phase** (D1, user decision). The form-correction frontend is cancelled; the backend is retained only as a Squat fallback; wiring OHP into the (UI-less) serving path adds no demo value. ROADMAP Phase-6 success-criterion 3 ("OHP selectable through the inference API") is reinterpreted: OHP-01 is satisfied by trained/evaluated models + the paper comparison + the visualization pack. (If a future UI milestone revives serving, OHP integration goes there — see `<deferred>`.)
- Image-based errors / Shallow-Squat / BarbellRow (Phase 7).
- The cross-method ensemble + full 3-exercise comparison pack (Phase 8).
- TTA as a shipped default (evaluated for completeness per Phase-4 precedent; Phase 4 found it didn't help — expected same).
- Graded severity (out of scope — labels binary).

**Requirements satisfied:** **OHP-01** (OHP Elbow/Knees baseline + MD-SSL trained; F1 per error on the official OHP splits; compared to published numbers). The "OHP selectable through the inference API" ROADMAP criterion is descoped by D1.

</domain>

<decisions>
## Implementation Decisions

### D1 — Scope: model + benchmark eval + visualization pack; NO backend API integration [LOCKED via discussion]
- **User decision (2026-05-27):** "Model + eval + notebooks only." The form-correction tab was removed from the app shell last session; the backend was kept only as a Squat fallback. OHP is **not** wired into the inference API this phase.
- ROADMAP Phase-6 **SC3 ("OHP selectable through the inference API") is descoped** — reason: the consuming frontend is cancelled, so serving code/tests no UI exercises add no value and contradict the user's stated deliverable ("paper-dependent — all the details, visualizations, results, comparison with the main paper").
- **Deliverable = the Squat deliverable pattern, for OHP:** OHP checkpoints + `results.pkl` + F1-vs-paper comparison + the 4-notebook pack + figures + `FINDINGS.md`, mirroring `docs/notebooks/` (`01_squat_eda` … `04_squat_evaluation` + `README.md`), `docs/eval/FINDINGS.md`, `docs/figures/`.
- **Action at plan/close:** reflect the SC3 descope in ROADMAP/REQUIREMENTS via `gsd-sdk` (not a direct edit) — note OHP-01 met by model+eval+viz, API criterion descoped with this reason.

### D2 — Detector structure: shared backbone, joint 2-output multi-label head [LOCKED via discussion]
- **User decision:** shared MD-SSL backbone → `Dropout(0.2)+Linear(512,2)`, two sigmoid outputs (Elbows, Knees). `BCEWithLogitsLoss(pos_weight)` treats the two labels independently. Mirror Squat exactly (`build_finetune_model` shape); one SSL pretrain + one 3-seed fine-tune yields both errors.
- **Data caveat (verified this discussion — differs from Squat):** OHP Elbows (upper-body) and Knees (lower-body) are **largely independent** — train co-occurrence: both+ 85, Elbow-only 322, Knees-only 456 (only 85/407 Elbow+ are also Knees+). Squat's KIE/KFE co-occurred 83% (192/232), which was the *original* justification for the joint head. For OHP the joint head is justified instead by (a) the shared MD-SSL representation being the contribution, (b) multi-label BCE handling independent labels correctly, (c) efficiency (~½ the compute of two models). **Do NOT expect a co-occurrence-driven joint lift; set defense expectations accordingly.**

### D3 — Method & recipe: faithful reuse of the P3 baseline + P4 MD-SSL pipeline [LOCKED from precedent]
- **Preprocessing (P2/P3 D3/D5, reuse verbatim):** 32-frame uniform sampling (±2-frame jitter on train, deterministic on val/test), short-side resize 128 → 112² (random crop train / center crop val/test), Kinetics-400 normalization, **horizontal flip OFF**. `crop_size`/`num_frames` parameterized.
- **Loss:** `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)`, train-derived (Elbows ≈ 2.89, Knees ≈ 1.92 — verified, see `<specifics>`).
- **Supervised baseline:** R(2+1)D-18 Kinetics-V1 init, full fine-tune, Adam lr 1e-4 (or the P3 recipe the researcher confirms), 50ep/8-patience cosine, batch ~16 — the Phase-3 recipe.
- **Fine-tune (from MD backbone):** AdamW wd 1e-4, dropout 0.2 head-only, lr 1e-4, 50ep/8-patience cosine — the Phase-4 D3 recipe + the D6 val/train overfit monitor.
- **MD-SSL pretrain (Phase-4 D2 + Plan-02 findings):** half-cycle contrast on barbell-trajectory-split reps; **strong augs (the validated Phase-4 v2 set: translation/zoom/blur/channel-swap; rotation default-OFF)**; AdamW; NT-Xent/triplet loss; projection head discarded at fine-tune; **linear-probe convergence monitor + effective-rank collapse guard**; backbone = the linear-probe-best epoch (Phase-4 v2 converged ~ep5). Do NOT re-run the weak-vs-strong-aug ablation — Phase 4 already established strong augs are required; carry the v2 recipe as default.
- **Ensemble (P4 D4):** 3 seeds (42/1337/7), one shared SSL pretrain → 3 fine-tunes, **mean-of-sigmoids**, single per-head threshold tuned on the ensemble val scores. Per-seed F1 reported as variance evidence.
- **Threshold sweep (P3 D7):** per-error independent sweep on val, applied to test. PR-AUC threshold-free.
- **TTA (P4 D5):** evaluated via val-tuning for completeness; Phase 4 found it reversed on test and was not adopted — report both numbers, expect no-TTA headline.
- **Eval (P3 D9):** `eval/metrics.py` F1-per-error + PR-AUC + confusion + threshold-sweep, sklearn-backed, reused UNCHANGED. Official splits verbatim.

### D4 — OHP data contract [VERIFIED this discussion — LOCKED]
- **Labeled:** 2,260 clips — train 1,582 / val 339 / test 339 (reconciles to the master plan exactly). Split files: `OHP/Labeled_Dataset/Splits/{train,val,test}_keys.json`, JSON lists of `{video}_{rep}` ids (same id format as Squat).
- **Labels:** `OHP/Labeled_Dataset/Labels/error_elbows.json` and `error_knees.json` (note the `Labels/` subdir — same convention as Squat, which is also under `Labeled_Dataset/Labels/`). Each is a dict keyed by clip_id → list of `[start,end]` intervals; **non-empty list = positive**. Identical semantics to Squat's interval labels.
- **Class balance (train, 1,582):** Elbows+ 407 (25.7%) → pos_weight ≈ **2.887**; Knees+ 541 (34.2%) → pos_weight ≈ **1.924**. Two moderate-minority classes — different profile from Squat (rare KIE 14% w≈6.10 + majority KFE 68% w≈0.45). Both pos_weights >1.
- **Labeled videos.zip:** 2,367 mp4s (107 extra beyond the 2,260 official — dropped by using the split files as the id source of truth; mirrors Squat's 1,739-vs-1,623).
- **Unlabeled:** 5,490 clips (`OHP/Unlabeled_Dataset/videos.zip`) + 5,490 trajectory JSONs (`OHP/Unlabeled_Dataset/bar_trajectories_raw/{clip_id}.json`), 1:1 by clip_id.
- **Archive layout / consolidation:** OHP is split across release folders — `…release-…-3-001/…/OHP/` holds Labeled_Dataset + the trajectory zip; `…release-…-3-002/…/OHP/Unlabeled_Dataset/videos.zip` holds the unlabeled videos. Like Phase 1's Squat consolidation, staging to Drive + local extraction must gather both. (The training reads from Drive/Colab, not the local archive — local is for offline inspection only, per P2 precedent.)

### D5 — Trajectory format: gated probe before SSL [LOCKED approach; mechanics RESEARCHER/EXECUTE-RESOLVE]
- OHP trajectories are `bar_trajectories_raw/{clip_id}.json` — **"raw"** (the name suggests possibly un-interpolated, unlike Squat's interpolated lists). The exact content (flat y-list vs `[x,y]` pairs vs frame-indexed; NaN handling; frame↔trajectory alignment) **and the half-cycle bottom sign (argmin vs argmax)** MUST be confirmed by a gated `checkpoint:human-verify` probe on 2–5 real OHP clips **before** finalizing `ohp_ssl._load_trajectory` and **before** the SSL GPU burn — exactly as Phase 4 Wave-1 did for Squat (which found argmin + `rglob` nesting). **Do NOT assume it equals Squat.** `OHP/Unlabeled_Dataset/ReadMe.md.docx` may document the format — researcher reads it first.

### D6 — Code organization: new `ohp.py`/`ohp_ssl.py` from squat templates; reuse the agnostic modules [LOCKED]
- **New:** `datasets/ohp.py` (mirror `squat.py` → `OHPElbowsKneesDataset` + `build_loaders` + train-derived `pos_weight`; label fields `label_elbows`/`label_knees`), `datasets/ohp_ssl.py` (mirror `squat_ssl.py`).
- **Reuse UNCHANGED (verified exercise-agnostic):** `datasets/transforms.py` (Kinetics norm, `uniform_sample_indices`, `decode_clip`, `spatial_train/val`); `datasets/ssl_augs.py` (the strong-aug set); `eval/metrics.py`, `eval/ensemble.py`, `eval/tta.py` (operate on score arrays); the training *logic* in `harness/supervised_train.py`, `harness/md_pretrain.py`, `harness/md_finetune.py` (epoch loop, checkpoint schema, early-stop, D6 monitor, linear-probe, collapse metric).
- **Needs OHP variant/parameterization (NOT verbatim reuse — corrects the handoff's "splits.py reuse as-is"):**
  - `datasets/splits.py` is **Squat-hardcoded**: `_EXPECTED_COUNTS={train:1136,val:243,test:244}`, `_LABELS_SUBDIR`/`_SPLITS_SUBDIR` point at `…/Squat/…`, filenames `error_knees_inward/forward.json`, and `ClipRecord` has `label_kie`/`label_kfe`. OHP needs either a parameterized `index(exercise=…, label_files=…)` + a generic 2-label `ClipRecord`, or an OHP-specific split loader. The interval→binary `_label_value` logic is generic and transfers. **Planner decides factoring (parameterize vs OHP variant).**
  - `harness/colab.py` staging (`stage_squat_videos`, `stage_unlabeled_squat_videos`) is Squat-pathed → needs OHP staging analogs (labeled + unlabeled, with the trajectory zip).
  - The trainers' dataloader construction (`supervised_train._build_dataloaders`, and `md_finetune` which reuses it per P4 D9) likely constructs `SquatKIEKFEDataset` directly → must be parameterized to accept the OHP dataset/exercise or wrapped. **Planner/researcher resolves.**
- **Carry-forward landmines (P4 D7 — must NOT re-surface):** `persistent_workers=True` on every `num_workers>0` loader; `map_location='cpu'` for `load_latest_checkpoint`; load `metrics_history` from the `latest.txt`-pointed checkpoint, not `best.pt`; deliver notebooks as paired `.py` + `.ipynb` with Cell A bootstrap; **one runnable unit at a time, never pre-write task N+1**; restart runtime / `importlib.reload` after `git pull`; `update_latest=False` when writing `backbone.pt`.

### D7 — Working agreement [LOCKED from memories]
- **Interactive, one runnable unit at a time, paste-back** ([[feedback_interactive_execution]], [[feedback_one_cell_at_a_time_strict]]); no blind runs.
- **Heavy training → fresh L4 Colab notebook; prompt the user FIRST** before the first training cell ([[feedback_heavy_training_new_notebook]]). SSL pretrain on 5,490 clips ≈ Squat's 4,970 (12–24h); + 3-seed fine-tune ≈ 13–16h. Disconnect-safe by construction ([[feedback_notebook_disconnect_safe]]).
- **Deliver Colab notebooks as `.ipynb`** ([[feedback_deliver_colab_as_ipynb]]); new cells handed as chat code blocks to paste ([[reference_colab_notebook_cell_delivery]]).
- **Visualizations first-class** ([[project_supervisor_visualizations]]); **no fabricated metrics/curves** — every number/plot from real data or a real run ([[feedback_ai_correctness]]).
- **GSD discipline:** atomic commit per task (`feat(06)/fix(06)/test(06)/docs(06)` with task # + D-ID inline); SUMMARY per plan; STATE/ROADMAP/REQUIREMENTS kept current via `gsd-sdk`.
- **Git:** push `fresh-start` to origin without asking ([[feedback_dont_ask_to_push]]); never force-push; reconcile divergences by merge; stage files by name only (Fitness-AQA/ ~5.8 GB, videos_extracted/, weights are untracked/gitignored).
- **Tone:** direct, evidence-based, terse; no deadline-pressure framing ([[feedback_working_style]], [[feedback_dont_agree]]).

### Claude's Discretion
- Factoring of the shared split/staging/dataloader code (parameterize vs OHP variant) — planner/researcher.
- SSL/fine-tune batch sizing re-estimate on L4 (Phase 3 hit 15.2 GB backward at batch 16; SSL contrastive wants larger batches — re-measure before committing).
- Whether `ohp.py`'s `build_loaders` is even on the training path (Phase 3+ trainers build DataLoaders directly with `num_workers>0`; `build_loaders` clamps to 0) — likely a convenience/notebook-smoke factory only.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, pattern-mapper, planner, plan-checker) MUST read these before producing their artefacts.**

### Master plan & milestone authority
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master rebuild plan; **Phase 6 paragraph (line 104–105)**; the dataset table (OHP row, line 33); **comparison targets (lines 60–61: OHP Knees MD 0.845 / Elbow 0.455)**; method (baseline → MD-SSL); working agreement.
- `.planning/PROJECT.md` — milestone framing; OHP in Active requirements; constraints (PyTorch, official splits, F1 per error, Colab).
- `.planning/ROADMAP.md` — Phase 6 goal + success criteria (**SC3 descoped per D1**); `Depends on: Phase 5`.
- `.planning/REQUIREMENTS.md` — **OHP-01**.
- `~/.claude/plans/handoff-2026-05-27-phase6-ohp-start.md` — the continuation handoff (orienting; verify claims — this CONTEXT corrects its "splits.py reuse as-is" line in D6).

### Phase 4 (the DIRECT template — MD-SSL → fine-tune → ensemble → eval)
- `.planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md` — D1–D9 (scope, MD-SSL recipe, regularization, 3-seed ensemble, TTA, overfit safeguards, carry-forward landmines, reuse contracts).
- `.planning/phases/04-squat-motion-disentangling-ssl/04-01-SUMMARY.md` — Wave-0 module scaffolds + pure-function verification (half-cycle splitter, triplet loss, projector, ensemble/TTA) — the OHP analog re-uses these.
- `.planning/phases/04-squat-motion-disentangling-ssl/04-02-SUMMARY.md` — **the SSL pretrain reality**: weak-aug collapse, strong-aug v2 fix, linear-probe-best backbone, `bottom_is_argmax=False` (Squat), trajectory probe findings.
- `.planning/phases/04-squat-motion-disentangling-ssl/04-03-SUMMARY.md` — 3-seed fine-tune; D6 monitor (val/train ratio) fix; per-seed variance.
- `.planning/phases/04-squat-motion-disentangling-ssl/04-04-SUMMARY.md` — ensemble + TTA + test eval; the headline comparison-chart pattern; production-checkpoint declaration.

### Phase 3 (the baseline recipe)
- `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md` — D1–D14 (model construction, loss, data contract, num_workers, augmentation, optimizer/scheduler, threshold protocol, checkpoint schema, eval harness).
- `.planning/phases/03-squat-supervised-baseline/03-01-SUMMARY.md` — baseline result + overfit finding + carry-forward landmines.

### Phase 2 (the data pipeline + Colab harness)
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-CONTEXT.md` — pipeline decisions, spatial/temporal aug rationale, checkpoint/resume contract.
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md` — interfaces; the `Code_Release/motion_disentanglement/` empty-README finding (SSL reconstructed from paper); atomic-write contract.
- `.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md` — upstream official-code gap analysis.

### Phase 1 (dataset method)
- `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md` — the EDA/consolidation method to replicate for the OHP `01_ohp_eda` notebook.

### Code modules — TEMPLATES to mirror / REUSE (do NOT redefine the agnostic ones)
- `backend/training/aqa/datasets/squat.py` — **template for `datasets/ohp.py`** (dataset + build_loaders + pos_weight).
- `backend/training/aqa/datasets/squat_ssl.py` — **template for `datasets/ohp_ssl.py`** (SSL dataset + `_load_trajectory` + half-cycle pairing).
- `backend/training/aqa/datasets/splits.py` — Squat-hardcoded; **needs OHP generalization/variant** (D6).
- `backend/training/aqa/datasets/transforms.py`, `datasets/ssl_augs.py` — reuse UNCHANGED.
- `backend/training/aqa/harness/colab.py` — staging is Squat-pathed (needs OHP analogs); RNG/atomic-checkpoint/prune reused.
- `backend/training/aqa/harness/supervised_train.py`, `harness/md_pretrain.py`, `harness/md_finetune.py` — training logic reused; dataloader construction needs OHP parameterization (D6).
- `backend/training/aqa/eval/metrics.py`, `eval/ensemble.py`, `eval/tta.py` — reuse UNCHANGED.
- `backend/training/aqa/notebooks/03_squat_supervised_baseline.{py,ipynb}`, `04_squat_md_ssl.{py,ipynb}`, `05_squat_md_finetune.{py,ipynb}` — Colab training-notebook templates (Cell A, Step 0, disconnect-safe).

### Deliverable pattern to REPLICATE (the Squat visualization pack)
- `docs/notebooks/` — `01_squat_eda`, `02_squat_data_pipeline`, `03_squat_training`, `04_squat_evaluation` (+ `README.md`): the 4-notebook structure to mirror for OHP.
- `docs/eval/FINDINGS.md`, `docs/eval/squat_test_scores.csv`, `docs/figures/*.png` — the FINDINGS + per-clip CSV + figures pattern.
- `backend/scripts/eval_benchmark_viz.py`, `backend/scripts/eval_best_examples.py` — reproducible figure scripts.
- `.planning/phases/04-squat-motion-disentangling-ssl/figures/results.pkl` — the `results.pkl` schema (per-epoch curves + official per-clip scores) that made the offline notebooks possible.

### Fitness-AQA paper + dataset
- `Fitness-AQA Paper.pdf` (Parmar et al., ECCV 2022, [arXiv:2202.14019](https://arxiv.org/abs/2202.14019)) — **§4 MD-SSL pretext/loss; §5 downstream fine-tune; the OHP Elbow/Knees rows in Table 2** (Read tool can't render — extract via `fitz`/`pypdf`).
- Local OHP archives (offline inspection only): `Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/Fitness-AQA_dataset_release/OHP/` (Labeled_Dataset + Unlabeled_Dataset/bar_trajectories_raw.zip) and `…-3-002/…/OHP/Unlabeled_Dataset/videos.zip`. Drive mirror for Colab training.
- `OHP/Unlabeled_Dataset/ReadMe.md.docx` — may document the unlabeled set / trajectory format (researcher reads for D5).
- GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025) — independent comparison points if the paper reports OHP for them (Phase 8 mainly; mention if available).

### Working-agreement memories (load all)
[[feedback_interactive_execution]], [[feedback_one_cell_at_a_time_strict]], [[feedback_deliver_colab_as_ipynb]], [[feedback_notebook_disconnect_safe]], [[feedback_heavy_training_new_notebook]], [[feedback_run_gsd_autonomously]], [[feedback_ai_correctness]], [[feedback_dont_agree]], [[feedback_working_style]], [[feedback_dont_ask_to_push]], [[reference_pytorch_persistent_workers]], [[reference_colab_module_reload_after_git_pull]], [[reference_colab_notebook_cell_delivery]], [[reference_best_pt_metrics_history_is_stale]], [[reference_fitness_aqa]], [[project_form_correction_status]], [[project_supervisor_visualizations]], [[project_realtime_demo_expectation]], [[project_gsd_adoption]]

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (verified exercise-agnostic)
- `datasets/transforms.py` — `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip` (F11 windowed), `spatial_train`, `spatial_val`. No Squat coupling.
- `datasets/ssl_augs.py` — the strong-aug set (translation/zoom/blur/channel-swap + gated rotation) validated in Phase 4.
- `eval/metrics.py`, `eval/ensemble.py` (`aggregate_sigmoid_mean`), `eval/tta.py` (`select_tta_recipe`/`tta_forward`) — operate on score arrays, modality/exercise-agnostic.
- Training logic in `harness/supervised_train.py` (`run_supervised_epoch`, `build_model`, `seed_worker`), `harness/md_pretrain.py` (`run_md_pretrain_epoch`, `_linear_probe`, `_embedding_collapse_metrics`, `ProjectionHead`, `md_triplet_loss`, `build_md_model`), `harness/md_finetune.py` (`run_md_finetune_epoch`, `build_finetune_model`, `_d6_overfit_abort`) — epoch loop / checkpoint / early-stop / monitors reused.
- `harness/colab.py` — `mount_drive`, atomic checkpoint (tmp + round-trip verify + `os.replace` + `latest.txt`-last + `update_latest` kwarg), 4-RNG capture/restore, `prune_checkpoints`.

### Templates (mirror, don't reuse verbatim)
- `datasets/squat.py` → `datasets/ohp.py` (rename labels kie/kfe → elbows/knees; pos_weight from train; same `__getitem__` decode→sample→spatial path).
- `datasets/squat_ssl.py` → `datasets/ohp_ssl.py` (trajectory load + half-cycle pairing; sign + format from the D5 probe).

### Needs OHP parameterization/variant (the real net-new work besides training)
- `datasets/splits.py` — Squat-hardcoded subdir/filenames/`ClipRecord.label_kie/kfe`/expected counts.
- `harness/colab.py` staging functions (`stage_squat_videos`, `stage_unlabeled_squat_videos`) — Squat-pathed.
- `supervised_train._build_dataloaders` (and `md_finetune` reusing it) — likely constructs `SquatKIEKFEDataset` directly.

### Integration Points
- Net-new code lives under `backend/training/aqa/` (datasets/ohp*, harness staging additions, notebooks). **Nothing touches the FastAPI runtime** (D1 — API integration descoped).
- Deliverable notebooks/figures land under `docs/notebooks/`, `docs/eval/`, `docs/figures/` (OHP files alongside the Squat ones).

</code_context>

<specifics>
## Specific Targets, Numbers, and Patterns

- **Comparison targets (paper MD, master plan lines 60–61):** OHP **Knees 0.845**, **Elbow 0.455**. **No paper Kinetics (supervised) baseline exists for OHP** — so the headline is *our baseline → our MD-SSL → paper MD*, with the baseline as the measured SSL-lift control (unlike Squat, where the paper gave both rows).
- **Verified OHP counts:** labeled 2,260 (train 1,582 / val 339 / test 339); unlabeled 5,490 clips + 5,490 trajectory JSONs.
- **Verified train balance:** Elbows+ 407 (25.7%, pos_weight ≈ 2.887); Knees+ 541 (34.2%, pos_weight ≈ 1.924). (val: Elbows 24.5% / Knees 31.9%; test: Elbows 25.4% / Knees 37.8%.)
- **Verified co-occurrence (train):** both+ 85, Elbow-only 322, Knees-only 456 — errors largely independent (drives the D2 caveat).
- **Label format:** `Labels/error_{elbows,knees}.json` dicts keyed by `{video}_{rep}` → `[start,end]` interval lists; non-empty = positive.
- **Trajectory format:** `bar_trajectories_raw/{clip_id}.json`, 1:1 with unlabeled clips — "raw"; exact schema + half-cycle sign via the D5 gated probe.
- **Run-name convention (mirror P4):** `ohp_supervised_v1` (baseline), `ohp_md_pretrain_v1/v2` (SSL), `ohp_md_finetune_seed{42,1337,7}` (fine-tunes). Drive layout `My Drive/FitNova/checkpoints/phase06/{run_name}/`.
- **Wall-time anchor:** Squat fine-tune ~6.1 min/epoch on L4; SSL pretrain 12–24h; 3-seed fine-tune ~13–16h. Resumable harness mandatory.
- **Plan shape:** Phase 6 spans P3+P4 scope (baseline → SSL → fine-tune → ensemble → eval + the notebook pack); expect multiple plans / waves like Phase 4 (Wave-0 module scaffolds, gated trajectory probe, SSL pretrain, 3-seed fine-tune, ensemble+eval+figures, deliverable notebooks).

</specifics>

<deferred>
## Deferred Ideas

- **OHP backend inference API integration** — descoped this phase (D1). If a future UI milestone revives form-correction serving, OHP integration belongs there (parity with the kept Squat `SquatFormService` path).
- **Image-based errors (Shallow-Squat, BarbellRow Lumbar/Torso) via CVCSPC** — Phase 7.
- **Cross-method (MD+CVCSPC) ensemble + the full 3-exercise identical-metric comparison pack (vs Parmar / GYMetricPose / LMM)** — Phase 8.
- **TTA as a shipped default** — evaluated for completeness; Phase-4 precedent says it won't help.
- **5-seed ensemble / weak-aug ablation re-run** — not needed; Phase 4 settled both.

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-overhead-press*
*Context gathered: 2026-05-27 via /gsd:discuss-phase 6*
