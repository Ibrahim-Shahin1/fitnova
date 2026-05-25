# Phase 5: Backend Inference Integration (Squat) - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Source:** `/gsd:discuss-phase 5`. The user authorized autonomous GSD operation and corrected an over-eager question-dump — these decisions are resolved from the **master plan** (`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`, which specifies the Phase 5 design + API contract verbatim), the Phase 4 production-protocol handoff (`04-04-SUMMARY.md`), the inherited data/preprocessing contract (Phase 3 `03-CONTEXT.md` D3), the codebase (`app.py`, `form_analyzer.py`, `form_session.py` read directly), and the working-agreement memories. Engineering calls not fixed by the docs are labeled **[DECIDED — Claude's discretion; researcher confirms]**. No blocking questions remained for prior sessions.

<domain>
## Phase Boundary

**This phase delivers** (satisfies **API-01, API-02, API-03, API-04**):

1. **Removal/archival of the old TF + MediaPipe form subsystem** — `backend/services/form_analyzer.py`, `form_session.py`, `form_geometry.py`, `mediapipe_config.py`, the form parts of `exercise_sanity.py`, plus the legacy form training code / `backend/models/form_model*` weight dirs / `pose_landmarker_full.task`. Archived (not hard-deleted) for dissertation-appendix evidence. **The TensorFlow recommender + plan-generation feature is UNTOUCHED.** (API-01)
2. **A new PyTorch Squat form-inference service** loaded once into `app.state` at FastAPI startup, holding the Phase-4 production model (the 3-seed MD-SSL ensemble) and running the inference protocol below. (API-01)
3. **A dedicated rep-segmentation component** — the dataset provides no model for this. **No pose** (the MediaPipe pose pipeline is explicitly not reused — the Fitness-AQA method is raw-pixel; the paper's premise is that pose is unreliable in-the-wild). Upload mode segments server-side and ships first; live mode does lightweight on-the-fly segmentation. (API-02)
4. **Redefined `POST /analyze-form-video`** (upload): clip → segment reps → per rep classify KIE/KFE → binary + timing response. (API-03)
5. **Redefined `WS /ws/form-session`** (live): camera frames → segment reps on the fly → per-rep KIE/KFE response. (API-04)
6. **API-level verification** (pytest/httpx + manual curl/WS — NOT the Flutter UI) and an **early real-camera domain-shift test** on a user-provided phone clip.
7. **Visualization**: rendered result overlaid on a sample clip (per `[[project_supervisor_visualizations]]`).

**This phase does NOT deliver:**

- **Polished Flutter form UI** — explicitly a later milestone (REQUIREMENTS v2 UI-01/UI-02; master plan "Polished Flutter UI is a later milestone"). The milestone bar is **"Model + inference API, verified via API."** The current Flutter form screens consume the OLD schema and will break; that is acceptable this milestone.
- OHP errors (Phase 6); image-based errors / Shallow-Squat / BarbellRow (Phase 7); the cross-method ensemble + full comparison pack (Phase 8).
- Graded-severity **ground truth** (out of scope — labels are binary; `severity_word` is a confidence-derived UX estimate, not a measured result).
- General backend security hardening (rate limiting, upload-size cap, CORS allowlist — see `CONCERNS.md`); only fold in if trivial and adjacent.

</domain>

<decisions>
## Implementation Decisions

### Inference model & protocol  *(LOCKED — Phase 4 `04-04-SUMMARY.md`)*
- **D-01 — Production model = the 3-seed MD-SSL ensemble.** Load `md_finetune_seed{42,1337,7}/best.pt`, **mean-of-sigmoids** across seeds, per-head decision thresholds **KIE 0.614 / KFE 0.385**, **NO TTA** (TTA's val gain reversed on test in Phase 4). Test macro-F1 0.6304 (KIE 0.4198 / KFE 0.8410).
- **D-02 — Clip contract = 32-frame, 112², Kinetics-normalized RGB** → `build_finetune_model` R(2+1)D-18 (`fc = Dropout(0.2)+Linear(512,2)`) → 2 logits → sigmoid. **Preprocessing MUST reuse the exact Phase-2/3 contract** (`transforms.KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_val`). Train/serve preprocessing skew is non-negotiable to avoid — it is what killed the OLD model (the B6 MediaPipe IMAGE-vs-VIDEO mismatch collapsed it 89%→11%). Mirror `05_squat_md_finetune` notebook's load→forward→threshold path exactly.

### Rep segmentation (API-02)  *(LOCKED approach — master plan lines 100-101, 126-131, 140; exact algorithm RESEARCHER-RESOLVES)*
- **D-03 — Dedicated component, NO pose.** Upload (ships first, simpler): server-side segmentation of the clip into reps, then per-rep 32-frame classification. Live: lightweight on-the-fly detection (sliding-window classification over the rolling frame buffer; "sliding-window for timing" per the master plan). The dataset clips are single-rep ~3 s @ 30 fps — **a single-rep clip is the simplest correct case and matches training**; multi-rep uploads need segmentation.
- **D-04 — Exact algorithm is the one genuinely new piece [DECIDED — researcher resolves + MEASURE].** Candidate no-pose signals: motion-energy / frame-difference, optical-flow magnitude, or a fixed sliding-window cadence. Researcher picks the pragmatic option grounded in the ~3 s single-rep clip shape; validate on a real clip before committing ([[feedback_ai_correctness]] — measure, don't guess).

### API output schema  *(LOCKED — master plan lines 126-131)*
- **D-05 — Per-rep response:** `{ exercise, errors: [ { type, detected: bool, confidence, severity_word, intervals: [[start,end]] } ] }`. Binary + timing. **No percentages shown to the user.** `severity_word` = confidence-derived word, presented as an estimate, not a measured result. `type` ∈ {KIE, KFE} for Squat. The old `FormFrameResult` / `FormSessionSummary` Pydantic models are **replaced, not shimmed** (UI is deferred, so back-compat with the current Flutter screens is not required).

### Old-subsystem removal & coexistence  *(LOCKED — master plan lines 113-124, 141)*
- **D-06 — Archive, don't hard-delete.** Move the listed form modules/weights to `backend/_archive_*/` (consistent with the existing archive pattern in `CONCERNS.md`). Reuse: FastAPI app structure, the `app.state` service-loading pattern, the REST/WS endpoint scaffolding, project conventions. Do NOT touch the recommender/plan-generation path.
- **D-07 — PyTorch + TF coexist in one FastAPI process** via separate `app.state` objects, both loaded at startup ("no conflict" per the master plan). **Verify import order + memory early** (the one integration risk to smoke-test). Avoid repeating the old `FormAnalyzer` concurrency bug — the model is stateless for inference; only frame-buffering is per-session, so keep per-connection buffers isolated and the shared model read-only.

### Ensemble vs single-seed (latency)  *(DECIDED — Claude's discretion; researcher confirms)*
- **D-08 — Default to the ensemble; MEASURE CPU latency first, fall back if needed.** The demo server is **CPU-only** (no Windows GPU for TF or torch — `CONCERNS.md`). 3× R(2+1)D-18 forward may be too slow for live. **First integration step measures per-clip latency** (ensemble 3× vs single-seed 1×) on the target machine (Phase-3 timing-probe precedent). If live latency is unacceptable, ship single-seed (~0.612, −0.018 macro) for live; upload can still afford the ensemble. **Make seed-count a config knob.**

### Weights transfer (Drive → server)  *(DECIDED — Claude's discretion; researcher confirms size/format)*
- **D-09 — Stage the `.pt` to `backend/models/`, not committed raw.** The 3 `best.pt` + backbone live on Drive (`MyDrive/FitNova/checkpoints/phase04/`), not the repo. R(2+1)D-18 ≈ ~120 MB fp32 each → ensemble ~360 MB — too large for a plain git commit. Stage to e.g. `backend/models/form_model_squat_md/` via a documented step; researcher confirms whether state-dict-only / fp16 shrinks it enough for Git-LFS, else a scripted/manual Drive download. **Action item: may need a user Drive export when we reach this step.**

### Plain-language feedback  *(DECIDED — Claude's discretion)*
- **D-10 — Deterministic templated feedback.** Per-rep text derived from `detected` + `severity_word` (e.g., "Knees caving inward — detected (moderate)"), matching the master plan's example ("Knees inward — detected") and "no percentages." Deterministic = reproducible, zero API cost, defensible ([[feedback_ai_correctness]]). The existing GPT-4o-mini coaching layer (`form_session._generate_llm_feedback`) is **optional session-level polish** — reuse only if cheap; not required for the API bar.

### Domain-shift early validation  *(LOCKED — `[[project_realtime_demo_expectation]]` + handoff)*
- **D-11 — Test the user's real phone-camera squat clip EARLY**, on a minimal end-to-end path (load ensemble → preprocess → forward → schema), **before** polishing the full API. Offline test macro 0.63 on curated Fitness-AQA clips ≠ live phone camera (domain shift: lighting, angle, framing, single-rep-vs-clip). Surface problems while there's time to mitigate (threshold/preprocessing/calibration). **Action item: user provides a real squat clip at this step.**

### Working agreement & venue  *(carry-forward)*
- **D-12 — Phase 5 is LOCAL backend work, NOT Colab.** This is the first non-training phase: the serving code runs on the local FastAPI server (Windows, CPU). **The Colab / L4 / Drive-checkpoint / disconnect-safe / cell-by-cell-paste machinery does NOT apply to the serving code** ([[feedback_heavy_training_new_notebook]] etc. are training-phase rules). Still in force: interactive one-unit-at-a-time (run the server locally, hit endpoints, paste outputs), **tests alongside implementation**, atomic commits `feat(05)/fix(05)/test(05)`, GSD discipline (SUMMARY per plan, STATE/ROADMAP/REQUIREMENTS updated), visualization deliverable, no deadline-pressure framing.

### Claude's Discretion
- **D-04** (rep-segmentation algorithm), **D-08** (ensemble-vs-single default + latency probe), **D-09** (weights transfer mechanism), **D-10** (deterministic feedback). All labeled above; the researcher confirms/overrides with evidence.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner) MUST read these before producing their artefacts.**

### Master plan & milestone authority
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — **the Phase 5 paragraph (lines 100-102), the redefined API contract (lines 126-131), the codebase delta / removal list (lines 113-124), and the risk table (lines 133-143) are the ground truth for this phase.**
- `.planning/PROJECT.md` — milestone framing; "Out of Scope" (Flutter UI deferred).
- `.planning/ROADMAP.md` — Phase 5 success criteria (old subsystem removed; upload → binary+timing; live → per-rep feedback); `Depends on: Phase 4`.
- `.planning/REQUIREMENTS.md` — **API-01 / API-02 (rep-segmentation) / API-03 (upload) / API-04 (live)**; v2 UI-01/UI-02 (deferred); out-of-scope MediaPipe pose.

### Phase 4 (the model being served — production protocol)
- `.planning/phases/04-squat-motion-disentangling-ssl/04-04-SUMMARY.md` — **the production-checkpoint declaration + inference protocol + thresholds + single-seed fallback (the Phase 5 input contract).**
- `.planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md` — D7 carry-forward landmines, D9 reuse contracts.

### Phase 3 (the data/preprocessing contract inference must match)
- `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md` — **D3 data contract: `num_frames=32`, `crop_size=112`, Kinetics-normalized, output `(3,32,112,112)`**; D1 model construction (`r2plus1d_18` + `Linear(512,2)`); D7 threshold protocol.

### Fitness-AQA paper (method fidelity — preprocessing parity)
- `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` (Parmar et al., ECCV 2022, arXiv:2202.14019) — the clip-sampling / normalization the served inference must reproduce; single-rep ~3 s clip definition. (Read tool can't render; extract text via `fitz`/`pypdf`.)

### Code modules — to REMOVE/ARCHIVE (do NOT extend)
- `backend/services/form_analyzer.py` — TF/MediaPipe per-frame analyzer (replaced).
- `backend/services/form_session.py` — per-connection session state machine; **its session-summary + LLM-coach structure is a pattern reference** for the new feedback layer even though the class is replaced.
- `backend/services/form_geometry.py`, `backend/services/mediapipe_config.py`, form parts of `backend/services/exercise_sanity.py`.

### Code modules — REUSE (contracts; do NOT redefine)
- `backend/app.py` — FastAPI entry; `lifespan` loads `app.state.*`; the `WS /ws/form-session` + `POST /analyze-form-video` handlers + `/health` (the redefinition targets); `app.state.form_analyzer` is the swap point.
- `backend/training/aqa/harness/md_finetune.py` — **`build_finetune_model`** (the architecture to load the `.pt` weights into).
- `backend/training/aqa/eval/ensemble.py` — **`aggregate_sigmoid_mean`** (the ensemble aggregation).
- `backend/training/aqa/eval/metrics.py` — thresholding/F1 helpers (for any API-level eval).
- `backend/training/aqa/datasets/transforms.py` — **`KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_val`** (the EXACT preprocessing inference must reuse — D-02).
- `backend/training/aqa/datasets/squat.py` — label schema (KIE/KFE) reference.
- `backend/training/aqa/notebooks/05_squat_md_finetune.{py,ipynb}` — **how the model was produced**; the reference for the exact load → forward → ensemble → threshold path to replicate at serve time.

### Codebase maps
- `.planning/codebase/INTEGRATIONS.md` — REST/WS transport; Flutter↔backend contract; no DB; single-process FastAPI.
- `.planning/codebase/CONCERNS.md` — **CPU-only inference on Windows**; the old `FormAnalyzer` shared-instance concurrency bug (don't repeat); upload-size / rate-limit gaps; dead-code/stale-weights inventory.

### Working-agreement memories (load all)
`[[feedback_interactive_execution]]`, `[[feedback_working_style]]`, `[[feedback_dont_agree]]`, `[[feedback_run_gsd_autonomously]]`, `[[feedback_ai_correctness]]`, `[[project_form_correction_status]]`, `[[project_realtime_demo_expectation]]`, `[[project_supervisor_visualizations]]`, `[[project_gsd_adoption]]`, `[[feedback_dont_ask_to_push]]`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`app.state` service-loading pattern** (`app.py` `lifespan`): the new `SquatFormService` loads once at startup exactly like `form_analyzer` did — swap the construction, keep the pattern.
- **REST/WS endpoint scaffolding** (`app.py`): `POST /analyze-form-video` (UploadFile + OpenCV decode loop) and `WS /ws/form-session` (start_session → frame → end_session JSON protocol) are redefined in place — the transport plumbing, temp-file handling, and message loop are reusable shells.
- **Eval primitives**: `eval/ensemble.py` (`aggregate_sigmoid_mean`) and `eval/metrics.py` (thresholding) move from notebook-eval to serve-time inference + any API-level check.
- **`build_finetune_model`** (`harness/md_finetune.py`): the architecture loader for the `.pt` state-dicts.
- **`transforms.py`** preprocessing: the single source of train/serve parity (D-02).
- **`form_session` LLM-coach + session-summary shape**: structural reference for the new (deterministic) feedback layer.

### Established Patterns
- Services instantiated once in `lifespan`, attached to `app.state`, held for process lifetime (no DB, no cache layer).
- Two transports: WS (live, per-frame stream) + REST (upload). Pydantic request/response models declared inline in `app.py`.
- Graceful degradation: the old analyzer returned neutral output when `model_ready` was false — keep an equivalent "model not loaded" path.

### Integration Points
- **`app.state.form_analyzer` → `app.state.squat_form_service`** (or similar) is the single swap point; the two endpoint handlers + `/health` call into it.
- `/health` should report the new model identity (ensemble seeds loaded, thresholds) so callers can verify load without parsing logs.

### Constraints / Lessons
- **CPU-only** serving (no Windows GPU for TF or torch) — drives the ensemble-vs-single latency decision (D-08).
- **TF + torch in one process** — verify import order/memory early (D-07).
- **Preprocessing parity is non-negotiable** — the old model's B6 failure (train/serve skew) is the cautionary tale (D-02).
- **No shared mutable inference state** — the old `FormAnalyzer` concurrency bug came from per-session state on a shared instance; keep the model read-only and buffers per-connection (D-07).

</code_context>

<specifics>
## Specific Ideas

- **Production protocol numbers** (verbatim from Phase 4): ensemble mean-of-sigmoids over seeds {42, 1337, 7}; thresholds **KIE 0.614 / KFE 0.385**; **no TTA**; clip **32-frame, 112², Kinetics-normalized**; single-seed fallback ~0.612.
- **Weights on Drive**: `MyDrive/FitNova/checkpoints/phase04/md_finetune_seed{42,1337,7}/best.pt` (+ `md_pretrain_v2/backbone.pt`). Not in the repo.
- **Response schema** (master plan): `{ exercise, errors: [{ type, detected, confidence, severity_word, intervals: [[start,end]] }] }`; error `type` ∈ {KIE, KFE}; no percentages surfaced.
- **Dataset clip shape**: single repetition, ~3 s @ 30 fps — informs the rep-segmentation default and the "single-rep clip = simplest correct case."
- **Expected behavior to set in feedback/UX**: strong KFE (~97% recall), modest KIE (~47% recall / 38% precision) — surface KIE as "possible" rather than definitive.

</specifics>

<deferred>
## Deferred Ideas

- **OHP inference** (Phase 6), **image-based errors** (Phase 7), **cross-method ensemble + full identical-metric comparison pack** (Phase 8).
- **Polished Flutter form UI** (v2 UI-01/UI-02) — including any compatibility shim so the *current* Flutter form screen keeps rendering. If the live demo needs the existing app to show something, that's a UI-milestone concern, not Phase 5.
- **Graded-severity ground truth** (v2 FB-01).
- **Backend security hardening** — upload-size cap, rate limiting, CORS allowlist (`CONCERNS.md`). Out of Phase 5 scope unless trivial and directly adjacent to the new endpoints.
- **Broader dead-code / legacy-weight-dir cleanup** beyond the form subsystem (`CONCERNS.md` stale-code inventory).
- **GPU serving / ONNX / TorchServe** — only if CPU latency proves unworkable for the demo (revisit under D-08).

</deferred>

---

*Phase: 05-backend-inference-integration-squat*
*Context gathered: 2026-05-25 via /gsd:discuss-phase 5*
