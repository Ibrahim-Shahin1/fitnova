# Codebase Concerns

**Analysis Date:** 2026-05-18

---

## Security Issues

### CRITICAL: Real OpenAI API key committed to git history

- Risk: The file `backend/.env` (310 bytes, containing `OPENAI_API_KEY=sk-proj-21j4Yw…`) is **tracked by git** and appears in at least two commits (`258a61d`, `8ad795c`). Even if removed from the working tree today, the key remains permanently readable in git history.
- Files: `backend/.env`
- `.gitignore` correctly lists `.env` under "Environment and Secrets" but this entry was added **after** the file was committed — it only prevents future staging, not removal from history.
- Current mitigation: None — the key is live in git history.
- Recommendations:
  1. Rotate the OpenAI key immediately in the OpenAI dashboard.
  2. Remove the file from history with `git filter-repo --path backend/.env --invert-paths` (or BFG Repo Cleaner) and force-push all branches.
  3. Verify `backend/.env` is listed in `.gitignore` before the next commit.

### Wildcard CORS with `allow_credentials=True`

- Risk: `backend/app.py` lines 219-222 set `allow_origins=["*"]` together with `allow_credentials=True`. The CORS spec requires credentials to be rejected when origin is `"*"`; FastAPI/Starlette silently accepts this combination, effectively allowing any origin to send credentialed cross-site requests.
- Files: `backend/app.py` (lines 219-222)
- Current mitigation: None.
- Recommendations: Replace `"*"` with an explicit allowlist (e.g., `["http://localhost:8000", "http://10.0.2.2:8000"]`) or remove `allow_credentials=True` if cookies/auth headers are not in use.

### No upload size limit on video endpoint

- Risk: `/analyze-form-video` (`backend/app.py` lines 397-454) calls `await file.read()` with no size cap, loading the entire video into memory before writing it to disk. A malicious client (or accidental large upload) can exhaust server memory.
- Files: `backend/app.py` (lines 407-420)
- Current mitigation: None.
- Recommendations: Add a `max_upload_size` check early in the handler, or configure uvicorn/FastAPI's `--limit-max-requests` and a streaming body limit.

### No rate limiting on LLM-backed endpoints

- Risk: `/generate-plan`, `/chat`, and `/analyze-form-video` all invoke the OpenAI API. There is no per-IP or per-session rate limit. Each `/generate-plan` call costs ~$0.01-0.05 in tokens; repeated automated calls can exhaust the project budget.
- Files: `backend/app.py` (lines 261-329, 397-454), `backend/services/llm_adapter.py`, `backend/services/chat_service.py`
- Current mitigation: None.
- Recommendations: Add `slowapi` (FastAPI rate limiter) with a per-IP limit on the three expensive endpoints.

### Temp file left on disk if video decode raises before `finally`

- Risk: `backend/app.py` lines 415-450 create a temp file with `delete=False`. The `os.unlink` is in the `finally` block, so it should clean up, but if the `NamedTemporaryFile` context-manager itself throws before `tmp_path` is assigned, `tmp_path` is undefined and `os.unlink(tmp_path)` will raise a `NameError`, leaving the orphaned temp file.
- Files: `backend/app.py` (lines 407-454)
- Current mitigation: Partial — the `finally` runs in most cases.
- Recommendations: Assign `tmp_path = None` before the `with` block and guard `os.unlink` with `if tmp_path`.

---

## Tech Debt

### FormAnalyzer shared across concurrent WebSocket sessions without a lock

- Issue: The docstring for `FormAnalyzer` (`backend/services/form_analyzer.py` line 60-61) explicitly warns: *"process_frame() mutates _ts_ms and _prev_result, so do NOT share a single FormAnalyzer across concurrent sessions without a lock."* Despite this warning, `backend/app.py` creates one `FormAnalyzer` at startup (`app.state.form_analyzer`) and passes the same instance to every `FormSession` — both the WebSocket handler (line 363) and the REST video handler (line 409).
- Files: `backend/app.py` (lines 203, 363, 409), `backend/services/form_analyzer.py` (lines 56-61)
- Impact: Under concurrent real-time sessions, `_ts_ms` and `_prev_result` race, corrupting MediaPipe's VIDEO-mode timestamp counter and producing wrong pose estimates across sessions.
- Fix approach: Either (a) create a separate `FormAnalyzer` per `FormSession` (memory cost: ~50-100 MB each for TF model + MediaPipe), or (b) wrap `process_frame()` with an `asyncio.Lock`.

### ENABLE_QUALITY_CYCLE_DETECTOR hard-disabled with no path to re-enable

- Issue: `backend/services/form_session.py` line 63 sets `ENABLE_QUALITY_CYCLE_DETECTOR = False` with a comment saying "When QEVD-trained v6 lands with a wider quality range, this can be set back to True." v6 weights are now in the repo but the flag was never re-evaluated. The flag controls the primary rep-detection path; with it disabled, rep counting falls back entirely to the time-based heuristic (`TIME_BASED_REP_INTERVAL = 30` frames), regardless of actual model version.
- Files: `backend/services/form_session.py` (lines 43-63)
- Impact: Rep counting is coarser than intended for v6 users. The quality-cycle signal is wasted.
- Fix approach: Add a model-version check at `FormSession.__init__`: enable the quality-cycle detector only when `model_version == "v6"`.

### ExerciseMismatchDetector silently inactive under v6

- Issue: `FormSession.__init__` instantiates `ExerciseMismatchDetector` but the detector uses FitNova UI exercise IDs (e.g. `"squat"`) while v6 QEVD class names use pluralised lowercase (`"squats"`). The mapping `V6_FITNOVA_TO_QEVD` in `form_session.py` (lines 105-112) resolves inbound exercise IDs for embedding lookup but the mismatch detector receives the raw UI ID.
- Files: `backend/services/form_session.py` (lines 88-112, 202-208), `backend/services/exercise_sanity.py`
- Impact: No exercise mismatch warnings fire for any v6 session, so a user who selects "deadlift" and does squats gets no warning.
- Fix approach: Pass the QEVD-mapped name to `ExerciseMismatchDetector` when `model_version == "v6"`, or extend the reverse map.

### v6.1 training script imports old v1 label builder (not v2)

- Issue: `backend/training/train_form_model_v6_1.py` line 349 imports `from backend.training.preprocessing.qevd_label_builder import QEVDLabels` — the original lexicon-based v1 builder. The repository also contains `qevd_label_builder_v2.py` (paper-faithful multi-label builder) which is the stated architecture for v6.1. The v6.1 training script and its companion `qevd_dataset_v6_1.py` were written to consume `v2`-style labels.
- Files: `backend/training/train_form_model_v6_1.py` (line 349), `backend/training/preprocessing/qevd_label_builder.py`, `backend/training/preprocessing/qevd_label_builder_v2.py`
- Impact: Running v6.1 training silently uses v1 lexicon-derived quality scalars (which the v6.1 architecture drops entirely) rather than the multi-label variation targets.
- Fix approach: Change the import on line 349 to `from backend.training.preprocessing.qevd_label_builder_v2 import QEVDLabels` and audit downstream usage.

### requirements.txt uses loose minimum-version pins

- Issue: All dependencies in `backend/requirements.txt` use `>=` lower-bound pins (e.g. `tensorflow>=2.13.0`, `openai>=1.30.0`). There is no upper bound and no lockfile equivalent. A `pip install -r requirements.txt` on a fresh machine today may install significantly newer versions that break compatibility (e.g. TF 2.18 changed `model.predict()` signature; OpenAI SDK v2 has breaking changes).
- Files: `backend/requirements.txt`
- Impact: Non-reproducible installs; Colab runs may silently diverge from local runs.
- Fix approach: Pin to exact versions (or at minimum use compatible-release `~=`) and commit a `pip freeze` lockfile.

### HTTP calls in `api_service.dart` have no timeout for chat/plan endpoints

- Issue: `lib/services/api_service.dart` calls `http.post` for both `/chat` (line 14) and `/generate-plan` (line 35) with no `.timeout()`. Only the video upload (line 69) has a timeout (`Duration(minutes: 4)`). An unresponsive backend will hang the Flutter UI indefinitely.
- Files: `lib/services/api_service.dart` (lines 14, 35)
- Impact: Poor UX; app appears frozen when the OpenAI API is slow or down.
- Fix approach: Add `.timeout(const Duration(seconds: 30))` to both `http.post` calls, with a `TimeoutException` handler in callers.

### `backend/.env` listed in `.gitignore` but was committed before the ignore rule

- Issue: The `.gitignore` entry `".env"` exists (line 50) but the file `backend/.env` was added to the index before this ignore rule took effect. Git continues tracking it. The file contains a live API key (see Security section above).
- Files: `backend/.env`, `.gitignore` (line 50)
- Fix approach: `git rm --cached backend/.env`, commit the removal, then add the ignore rule and commit that separately.

---

## Known Bugs

### `reality_check_v6.py` defaults to `--exercise squat` (singular) but v6 QEVD class is `squats` (plural)

- Symptoms: Running `reality_check_v6.py` without `--exercise` argument falls back to `__other__` (index 24), making D9 evaluation results appear to show the model detecting "other" for every squat clip.
- Files: `backend/training/evaluation/reality_check_v6.py`
- Trigger: Running the script with the default `--exercise squat` argument.
- Workaround: Always pass `--exercise squats` (with the QEVD plural form) explicitly.

### v6 `qevd_exercise_map.json` not present in `backend/models/form_model_v6/`

- Symptoms: `FormAnalyzer._load_v6_model()` raises `FileNotFoundError` with message "v6 missing qevd_exercise_map.json" at startup, falling back to v5.2 or v4 weights silently.
- Files: `backend/services/form_analyzer.py` (lines 285-293), `backend/models/form_model_v6/`
- Trigger: Starting the backend after training v6 weights without copying `qevd_exercise_map.json` to the model output directory.
- Workaround: Copy `backend/data/qevd_class_space_v6_1.json` (or the generated map) to `backend/models/form_model_v6/qevd_exercise_map.json`.

---

## Performance Bottlenecks

### Full video loaded into memory before frame sampling

- Problem: `/analyze-form-video` calls `contents = await file.read()` (all bytes into memory) before writing to a temp file for OpenCV to re-read. For a 10-second video at 1080p this is 20-50 MB held in process memory for the duration of the call.
- Files: `backend/app.py` (lines 413-417)
- Cause: FastAPI's `UploadFile.read()` is a coroutine that buffers the full body.
- Improvement path: Use `shutil.copyfileobj` in chunks, or pass the `SpooledTemporaryFile` directly to OpenCV via the file path without reading into Python memory first.

### MediaPipe landmarker recreated on every `reset_video_state()` call

- Problem: `FormAnalyzer.reset_video_state()` destroys and recreates the `PoseLandmarker` object on each new session to reset the timestamp counter. PoseLandmarker initialisation loads the `.task` model file from disk each time.
- Files: `backend/services/form_analyzer.py` (lines 88-113)
- Cause: MediaPipe Tasks API (VIDEO mode) has no public `reset()` method; the workaround is full teardown/reinit.
- Improvement path: Cache the `.task` file bytes in memory and pass them as a buffer rather than a file path; MediaPipe accepts a byte buffer and avoids disk I/O on reinit.

### TensorFlow running on CPU on Windows (no GPU support in TF >=2.11)

- Problem: TF 2.13+ on Windows does not support native GPU; all inference runs on CPU. For a 1M-parameter model this yields ~5-15 ms per window inference, which is acceptable at 10 fps but becomes a bottleneck if inference cadence is increased.
- Files: `backend/services/form_analyzer.py` (all `_predict_window_*` paths)
- Cause: Architectural limitation of TensorFlow on Windows post-2.11.
- Improvement path: Use WSL2 + TF-GPU, or switch to ONNX Runtime which has native Windows GPU support.

---

## Fragile Areas

### Multi-version model auto-detection in `app.py` lifespan

- Files: `backend/app.py` (lines 183-203)
- Why fragile: The model selection is a three-way `if/elif/else` that probes file existence at startup. Introducing a v7 weights file in `form_model_v6/` would silently shadow the v6 path because the probe checks only the exact filenames `v6_supervised.weights.h5` and `v5_2_supervised.weights.h5`. Adding new model versions requires touching both `app.py` and `form_analyzer.py` in sync.
- Safe modification: Always update both files together. The `FITNOVA_MODEL_DIR` env var bypass is the safest path for testing a new model dir without changing code.
- Test coverage: No test covers the three-branch model fallback logic in isolation.

### `FormSession._resolve_exercise_idx` falls back to index 0 (squat) silently

- Files: `backend/services/form_session.py` (lines 83-96, `V5_2_EXERCISE_TO_IDX`)
- Why fragile: Any exercise not in `V5_2_EXERCISE_TO_IDX` (15 exercises) silently maps to index 0, making the model treat the session as a squat session. The UI exposes 27 exercises. Under v6 the fallback goes to `__other__` (index 24) instead, but the silent-fallback pattern is the same.
- Safe modification: Add a warning log whenever the fallback is triggered; consider surfacing this to the frontend as a capability warning.

### Distillation labels corpus is incomplete (465 clips, 35 failures)

- Files: `backend/data/distillation/vlm_labels_fitcoach_train.jsonl` (465 lines), `backend/data/distillation/vlm_labels_fitcoach_train.failed.jsonl` (35 lines)
- Why fragile: The v7 distillation pipeline depends on VLM labels covering the full FIT-COACH training set. The 35 failure records in `vlm_labels_fitcoach_train.failed.jsonl` indicate clips for which GPT-4o-mini Vision returned no usable response. A training run started on this corpus will silently omit those clips. If failure distribution is exercise-biased, the distilled model will underperform on the affected exercises.
- Safe modification: Inspect the 35 failure records, retry with higher temperature or longer prompts, and verify corpus coverage before starting the full distillation training.

### No v7 distillation training script exists yet

- Files: `backend/training/distillation/` (contains only `__pycache__`)
- Why fragile: The VLM label collection pipeline is complete (`vlm_labels_fitcoach_train.jsonl` exists) but `backend/training/distillation/` contains no Python training script. The next phase (training the distilled model on VLM labels) has no implementation. With the defense deadline on 2026-06-03, this gap is on the critical path.
- Safe modification: Implement the distillation training script as the immediate next step.

---

## Stale / Dead Code

### `backend/services/form_geometry.py` — orphaned 505-line geometric validator

- Issue: The comment in `form_session.py` line 25-27 states "form_geometry removed from runtime (Day 0.5, 2026-05-11) — pure ML signal only. The module remains in the repo for history/ablation reuse but is not wired into any session." The only import of `form_geometry` in production paths is via `exercise_rules/__init__.py` which describes rules "consumed by form_geometry" — also unused at runtime.
- Files: `backend/services/form_geometry.py` (505 lines), `backend/config/exercise_rules/__init__.py`, `backend/config/exercise_rules/squat.py` (209 lines), `backend/services/test_form_geometry.py`
- Impact: Dead code clutters the services layer and misleads future developers. The companion `exercise_rules/` config directory is also stranded.
- Fix approach: Move to `backend/_archive_*/` or delete. Keep `test_form_geometry.py` only if ablation is needed.

### Multiple stale training scripts for superseded model versions

- Issue: The following training scripts target model architectures that have been superseded and are no longer the active training path:
  - `backend/training/train_form_model.py` — v4 MT-TCN (superseded by v6)
  - `backend/training/train_form_model_v6.py` — v6 multi-task heads (superseded by v6.1)
  - `backend/training/train_ssl_pretrain.py` — SSL pre-training for Fit3D (Fit3D pipeline abandoned)
  - `backend/training/preprocessing/dataset_builder.py`, `dataset_builder_v5.py`, `dataset_builder_v5_1.py` — Fit3D dataset builders (Fit3D data pipeline deprecated in favour of QEVD)
  - `backend/training/evaluation/reality_check.py`, `reality_check_v5.py`, `reality_check_v6.py` — Evaluation scripts for v4, v5, v6 (only `reality_check_v6_1.py` is current)
- Files: Listed above
- Impact: Increases maintenance surface; increases Colab bundle size when the repo is zipped.
- Fix approach: Archive to `backend/_archive_*/` with a README explaining they are kept for dissertation appendix evidence only.

### Multiple stale model weight directories committed to git

- Issue: The following model directories are committed to git and contain binary weight files that are no longer the active inference path:
  - `backend/models/form_model_v1_quality_broken_20260420_154019/` — 23 MB, explicitly named "broken"
  - `backend/models/form_model_mediapipe_backup_20260420_032708/` — 21 MB backup
  - `backend/models/form_model_v2_backup_20260424_154358/` — 23 MB backup
  - `backend/models/form_model_v5/` — 15 MB (v5 Fit3D weights, superseded by v6)
  - `backend/models/form_model_v5_1/` — 4.3 MB
  - `backend/models/form_model_v5_2/` — 4.3 MB (still a valid fallback path in `form_analyzer.py`)
  - `backend/_archive_v5_2/models/form_model_v5_2/` — 4.5 MB total in archive
- Combined size tracked in git: ~100+ MB of binary weight files.
- Files: Directories listed above
- Impact: Inflates clone size; makes it unclear which weights are authoritative. The `.gitignore` correctly ignores `*_backup_*` dirs but `v1_quality_broken` and the archive `v5_2` are not excluded.
- Fix approach: Add `backend/models/form_model_v1_quality_broken_*/` and `backend/_archive_v5_2/models/` to `.gitignore`; remove them from the index with `git rm -r --cached`.

### 16 root-level build/bundle scripts tracked in git

- Issue: 16 utility scripts (`_build_v5_colab_bundle.py`, `_build_v6_1_colab_bundle.py`, etc.) exist at project root and are tracked in git. These scripts regenerate Colab ZIP bundles for superseded model versions (v5, v5.1, v5.2, v6.1 D2, v6.1 D6). They serve no function during app development or inference.
- Files: `_build_v5_colab_bundle.py`, `_build_v5_1_colab_bundle.py`, `_build_v5_2_colab_bundle.py`, `_build_v6_1_colab_bundle.py`, `_build_v6_d2_colab_bundle.py`, `_build_v6_d6_colab_bundle.py`, `_build_fit3d_annotations_zip.py`, `_build_fitnova_src_zip.py`, `_update_notebook.py`, `_validate_v6_notebook.py`, `_verify_v6_bundle.py`, `bundle_for_colab.ps1`, `rebuild_src_zip.py`, `rebuild_videos_zip.py`, `smoke_test_content_filter.py`, `smoke_test_recommender.py`
- Impact: Root directory clutter; new contributors are uncertain which scripts are active.
- Fix approach: Move to `tools/` subdirectory; mark v5/v5.1/v5.2 bundle scripts as deprecated in a README.

### `docs/ARCHITECTURE.md` and `README.md` describe Gemini/Claude as the LLM

- Issue: `docs/ARCHITECTURE.md` (lines 39, 214, 272, 345) and `README.md` (line 14) refer to "Gemini Flash" and "Gemini/Claude" as the Layer 3 LLM. The actual backend uses `gpt-4o-mini` (OpenAI) in `llm_adapter.py`, `chat_service.py`, and `form_session.py`.
- Files: `docs/ARCHITECTURE.md`, `README.md`, `backend/services/llm_adapter.py` (line 162), `backend/services/chat_service.py` (lines 142, 172)
- Impact: Misleading for dissertation reviewers and committee members who read the docs.
- Fix approach: Update `docs/ARCHITECTURE.md` and `README.md` to reflect `gpt-4o-mini` as the actual provider.

### 9 old Colab training notebooks tracked (v5 through v6.1)

- Issue: `colab_notebooks/` contains notebooks for v5, v5.1, v5.2, v6-D2, v6-D6, and v6.1 training runs. The only active notebook is `FitNova_Colab.ipynb` at the project root. The `colab_notebooks/` subdirectory represents historical training iterations.
- Files: `colab_notebooks/v5_ssl_pretrain.ipynb`, `colab_notebooks/v5_supervised_train.ipynb`, `colab_notebooks/v5_1_ssl_pretrain.ipynb`, `colab_notebooks/v5_1_supervised_train.ipynb`, `colab_notebooks/v5_2_supervised_train.ipynb`, `colab_notebooks/v6_d2_extract.ipynb`, `colab_notebooks/v6_d6_train.ipynb`, `colab_notebooks/v6_1_train.ipynb`
- Impact: Repo navigation confusion; dissertation readers may run the wrong notebook.
- Fix approach: Add a clear `colab_notebooks/DEPRECATED.md` noting these are historical evidence; ensure `FitNova_Colab.ipynb` at root is clearly identified as the active notebook.

---

## Test Coverage Gaps

### No integration test for the full WebSocket form-analysis pipeline

- What is not tested: The `FormSession + FormAnalyzer` end-to-end path via WebSocket. Only unit-level tests exist in `backend/services/test_form_session_v6_routing.py` and `test_form_analyzer_v6.py` which mock the underlying inference.
- Files: `backend/services/test_form_session_v6_routing.py`, `backend/services/test_form_analyzer_v6.py`
- Risk: A regression in the WebSocket message protocol (e.g. `start_session` → `frame` → `end_session` sequence) would not be caught before deployment.
- Priority: Medium

### No Flutter widget tests beyond the splash-screen smoke test

- What is not tested: `test/widget_test.dart` contains one test verifying the splash screen renders. There are no widget tests for the form check screen, plan screen, chat screen, or video upload screen.
- Files: `test/widget_test.dart`
- Risk: UI regressions in core screens go undetected.
- Priority: Medium (low-value for graduation project scope, but important for committee demo stability)

### No test for the three-branch model-version auto-detection in `app.py`

- What is not tested: The `lifespan` handler's v6 → v5.2 → v4 fallback logic has no test. If the v6 weights file is present but `qevd_exercise_map.json` is missing, the `FormAnalyzer` raises inside its constructor but the lifespan swallows it silently.
- Files: `backend/app.py` (lines 183-203), `backend/tests/test_app.py`
- Risk: Silent fallback to v4 weights in production without any visible error.
- Priority: High

### `backend/services/form_geometry.py` tests will pass but test dead code

- What is not tested correctly: `backend/services/test_form_geometry.py` tests a module that is no longer wired into any runtime path. Tests pass but provide zero coverage benefit for the live system.
- Files: `backend/services/test_form_geometry.py`, `backend/services/form_geometry.py`
- Risk: Creates false sense of test coverage.
- Priority: Low (resolve by archiving the module)

---

## Scaling Limits

### Synthetic interaction dataset ceiling

- Current capacity: The NeuMF recommendation model is trained on 809,439 synthetic interactions across 3,048 programs.
- Limit: All interactions are procedurally generated, not from real user behaviour. The NCF cold-start strategy (nearest-neighbour mapping to a synthetic user) is a known approximation that degrades as real-user behaviour diverges from synthetic assumptions.
- Scaling path: Collect real anonymised interaction data (program selected → feedback) and retrain NeuMF once enough real interactions exist (recommended minimum: ~50k interactions per literature).

### Single-process FastAPI, no horizontal scaling support

- Current capacity: The app runs as a single uvicorn process. The `FormAnalyzer` singleton is not process-safe for multi-process deployment.
- Limit: Vertical scaling only; adding uvicorn workers would give each worker its own `FormAnalyzer` copy, consuming ~300-500 MB per worker for the TF model.
- Scaling path: If multi-worker deployment is needed, load `FormAnalyzer` lazily per-request (not at startup) and use `--workers 2-4` in uvicorn. For GPU inference, switch to TF Serving or TorchServe.

---

*Concerns audit: 2026-05-18*
