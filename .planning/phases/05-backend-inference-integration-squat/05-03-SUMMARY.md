---
phase: 05-backend-inference-integration-squat
plan: "03"
subsystem: backend-api
tags: [fastapi, pytorch, squat-form, d05-schema, upload-endpoint, security]
dependency_graph:
  requires: [05-01, 05-02]
  provides: [app.state.squat_form_service, POST /analyze-form-video D-05, GET /health squat fields]
  affects: [backend/app.py, backend/tests/test_squat_form_api.py, backend/tests/test_app.py]
tech_stack:
  added: []
  patterns:
    - "SquatFormService loaded once at startup into app.state; shared read-only"
    - "classify_clip_async via run_in_threadpool — PyTorch forward off the event loop"
    - "Chunked read with running-total cap (T-05-07: 100 MB, defend missing Content-Length)"
    - "tmp_path=None guard before NamedTemporaryFile; if tmp_path: in finally (T-05-08)"
    - "D-05 binary+timing schema: ErrorDetection / RepResult / UploadResponse / SessionSummary"
key_files:
  created: []
  modified:
    - backend/app.py
    - backend/tests/test_squat_form_api.py
    - backend/tests/test_app.py
decisions:
  - "WS /ws/form-session is stubbed with a placeholder error; full implementation deferred to Plan 04"
  - "decode_clip_cv2/get_frame_count_and_fps used instead of decode_clip/read_video_timestamps (see Deviations)"
  - "Exercise allowlist is {squat} for Phase 5; 400 on any other value (T-05-09)"
  - "Graceful model-not-loaded: neutral D-05 dict returns 200, not 500 (T-05-02)"
metrics:
  duration_minutes: 15
  completed_date: "2026-05-26"
  tasks_completed: 2
  files_changed: 3
requirements_completed: [API-01, API-03]
---

# Phase 5 Plan 03: App.py Swap — SquatFormService + D-05 Upload Endpoint Summary

**One-liner:** Replaced TF/MediaPipe form subsystem with PyTorch SquatFormService in FastAPI lifespan; redefined POST /analyze-form-video with rep-segmentation, per-rep ensemble classification, D-05 binary+timing schema, and 100 MB upload cap.

## What Was Built

### Task 1 — Lifespan + /health + D-05 Pydantic models

- Removed `FormAnalyzer` and `FormSession` imports from `backend/app.py` (no longer imported at module level or anywhere)
- Added `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` before TF-dependent service imports (RESEARCH Pitfall 5)
- Replaced `FormFrameResult` + `FormSessionSummary` with four D-05 models:
  - `ErrorDetection`: type `^(KIE|KFE)$`, detected bool, confidence ge=0 le=1, severity_word `^(none|possible|moderate|strong)$`, intervals list[list[float]]
  - `RepResult`: exercise str, errors list[ErrorDetection]
  - `UploadResponse`: exercise str, total_reps int, reps list[RepResult]
  - `SessionSummary`: type="session_summary", exercise, total_reps, rep_results, session_feedback
- Lifespan: replaced v6/v5.2/v4 detection + FormAnalyzer load with `SquatFormService(model_dir=FITNOVA_MODEL_DIR or .../form_model_squat_md)`; logs model_ready + seeds_loaded; no mp_pose cleanup
- `/health`: returns `squat_model_ready`, `seeds_loaded`, `kie_threshold`, `kfe_threshold`; removed `model_version`/`n_exercises`/`model_dir`
- All merged router imports (`plan`, `coach`, `logs`) and `app.include_router(...)` calls PRESERVED

### Task 2 — POST /analyze-form-video + REST tests + test_app.py reconciliation

**Endpoint rewrite (`backend/app.py`):**
- `MAX_UPLOAD_BYTES = 100 * 1024 * 1024` (100 MB)
- HTTP 413 on `Content-Length` header pre-check (fast path for well-behaved clients)
- Chunked 1 MB streaming into temp file; running total check aborts at 100 MB (defends missing/forged header — T-05-07)
- `tmp_path: str | None = None` before `NamedTemporaryFile`; `if tmp_path: os.unlink(tmp_path)` in finally (T-05-08)
- Exercise allowlist `{"squat"}` else HTTP 400 (T-05-09)
- Inference: `get_frame_count_and_fps` → `segment_reps_by_motion_energy` → per-rep `uniform_sample_indices` + `decode_clip_cv2` + `await svc.classify_clip_async(frames.numpy())` (off event loop via run_in_threadpool)
- Timing: `intervals = [[start/fps, (end+1)/fps]]` attached to each detected error
- Graceful: `model_not_loaded` neutral dict returns 200, not 500

**Tests (`backend/tests/test_squat_form_api.py`):**
- `mock_services_squat` fixture: real `async def _classify_async` coroutine returning valid D-05 dict; `model_ready=True`, `_models=[1,2,3]`, thresholds set
- `squat_client` fixture: `TestClient(app, raise_server_exceptions=False)`
- `test_health_schema`: squat fields present, no model_version, seeds_loaded=3
- `test_upload_response_schema`: synthetic mp4 + exercise=squat → 200, exercise=="squat", total_reps>=1, errors type in {KIE,KFE}
- `test_upload_no_model`: neutral dict → 200 (graceful degradation)
- `test_upload_too_large`: Content-Length > MAX_UPLOAD_BYTES → 413
- `test_upload_bad_exercise`: exercise="bench" → 400

**test_app.py reconciliation:**
- `mock_services` fixture: added `app.state.squat_form_service = MagicMock()` so /health doesn't AttributeError on None
- `test_health_check`: changed from exact equality `{"status":"healthy"}` to `status_code==200` + `data["status"]=="healthy"` + `"squat_model_ready" in data`

## Deviations from Plan

### Auto-fixed Issues

**1. [Critical correction — per execution prompt] decode_clip/read_video_timestamps replaced with cv2 serving helpers**
- **Found during:** Task 2 implementation
- **Issue:** Plan's `<interfaces>` and Task 2 action block specified `decode_clip(tmp_path, idx)` (from `transforms.py`) and `read_video_timestamps` for fps/frame-count. Both call `torchvision.io.read_video` and `torchvision.io.read_video_timestamps`, which were removed in torchvision 0.27 (TorchCodec migration). Both raise `AttributeError: module 'torchvision.io' has no attribute 'read_video'` on the serving machine.
- **Fix:** Used `backend.services.clip_decode.get_frame_count_and_fps(tmp_path)` for total_frames + fps, and `backend.services.clip_decode.decode_clip_cv2(tmp_path, idx)` for clip decoding. Both were built and committed in Phase 05-02 specifically as the cv2-based serving-path replacements. Output contract identical: uint8 `[32, 3, H, W]` RGB tensor.
- **Files modified:** `backend/app.py`
- **Commits:** 7a24880

**2. [Rule 3 — Blocking] WS handler references to FormSession/form_analyzer replaced with stub**
- **Found during:** Task 1 — removing FormSession import
- **Issue:** The existing `/ws/form-session` WebSocket handler referenced `FormSession` and `websocket.app.state.form_analyzer`, which were removed by Task 1. Leaving these references would break `import backend.app`.
- **Fix:** Replaced the WS handler with a minimal stub that accepts connections and returns a "not yet available — see Plan 04" error on any message. Plan 04 is the scheduled owner of the full WS rewrite (SquatLiveSession). The stub preserves the route path and WebSocket protocol shape.
- **Files modified:** `backend/app.py`
- **Commits:** 389d199

## Known Stubs

| Stub | File | Line(s) | Reason |
|------|------|---------|--------|
| `/ws/form-session` WebSocket handler | backend/app.py | ~344-368 | Returns "not yet available" for all messages; full SquatLiveSession implementation is Plan 04 |

This stub does NOT prevent Plan 03's goal (upload endpoint + REST tests). Plan 04 resolves it.

## Self-Check

Files exist:
- `backend/app.py` ✓
- `backend/tests/test_squat_form_api.py` ✓
- `backend/tests/test_app.py` ✓

Commits:
- `389d199` feat(05-03): swap lifespan/health/D-05 models; remove FormAnalyzer+FormSession imports ✓
- `7a24880` feat(05-03): redefine POST /analyze-form-video + REST tests + reconcile test_app.py ✓

Tests: `python -m pytest backend/tests/test_squat_form_api.py backend/tests/test_app.py -x -q` → 21 passed ✓

Import: `python -c "import backend.app"` → exits 0 ✓

## Self-Check: PASSED
