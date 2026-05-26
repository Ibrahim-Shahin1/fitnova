---
phase: 05-backend-inference-integration-squat
verified: 2026-05-26T18:30:00Z
status: passed
score: 10/10 must-haves verified
e2e_closure: "Human items 1-2 closed 2026-05-26 via real-weights FastAPI TestClient E2E (no mocks); item 3 (serving-threshold calibration) tracked in 05-HUMAN-UAT.md (non-blocking)."
overrides_applied: 0
human_verification:
  - test: "Run the live WebSocket endpoint with a real phone-camera feed — open the WS, stream ~50 actual JPEG frames, verify a rep_result fires and the session_summary closes cleanly."
    expected: "rep_result arrives with rep_number >= 1, errors[*].type in {KIE,KFE}, session_summary has total_reps >= 1 and non-empty session_feedback."
    why_human: "TestClient WS test uses mocked classify_clip_async; the real model path (staged weights) and real-device JPEG decode are not exercised by the automated suite."
  - test: "Run the upload endpoint on a real phone-camera squat clip via curl or Postman (weights must be staged): POST /analyze-form-video -F 'file=@clip.mp4' -F 'exercise=squat'"
    expected: "200 response with D-05 JSON; KFE confidence > threshold (0.385) on a bad-form clip; KIE confidence directionally above zero; reps >= 1; timing intervals attached to detected errors."
    why_human: "Upload integration tests use mocked classify_clip_async; the full pipeline (cv2 decode -> kneeaware_spatial_val -> real model forward -> thresholds) is only exercised with real weights and a real clip."
  - test: "Verify the serving-threshold calibration note (D-11 follow-up): run probe_squat_inference.py on 3+ additional real clips (good-form and bad-form) and compare KIE/KFE confidence distributions."
    expected: "KFE good vs bad clips show >= 0.10 confidence gap; KIE shows directional separation (bad-form clips score higher even if below 0.614 threshold)."
    why_human: "The D-11 fix (kneeaware_spatial_val) was validated on 4 clips; the serving threshold for KIE (0.614) may need calibration for the phone-camera domain — this is the documented follow-up from Plans 02 and 05 SUMMARYs."
---

# Phase 5: Backend Inference Integration (Squat) Verification Report

**Phase Goal:** "The Squat form detector served through the backend in live and video-upload modes."
**Verified:** 2026-05-26T18:30:00Z
**Status:** passed (human items 1-2 closed via real-endpoint E2E on 2026-05-26; item 3 is a tracked non-blocking follow-up — see 05-HUMAN-UAT.md)
**Re-verification:** No — initial verification

> **E2E Closure (2026-05-26):** Human items 1 (live WS) + 2 (upload) were closed by a real-weights FastAPI `TestClient` run through the ACTUAL endpoints (no mocks): `/health` → `squat_model_ready: true`, 3 seeds; `POST /analyze-form-video` `BadSquat_45` → 200, **KFE detected 0.482** + timing `[0.0, 4.0]`, `GoodSquat_45` → 200, KFE 0.408, `exercise=bench` → **400**; `WS /ws/form-session` (60 real JPEG frames) → `session_started` → `rep_result` **KFE detected 0.852** (KIE clear) → `session_summary` (1 rep, deterministic feedback). Item 3 (serving-threshold calibration — esp. KIE 0.614) remains the documented **non-blocking** follow-up: `GoodSquat`'s 0.408 KFE confirms the offline KFE threshold over-detects on phone-camera framing, though the model still separates good vs bad (live WS bad 0.852 vs good ≈0.41).

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|--- |-------|--------|---------|
| 1 | The old form-analysis subsystem (form_analyzer, form_session, form_geometry, mediapipe_config, exercise_sanity) is removed from backend/services/ and present under backend/_archive_form_v4_v6/ | VERIFIED | `backend/services/` has none of the 5 old modules; all 5 confirmed under `backend/_archive_form_v4_v6/services/`. Archive README tracked, weight binaries gitignored. |
| 2 | A PyTorch form-inference service loads in FastAPI (app.state.squat_form_service) | VERIFIED | `backend/app.py` lifespan constructs `SquatFormService(model_dir=_squat_dir)` at startup; `app.state.squat_form_service` wired at lines 344–348. Zero occurrences of `FormAnalyzer`/`FormSession` in non-comment code. `python -c "import backend.app"` exits 0. |
| 3 | Uploading a squat clip returns binary error detections with timing (D-05 schema) | VERIFIED (automated) | POST /analyze-form-video: segments reps via `segment_reps_by_motion_energy`, decodes each rep with `decode_clip_cv2`, classifies via `classify_clip_async`, attaches timing intervals per detected error, returns `UploadResponse` (D-05 shape). `test_upload_response_schema` green. Real-weight execution requires human check (see human_verification item 2). |
| 4 | A live WebSocket session segments reps and returns per-rep error feedback | VERIFIED (automated) | `SquatLiveSession` buffers frames in rolling deque, uses `LiveWindowTrigger` for sliding-window gate (>=32 frames AND >=45 since last), classifies via `classify_clip_async(n_seeds=1)`, sends rep_result only on trigger fire, closes with deterministic session_summary. Tests `test_ws_session_protocol`, `test_ws_rep_result_schema`, `test_ws_malformed_frame` all green. Real-device exercise requires human check (see human_verification item 1). |
| 5 | RepSegmenter returns >= 1 rep interval for any decodable clip with guaranteed single-rep fallback | VERIFIED | `segment_reps_by_motion_energy` fallback to `[(0, total_frames-1)]` is present and tested. `test_rep_segmenter_synthetic` green. `LIVE_REP_WINDOW_FRAMES=32`, `LIVE_MIN_GAP_FRAMES=45`, `LIVE_BUFFER_MAX_FRAMES=90` all match spec. |
| 6 | Live sliding-window trigger fires only after >=32 buffered frames AND >=LIVE_MIN_GAP_FRAMES since last trigger | VERIFIED | `LiveWindowTrigger.should_fire()` enforces both conditions; resets counter on fire. `test_live_window_trigger` confirms fires exactly once across LIVE_MIN_GAP_FRAMES calls at buffer_len=32, never fires at buffer_len=10. |
| 7 | SquatFormService uses the EXACT Phase-4 inference path (kneeaware_spatial_val, aggregate_sigmoid_mean on raw logits, strict=True, thresholds 0.614/0.385) | VERIFIED | All anti-pattern guards confirmed in code: `weights=None`, `spatial_train` absent, `KINETICS400_V1` absent, `.half(` absent, `aggregate_sigmoid_mean` on raw logits, `strict=True` on best.pt, `run_in_threadpool` wrapper. `kneeaware_spatial_val` applied (D-11 fix: portrait lower-body pre-crop). `test_classify_deterministic` proves spatial_val determinism. |
| 8 | When weights are absent model_ready is False and classify_clip returns a neutral D-05 response | VERIFIED | `test_neutral_response_schema` uses empty tmp_path dir, confirms `model_ready=False`, `model_not_loaded=True`, `errors[0].type=="KIE"`, `errors[1].type=="KFE"`, no exception raised. |
| 9 | The torch forward runs off the event loop via run_in_threadpool (upload and live paths) | VERIFIED | `classify_clip_async` wraps `classify_clip` via `functools.partial` + `run_in_threadpool`. Upload endpoint awaits `classify_clip_async`. WS `add_frame` awaits `classify_clip_async(n_seeds=1)`. |
| 10 | POST /analyze-form-video rejects oversized uploads (413) and unsupported exercises (400); tmp file is cleaned up | VERIFIED | `MAX_UPLOAD_BYTES = 100 * 1024 * 1024`; content-length pre-check + chunked read cap both present; `tmp_path: str | None = None` before try block; `if tmp_path: os.unlink(tmp_path)` in finally. `test_upload_too_large` -> 413, `test_upload_bad_exercise` -> 400 both green. |

**Score:** 10/10 truths verified (automated). 3 items route to human verification due to mocked services in the test suite.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/services/squat_form_service.py` | SquatFormService + classify_clip + classify_clip_async + neutral degradation | VERIFIED | 319 lines; all required symbols present; anti-pattern guards enforced. |
| `backend/services/rep_segmenter.py` | motion-energy segmentation + LiveWindowTrigger | VERIFIED | 236 lines; constants match spec; guaranteed fallback present. |
| `backend/services/clip_decode.py` | cv2 decode backend + kneeaware_spatial_val (D-11 fix) | VERIFIED | Documented deviation from torchvision.io.read_video (removed in 0.27); cv2 drop-in with BGR→RGB conversion + portrait lower-body pre-crop; parity-APPROX documented. |
| `backend/tests/test_squat_form_api.py` | All 16 required test functions + helpers | VERIFIED | All 16 test functions confirmed present; 16 tests collected, all green. |
| `backend/app.py` | SquatFormService lifespan + D-05 Pydantic models + /health + upload endpoint + WS endpoint | VERIFIED | All required classes and patterns confirmed; zero FormAnalyzer/FormSession references in executable code. |
| `backend/_archive_form_v4_v6/README.md` | Archive manifest with form_analyzer | VERIFIED | Exists, tracked, contains "form_analyzer"; manifest covers all 5 services + tests + training scripts + weight dirs. |
| `backend/_archive_form_v4_v6/services/` | 5 old form modules archived | VERIFIED | All 5 modules confirmed under archive directory. |
| `backend/scripts/probe_squat_inference.py` | Latency probe + domain-shift test | VERIFIED | Exists; imports SquatFormService, contains classify_clip. Human-run on BadSquat_45.mp4 per Plan 02 SUMMARY. |
| `backend/scripts/render_squat_result.py` | KIE/KFE overlay visualization | VERIFIED | Exists; imports classify_clip, matplotlib (Agg), uniform_sample_indices; no .half() calls in executable code (AST-confirmed); figure produced on BadSquat_45.mp4. |
| `backend/models/form_model_squat_md/README.md` | Staging docs with Drive paths | VERIFIED | Exists; contains "md_finetune_seed"; documents seed42/1337/7 layout and thresholds. |
| `backend/training/aqa/` (inference contract) | transforms.py, eval/ensemble.py, harness/md_finetune.py INTACT | VERIFIED | All 4 required contract files confirmed present; NOT archived. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `backend/app.py` | `backend/services/squat_form_service.py` | `app.state.squat_form_service = SquatFormService(...)` in lifespan | WIRED | Lines 344–348. |
| `backend/app.py` | `backend/services/rep_segmenter.py` | `segment_reps_by_motion_energy(` in upload handler; `LiveWindowTrigger` in SquatLiveSession | WIRED | Lines 41–45 (module-level import) + line 631. |
| `backend/services/squat_form_service.py` | `backend/training/aqa/eval/ensemble.py` | `from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean` | WIRED | Line 44 of squat_form_service.py. |
| `backend/services/squat_form_service.py` | `backend/services/clip_decode.py` | `from backend.services.clip_decode import kneeaware_spatial_val` | WIRED | Line 45 of squat_form_service.py. |
| `backend/services/clip_decode.py` | `backend/training/aqa/datasets/transforms.py` | `from backend.training.aqa.datasets.transforms import spatial_val` | WIRED | Line 48 of clip_decode.py; transitively satisfies Plan 01 key_link to transforms.py. |
| `backend/app.py` (upload) | `backend/services/clip_decode.py` | `decode_clip_cv2(tmp_path, idx)` | WIRED | Lazy import at line 622; called at line 640. |
| `backend/scripts/render_squat_result.py` | `backend/services/squat_form_service.py` | `SquatFormService.classify_clip` | WIRED | Confirmed present in render script. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `POST /analyze-form-video` handler | `reps: list[RepResult]` | `decode_clip_cv2` -> `classify_clip_async` -> `ErrorDetection` construction | Yes — real cv2 frame decode, real (or mocked in tests) model inference, timing from fps calculation | FLOWING (with staged weights); mocked in automated tests |
| `WS /ws/form-session` handler | `rep_result` dict | `cv2.imdecode` -> RGB convert -> buffer -> trigger -> `classify_clip_async` -> `self._rep_results.append` | Yes — real JPEG decode, real (or mocked) model inference | FLOWING (with staged weights); mocked in automated tests |
| `SquatFormService.classify_clip` | `scores` ndarray | `r2plus1d_18` forward -> `aggregate_sigmoid_mean` on raw logits | Yes — real PyTorch forward when weights loaded; neutral dict otherwise | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `backend.app` imports cleanly | `python -c "import backend.app"` | Exit 0, no output | PASS |
| Scoped test suite (Phase 5 only) | `python -m pytest backend/tests/test_squat_form_api.py backend/tests/test_app.py -q` | 24 passed in 15.65s | PASS |
| FormAnalyzer absent from app.py | grep non-comment app.py for FormAnalyzer | 0 occurrences | PASS |
| FormSession absent from app.py | grep non-comment app.py for FormSession | 0 occurrences | PASS |
| seed42/best.pt gitignored | `git check-ignore backend/models/form_model_squat_md/seed42/best.pt` | Path printed (ignored) | PASS |
| aqa/ inference contract intact | File existence check on transforms.py, ensemble.py, metrics.py, md_finetune.py | All 4 present | PASS |

---

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes are present for this phase. The equivalent role is filled by `backend/scripts/probe_squat_inference.py` (a Python CLI), which requires staged weights and a real clip. Per Plan 02 SUMMARY, it was run on `BadSquat_45.mp4` and produced:

```
KFE detected=True  confidence=0.4817  severity=possible
KIE detected=False  confidence=0.0603  severity=none
```

This is logged as a human-verified datapoint in the SUMMARY, not reproducible programmatically from the CI-clean repo state.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| API-01 | Plans 01, 03, 05 | Old form-analysis subsystem removed; PyTorch form-inference service loaded in FastAPI | SATISFIED | 5 old modules git-moved to `_archive_form_v4_v6/`; `SquatFormService` in lifespan; zero FormAnalyzer/FormSession imports in app.py. |
| API-02 | Plans 01, 03, 04 | Rep-segmentation component for live and upload inference paths | SATISFIED | `segment_reps_by_motion_energy` used in upload handler; `LiveWindowTrigger` used in `SquatLiveSession`; both green under test. |
| API-03 | Plan 03 | Video-upload REST endpoint returning binary error detections with timing | SATISFIED | `POST /analyze-form-video` segments reps, classifies each, attaches timing intervals, returns `UploadResponse` (D-05). `test_upload_response_schema` green. |
| API-04 | Plan 04 | Live WebSocket endpoint returning per-rep error feedback | SATISFIED | `WS /ws/form-session` with `SquatLiveSession` fires per-rep results, closes with deterministic `session_summary`. WS tests green. |

**All 4 Phase-5 requirements (API-01 through API-04) are satisfied.**

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `backend/services/squat_form_service.py` | `_neutral_response()` returns `detected=False, confidence=0.0` for all errors | INFO (intentional) | Correct behavior when `model_ready=False`; documented graceful-degradation pattern, not a stub. |
| `backend/services/clip_decode.py` | cv2 BGR decoder instead of torchvision read_video | INFO (documented deviation) | torchvision 0.27 removed read_video; cv2 is parity-APPROX. Documented in module docstring and SUMMARY. Not a blocker — the phase goal is "serve the detector," not "bit-identical training/serving parity." |

No TBD, FIXME, or XXX markers found in any Phase-5 modified file. No unreferenced debt markers.

---

### Human Verification Required

The automated suite uses mocked `classify_clip_async` throughout (both REST and WS integration tests). The following require execution with real staged weights and real input:

#### 1. Live WebSocket with Real Phone-Camera Feed

**Test:** Start the backend (`uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`), connect via wscat or a Flutter client, send `start_session`, stream 50+ real JPEG frames from a phone-camera squat, send `end_session`.

**Expected:** At least one `rep_result` message arrives with `rep_number >= 1`, `errors[*].type` in {KIE, KFE}; `session_summary` arrives with `total_reps >= 1` and a non-empty `session_feedback` string describing knee errors.

**Why human:** Mocked classify_clip_async bypasses the real PyTorch forward, the kneeaware_spatial_val portrait pre-crop, and real JPEG decode from a device. The automated WS tests confirm protocol correctness; they do not confirm the model produces sensible outputs on real input.

#### 2. Upload Endpoint with Real Squat Clip

**Test:** `curl -X POST http://localhost:8000/analyze-form-video -F "file=@clip.mp4" -F "exercise=squat"` using a real phone-camera squat clip with staged weights.

**Expected:** 200 response; D-05 JSON with `total_reps >= 1`; KFE confidence > 0.385 on a bad-form clip; timing intervals populated on detected errors. Confirm KIE confidence is non-zero (even if below 0.614 — directional correctness).

**Why human:** Full pipeline from cv2 decode to kneeaware_spatial_val to real ensemble forward. The mocked test confirms schema; this confirms model behavior.

#### 3. Serving-Threshold Calibration Follow-Up (D-11 — documented, not blocking)

**Test:** Run `python -m backend.scripts.probe_squat_inference <clip.mp4>` on >= 3 additional good-form and bad-form clips. Record KIE and KFE confidences per clip.

**Expected:** KFE: good-form clips score consistently below 0.385; bad-form clips score above. KIE: directional separation (bad-form higher than good-form) even if both below 0.614.

**Why human:** Plan 02 SUMMARY documents this as a deferred follow-up — the threshold was tuned on offline Fitness-AQA val clips; the phone-camera domain shift may require recalibration on user clips. This is explicitly NOT treated as a phase-goal failure (per `<important_context>` in the verification request). It is surfaced here for the developer's awareness.

---

### Gaps Summary

No automated gaps were found. All 10 must-have truths verified. All 4 requirements (API-01 through API-04) covered. No blocker anti-patterns or unreferenced debt markers.

The 3 human verification items above are the only open items. They all relate to real-model / real-device behavior that cannot be confirmed from a CI-clean repo without staged weights (~378 MB, gitignored) and a phone-camera clip.

The D-11 serving-threshold follow-up is a documented improvement item, not a goal failure — the phase goal "serve the detector via API" is delivered: the upload and live endpoints return the D-05 schema and the model discriminates with knees in frame.

---

_Verified: 2026-05-26T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
