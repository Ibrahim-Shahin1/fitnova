---
phase: 05-backend-inference-integration-squat
plan: "04"
subsystem: backend/form-ws
tags: [websocket, live-inference, squat-form, api, pytorch]
dependency_graph:
  requires: [05-01, 05-02, 05-03]
  provides: [WS /ws/form-session live rep feedback, SquatLiveSession]
  affects: [backend/app.py, backend/tests/test_squat_form_api.py]
tech_stack:
  added: []
  patterns:
    - "LiveWindowTrigger sliding-window gate for live rep detection"
    - "cv2.imdecode + HWC->TCHW permute for per-frame JPEG decode"
    - "conditional-send: rep_result only on trigger fire"
    - "deterministic session_summary feedback (no LLM)"
key_files:
  created: []
  modified:
    - backend/app.py
    - backend/tests/test_squat_form_api.py
decisions:
  - "Single-seed classify_clip_async(n_seeds=1) for ~0.93s live latency (D-08)"
  - "Deterministic session_feedback from per-rep KIE/KFE counts — no LLM (D-10)"
  - "deque(maxlen=90) caps per-session buffer at ~3s of frames (T-05-14)"
  - "Fresh SquatLiveSession per WebSocket accept — no shared mutable inference state (T-05-15)"
metrics:
  duration_minutes: 25
  tasks_completed: 2
  files_modified: 2
  completed_date: "2026-05-26"
---

# Phase 05 Plan 04: SquatLiveSession + Live WS Handler Summary

**One-liner:** Per-connection `SquatLiveSession` buffers JPEG frames, fires the
`LiveWindowTrigger` sliding-window gate, classifies single-seed off the event loop,
emits per-rep `rep_result` (conditional-send), and closes with a deterministic
`session_summary` — API-04 fully implemented.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | SquatLiveSession + redefined WS /ws/form-session handler | 8a07474 | backend/app.py |
| 2 | WS protocol + per-rep integration tests | b3cbcfc | backend/tests/test_squat_form_api.py |

## What Was Built

### Task 1 — SquatLiveSession + WS Handler

`class SquatLiveSession` added to `backend/app.py`:

- `__init__`: `deque(maxlen=LIVE_BUFFER_MAX_FRAMES=90)` of raw RGB frames,
  `LiveWindowTrigger()`, `_rep_results`, `_rep_number`.
- `add_frame(jpeg_bytes, timestamp_ms)`: `cv2.imdecode` decode + T-05-10 None-guard,
  BGR→RGB, buffer append, `should_fire()` check, `np.stack()` + `permute(0,3,1,2)`,
  `classify_clip_async(n_seeds=1)`, returns `rep_result` dict or `None`.
- `end_session()`: deterministic `_build_feedback()` from per-rep KIE/KFE counts,
  returns `session_summary` dict (D-05 schema).

`@app.websocket("/ws/form-session")` redefined:

- `start_session` → constructs fresh `SquatLiveSession`, sends `{"type":"session_started"}`.
- `frame` → decodes base64, calls `add_frame`, sends `rep_result` **only if not None**
  (conditional-send — the key behavioral change from the stub).
- `end_session` → `session.end_session()` → sends `session_summary`, breaks loop.
- `WebSocketDisconnect` / generic `Exception` handlers preserved from Plan-03 shell.

### Task 2 — WS Integration Tests

Three tests appended to `backend/tests/test_squat_form_api.py`:

- `test_ws_session_protocol`: start→started; end→summary with `exercise`+`rep_results`.
- `test_ws_rep_result_schema`: 47 frames → trigger fires → `rep_result` with
  `rep_number>=1` and `errors[*].type in {KIE,KFE}`.
- `test_ws_malformed_frame`: `b"not-a-jpeg"` frame → session survives → `session_summary`
  returned normally (T-05-10 None-guard verified).

All 16 tests in `test_squat_form_api.py` pass (13 from Plans 01+03, 3 new WS tests).

## Verification

```
python -c "import backend.app"         # exits 0
python -m pytest backend/tests/test_squat_form_api.py -x -q   # 16 passed
grep -c 'class SquatLiveSession' backend/app.py  # 1
grep -c 'n_seeds=1' backend/app.py               # 1
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing `base64` import in test module**

- **Found during:** Task 2 — `test_ws_malformed_frame` raised `NameError: name 'base64' is not defined`
- **Issue:** `base64` was used in the new WS tests but not imported in
  `backend/tests/test_squat_form_api.py` (Plan 03 test module).
- **Fix:** Added `import base64` at module top.
- **Files modified:** `backend/tests/test_squat_form_api.py`
- **Commit:** b3cbcfc (bundled with Task 2)

**2. [Rule 1 - Bug] `test_ws_rep_result_schema` would deadlock under original design**

- **Found during:** Task 2 — test hung indefinitely when calling `ws.receive_json()`
  inside the per-frame loop.
- **Issue:** Starlette TestClient's synchronous `receive_json()` blocks until a message
  arrives. With conditional-send, most frames produce no server reply, so the receive
  call blocked forever.
- **Fix:** Restructured the test to send all 47 frames first (no receive in loop),
  then call `receive_json()` exactly once to collect the single queued `rep_result`,
  then `end_session`. This matches the conditional-send semantics correctly.
- **Files modified:** `backend/tests/test_squat_form_api.py`
- **Commit:** b3cbcfc

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes beyond the planned WS redefinition.
The T-05-10/T-05-11/T-05-14/T-05-15 mitigations from the plan's threat register are all
implemented as specified.

## Self-Check: PASSED

- `backend/app.py` — modified, exists
- `backend/tests/test_squat_form_api.py` — modified, exists
- Commit 8a07474 — `feat(05-04): add SquatLiveSession + redefine WS /ws/form-session handler`
- Commit b3cbcfc — `feat(05-04): add WS integration tests — protocol, rep_result schema, malformed frame`
- `python -m pytest backend/tests/test_squat_form_api.py -x -q` → 16 passed
