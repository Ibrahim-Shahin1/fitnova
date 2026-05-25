# Phase 5: Backend Inference Integration (Squat) - Pattern Map

**Mapped:** 2026-05-25
**Files analyzed:** 7 new/modified files
**Analogs found:** 6 / 7 (1 has no codebase analog — RepSegmenter)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `backend/services/squat_form_service.py` | service | request-response (clip-in, dict-out) | `backend/services/form_analyzer.py` | role-match |
| `backend/services/rep_segmenter.py` | utility | transform (video-in, intervals-out) | `backend/app.py` `/analyze-form-video` OpenCV loop | partial (frame-source pattern only) |
| `backend/app.py` (modified) | controller | request-response + WS | `backend/app.py` (self) | exact (redefined in place) |
| `backend/tests/test_squat_form_api.py` | test | request-response | `backend/tests/test_app.py` | exact |
| `backend/models/form_model_squat_md/` (staged) | config/artifact | — | `backend/models/form_model_v6/` (old) | structure-match |
| `backend/_archive_form_v4_v6/` (new dir) | archive | — | existing `backend/models/` dirs | structural |
| `.gitignore` (modified) | config | — | existing `.gitignore` | — |

---

## Pattern Assignments

### `backend/services/squat_form_service.py` (service, request-response)

**Analog:** `backend/services/form_analyzer.py`

**The class this replaces.** Copy the structural shell: module docstring, `__init__` signature taking `model_dir`, `model_ready` property, `_load` private method, and the neutral-output degradation pattern. Everything MediaPipe/TF specific is discarded; everything about startup load, `model_ready` guard, and graceful degradation is kept.

**Imports pattern** (`form_analyzer.py` lines 1-41):
```python
"""
SquatFormService — PyTorch R(2+1)D-18 ensemble inference service.

Loaded once at FastAPI startup into app.state.squat_form_service.
Shared read-only across all sessions — NO per-call mutable state
(lesson from old FormAnalyzer: _ts_ms/_prev_result on a shared instance
corrupted concurrent sessions — CONCERNS.md).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from starlette.concurrency import run_in_threadpool
from torchvision.models.video import r2plus1d_18

from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    spatial_val,
    uniform_sample_indices,
)

logger = logging.getLogger(__name__)
```

**`model_ready` property pattern** (`form_analyzer.py` lines 335-342):
```python
# Old analog (copy the shape, not the TF logic):
@property
def model_ready(self) -> bool:
    if self._model is None:
        return False
    # v6 + v5.2 don't use the AngleNormalizer; v4 requires it.
    if getattr(self, "_model_version", "v4") in ("v6", "v5.2"):
        return True
    return self._normalizer is not None

# New version (simpler — no normalizer needed):
@property
def model_ready(self) -> bool:
    return self._model_ready  # bool set in _load()
```

**`_load` / startup model-load pattern** (`form_analyzer.py` lines 151-222, v4 path as structure reference):
```python
def _load_model(self):
    """Detect model version from model_dir and load accordingly."""
    v6_weights   = os.path.join(self.model_dir, "v6_supervised.weights.h5")
    # ...file-presence detection...
    if not os.path.exists(weights_path):
        logger.warning(
            f"No v5.2 or v4 weights found in {self.model_dir}. "
            "Train the model first."
        )
        return  # ← graceful: model_ready stays False; endpoints return neutral output

    # ...load and set self._model...
    logger.info("v4 MT-TCN weights loaded (%.1f MB).", os.path.getsize(weights_path) / 1e6)
```

**Architecture construction (exact pattern from `harness/md_finetune.py` lines 84-108 — use verbatim):**
```python
# harness/md_finetune.py:84-108
def build_finetune_model(md_backbone_path: str, *, dropout: float = 0.2) -> nn.Module:
    from torchvision.models.video import r2plus1d_18

    # ANTI-PATTERN GUARD: weights=None — NOT Kinetics. MD backbone IS the initialization.
    model = r2plus1d_18(weights=None)
    assert model.fc.in_features == 512, (
        f"R(2+1)D-18 fc.in_features={model.fc.in_features}, expected 512"
    )
    # strict=False for backbone-only checkpoint (backbone.pt has no fc keys)
    ckpt = torch.load(md_backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))
    return model
```

**At serve time use `strict=True` for the fine-tune `best.pt` (DIFFERENT from training):**
```python
# For best.pt (full fine-tuned model — backbone + new fc head):
ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)  # strict=True — full model
model.eval()
```

**Neutral-output degradation pattern** (`form_analyzer.py` lines 507-513):
```python
# Old analog — copy the shape, adapt the fields:
def predict_window(self, ...):
    if not self.model_ready:
        return {
            "exercise": "unknown", "exercise_confidence": 0.0,
            "boundary": [0.0] * 64, "rep_count": 0.0,
            "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
            "model_version": "none",
        }
```

**New neutral-output for `classify_clip`:**
```python
def _neutral_response(self) -> dict:
    """Returned when model_ready is False — matches D-05 schema."""
    return {
        "exercise": "squat",
        "errors": [
            {"type": "KIE", "detected": False, "confidence": 0.0,
             "severity_word": "none", "intervals": []},
            {"type": "KFE", "detected": False, "confidence": 0.0,
             "severity_word": "none", "intervals": []},
        ],
        "model_not_loaded": True,
    }
```

**Core inference pattern — preprocessing (exact from `transforms.py` lines 33-215 — DO NOT deviate):**
```python
# transforms.py:190-215  ← spatial_val (val/test path — NOT spatial_train)
def spatial_val(
    clip_tchw: torch.Tensor,
    *,
    crop_size: int = 112,
    resize_short: int = 128,
) -> torch.Tensor:
    """Val/test spatial pipeline: resize → center crop → /255 → Kinetics norm → permute.
    Deterministic — no generator parameter."""
    resized = _resize_short_side(clip_tchw, resize_short)
    _, _, h, w = resized.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    cropped = resized[:, :, top : top + crop_size, left : left + crop_size]
    floated = cropped.to(torch.float32) / 255.0
    normed = _normalize_kinetics(floated)
    return normed.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

# KINETICS constants (transforms.py:24-25):
KINETICS_MEAN: Final[tuple] = (0.43216, 0.394666, 0.37645)
KINETICS_STD:  Final[tuple] = (0.22803, 0.22145, 0.216989)
```

**Ensemble aggregation pattern (exact from `eval/ensemble.py` lines 21-48 — use verbatim):**
```python
# ensemble.py:21-48
def aggregate_sigmoid_mean(per_seed_logits: list[np.ndarray]) -> np.ndarray:
    """Mean of per-seed sigmoid scores (D4: mean-of-sigmoids).
    Input: list of (N, 2) raw logits — NOT pre-sigmoidized.
    Output: (N, 2) float in [0, 1]."""
    sigmoid_scores = [1.0 / (1.0 + np.exp(-a)) for a in arrays]
    ensemble = np.mean(sigmoid_scores, axis=0)
    # ...
```

**`run_in_threadpool` async wrapper pattern** (FastAPI idiom — no existing analog in this codebase, but confirmed by RESEARCH):
```python
# Pattern sourced from RESEARCH.md §Standard Stack — starlette.concurrency:
async def classify_clip_async(self, frames_tchw: np.ndarray) -> dict:
    """Async wrapper — must be called from async endpoint handlers."""
    return await run_in_threadpool(self.classify_clip, frames_tchw)
```

---

### `backend/services/rep_segmenter.py` (utility, transform)

**Analog:** `backend/app.py` `/analyze-form-video` endpoint (lines 419-451) — frame-source/decode loop pattern only. No analog for the segmentation algorithm itself.

**Frame-decode loop pattern to reuse** (`app.py` lines 419-451):
```python
# app.py:419-451 — the OpenCV frame-decode loop that feeds frames
cap = cv2.VideoCapture(tmp_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_idx = 0
SKIP = max(1, int(fps / 10))  # sample ~10 fps

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % SKIP == 0:
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        ts_ms = int((frame_idx / fps) * 1000)
        # ... process frame ...
    frame_idx += 1

cap.release()
```

**For `RepSegmenter`, the frame decode loop becomes a low-res grayscale scan:**
```python
# Adapted from app.py pattern — full-res not needed for motion energy
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames_gray = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    small = cv2.resize(frame, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    frames_gray.append(gray.astype(np.float32))
cap.release()
# ... motion-energy segmentation on frames_gray ...
```

**Temp-file handling pattern to copy** (`app.py` lines 413-418 and 449-450):
```python
# app.py:413-418 — write UploadFile to temp disk location
contents = await file.read()
suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(contents)
    tmp_path = tmp.name

# app.py:449-450 — always clean up in finally:
try:
    # ... process ...
finally:
    os.unlink(tmp_path)
```

**Fallback pattern (always return at least one interval):**
```python
# When no segmentation is found, treat entire clip as one rep — the single-rep
# Fitness-AQA case is the simplest correct case (CONTEXT D-03).
if not rep_intervals:
    total_frames = len(frames_gray)
    rep_intervals = [(0, total_frames - 1)]
    logger.info("rep_segmenter: no boundaries detected — treating as single rep")
return rep_intervals
```

---

### `backend/app.py` (modified — redefined in place)

**Analog:** `backend/app.py` (self — redefined in place; reuse the shell, replace the form-specific parts)

**Lifespan pattern to reuse** (`app.py` lines 167-209):
```python
# app.py:167-209 — the lifespan pattern for loading services at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading recommendation engine (Layers 1+2)...")
    app.state.recommender = Recommender()
    logger.info("Loading LLM adapter (Layer 3)...")
    app.state.llm_adapter = LLMAdapter()
    logger.info("Loading chat service...")
    app.state.chat_service = ChatService()
    logger.info("Loading form analyzer (Layer 4)...")
    # Auto-detection priority: v6 -> v5.2 -> v4. Override with FITNOVA_MODEL_DIR
    _override = os.environ.get("FITNOVA_MODEL_DIR")
    if _override:
        _model_dir = _override
        logger.info("FITNOVA_MODEL_DIR override -> %s", _model_dir)
    else:
        _v6_dir = os.path.join(os.path.dirname(__file__), "models", "form_model_v6")
        # ... file-presence detection ...
    app.state.form_analyzer = FormAnalyzer(model_dir=_model_dir)
    logger.info("FitNova backend ready.")
    yield
    logger.info("Shutting down FitNova backend.")
    if hasattr(app.state, "form_analyzer") and app.state.form_analyzer.mp_pose:
        app.state.form_analyzer.mp_pose.close()
```

**New lifespan replaces the form-analyzer block (lines 175-208) with:**
```python
# Replace lines 175-208 with:
logger.info("Loading Squat form service (PyTorch)...")
from backend.services.squat_form_service import SquatFormService
_squat_dir = os.path.join(os.path.dirname(__file__), "models", "form_model_squat_md")
app.state.squat_form_service = SquatFormService(model_dir=_squat_dir)
logger.info(
    "SquatFormService loaded (model_ready=%s)",
    app.state.squat_form_service.model_ready,
)
logger.info("FitNova backend ready.")
yield
logger.info("Shutting down FitNova backend.")
# No mp_pose.close() needed — PyTorch CPU inference needs no cleanup
```

**`/health` pattern to update** (`app.py` lines 236-247):
```python
# app.py:236-247 — keep the structure, update the field names:
@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    analyzer = getattr(request.app.state, "form_analyzer", None)
    model_info = {
        "model_version": getattr(analyzer, "_model_version", None) if analyzer else None,
        "model_ready":   bool(analyzer.model_ready) if analyzer else False,
        "model_dir":     getattr(analyzer, "model_dir", None) if analyzer else None,
        "n_exercises":   len(getattr(analyzer, "_exercise_labels", {})) if analyzer else 0,
    }
    return {"status": "healthy", **model_info}
```

**New `/health` mirrors this exactly but with `squat_form_service` fields:**
```python
# New fields: squat_model_ready, seeds_loaded, thresholds
svc = getattr(request.app.state, "squat_form_service", None)
model_info = {
    "squat_model_ready": bool(svc.model_ready) if svc else False,
    "seeds_loaded":      len(svc._models) if svc else 0,
    "kie_threshold":     svc.kie_threshold if svc else None,
    "kfe_threshold":     svc.kfe_threshold if svc else None,
}
return {"status": "healthy", **model_info}
```

**Pydantic model pattern to replace** (`app.py` lines 138-160):
```python
# app.py:138-160 — old FormFrameResult + FormSessionSummary (REPLACED, not shimmed)
class FormFrameResult(BaseModel):
    timestamp_ms: int
    landmarks: list[list[float]] | None = None
    joint_errors: list[float]            # 10 values, 0-1
    quality_score: float
    rep_count: int
    exercise_detected: str
    confidence: float
    status: str

class FormSessionSummary(BaseModel):
    exercise: str
    total_reps: int
    duration_seconds: int
    average_quality: float
    per_rep_scores: list[float]
    common_errors: dict[str, int]
    quality_trend: str
    llm_feedback: str
```

**New Pydantic models follow the same inline-in-app.py pattern with `Field(...)` validators:**
```python
# New models — declare inline in app.py, same pattern as UserProfileRequest above:
class ErrorDetection(BaseModel):
    type: str = Field(..., pattern=r"^(KIE|KFE)$")
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity_word: str = Field(..., pattern=r"^(none|possible|moderate|strong)$")
    intervals: list[list[float]] = Field(default_factory=list)

class RepResult(BaseModel):
    exercise: str
    errors: list[ErrorDetection]

class SessionSummary(BaseModel):
    type: str = "session_summary"
    exercise: str
    total_reps: int
    rep_results: list[RepResult]
    session_feedback: str
```

**WebSocket handler pattern to reuse** (`app.py` lines 336-390 — keep transport plumbing verbatim, replace session class):
```python
# app.py:336-390 — the entire WS handler shell is reused:
@app.websocket("/ws/form-session")
async def form_session_ws(websocket: WebSocket):
    await websocket.accept()
    session: FormSession | None = None  # ← swap FormSession → SquatLiveSession

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_session":
                selected = data.get("selected_exercise") or data.get("exercise_hint")
                session = FormSession(                    # ← swap to SquatLiveSession
                    analyzer=websocket.app.state.form_analyzer,
                    selected_exercise=selected,
                )
                await websocket.send_json({"type": "session_started"})

            elif msg_type == "frame":
                if session is None:
                    session = FormSession(analyzer=websocket.app.state.form_analyzer)
                jpeg_bytes   = base64.b64decode(data["data"])
                timestamp_ms = int(data.get("timestamp_ms", 0))
                result = session.add_frame(jpeg_bytes, timestamp_ms)
                await websocket.send_json(result)          # ← only send if result is not None

            elif msg_type == "end_session":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No active session"})
                else:
                    summary = session.end_session()
                    await websocket.send_json(summary)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception(f"WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
```

**REST upload handler pattern to reuse** (`app.py` lines 397-454 — keep UploadFile + temp file + try/finally shell):
```python
# app.py:397-454 — UploadFile handling, temp file, cap.read() loop shells are reused.
# The session.add_frame() loop is replaced by:
#   1. RepSegmenter.segment_reps_by_motion_energy(tmp_path) → [(start, end), ...]
#   2. For each rep interval: uniform_sample_indices + decode_clip + svc.classify_clip_async()
@app.post("/analyze-form-video", tags=["Form"])
async def analyze_form_video(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form(default="squat"),
):
    svc = request.app.state.squat_form_service
    contents = await file.read()
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ... RepSegmenter + per-rep classify_clip_async ...
    finally:
        os.unlink(tmp_path)   # ← same finally pattern as original
```

---

### `backend/tests/test_squat_form_api.py` (test, request-response + WS)

**Analog:** `backend/tests/test_app.py` (exact match — same TestClient pattern, same fixture structure)

**Module docstring + imports pattern** (`test_app.py` lines 1-11):
```python
"""
Tests for backend/app.py — FastAPI endpoints.
All heavy services (Recommender, LLMAdapter) are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.app import app
```

**`autouse` fixture pattern for mocking services** (`test_app.py` lines 60-85):
```python
# test_app.py:60-85 — the mock_services autouse fixture pattern
@pytest.fixture(autouse=True)
def mock_services():
    mock_rec = MagicMock()
    mock_rec.recommend.return_value = { ... }

    mock_llm = MagicMock()
    mock_llm.generate_plan.return_value = { ... }

    app.state.recommender = mock_rec
    app.state.llm_adapter = mock_llm
    yield mock_rec, mock_llm
```

**Adapt for `squat_form_service` mock:**
```python
@pytest.fixture(autouse=True)
def mock_services_squat():
    mock_svc = MagicMock()
    mock_svc.model_ready = True
    mock_svc.kie_threshold = 0.614
    mock_svc.kfe_threshold = 0.385
    mock_svc.classify_clip.return_value = {
        "exercise": "squat",
        "errors": [
            {"type": "KIE", "detected": False, "confidence": 0.3,
             "severity_word": "none", "intervals": []},
            {"type": "KFE", "detected": True, "confidence": 0.82,
             "severity_word": "strong", "intervals": []},
        ],
    }
    # classify_clip_async must be a coroutine mock:
    async def _async_classify(frames):
        return mock_svc.classify_clip.return_value
    mock_svc.classify_clip_async = _async_classify

    app.state.recommender = MagicMock()     # keep existing services mocked
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.squat_form_service = mock_svc
    yield mock_svc
```

**`client` fixture pattern** (`test_app.py` lines 83-85):
```python
@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)
```

**REST test pattern** (`test_app.py` lines 93-96 — adapt for `/analyze-form-video`):
```python
# test_app.py:93-96
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

**WebSocket test pattern** (Starlette TestClient WS — no existing analog in this codebase; RESEARCH.md §7 provides the pattern):
```python
# RESEARCH.md §7 — starlette.testclient.WebSocketTestSession
def test_ws_session_protocol(client):
    with client.websocket_connect("/ws/form-session") as ws:
        ws.send_json({"type": "start_session", "exercise_hint": "squat"})
        resp = ws.receive_json()
        assert resp["type"] == "session_started"
        # ...send frames, end_session...
        ws.send_json({"type": "end_session"})
        summary = ws.receive_json()
        assert summary["type"] == "session_summary"
```

**Error + failure test pattern** (`test_app.py` lines 139-153 — copy for service failure cases):
```python
# test_app.py:139-153
def test_generate_plan_recommender_failure(client, mock_services):
    mock_rec, _ = mock_services
    mock_rec.recommend.side_effect = RuntimeError("boom")
    response = client.post("/generate-plan", json=VALID_BODY)
    assert response.status_code == 500
    assert "Recommendation failed" in response.json()["detail"]
```

**pytest.importorskip pattern for torch** (`test_md_finetune.py` lines 12-13):
```python
# test_md_finetune.py:12-13 — skip when torch unavailable
torch = pytest.importorskip("torch")
```

---

### `backend/models/form_model_squat_md/` (staged artifact directory)

**Structure analog:** `backend/models/form_model_v6/` (old dir, now archived)

No code pattern — this is a directory holding `seed42/best.pt`, `seed1337/best.pt`, `seed7/best.pt` staged from Drive. The planner must include a human action step for the user to download from `MyDrive/FitNova/checkpoints/phase04/md_finetune_seed{42,1337,7}/best.pt`. Commit a `README.md` documenting the paths; add `backend/models/form_model_squat_md/` to `.gitignore` for the `.pt` files.

---

## Shared Patterns

### 1. `app.state` Service Load Pattern
**Source:** `backend/app.py` lines 167-208 (`lifespan` context manager)
**Apply to:** `squat_form_service` load block in the new lifespan
```python
# Single pattern used for ALL services: construct in lifespan, attach to app.state.*,
# log the result. No lazy init — fail fast at startup.
app.state.recommender = Recommender()
# ...
app.state.form_analyzer = FormAnalyzer(model_dir=_model_dir)
logger.info("FitNova backend ready.")
yield
logger.info("Shutting down FitNova backend.")
```

### 2. Graceful Degradation (model_ready guard)
**Source:** `backend/services/form_analyzer.py` lines 335-342 + lines 507-513
**Apply to:** `SquatFormService.classify_clip`, both endpoint handlers
```python
# form_analyzer.py:507-513 — neutral output when model not ready:
if not self.model_ready:
    return {
        "exercise": "unknown", "exercise_confidence": 0.0,
        "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
        "model_version": "none",
    }
```

### 3. Endpoint Exception → HTTPException Pattern
**Source:** `backend/app.py` lines 272-295 (`generate_plan` handler)
**Apply to:** Both new endpoint handlers
```python
# app.py:272-295
try:
    rec_result = request.app.state.recommender.recommend(user_profile)
except Exception as exc:
    logger.exception("Recommendation engine failed")
    raise HTTPException(
        status_code=500,
        detail=f"Recommendation failed: {exc}",
    )
```

### 4. Deterministic Fallback Feedback
**Source:** `backend/services/form_session.py` lines 727-746 (`_fallback_feedback` static method)
**Apply to:** `SquatLiveSession.end_session()` session feedback generation (D-10)
```python
# form_session.py:727-746
@staticmethod
def _fallback_feedback(
    exercise: str,
    per_rep_scores: list,
    common_errors: dict,
) -> str:
    """Deterministic fallback if LLM is unavailable."""
    avg = round(float(np.mean(per_rep_scores)) * 100 if per_rep_scores else 50)
    name = exercise.replace("_", " ").title()
    top_error = next(iter(common_errors), None)
    if top_error:
        return (
            f"Good effort on the {name}! Your average form score was {avg}%. "
            f"Pay attention to your {top_error.lower()} — focus on controlled movement "
            f"and maintaining proper alignment throughout each rep."
        )
    return (
        f"Great work on the {name}! Your average form score was {avg}%. "
        f"Keep focusing on controlled movement and full range of motion."
    )
```

**Adapted for binary KIE/KFE schema (D-10 template):**
```python
# RESEARCH.md §Code Examples — Deterministic Feedback Templates
ERROR_FEEDBACK_TEMPLATES = {
    "KIE": {
        True:  "Knees caving inward — {severity} signal detected.",
        False: "Knees inward error not detected.",
    },
    "KFE": {
        True:  "Knees traveling too far forward — {severity} signal detected.",
        False: "Knee forward error not detected.",
    },
}
```

### 5. Per-Connection Session State Pattern
**Source:** `backend/services/form_session.py` lines 115-208 (`FormSession.__init__`)
**Apply to:** New `SquatLiveSession` class (per-WebSocket-connection state)
```python
# form_session.py:115-208 — the per-connection stateful session pattern:
class FormSession:
    """Stateful per-connection session. Thread-safety: each WebSocket has its own instance."""
    def __init__(self, analyzer, ...):
        self.analyzer = analyzer
        # Sliding window buffers
        self._angle_buf: deque = deque(maxlen=WINDOW_SIZE)
        self._joint_buf: deque = deque(maxlen=WINDOW_SIZE)
        # Frame counters
        self._frame_count    = 0
        self._last_inference = 0
        # Rep tracking
        self._rep_count = 0
        # Session history
        self._all_reps: List[dict] = []
        self._start_time = time.time()
```

**Key difference:** `SquatLiveSession` holds `deque(maxlen=LIVE_BUFFER_MAX_FRAMES)` of raw frame arrays (not angle/joint features), because the new model is raw-pixel CNN, not pose-based.

### 6. Module-Level Constants Pattern
**Source:** `backend/services/form_session.py` lines 35-81
**Apply to:** `backend/services/rep_segmenter.py` and `SquatLiveSession`
```python
# form_session.py:35-81 — SCREAMING_SNAKE constants block at module top
WINDOW_SIZE         = 64    # frames
INFERENCE_INTERVAL  = 10    # run model every N frames (~1s at 10fps)
BOUNDARY_THRESHOLD  = 0.55  # boundary probability to count as a rep boundary
MIN_REP_GAP_FRAMES  = 20    # minimum frames between two rep boundaries
```

**New version:**
```python
# rep_segmenter.py and squat_live_session (same convention):
LIVE_REP_WINDOW_FRAMES   = 32      # clip window sent to classifier
LIVE_MIN_GAP_FRAMES      = 45      # ~1.5 s @ 30fps between consecutive window captures
LIVE_BUFFER_MAX_FRAMES   = 90      # rolling buffer depth (~3 s of frames)
```

### 7. `autouse` Fixture Mock Pattern
**Source:** `backend/tests/test_app.py` lines 60-85
**Apply to:** `backend/tests/test_squat_form_api.py`
```python
# test_app.py:60-85 — always mock heavy services; direct app.state assignment
@pytest.fixture(autouse=True)
def mock_services():
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    yield
```

### 8. `from __future__ import annotations` + logger
**Source:** Every backend Python module
**Apply to:** All new modules
```python
from __future__ import annotations
# ...
logger = logging.getLogger(__name__)
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `backend/services/rep_segmenter.py` (algorithm) | utility | transform | No motion-energy or optical-flow segmentation in the codebase. The frame-decode loop analog is in `app.py` but the actual algorithm (frame-difference energy, smoothing, contiguous-region detection) is new. RESEARCH.md §Rep Segmentation provides the design. |
| WS `send_json` conditional (only send when rep fires) | pattern | event-driven | Old WS sends a result on EVERY frame. New WS only sends `rep_result` when the inference trigger fires (buffer + gap check). No analog — this conditional-send pattern is new. |

---

## Critical Anti-Patterns (Documented in RESEARCH.md — Planner Must Reference in Every Plan Action)

1. **`spatial_train` at serve time** — `transforms.py:149-187`. Always use `spatial_val` (`transforms.py:190-215`). `spatial_train` uses random crop (non-deterministic); `spatial_val` uses center crop (deterministic). Wrong choice = B6-class train/serve skew.
2. **`weights=R2Plus1D_18_Weights.KINETICS400_V1` when building the serve model** — `md_finetune.py:96`. Use `weights=None`. MD backbone IS the initialization; Kinetics weights overwrite SSL pretraining.
3. **`strict=False` on the fine-tune `best.pt`** — `md_finetune.py:104`. That uses `strict=False` for the backbone-only `backbone.pt`. For `best.pt` (full fine-tuned model), use `strict=True`.
4. **Calling `classify_clip()` directly in async handler** — `run_in_threadpool` required (RESEARCH §Pitfall 3). ~0.93–2.8 s forward blocks the event loop.
5. **`.half()` on CPU** — confirmed hang on target machine (RESEARCH §Pitfall 4). fp32 only.
6. **Shared mutable state on `SquatFormService`** — `form_analyzer.py:62-63` docstring explains the concurrency bug. `classify_clip()` must be a pure function (inputs in, outputs out).

---

## Metadata

**Analog search scope:** `backend/services/`, `backend/app.py`, `backend/tests/`, `backend/training/aqa/`
**Files scanned:** 12 source files read in full
**Pattern extraction date:** 2026-05-25
