"""
FitNova — FastAPI Server
========================
Pipeline layers:
  Layer 1: Content-Based Filter  (backend/services/content_filter.py)
  Layer 2: NeuMF Re-ranker       (backend/services/neumf_ranker.py)
  Layer 3: LLM Adaptation        (backend/services/llm_adapter.py)
  Layer 4: Form Detection        (backend/services/squat_form_service.py)

Run with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import io
import logging
from collections import deque
from contextlib import asynccontextmanager

import os

import cv2
import numpy as np

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from backend.services.chat_service import ChatService
from backend.services.llm_adapter import LLMAdapter
from backend.services.recommender import Recommender
from backend.services.coach_service import CoachChatService
from backend.services.rep_segmenter import (
    LiveRepDetector,
    LIVE_BUFFER_MAX_FRAMES,
)

from backend.deps.auth import AuthUser, require_user
from backend.routers.plan import router as plan_router
from backend.routers.coach import router as coach_router
from backend.routers.logs import router as logs_router

logger = logging.getLogger("fitnova")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class UserProfileRequest(BaseModel):
    experience_level: int = Field(
        ..., ge=1, le=3, description="1=Beginner, 2=Intermediate, 3=Advanced"
    )
    workout_type: str = Field(
        ..., pattern=r"^(Strength|Cardio|Yoga|HIIT)$",
        description="One of: Strength, Cardio, Yoga, HIIT"
    )
    session_duration_hours: float = Field(
        ..., gt=0, le=4.0, description="Hours per session (e.g. 1.0, 1.5)"
    )
    workout_frequency: int = Field(
        ..., ge=1, le=7, description="Training days per week"
    )
    age: int = Field(default=30, ge=10, le=100)
    gender: str = Field(default="Male", pattern=r"^(Male|Female)$")
    bmi: float = Field(default=25.0, gt=10, lt=60)
    injuries: list[str] = Field(
        default_factory=list,
        description="Injury area IDs e.g. ['lower_back', 'knees']",
    )
    training_focus: str | None = Field(
        default=None,
        pattern=r"^(bodybuilding|powerbuilding|powerlifting|cardio|general)$",
    )
    years_training: int | None = Field(default=None, ge=0, le=50)
    equipment: list[str] = Field(default_factory=list)


class ExerciseDetail(BaseModel):
    exercise_name: str
    sets: int
    reps: str
    rest_seconds: int
    coaching_cue: str
    media_url: str | None = None


class DayPlan(BaseModel):
    day_number: int
    focus: str
    is_rest_day: bool
    exercises: list[ExerciseDetail]


class GeneratePlanResponse(BaseModel):
    program_id: int
    program_title: str
    similar_user_id: int
    personalization_notes: str
    source: str
    weekly_plan: dict[str, DayPlan]


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str


class UserContext(BaseModel):
    workout_type: str = Field(
        ..., pattern=r"^(Strength|Cardio|Yoga|HIIT)$",
        description="One of: Strength, Cardio, Yoga, HIIT"
    )
    age: int = Field(default=30, ge=10, le=100)
    gender: str = Field(default="Male", pattern=r"^(Male|Female)$")
    bmi: float = Field(default=25.0, gt=10, lt=60)
    injuries: list[str] = Field(
        default_factory=list,
        description="Injury area IDs e.g. ['lower_back', 'knees']",
    )
    training_focus: str | None = Field(
        default=None,
        pattern=r"^(bodybuilding|powerbuilding|powerlifting|cardio|general)$",
    )


class ChatRequest(BaseModel):
    conversation: list[ChatMessage] = Field(
        default_factory=list,
        description="Chat history so far (can be empty for initial greeting)"
    )
    user_context: UserContext


class ChatResponse(BaseModel):
    status: str = Field(..., description='"continue" or "ready"')
    message: str = Field(..., description="Assistant's reply")
    extracted: dict = Field(
        default_factory=dict,
        description="Extracted fields so far (experience_level, session_duration_hours, workout_frequency)"
    )


# ── Form detection models (D-05 schema) ───────────────────────────────────────

class ErrorDetection(BaseModel):
    type: str = Field(..., pattern=r"^(KIE|KFE)$")
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity_word: str = Field(..., pattern=r"^(none|possible|moderate|strong)$")
    intervals: list[list[float]] = Field(default_factory=list)


class RepResult(BaseModel):
    exercise: str
    errors: list[ErrorDetection]


class ModelView(BaseModel):
    """The actual 112x112 frames the model analyzed (kneeaware crop, denormalized),
    as base64-encoded JPEGs — 'what the model sees' for the upload replay viz."""
    frames: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    exercise: str
    total_reps: int
    reps: list[RepResult]
    duration_s: float = 0.0
    model_view: ModelView | None = None


class SessionSummary(BaseModel):
    type: str = "session_summary"
    exercise: str
    total_reps: int
    rep_results: list[RepResult]
    session_feedback: str


# ─────────────────────────────────────────────────────────────────────────────
# Live WebSocket session — SquatLiveSession (Plan 04, API-04)
# ─────────────────────────────────────────────────────────────────────────────

# Deterministic feedback templates (D-10, NO LLM).
_ERROR_FEEDBACK = {
    "KIE": {
        True: "Knees caving inward — {severity} signal detected.",
        False: "Knees inward error not detected.",
    },
    "KFE": {
        True: "Knees traveling too far forward — {severity} signal detected.",
        False: "Knee forward error not detected.",
    },
}


class SquatLiveSession:
    """Per-WebSocket-connection session for live squat form analysis (API-04).

    Buffers incoming base64-decoded JPEG frames in a rolling deque, fires the
    LiveWindowTrigger sliding-window gate, classifies each triggered window
    single-seed (low-latency — D-08), and accumulates per-rep results.

    Thread safety: NOT thread-safe.  Each WebSocket connection must use its own
    SquatLiveSession instance (constructed fresh in the WS handler).
    Model is read-only — no shared mutable inference state (T-05-15).

    Buffer cap: deque(maxlen=LIVE_BUFFER_MAX_FRAMES=90) — evicts old frames,
    caps per-session memory at ~3 s of frames (T-05-14).
    """

    def __init__(self, service) -> None:
        self._service = service
        # Raw RGB np.ndarray frames from cv2; LIVE_BUFFER_MAX_FRAMES caps memory (T-05-14).
        self._buffer: deque = deque(maxlen=LIVE_BUFFER_MAX_FRAMES)
        self._detector = LiveRepDetector()
        self._rep_results: list[dict] = []
        self._rep_number: int = 0
        self._pending_span: int = 0

    @property
    def next_rep_number(self) -> int:
        """Rep number the next classify_pending() will assign (for the 'analyzing' msg)."""
        return self._rep_number + 1

    def push_frame(self, jpeg_bytes: bytes) -> bool:
        """Decode + buffer a frame and run rep-end detection (fast — no inference).

        Returns True when a rep just ended (caller should send 'analyzing' then await
        classify_pending()); False otherwise.

        Security (T-05-10): a malformed JPEG (cv2.imdecode -> None) is dropped and
        False is returned — the session stays alive.
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # model trained on RGB
        self._buffer.append(frame_rgb)
        span = self._detector.push(frame_rgb)
        if span:
            self._pending_span = min(span, len(self._buffer))
            return True
        return False

    async def classify_pending(self) -> dict:
        """Classify the just-finished rep's frames and return a rep_result dict.

        Samples the rep's frame span to the 32-frame model window, then runs the FULL
        ensemble off the event loop (no quality tradeoff; ~1.5 s on CPU with ONNX).
        SQUAT_INFERENCE_SEEDS=1 drops to single-seed (~0.5 s) without a rebuild.
        """
        from backend.training.aqa.datasets.transforms import uniform_sample_indices

        frames = (
            list(self._buffer)[-self._pending_span:]
            if self._pending_span
            else list(self._buffer)
        )
        if not frames:
            self._rep_number += 1
            return {"type": "rep_result", "rep_number": self._rep_number,
                    "exercise": "squat", "errors": []}

        idx = uniform_sample_indices(len(frames), 32, jitter=0)
        stacked = np.stack([frames[int(i)] for i in idx])   # [32, H, W, 3] HWC uint8
        frames_tchw = stacked.transpose(0, 3, 1, 2)          # [32, 3, H, W]

        self._rep_number += 1
        result = await self._service.classify_clip_async(frames_tchw)  # ensemble default
        self._rep_results.append(result)
        return {"type": "rep_result", "rep_number": self._rep_number, **result}

    def end_session(self) -> dict:
        """Finalize the session and return a deterministic session_summary dict (D-10).

        session_feedback is assembled from per-rep KIE/KFE detections using
        _ERROR_FEEDBACK templates — no LLM call (D-10).

        Returns:
            Dict matching the SessionSummary D-05 shape:
              type, exercise, total_reps, rep_results, session_feedback.
        """
        n_reps = self._rep_number
        feedback = self._build_feedback()
        return {
            "type": "session_summary",
            "exercise": "squat",
            "total_reps": n_reps,
            "rep_results": self._rep_results,
            "session_feedback": feedback,
        }

    def _build_feedback(self) -> str:
        """Build deterministic session feedback from per-rep detections (D-10, NO LLM)."""
        if not self._rep_results:
            return (
                "Session complete. No reps were classified — try recording for longer "
                "or ensure your full body is visible."
            )

        n = len(self._rep_results)

        # Count reps where each error type was detected.
        kie_count = sum(
            1 for r in self._rep_results
            for e in r.get("errors", [])
            if e.get("type") == "KIE" and e.get("detected")
        )
        kfe_count = sum(
            1 for r in self._rep_results
            for e in r.get("errors", [])
            if e.get("type") == "KFE" and e.get("detected")
        )

        lines: list[str] = [f"Session complete — {n} rep(s) analyzed."]

        if kie_count > 0:
            lines.append(
                f"Knees caving inward detected on {kie_count} of {n} rep(s). "
                "Focus on pushing your knees out in line with your toes throughout the squat."
            )
        else:
            lines.append("Good knee tracking — no inward caving detected.")

        if kfe_count > 0:
            lines.append(
                f"Knees traveling too far forward detected on {kfe_count} of {n} rep(s). "
                "Try initiating the squat by pushing your hips back first."
            )
        else:
            lines.append("Good depth control — knee forward error not detected.")

        return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading recommendation engine (Layers 1+2)...")
    app.state.recommender = Recommender()
    logger.info("Loading LLM adapter (Layer 3)...")
    app.state.llm_adapter = LLMAdapter()
    logger.info("Loading chat service...")
    app.state.chat_service = ChatService()
    logger.info("Loading coach service...")
    app.state.coach_service = CoachChatService()
    logger.info("Loading squat form service (Layer 4 — PyTorch R(2+1)D-18 ensemble)...")
    from backend.services.squat_form_service import SquatFormService
    _squat_dir = os.environ.get("FITNOVA_MODEL_DIR") or os.path.join(
        os.path.dirname(__file__), "models", "form_model_squat_md"
    )
    app.state.squat_form_service = SquatFormService(model_dir=_squat_dir)
    logger.info(
        "SquatFormService ready: model_ready=%s, seeds_loaded=%d, onnx_enabled=%s",
        app.state.squat_form_service.model_ready,
        len(app.state.squat_form_service._models),
        app.state.squat_form_service.onnx_enabled,
    )
    logger.info("FitNova backend ready.")
    yield
    logger.info("Shutting down FitNova backend.")


app = FastAPI(
    title="FitNova API",
    version="1.0.0",
    description="AI-powered personalised fitness plan generator",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(plan_router)
app.include_router(coach_router)
app.include_router(logs_router)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    """Liveness check — reports PyTorch squat model status and per-head thresholds."""
    svc = getattr(request.app.state, "squat_form_service", None)
    return {
        "status": "healthy",
        "squat_model_ready": bool(svc.model_ready) if svc else False,
        "seeds_loaded":      len(svc._models) if svc else 0,
        "onnx_enabled":      bool(getattr(svc, "onnx_enabled", False)) if svc else False,
        "kie_threshold":     svc.kie_threshold if svc else None,
        "kfe_threshold":     svc.kfe_threshold if svc else None,
    }


@app.get("/api/exercises", tags=["Form"])
def list_exercises():
    """Return the full exercise metadata SSOT.

    Flutter fetches this on app start to render the exercise picker + the
    per-exercise camera-setup guidelines screen.
    """
    from backend.config.exercises import all_exercises
    return all_exercises()


@app.get("/api/me", tags=["Auth"])
async def whoami(user: AuthUser = Depends(require_user)):
    """Debug endpoint — echoes the authenticated user's id + email.

    Returns 401 if the bearer token is missing or invalid. Used to verify the
    Supabase JWT verification path end-to-end before wiring the real per-user
    endpoints (plan persistence, workout logs, coach chat).
    """
    return {"user_id": str(user.id), "email": user.email}


@app.post("/generate-plan", response_model=GeneratePlanResponse, tags=["Plans"])
async def generate_plan(profile: UserProfileRequest, request: Request):
    """
    Full pipeline: user profile → recommended program → personalised 7-day plan.

    1. Layer 1+2 (Recommender): content filter → NeuMF re-ranking → top program
    2. Layer 3  (LLM Adapter):  GPT-4o-mini adapts the program into a weekly plan
    """
    user_profile = profile.model_dump()

    # ── Layer 1+2: recommendation ─────────────────────────────────────────
    try:
        rec_result = request.app.state.recommender.recommend(user_profile)
    except Exception as exc:
        logger.exception("Recommendation engine failed")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}",
        )

    program_id: int = rec_result["program_id"]
    similar_user_id: int = rec_result["similar_user_id"]

    # ── Layer 3: LLM adaptation ───────────────────────────────────────────
    try:
        plan_result = request.app.state.llm_adapter.generate_plan(
            program_id, user_profile
        )
    except Exception as exc:
        logger.exception("Plan generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Plan generation failed: {exc}",
        )

    return GeneratePlanResponse(
        program_id=program_id,
        program_title=plan_result["program_title"],
        similar_user_id=similar_user_id,
        personalization_notes=plan_result["personalization_notes"],
        source=plan_result["source"],
        weekly_plan=plan_result["plan"],
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest, request: Request):
    """
    Conversational intake: the AI asks natural questions to gather the user's
    experience level, session duration, and workout frequency.

    Send an empty ``conversation`` list to get the initial greeting.
    Keep sending the full conversation history with each turn.
    When ``status`` is ``"ready"``, all fields are gathered — call
    ``/generate-plan`` with the complete profile.
    """
    try:
        result = request.app.state.chat_service.process_message(
            conversation=[m.model_dump() for m in req.conversation],
            user_context=req.user_context.model_dump(),
        )
    except Exception as exc:
        logger.exception("Chat service failed")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        )

    return ChatResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# Form Detection — WebSocket (live streaming) — redefined in Plan 04
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/form-session")
async def form_session_ws(websocket: WebSocket):
    """Real-time squat form analysis via WebSocket (API-04).

    Protocol:
      1. Client sends {"type": "start_session", "exercise_hint": "squat"}
         Server replies {"type": "session_started"}

      2. Client streams frames:
         {"type": "frame", "data": "<base64-JPEG>", "timestamp_ms": <int>}
         Motion-energy rep-end detection runs per frame (no inference). When a rep
         ends the server replies {"type": "analyzing", "rep_number": N} immediately,
         then {"type": "rep_result", "rep_number": N, "exercise": "squat",
               "errors": [...]} once classification (~1.5s) completes. Most frames
         get no reply (buffered only).

      3. Client sends {"type": "end_session"}
         Server replies {"type": "session_summary", "exercise": "squat",
                         "total_reps": N, "rep_results": [...], "session_feedback": "..."}
         and closes the loop.

    Per-connection isolation: a fresh SquatLiveSession is constructed for each
    accepted WebSocket (T-05-15 — no shared mutable state).
    Malformed JPEG frames are silently dropped; the session continues (T-05-10).
    """
    await websocket.accept()
    session: SquatLiveSession | None = None
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_session":
                # Construct a fresh per-connection session (T-05-15).
                session = SquatLiveSession(websocket.app.state.squat_form_service)
                await websocket.send_json({"type": "session_started"})

            elif msg_type == "frame":
                if session is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Send start_session before frame"}
                    )
                    continue
                jpeg_bytes = base64.b64decode(data["data"])
                # push_frame is fast (decode + motion rep-end detection, no inference).
                if session.push_frame(jpeg_bytes):
                    # A rep just ended — tell the UI we're analyzing, then classify it.
                    await websocket.send_json(
                        {"type": "analyzing", "rep_number": session.next_rep_number}
                    )
                    result = await session.classify_pending()
                    await websocket.send_json(result)

            elif msg_type == "end_session":
                if session is None:
                    await websocket.send_json(
                        {"type": "error", "message": "No active session to end"}
                    )
                else:
                    await websocket.send_json(session.end_session())
                break

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Form Detection — REST (video upload)
# ─────────────────────────────────────────────────────────────────────────────

# T-05-07: Reject uploads > 100 MB before buffering body into memory.
MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024  # 100 MB


def _model_view_frames(frames_tchw_uint8, n_show: int = 6) -> list[str]:
    """Base64 JPEGs of the actual 112x112 frames the model analyzes (kneeaware crop,
    denormalized) — the literal 'eyes of the model' for the replay. Independent of
    whether the model loaded (it's just the preprocessing)."""
    import torch
    from backend.services.clip_decode import kneeaware_spatial_val
    from backend.training.aqa.datasets.transforms import KINETICS_MEAN, KINETICS_STD

    clip = kneeaware_spatial_val(torch.from_numpy(frames_tchw_uint8))  # [3,T,112,112] norm
    mean = torch.as_tensor(KINETICS_MEAN).view(-1, 1, 1, 1)
    std = torch.as_tensor(KINETICS_STD).view(-1, 1, 1, 1)
    vis = ((clip * std + mean).clamp(0, 1) * 255).to(torch.uint8)      # [3,T,112,112]
    n_t = vis.shape[1]
    picks = sorted({int(round(i)) for i in np.linspace(0, n_t - 1, min(n_show, n_t))})
    out: list[str] = []
    for t in picks:
        rgb = vis[:, t].permute(1, 2, 0).numpy()                       # [112,112,3] RGB
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            out.append(base64.b64encode(buf.tobytes()).decode())
    return out


@app.post("/analyze-form-video", tags=["Form"])
async def analyze_form_video(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form(default="squat"),
):
    """
    Upload a pre-recorded SINGLE-REP squat video for form analysis.
    The clip is treated as one rep (the model's training distribution — no
    segmentation), classified through the PyTorch R(2+1)D-18 ensemble, and returned
    with the exact KIE/KFE scores + the 112x112 frames the model actually analyzed.
    """
    import tempfile

    # T-05-09: exercise allowlist — Phase 5 is squat-only.
    if exercise not in {"squat"}:
        raise HTTPException(status_code=400, detail="Only 'squat' supported in this phase")

    # T-05-07: content-length pre-check (fast-path; defends well-behaved clients).
    content_length_hdr = request.headers.get("content-length")
    if content_length_hdr is not None:
        try:
            if int(content_length_hdr) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Video too large")
        except ValueError:
            pass  # non-numeric header — proceed to chunked read cap

    # T-05-08: tmp_path None-guard so finally: never NameErrors on early exit.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            total_read = 0
            # T-05-07: chunked read cap — defends missing/forged content-length.
            while True:
                chunk = await file.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video too large")
                tmp.write(chunk)

        from backend.services.clip_decode import decode_clip_cv2, get_frame_count_and_fps
        from backend.training.aqa.datasets.transforms import uniform_sample_indices

        total_frames, fps = get_frame_count_and_fps(tmp_path)
        svc = request.app.state.squat_form_service

        # SINGLE-REP: the uploaded clip IS one rep (the model's training distribution).
        # Classify the whole clip directly — no segmentation, no rep-counting.
        try:
            idx = uniform_sample_indices(total_frames, 32, jitter=0).clamp(
                0, max(total_frames - 1, 0)
            )
            frames = decode_clip_cv2(tmp_path, idx)          # uint8 [32,3,H,W] RGB
            result = await svc.classify_clip_async(frames.numpy())
            duration_s = total_frames / fps if fps else 0.0

            errors: list[ErrorDetection] = []
            for err in result.get("errors", []):
                intervals = [[0.0, round(duration_s, 2)]] if err.get("detected") else []
                errors.append(ErrorDetection(
                    type=err["type"],
                    detected=err["detected"],
                    confidence=err["confidence"],
                    severity_word=err["severity_word"],
                    intervals=intervals,
                ))
            model_view = ModelView(frames=_model_view_frames(frames.numpy()))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Form analysis inference failed")
            raise HTTPException(status_code=500, detail=f"Form analysis failed: {exc}")

        return UploadResponse(
            exercise="squat",
            total_reps=1,
            reps=[RepResult(exercise="squat", errors=errors)],
            duration_s=round(duration_s, 2),
            model_view=model_view,
        )

    finally:
        # T-05-08: clean up temp file; guard against None if NamedTemporaryFile raised.
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
