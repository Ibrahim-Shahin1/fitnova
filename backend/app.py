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
    LiveWindowTrigger,
    LIVE_REP_WINDOW_FRAMES,
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


class UploadResponse(BaseModel):
    exercise: str
    total_reps: int
    reps: list[RepResult]


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
        self._trigger = LiveWindowTrigger()
        self._rep_results: list[dict] = []
        self._rep_number: int = 0

    async def add_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> dict | None:
        """Decode a JPEG, buffer it, and classify if the sliding-window trigger fires.

        Args:
            jpeg_bytes:    raw JPEG bytes (decoded from base64 by the WS handler).
            timestamp_ms:  client-side timestamp in milliseconds (unused internally
                           but accepted for protocol compatibility).

        Returns:
            A rep_result dict when the trigger fires (classification complete), or
            None when the trigger has not fired (frame buffered only).

        Security (T-05-10):
            If cv2.imdecode returns None (malformed JPEG), the frame is silently
            dropped and None is returned — the session stays alive.
        """
        # T-05-10: malformed-JPEG guard — never pass None into the model.
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # BGR→RGB conversion before buffering (model trained on RGB).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._buffer.append(frame_rgb)

        if self._trigger.should_fire(len(self._buffer)):
            # Stack the last LIVE_REP_WINDOW_FRAMES frames: [32, H, W, 3] HWC.
            window_frames = list(self._buffer)[-LIVE_REP_WINDOW_FRAMES:]
            stacked = np.stack(window_frames)  # [32, H, W, 3] HWC uint8

            # Permute HWC → TCHW: [32, H, W, 3] → [32, 3, H, W].
            # classify_clip_async (kneeaware_spatial_val path) expects [T, C, H, W].
            import torch  # lazy import — torch is heavy; keep module-level imports slim
            frames_tchw = torch.from_numpy(stacked).permute(0, 3, 1, 2).numpy()

            self._rep_number += 1
            # Single-seed for live latency (D-08 — ~0.93 s; T-05-11 offloaded to thread).
            result = await self._service.classify_clip_async(frames_tchw, n_seeds=1)
            self._rep_results.append(result)
            return {"type": "rep_result", "rep_number": self._rep_number, **result}

        return None

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
        "SquatFormService ready: model_ready=%s, seeds_loaded=%d",
        app.state.squat_form_service.model_ready,
        len(app.state.squat_form_service._models),
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
         Server replies {"type": "rep_result", "rep_number": N, "exercise": "squat",
                         "errors": [...]} ONLY when the sliding-window trigger fires
         (i.e. NOT on every frame — conditional-send).

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
                result = await session.add_frame(
                    jpeg_bytes, int(data.get("timestamp_ms", 0))
                )
                # Conditional-send: emit rep_result ONLY when the trigger fired
                # (add_frame returns None on buffered-only frames).
                if result is not None:
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


@app.post("/analyze-form-video", tags=["Form"])
async def analyze_form_video(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form(default="squat"),
):
    """
    Upload a pre-recorded squat video for form analysis.
    Segments reps by motion energy, classifies each rep through the PyTorch
    R(2+1)D-18 ensemble, and returns the D-05 binary+timing schema.
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
        from backend.services.rep_segmenter import segment_reps_by_motion_energy
        from backend.training.aqa.datasets.transforms import uniform_sample_indices

        total_frames, fps = get_frame_count_and_fps(tmp_path)

        svc = request.app.state.squat_form_service

        # If model is not loaded, classify_clip_async returns the neutral dict (200, not 500).
        rep_intervals = segment_reps_by_motion_energy(tmp_path)

        reps: list[RepResult] = []
        try:
            for start, end in rep_intervals:
                span = end - start + 1
                raw_idx = uniform_sample_indices(span, 32, jitter=0)
                # Offset into the full clip and clamp to valid range.
                idx = (raw_idx + start).clamp(0, max(total_frames - 1, 0))
                frames = decode_clip_cv2(tmp_path, idx)  # uint8 [32,3,H,W] RGB tensor
                result = await svc.classify_clip_async(frames.numpy())

                # Attach timing intervals to each detected error.
                t_start = start / fps
                t_end = (end + 1) / fps
                errors: list[ErrorDetection] = []
                for err in result.get("errors", []):
                    intervals = [[t_start, t_end]] if err.get("detected") else []
                    errors.append(ErrorDetection(
                        type=err["type"],
                        detected=err["detected"],
                        confidence=err["confidence"],
                        severity_word=err["severity_word"],
                        intervals=intervals,
                    ))
                reps.append(RepResult(exercise="squat", errors=errors))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Form analysis inference failed")
            raise HTTPException(status_code=500, detail=f"Form analysis failed: {exc}")

        return UploadResponse(exercise="squat", total_reps=len(rep_intervals), reps=reps)

    finally:
        # T-05-08: clean up temp file; guard against None if NamedTemporaryFile raised.
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
