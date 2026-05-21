"""
FitNova — FastAPI Server
========================
Pipeline layers:
  Layer 1: Content-Based Filter  (backend/services/content_filter.py)
  Layer 2: NeuMF Re-ranker       (backend/services/neumf_ranker.py)
  Layer 3: LLM Adaptation        (backend/services/llm_adapter.py)
  Layer 4: Form Detection        (backend/services/form_analyzer.py)

Run with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager

import os

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services.chat_service import ChatService
from backend.services.llm_adapter import LLMAdapter
from backend.services.recommender import Recommender
from backend.services.form_analyzer import FormAnalyzer
from backend.services.form_session import FormSession

from backend.deps.auth import AuthUser, require_user

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
        pattern=r"^(powerbuilding|powerlifting|hypertrophy|general)$",
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
        pattern=r"^(powerbuilding|powerlifting|hypertrophy|general)$",
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


# ── Form detection models ──────────────────────────────────────────────────────

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
    logger.info("Loading form analyzer (Layer 4)...")
    # Auto-detection priority: v6 -> v5.2 -> v4. Override with FITNOVA_MODEL_DIR
    # env var. The form_analyzer itself does file-presence detection inside the
    # chosen dir; this just picks WHICH dir to scan.
    _override = os.environ.get("FITNOVA_MODEL_DIR")
    if _override:
        _model_dir = _override
        logger.info("FITNOVA_MODEL_DIR override -> %s", _model_dir)
    else:
        _v6_dir   = os.path.join(os.path.dirname(__file__), "models", "form_model_v6")
        _v5_2_dir = os.path.join(os.path.dirname(__file__), "models", "form_model_v5_2")
        _v4_dir   = os.path.join(os.path.dirname(__file__), "models", "form_model")
        if os.path.isfile(os.path.join(_v6_dir, "v6_supervised.weights.h5")):
            _model_dir = _v6_dir
            logger.info("v6 weights detected -> using %s", _model_dir)
        elif os.path.isfile(os.path.join(_v5_2_dir, "v5_2_supervised.weights.h5")):
            _model_dir = _v5_2_dir
            logger.info("v5.2 weights detected -> using %s", _model_dir)
        else:
            _model_dir = _v4_dir
            logger.info("falling back to v4 -> using %s", _model_dir)

    if not os.path.isdir(_model_dir):
        logger.warning(
            "model_dir=%s does not exist; falling back to legacy v4 form_model/",
            _model_dir,
        )
        _model_dir = os.path.join(os.path.dirname(__file__), "models", "form_model")
    app.state.form_analyzer = FormAnalyzer(model_dir=_model_dir)
    logger.info("FitNova backend ready.")
    yield
    logger.info("Shutting down FitNova backend.")
    if hasattr(app.state, "form_analyzer") and app.state.form_analyzer.mp_pose:
        app.state.form_analyzer.mp_pose.close()


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


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    """Liveness check — also exposes the loaded model version so callers can
    verify v6/v5.2/v4 detection without parsing boot logs."""
    analyzer = getattr(request.app.state, "form_analyzer", None)
    model_info = {
        "model_version": getattr(analyzer, "_model_version", None) if analyzer else None,
        "model_ready":   bool(analyzer.model_ready) if analyzer else False,
        "model_dir":     getattr(analyzer, "model_dir", None) if analyzer else None,
        "n_exercises":   len(getattr(analyzer, "_exercise_labels", {})) if analyzer else 0,
    }
    return {"status": "healthy", **model_info}


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
# Form Detection — WebSocket (live streaming)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/form-session")
async def form_session_ws(websocket: WebSocket):
    """
    Real-time form analysis via WebSocket.

    Protocol (client → server):
        {"type": "start_session", "exercise_hint": "squat"}
        {"type": "frame", "data": "<base64_jpeg>", "timestamp_ms": 12345}
        {"type": "end_session"}

    Protocol (server → client):
        per-frame: FormFrameResult JSON
        on end:    FormSessionSummary JSON
    """
    await websocket.accept()
    session: FormSession | None = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_session":
                # Prefer new key; fall back to legacy exercise_hint
                selected = data.get("selected_exercise") or data.get("exercise_hint")
                session = FormSession(
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
                await websocket.send_json(result)

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


# ─────────────────────────────────────────────────────────────────────────────
# Form Detection — REST (video upload)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze-form-video", tags=["Form"])
async def analyze_form_video(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form(default="unknown"),
):
    """
    Upload a pre-recorded exercise video for form analysis.
    Returns the same FormSessionSummary as the WebSocket end_session response.
    """
    import tempfile, cv2

    analyzer: FormAnalyzer = request.app.state.form_analyzer
    session = FormSession(analyzer=analyzer, selected_exercise=exercise)

    # Write uploaded file to temp disk location so OpenCV can read it
    contents = await file.read()
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    timeline: list[dict] = []
    try:
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
                out = session.add_frame(jpeg.tobytes(), ts_ms)
                # Capture per-frame state so the Flutter replay screen can
                # overlay the skeleton + joint errors at each video timestamp.
                # Only record frames where MediaPipe got a pose (raw or
                # interpolated); skip "no_pose" frames to keep payload tight.
                if out.get("status") in ("ok", "pose_interpolated"):
                    timeline.append({
                        "timestamp_ms":  ts_ms,
                        "landmarks":     out.get("landmarks"),
                        "joint_errors":  out.get("joint_errors"),
                        "quality_score": out.get("quality_score"),
                        "rep_count":     out.get("rep_count"),
                    })
            frame_idx += 1

        cap.release()
    finally:
        os.unlink(tmp_path)

    summary = session.end_session()
    summary["timeline"] = timeline
    return summary
