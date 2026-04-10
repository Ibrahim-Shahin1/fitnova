"""
FitNova — FastAPI Server
========================
Single endpoint that chains all 3 pipeline layers:
  Layer 1: Content-Based Filter  (backend/services/content_filter.py)
  Layer 2: NeuMF Re-ranker       (backend/services/neumf_ranker.py)
  Layer 3: LLM Adaptation        (backend/services/llm_adapter.py)

Run with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.services.chat_service import ChatService
from backend.services.llm_adapter import LLMAdapter
from backend.services.recommender import Recommender

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


class ExerciseDetail(BaseModel):
    exercise_name: str
    sets: int
    reps: str
    rest_seconds: int
    coaching_cue: str


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


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness check — returns immediately without touching the services."""
    return {"status": "healthy"}


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
