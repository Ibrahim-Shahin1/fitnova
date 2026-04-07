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


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading recommendation engine (Layers 1+2)...")
    app.state.recommender = Recommender()
    logger.info("Loading LLM adapter (Layer 3)...")
    app.state.llm_adapter = LLMAdapter()
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
