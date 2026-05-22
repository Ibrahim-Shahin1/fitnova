"""
Plan router — authenticated plan generation + retrieval.

POST /api/plan/generate runs the existing recommender + LLM adapter pipeline
(unchanged) and persists the result via plan_repo, scoped to the authenticated
user. GET /api/plan/active returns the user's current plan.

Defined as its own request model (not importing app.py's UserProfileRequest) to
avoid a circular import — app.py imports this router.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.db.repositories import plan_repo, profile_repo
from backend.deps.auth import AuthUser, require_user

logger = logging.getLogger("fitnova.plan")

router = APIRouter(prefix="/api/plan", tags=["Plan"])


class PlanGenerateRequest(BaseModel):
    experience_level: int = Field(..., ge=1, le=3)
    workout_type: str = Field(
        default="Strength", pattern=r"^(Strength|Cardio|Yoga|HIIT)$"
    )
    session_duration_hours: float = Field(..., gt=0, le=4.0)
    workout_frequency: int = Field(..., ge=1, le=7)
    age: int = Field(default=30, ge=10, le=100)
    gender: str = Field(default="Male", pattern=r"^(Male|Female)$")
    bmi: float = Field(default=25.0, gt=10, lt=60)
    injuries: list[str] = Field(default_factory=list)
    training_focus: str | None = Field(
        default=None, pattern=r"^(powerbuilding|powerlifting|hypertrophy|general)$"
    )
    years_training: int | None = Field(default=None, ge=0, le=50)
    equipment: list[str] = Field(default_factory=list)


@router.post("/generate")
async def generate_and_persist(
    req: PlanGenerateRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
):
    """Recommend → LLM-adapt → persist. Returns the saved plan id + the plan."""
    profile = req.model_dump()

    try:
        rec = request.app.state.recommender.recommend(profile)
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}")

    try:
        plan_result = request.app.state.llm_adapter.generate_plan(
            rec["program_id"], profile, rec.get("content_candidates")
        )
    except Exception as exc:
        logger.exception("Plan generation failed")
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {exc}")

    try:
        plan_id = plan_repo.insert_plan(user.id, rec, plan_result)
    except Exception as exc:
        logger.exception("Plan persistence failed")
        raise HTTPException(status_code=500, detail=f"Saving plan failed: {exc}")

    return {
        "plan_id": plan_id,
        "program_id": rec.get("program_id"),
        "program_title": plan_result.get("program_title"),
        "personalization_notes": plan_result.get("personalization_notes"),
        "source": plan_result.get("source"),
        "quality_report": plan_result.get("quality_report"),
        "weekly_plan": plan_result.get("plan", {}),
    }


@router.get("/active")
async def get_active(user: AuthUser = Depends(require_user)):
    """Return the user's active plan (with days + exercises), or {plan: null}."""
    try:
        plan = plan_repo.fetch_active(user.id)
    except Exception as exc:
        logger.exception("Fetching active plan failed")
        raise HTTPException(status_code=500, detail=f"Fetching plan failed: {exc}")
    return {"plan": plan}


class CompletionRequest(BaseModel):
    completed: bool


@router.patch("/exercises/{exercise_id}/complete")
async def complete_exercise(
    exercise_id: UUID,
    req: CompletionRequest,
    user: AuthUser = Depends(require_user),
):
    """Mark a plan exercise complete/incomplete (RLS-safe: cross-user → 404)."""
    try:
        row = plan_repo.set_exercise_completed(user.id, exercise_id, req.completed)
    except Exception as exc:
        logger.exception("Exercise completion failed")
        raise HTTPException(status_code=500, detail=f"Completion failed: {exc}")
    if row is None:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return {"exercise": row}


@router.patch("/days/{day_id}/complete")
async def complete_day(
    day_id: UUID,
    req: CompletionRequest,
    user: AuthUser = Depends(require_user),
):
    """Mark a plan day complete/incomplete (RLS-safe: cross-user → 404)."""
    try:
        row = plan_repo.set_day_completed(user.id, day_id, req.completed)
    except Exception as exc:
        logger.exception("Day completion failed")
        raise HTTPException(status_code=500, detail=f"Completion failed: {exc}")
    if row is None:
        raise HTTPException(status_code=404, detail="Day not found")
    return {"day": row}


class PlanStreamRequest(BaseModel):
    """Overrides gathered by the coach; missing fields fall back to the profile."""
    workout_type: str | None = None
    experience_level: int | None = Field(default=None, ge=1, le=3)
    workout_frequency: int | None = Field(default=None, ge=1, le=7)
    session_duration_hours: float | None = Field(default=None, gt=0, le=4.0)
    injuries: list[str] | None = None
    equipment: list[str] | None = None
    notes: str | None = None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/generate/stream")
async def generate_stream(
    req: PlanStreamRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
):
    """Run the CrewAI pipeline and stream per-agent progress as SSE, persisting
    the final plan. Drives the live 4-agent screen. Events:
    started → agent(running/done) … → done{plan_id} | error{error}."""
    from backend.services import plan_crew
    from backend.services.coach_tools import _build_recommender_profile

    rec_engine = request.app.state.recommender
    adapter = request.app.state.llm_adapter
    uid = user.id
    overrides = req.model_dump(exclude_none=True)

    def stream():
        try:
            base = profile_repo.get_profile(uid) or {}
            profile = _build_recommender_profile(base, overrides)
            rec = rec_engine.recommend(profile)
            program = adapter._get_program(rec["program_id"])
            yield _sse({"event": "started"})
            for ev in plan_crew.build_plan_streamed(
                    adapter, program, profile, rec.get("content_candidates")):
                if ev.get("event") == "plan":
                    try:
                        plan_id = plan_repo.insert_plan(uid, rec, ev)
                    except Exception as exc:
                        logger.exception("Persisting streamed plan failed")
                        yield _sse({"event": "error", "error": f"save failed: {exc}"})
                        return
                    yield _sse({"event": "done", "plan_id": plan_id,
                                "program_title": ev.get("program_title"),
                                "quality_report": ev.get("quality_report")})
                else:
                    yield _sse(ev)
        except Exception as exc:
            logger.exception("Streamed generation failed")
            yield _sse({"event": "error", "error": str(exc)[:300]})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
