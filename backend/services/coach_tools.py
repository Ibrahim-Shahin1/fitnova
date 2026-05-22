"""
Coach tools — the function-calling tools the AI coach can invoke.

Each handler returns a JSON-safe dict (primitives/lists only — no UUIDs or
datetimes) so it can be serialized straight into the LLM tool-result message.
Handlers never raise out to the caller; failures come back as {"error": ...}
so the coach can explain them conversationally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from backend.db.repositories import plan_repo, profile_repo

logger = logging.getLogger("fitnova.coach.tools")


@dataclass
class ToolContext:
    """Everything a tool handler needs: the authed user + the ML services."""
    user_id: UUID
    recommender: object
    llm_adapter: object


# ── OpenAI function-calling schemas ──────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Read the user's saved profile (experience level 1-3, "
            "training focus, weekly frequency, session length, body stats, "
            "injuries, equipment). Call this before generating a plan so you "
            "know their baseline.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_plan",
            "description": "Read the user's current active 7-day plan (focus per "
            "day + exercises). Use when the user asks about their plan or before "
            "discussing/editing it.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_workout_plan",
            "description": "Generate and save a new 7-day plan for the user. This "
            "REPLACES any existing active plan, so confirm with the user first. "
            "Pull defaults from their profile; pass overrides only for what they "
            "tell you in chat (e.g. injuries to work around, available equipment).",
            "parameters": {
                "type": "object",
                "properties": {
                    "workout_type": {
                        "type": "string",
                        "enum": ["Strength", "Cardio", "Yoga", "HIIT"],
                        "description": "Primary training style. Default Strength "
                        "for powerbuilding/powerlifting/hypertrophy focuses.",
                    },
                    "experience_level": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "1=Beginner, 2=Intermediate, 3=Advanced. "
                        "Override the profile only if the user states it in chat.",
                    },
                    "workout_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "description": "Training days per week. Override the profile "
                        "only if the user states it in chat.",
                    },
                    "session_duration_hours": {
                        "type": "number",
                        "description": "Hours per session (e.g. 1.5). Override the "
                        "profile only if the user states it in chat.",
                    },
                    "injuries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Injury areas to work around (e.g. "
                        "['lower_back','shoulders']). Overrides the profile if given.",
                    },
                    "equipment": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Available equipment (e.g. "
                        "['barbell','dumbbells']). Overrides the profile if given.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any extra preferences the user mentioned "
                        "(e.g. 'prioritise deadlift', 'no overhead pressing').",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Handlers ─────────────────────────────────────────────────────────────────

_PROFILE_FIELDS = (
    "display_name", "age", "gender", "height_cm", "weight_kg", "training_focus",
    "experience_level", "workout_frequency", "session_duration_hours",
    "injuries", "equipment", "years_training",
)


def get_user_profile(ctx: ToolContext, args: dict) -> dict:
    p = profile_repo.get_profile(ctx.user_id)
    if not p:
        return {"error": "no_profile"}
    out = {k: p.get(k) for k in _PROFILE_FIELDS}
    # numeric/decimal → float for JSON safety
    for k in ("height_cm", "weight_kg", "session_duration_hours"):
        if out.get(k) is not None:
            out[k] = float(out[k])
    return out


def get_active_plan(ctx: ToolContext, args: dict) -> dict:
    plan = plan_repo.fetch_active(ctx.user_id)
    if not plan:
        return {"error": "no_active_plan"}
    return {
        "program_title": plan.get("program_title"),
        "source": plan.get("source"),
        "days": [
            {
                "day_number": d.get("day_number"),
                "focus": d.get("focus"),
                "is_rest_day": d.get("is_rest_day"),
                "exercises": [
                    {
                        "exercise_name": e.get("exercise_name"),
                        "sets": e.get("sets"),
                        "reps": e.get("reps"),
                        "rest_seconds": e.get("rest_seconds"),
                    }
                    for e in d.get("exercises", [])
                ],
            }
            for d in plan.get("days", [])
        ],
    }


def _build_recommender_profile(p: dict, args: dict) -> dict:
    h, w = p.get("height_cm"), p.get("weight_kg")
    bmi = (float(w) / ((float(h) / 100) ** 2)) if (h and w) else 25.0
    bmi = max(10.1, min(59.9, bmi))
    return {
        "experience_level": int(args.get("experience_level") or p.get("experience_level") or 2),
        "workout_type": args.get("workout_type") or "Strength",
        "session_duration_hours": float(
            args.get("session_duration_hours") or p.get("session_duration_hours") or 1.0
        ),
        "workout_frequency": int(args.get("workout_frequency") or p.get("workout_frequency") or 3),
        "age": int(p.get("age") or 30),
        "gender": p.get("gender") or "Male",
        "bmi": round(bmi, 1),
        "injuries": args.get("injuries") or p.get("injuries") or [],
        "training_focus": p.get("training_focus"),
        "years_training": p.get("years_training"),
        "equipment": args.get("equipment") or p.get("equipment") or [],
        "notes": args.get("notes"),
    }


def generate_workout_plan(ctx: ToolContext, args: dict) -> dict:
    p = profile_repo.get_profile(ctx.user_id) or {}
    profile = _build_recommender_profile(p, args)
    rec = ctx.recommender.recommend(profile)
    plan_result = ctx.llm_adapter.generate_plan(
        rec["program_id"], profile, rec.get("content_candidates"))
    plan_id = plan_repo.insert_plan(ctx.user_id, rec, plan_result)
    weekly = plan_result.get("plan", {})
    return {
        "plan_id": plan_id,
        "program_title": plan_result.get("program_title"),
        "personalization_notes": plan_result.get("personalization_notes"),
        "source": plan_result.get("source"),
        "quality": plan_result.get("quality_report"),
        "day_count": len(weekly),
        "training_days": [
            d.get("focus")
            for d in weekly.values()
            if not d.get("is_rest_day")
        ],
    }


_DISPATCH = {
    "get_user_profile": get_user_profile,
    "get_active_plan": get_active_plan,
    "generate_workout_plan": generate_workout_plan,
}


def dispatch(ctx: ToolContext, name: str, args: dict) -> dict:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown_tool:{name}"}
    try:
        return fn(ctx, args)
    except Exception as exc:  # never raise out of a tool — coach explains it
        logger.exception("Tool %s failed", name)
        return {"error": "tool_failed", "detail": str(exc)[:200]}
