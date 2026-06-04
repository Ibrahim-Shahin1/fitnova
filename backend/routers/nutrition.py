"""
Nutrition router — BMI/calorie targeting + AI meal-plan generation.

Stateless (no auth/DB) to keep the feature self-contained and demoable, mirroring
the legacy /generate-plan and /chat endpoints. Three endpoints:

  POST /api/nutrition/targets         body stats + goal -> BMI + calorie/macro target
  POST /api/nutrition/generate        + diet/prefs       -> full meal plan (4-agent crew)
  POST /api/nutrition/generate/stream                     -> SSE per-agent progress + plan

The deterministic engine + planner guarantee correctness; the crew adds language.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.nutrition_engine import compute_targets

logger = logging.getLogger("fitnova.nutrition")

router = APIRouter(prefix="/api/nutrition", tags=["Nutrition"])


class TargetsRequest(BaseModel):
    weight_kg: float = Field(..., gt=20, lt=400)
    height_cm: float = Field(..., gt=100, lt=260)
    age: int = Field(..., ge=10, le=100)
    sex: str = Field(..., pattern=r"^(male|female|Male|Female)$")
    activity_level: str = Field(default="moderate")
    goal: str = Field(default="maintain")


class MealPlanRequest(TargetsRequest):
    days: int = Field(default=7, ge=1, le=7)
    meals_per_day: int = Field(default=4, ge=3, le=4)
    diet: list[str] = Field(default_factory=list)
    exclude_ingredients: list[str] = Field(default_factory=list)
    cuisine: str | None = None


class RecipeInstructionsRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    ingredients: list[str] = Field(..., min_length=1)
    calorie_level: str = Field(default="medium", pattern=r"^(low|medium|high)$")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=1500)


class RecipeChatRequest(BaseModel):
    recipe: dict = Field(...)  # {name, ingredients[], steps[], ai_instructions?}
    messages: list[ChatMessage] = Field(..., min_length=1)


class PlanChatRequest(BaseModel):
    plan: dict = Field(...)  # the current MealPlan json (days, targets, diet, excluded…)
    message: str = Field(..., min_length=1, max_length=1000)


@router.post("/targets")
async def targets(req: TargetsRequest):
    """Body stats + goal → BMI, BMR, TDEE, calorie target, and macro grams.
    Pure formula (Mifflin-St Jeor); instant, no model."""
    try:
        t = compute_targets(
            weight_kg=req.weight_kg, height_cm=req.height_cm, age=req.age,
            sex=req.sex, activity_level=req.activity_level, goal=req.goal,
        )
    except Exception as exc:
        logger.exception("Targets computation failed")
        raise HTTPException(status_code=500, detail=f"Targets failed: {exc}")
    return t.as_dict()


@router.post("/generate")
async def generate(req: MealPlanRequest, request: Request):
    """Compute targets, then run the 4-agent crew over the deterministic plan."""
    crew = getattr(request.app.state, "nutrition_crew", None)
    if crew is None:
        raise HTTPException(status_code=503, detail="Nutrition service unavailable")
    t = compute_targets(
        weight_kg=req.weight_kg, height_cm=req.height_cm, age=req.age,
        sex=req.sex, activity_level=req.activity_level, goal=req.goal,
    )
    try:
        plan = crew.build(
            t, days=req.days, meals_per_day=req.meals_per_day, diet=req.diet,
            exclude_ingredients=req.exclude_ingredients, cuisine=req.cuisine,
        )
    except Exception as exc:
        logger.exception("Meal plan generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
    return plan


@router.post("/recipe-instructions")
async def recipe_instructions(req: RecipeInstructionsRequest, request: Request):
    """Generate cooking instructions for a recipe via the fine-tuned DistilGPT-2
    (the N2 model that beats the paper on BLEU/ROUGE/Distinct). Returns 503 if the
    model isn't installed (it's gitignored — copy from Drive FoodCom/model_tf)."""
    gen = getattr(request.app.state, "recipe_generator", None)
    if gen is None or not gen.available:
        raise HTTPException(
            status_code=503,
            detail="Recipe generator not installed (copy model to backend/models/recipe_model_tf).")
    result = gen.generate(
        name=req.name, ingredients=req.ingredients, calorie_level=req.calorie_level)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
    return result


@router.post("/recipe-chat")
async def recipe_chat(req: RecipeChatRequest, request: Request):
    """Conversational chat about one recipe (substitutions, scaling, technique),
    via GPT-4o-mini grounded in the recipe's real name/ingredients/dataset steps."""
    crew = getattr(request.app.state, "nutrition_crew", None)
    if crew is None:
        raise HTTPException(status_code=503, detail="Nutrition service unavailable")
    result = crew.chat_about_recipe(
        req.recipe, [m.model_dump() for m in req.messages])
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result.get("error", "Chat failed"))
    return result


@router.post("/plan-chat")
async def plan_chat(req: PlanChatRequest, request: Request):
    """Chat to MODIFY an existing meal plan: swap a meal, or add a constraint
    ('no fish') and re-pick every violating meal — deterministically, so the
    result is guaranteed constraint-clean. Returns {ok, reply, plan?}."""
    crew = getattr(request.app.state, "nutrition_crew", None)
    if crew is None:
        raise HTTPException(status_code=503, detail="Nutrition service unavailable")
    result = crew.plan_chat(req.plan, req.message)
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result.get("error", "Chat failed"))
    return result


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/generate/stream")
async def generate_stream(req: MealPlanRequest, request: Request):
    """Stream the 4-agent crew's progress as SSE, then the final plan. Drives the
    live nutrition agent screen. Events: started → agent(running/done)… → plan | error."""
    crew = getattr(request.app.state, "nutrition_crew", None)
    if crew is None:
        raise HTTPException(status_code=503, detail="Nutrition service unavailable")
    t = compute_targets(
        weight_kg=req.weight_kg, height_cm=req.height_cm, age=req.age,
        sex=req.sex, activity_level=req.activity_level, goal=req.goal,
    )

    def stream():
        try:
            yield _sse({"event": "started", "targets": t.as_dict()})
            for ev in crew.build_streamed(
                t, days=req.days, meals_per_day=req.meals_per_day, diet=req.diet,
                exclude_ingredients=req.exclude_ingredients, cuisine=req.cuisine,
            ):
                yield _sse(ev)
        except Exception as exc:
            logger.exception("Streamed meal plan failed")
            yield _sse({"event": "error", "error": str(exc)[:300]})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
