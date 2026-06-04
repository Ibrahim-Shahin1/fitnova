"""
FitNova - Nutrition Layer 3: 4-Agent Meal-Plan Crew (in-process)
================================================================
A 4-agent GPT-4o-mini pipeline that ENRICHES the deterministic meal plan:

  Dietitian  -> explains the calorie/macro targets for the user's goal
  Composer   -> names the plan + labels each day's theme from its actual recipes
  Critic     -> reviews the plan against the targets (grounded in real match %)
  Coach      -> writes practical coaching notes + per-meal tips + swap ideas

Design choice vs the fitness feature's CrewAI-in-subprocess: this runs in the
main backend env with direct OpenAI calls (the `openai` package is already a
dependency), so there's no second heavy `.crewenv`. Critically, the agents NEVER
pick food or change numbers — the deterministic MealPlanner already guarantees
calorie/macro/allergen correctness; the LLM only adds language on top. That makes
hallucinated recipes or broken macros structurally impossible.

Degrades gracefully: with no OpenAI client (or on any agent failure) it returns
the deterministic plan with template annotations, so the feature always works.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from backend.services.nutrition_engine import NutritionTargets
from backend.services.nutrition_meal_planner import MealPlanner

logger = logging.getLogger("fitnova.nutrition.crew")

_MODEL = "gpt-4o-mini"
_GOAL_WORDS = {"lose": "fat loss", "maintain": "maintenance", "gain": "muscle gain"}


class NutritionMealCrew:
    def __init__(self, client=None, planner: MealPlanner | None = None):
        self.client = client
        self.planner = planner or MealPlanner()

    # ── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        targets: NutritionTargets,
        *,
        days: int = 7,
        meals_per_day: int = 4,
        diet: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
        cuisine: str | None = None,
    ) -> dict:
        """Run the full pipeline and return the enriched plan (non-streamed)."""
        plan = self.planner.build_week(
            targets, days=days, meals_per_day=meals_per_day, diet=diet,
            exclude_ingredients=exclude_ingredients, cuisine=cuisine,
        )
        annotations = {}
        for ev in self._run_agents(plan, targets, diet, exclude_ingredients):
            if ev.get("event") == "annotations":
                annotations = ev["data"]
        return self._assemble(plan, targets, annotations)

    def build_streamed(
        self,
        targets: NutritionTargets,
        *,
        days: int = 7,
        meals_per_day: int = 4,
        diet: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
        cuisine: str | None = None,
    ) -> Iterator[dict]:
        """Generator: yields per-agent progress events, then a final
        {'event':'plan', ...}. Drives the live 4-agent Flutter screen."""
        plan = self.planner.build_week(
            targets, days=days, meals_per_day=meals_per_day, diet=diet,
            exclude_ingredients=exclude_ingredients, cuisine=cuisine,
        )
        annotations = {}
        for ev in self._run_agents(plan, targets, diet, exclude_ingredients):
            if ev.get("event") == "annotations":
                annotations = ev["data"]
            else:
                yield ev
        yield {"event": "plan", **self._assemble(plan, targets, annotations)}

    # ── Agent orchestration ─────────────────────────────────────────────────

    def _run_agents(self, plan, targets, diet, exclude) -> Iterator[dict]:
        ann: dict = {}
        agents = [
            ("Dietitian", self._dietitian),
            ("Composer", self._composer),
            ("Critic", self._critic),
            ("Coach", self._coach),
        ]
        for name, fn in agents:
            yield {"event": "agent", "name": name, "status": "running"}
            try:
                if self.client is None:
                    out = fn(plan, targets, diet, exclude, template=True)
                else:
                    out = fn(plan, targets, diet, exclude, template=False)
            except Exception as exc:
                logger.warning("Nutrition agent %s failed: %s — using template", name, exc)
                out = fn(plan, targets, diet, exclude, template=True)
            ann.update(out)
            yield {"event": "agent", "name": name, "status": "done", "detail": out}
        yield {"event": "annotations", "data": ann}

    def _call(self, system: str, user: str) -> dict:
        resp = self.client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=900,
            timeout=60,
        )
        return json.loads(resp.choices[0].message.content)

    # ── Plan-level chat (modify the existing plan via deterministic actions) ──

    def plan_chat(self, plan: dict, message: str) -> dict:
        """Chat about an EXISTING plan. The LLM picks an action; the planner
        EXECUTES it deterministically so the result is guaranteed constraint-clean.

        Actions:
          - swap_meal(day, slot): re-pick one meal
          - add_exclusion(items): add constraints, re-pick every violating meal
          - answer(text): just answer a question (no plan change)

        Returns {ok, reply, plan?}  (plan present only when it changed).
        """
        if self.client is None:
            return {"ok": False, "error": "Chat needs OPENAI_API_KEY."}
        targets = NutritionTargets(**plan["targets"])
        diet = list(plan.get("diet") or [])
        excluded = list(plan.get("excluded_ingredients") or [])
        slots = [m["slot"] for d in plan.get("days", []) for m in d["meals"]]
        slot_set = sorted(set(slots))
        day_count = len(plan.get("days", []))

        tools = [{
            "type": "function",
            "function": {
                "name": "modify_plan",
                "description": "Decide how to handle the user's request about their meal plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["swap_meal", "add_exclusion", "answer"]},
                        "day": {"type": "integer",
                                "description": f"1-{day_count}, for swap_meal (0 = all days)"},
                        "slot": {"type": "string",
                                 "description": f"one of {slot_set}, for swap_meal"},
                        "exclude": {"type": "array", "items": {"type": "string"},
                                    "description": "ingredients/categories to exclude, "
                                    "for add_exclusion (e.g. ['fish','tuna'])"},
                        "reply": {"type": "string",
                                  "description": "friendly message to the user"},
                    },
                    "required": ["action", "reply"],
                },
            },
        }]
        sys = ("You manage a user's existing weekly meal plan. Decide ONE action:\n"
               "- swap_meal: they want a different meal for a day/slot.\n"
               "- add_exclusion: they don't want an ingredient/food. Map to CATEGORY "
               "words the engine understands and err toward the user's clear intent:\n"
               "    'no fish' or 'no seafood'  -> exclude ['fish','shellfish']\n"
               "    'no shellfish'             -> exclude ['shellfish']\n"
               "    'no meat'                  -> exclude ['meat']\n"
               "    'no dairy'                 -> exclude ['dairy']; 'no nuts' -> ['nut']\n"
               "  A normal person saying 'no fish' does NOT want shrimp either, so "
               "include shellfish unless they specifically said only fish. For a single "
               "ingredient (e.g. 'no mushrooms') exclude that exact word.\n"
               "- answer: a general question, no change needed.\n"
               f"Plan: {day_count} days, slots={slot_set}, goal={targets.goal}, "
               f"diet={diet or 'none'}, already-excluded={excluded or 'none'}.")
        try:
            resp = self.client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": message[:1000]}],
                tools=tools, tool_choice={"type": "function",
                                          "function": {"name": "modify_plan"}},
                temperature=0.3, timeout=60,
            )
            msg = resp.choices[0].message
            tcs = msg.tool_calls or []
            if not tcs:
                # Model returned prose instead of the forced tool call — degrade to
                # a plain answer rather than crashing.
                return {"ok": True,
                        "reply": (msg.content or "").strip()
                        or "I can swap a meal or add a restriction — tell me which."}
            args = json.loads(tcs[0].function.arguments)
        except Exception as exc:
            logger.exception("plan_chat tool call failed")
            return {"ok": False, "error": str(exc)[:200]}

        action = args.get("action", "answer")
        reply = args.get("reply", "")

        if action == "answer":
            return {"ok": True, "reply": reply or "How can I adjust your plan?"}

        if action == "add_exclusion":
            new_terms = [str(x).lower().strip() for x in (args.get("exclude") or []) if x]
            if not new_terms:
                return {"ok": True, "reply": reply or "What would you like to avoid?"}
            excluded = sorted(set(excluded) | set(new_terms))
            changed = self._repick_violations(plan, targets, diet, excluded)
            plan["excluded_ingredients"] = excluded
            self._recompute(plan, targets)
            return {"ok": True,
                    "reply": (reply or f"Done — removed {', '.join(new_terms)}.")
                    + f" Updated {changed} meal(s) and re-checked the whole plan.",
                    "plan": plan}

        if action == "swap_meal":
            day = int(args.get("day", 0) or 0)
            slot = str(args.get("slot", "")).lower().strip()
            days_to_do = ([d for d in plan["days"] if d["day_number"] == day]
                          if day else plan["days"])
            used = {m["recipe_id"] for d in plan["days"] for m in d["meals"]}
            swapped = 0
            for d in days_to_do:
                for i, m in enumerate(d["meals"]):
                    if slot and m["slot"] != slot:
                        continue
                    new_meal = self.planner.swap_meal(
                        targets, d["day_number"] - 1, m["slot"],
                        exclude_recipe_ids=used, diet=diet,
                        exclude_ingredients=excluded)
                    if new_meal:
                        used.discard(m["recipe_id"]); used.add(new_meal["recipe_id"])
                        d["meals"][i] = new_meal
                        swapped += 1
                    if slot:
                        break
            self._recompute(plan, targets)
            return {"ok": True,
                    "reply": (reply or "Swapped it.") + f" ({swapped} meal(s) changed.)",
                    "plan": plan}

        return {"ok": True, "reply": reply or "Done."}

    def _repick_violations(self, plan, targets, diet, excluded) -> int:
        """Re-pick every meal that violates the (new) exclusion set. Returns count."""
        from backend.services.nutrition_constraints import expand_exclusions, meal_violations
        expanded = expand_exclusions(excluded, diet)
        used = {m["recipe_id"] for d in plan["days"] for m in d["meals"]}
        changed = 0
        for d in plan["days"]:
            for i, m in enumerate(d["meals"]):
                if meal_violations(m, expanded):
                    new_meal = self.planner.swap_meal(
                        targets, d["day_number"] - 1, m["slot"],
                        exclude_recipe_ids=used, diet=diet, exclude_ingredients=excluded)
                    if new_meal:
                        used.discard(m["recipe_id"]); used.add(new_meal["recipe_id"])
                        d["meals"][i] = new_meal
                        changed += 1
        return changed

    @staticmethod
    def _recompute(plan, targets) -> None:
        """Recompute per-day totals + match after meals change."""
        for d in plan["days"]:
            t = {"calories": round(sum(m["calories"] for m in d["meals"])),
                 "protein_g": round(sum(m["protein_g"] for m in d["meals"])),
                 "carbs_g": round(sum(m["carbs_g"] for m in d["meals"])),
                 "fat_g": round(sum(m["fat_g"] for m in d["meals"]))}
            d["totals"] = t
            d["calorie_match_pct"] = round(
                100 * (1 - abs(t["calories"] - targets.target_calories)
                       / max(targets.target_calories, 1)), 1)
        plan["avg_calorie_match_pct"] = round(
            sum(d["calorie_match_pct"] for d in plan["days"]) / max(len(plan["days"]), 1), 1)

    # ── Recipe chat (conversational, grounded in one recipe) ─────────────────

    def chat_about_recipe(self, recipe: dict, messages: list[dict]) -> dict:
        """Free-form chat scoped to ONE recipe (substitutions, scaling, technique,
        dietary tweaks), grounded in the recipe's real name/ingredients/steps so it
        stays on topic. Returns {ok, reply} or {ok: False, error}."""
        if self.client is None:
            return {"ok": False,
                    "error": "Chat needs the OpenAI client (set OPENAI_API_KEY)."}
        name = str(recipe.get("name", "this recipe"))
        ingredients = ", ".join(str(i) for i in recipe.get("ingredients", []) if i)
        steps = recipe.get("steps") or []
        steps_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) or "(not provided)"
        ai_instr = recipe.get("ai_instructions")
        system = (
            "You are FitNova's friendly cooking assistant. Help the user cook, adapt, "
            "scale, or substitute ingredients for THIS specific recipe. Be practical "
            "and concise (2-4 sentences unless they ask for more). Suggest realistic "
            "substitutions and technique tips. Do NOT invent a completely different "
            "recipe, and make no medical claims.\n\n"
            f"RECIPE: {name}\nINGREDIENTS: {ingredients}\n"
            f"ORIGINAL STEPS (Food.com dataset):\n{steps_txt}"
            + (f"\n\nOUR MODEL'S GENERATED INSTRUCTIONS:\n{ai_instr}" if ai_instr else "")
        )
        convo = [{"role": "system", "content": system}]
        for m in messages[-12:]:
            role = m.get("role")
            if role in ("user", "assistant") and m.get("content"):
                convo.append({"role": role, "content": str(m["content"])[:1500]})
        try:
            resp = self.client.chat.completions.create(
                model=_MODEL, messages=convo, temperature=0.6,
                max_tokens=400, timeout=60,
            )
            return {"ok": True, "reply": (resp.choices[0].message.content or "").strip()}
        except Exception as exc:
            logger.exception("Recipe chat failed")
            return {"ok": False, "error": str(exc)[:200]}

    # ── Agents ──────────────────────────────────────────────────────────────

    def _dietitian(self, plan, targets, diet, exclude, template) -> dict:
        if template:
            goal = _GOAL_WORDS.get(targets.goal, targets.goal)
            return {"targets_rationale": (
                f"Your BMI is {targets.bmi} ({targets.bmi_category}). For {goal}, "
                f"we set {targets.target_calories} kcal/day "
                f"(BMR {targets.bmr} x activity, adjusted for your goal) with "
                f"{targets.protein_g}g protein, {targets.carbs_g}g carbs, "
                f"{targets.fat_g}g fat.")}
        sys = ("You are a registered dietitian. Explain the given calorie and macro "
               "targets for the user's goal in 2-3 plain-language sentences. Return "
               'JSON: {"targets_rationale": str}. Do not change any numbers.')
        usr = (f"BMI {targets.bmi} ({targets.bmi_category}), goal {targets.goal}, "
               f"BMR {targets.bmr}, TDEE {targets.tdee}, target {targets.target_calories} kcal, "
               f"protein {targets.protein_g}g, carbs {targets.carbs_g}g, fat {targets.fat_g}g.")
        return self._call(sys, usr)

    def _composer(self, plan, targets, diet, exclude, template) -> dict:
        if template:
            goal = _GOAL_WORDS.get(targets.goal, targets.goal).title()
            return {"plan_title": f"{goal} Meal Plan - {targets.target_calories} kcal/day",
                    "day_themes": {}}
        day_summaries = []
        for d in plan["days"]:
            names = ", ".join(m["name"] for m in d["meals"])
            day_summaries.append(f"Day {d['day_number']}: {names}")
        sys = ("You name meal plans. Given the recipes per day, return JSON: "
               '{"plan_title": str, "day_themes": {"day_1": str, ...}}. The title is '
               "descriptive (goal + calories). Each day theme is 2-4 words based on its "
               "ACTUAL recipes (e.g. 'Mediterranean & Light'). Do not invent recipes.")
        usr = (f"Goal {targets.goal}, {targets.target_calories} kcal/day.\n"
               + "\n".join(day_summaries))
        return self._call(sys, usr)

    def _critic(self, plan, targets, diet, exclude, template) -> dict:
        avg = plan.get("avg_calorie_match_pct", 0)
        # AUTHORITATIVE deterministic check — every meal vs the expanded exclusion
        # set + diet. This is what actually catches "tuna when fish is excluded".
        from backend.services.nutrition_meal_planner import MealPlanner
        verify = MealPlanner.verify_plan(plan, diet, exclude)
        # Hard cap: if the plan isn't constraint-clean, the score is capped low and
        # the violations are surfaced (the LLM can't paper over a real violation).
        if template or not verify["clean"]:
            score = round(min(10, avg / 10)) if verify["clean"] else 3
            note = (f"Verified {verify['meals_checked']} meals: "
                    + ("all constraints satisfied; "
                       if verify["clean"] else
                       f"{len(verify['violations'])} violation(s) found; ")
                    + f"averages {avg}% of the {targets.target_calories} kcal target.")
            return {"quality_score": score, "quality_note": note, "verification": verify}
        per_day = [{"day": d["day_number"], "kcal": d["totals"]["calories"],
                    "target": d["target_calories"], "match_pct": d["calorie_match_pct"]}
                   for d in plan["days"]]
        sys = ("You are a nutrition QA reviewer. A DETERMINISTIC checker has already "
               "verified calorie/macro/allergen/diet correctness (report given). "
               "Judge overall quality and variety. Return JSON: "
               '{"quality_score": int 1-10, "quality_note": str (1-2 sentences)}.')
        usr = (f"Target {targets.target_calories} kcal. Per-day: {json.dumps(per_day)}. "
               f"Avg match {avg}%. Diet={diet or 'none'}, excluded={exclude or 'none'}. "
               f"Deterministic check: {verify['meals_checked']} meals, "
               f"{len(verify['violations'])} violations.")
        out = self._call(sys, usr)
        out["verification"] = verify
        return out

    def _coach(self, plan, targets, diet, exclude, template) -> dict:
        if template:
            tips = ["Prep proteins in bulk to save time on busy days.",
                    "Hit your protein target at each meal to preserve muscle.",
                    "Drink water before meals and aim for vegetables at lunch and dinner."]
            return {"coaching_notes": " ".join(tips)}
        sample = [m["name"] for d in plan["days"][:1] for m in d["meals"]]
        sys = ("You are a friendly nutrition coach. Write practical guidance for "
               "following this plan: prep tips, adherence advice, and how to swap a "
               "meal while keeping calories similar. Return JSON: "
               '{"coaching_notes": str (3-4 sentences)}. No medical claims.')
        usr = (f"Goal {targets.goal}, {targets.target_calories} kcal/day, "
               f"{targets.protein_g}g protein. Example day: {sample}. "
               f"Diet={diet or 'none'}, avoid={exclude or 'none'}.")
        return self._call(sys, usr)

    # ── Assembly ────────────────────────────────────────────────────────────

    def _assemble(self, plan: dict, targets: NutritionTargets, ann: dict) -> dict:
        # Apply day themes onto the plan days (LLM annotation only).
        themes = ann.get("day_themes") or {}
        for d in plan["days"]:
            theme = themes.get(f"day_{d['day_number']}")
            if theme:
                d["theme"] = theme
        return {
            "plan_title": ann.get("plan_title")
            or f"{_GOAL_WORDS.get(targets.goal, targets.goal).title()} Meal Plan",
            "targets": targets.as_dict(),
            "targets_rationale": ann.get("targets_rationale", ""),
            "coaching_notes": ann.get("coaching_notes", ""),
            "quality_report": {
                "score": ann.get("quality_score"),
                "note": ann.get("quality_note", ""),
                "avg_calorie_match_pct": plan.get("avg_calorie_match_pct"),
                "verification": ann.get("verification", {}),
            },
            "source": "llm" if self.client is not None else "deterministic",
            "days": plan["days"],
            "meals_per_day": plan["meals_per_day"],
            "diet": plan["diet"],
            "excluded_ingredients": plan["excluded_ingredients"],
        }
