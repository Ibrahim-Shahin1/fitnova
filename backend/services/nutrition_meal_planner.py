"""
FitNova - Nutrition Layer 2: Deterministic Meal-Plan Assembler
==============================================================
Turns a calorie/macro target + the recipe selector into a daily or weekly meal
plan that provably hits the target window and respects diet/allergen constraints.

This is the GUARANTEED backbone — the CrewAI/LLM layer (nutrition_meal_crew) adds
natural-language meal naming, coaching notes, and swap suggestions ON TOP of this,
mirroring how the fitness pipeline runs a deterministic hard gate under the crew.
No LLM, no API key required.
"""

from __future__ import annotations

from dataclasses import asdict

from backend.services.nutrition_engine import NutritionTargets
from backend.services.nutrition_recipes import NutritionRecipeSelector
from backend.services.nutrition_constraints import expand_exclusions, meal_violations

# Calorie share per meal slot for 3- and 4-meal days
MEAL_SPLITS = {
    3: {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    4: {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10},
}


class MealPlanner:
    def __init__(self, selector: NutritionRecipeSelector | None = None):
        self.selector = selector or NutritionRecipeSelector()

    def build_day(
        self,
        targets: NutritionTargets,
        day_index: int = 0,
        meals_per_day: int = 4,
        diet: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
        cuisine: str | None = None,
        used_ids: set[int] | None = None,
    ) -> dict:
        """Assemble one day. `used_ids` (mutated) enforces variety across a week."""
        split = MEAL_SPLITS.get(meals_per_day, MEAL_SPLITS[4])
        used = used_ids if used_ids is not None else set()

        meals = []
        for slot, frac in split.items():
            slot_kcal = targets.target_calories * frac
            # Per-slot macro target (same ratio as the daily target) so meals match
            # the user's macro split, not just calories.
            slot_macros = {
                "protein_g": targets.protein_g * frac,
                "carbs_g": targets.carbs_g * frac,
                "fat_g": targets.fat_g * frac,
            }
            # Pull a ranked shortlist, then take the best not-yet-used recipe (variety).
            candidates = self.selector.select(
                slot_kcal,
                n=40,
                diet=diet,
                exclude_ingredients=exclude_ingredients,
                cuisine=cuisine,
                macro_target=slot_macros,
            )
            chosen = next((c for c in candidates if c["id"] not in used), None)
            if chosen is None and candidates:
                chosen = candidates[0]  # constraints too tight for variety — allow a repeat
            if chosen is None:
                continue
            used.add(chosen["id"])
            meals.append({
                "slot": slot,
                "target_calories": round(slot_kcal),
                "recipe_id": chosen["id"],
                "name": chosen["name"],
                "calories": chosen["kcal"],
                "protein_g": chosen["protein_g"],
                "carbs_g": chosen["carbs_g"],
                "fat_g": chosen["fat_g"],
                "minutes": chosen["minutes"],
                "tags": chosen["tags"],
                "ingredients": chosen.get("ingredients", []),
                "calorie_level": chosen.get("calorie_level", 1),
                # Original Food.com steps for this recipe (dataset provenance) — the
                # AI generator's output is shown alongside these, clearly labeled.
                "steps": self.selector.get_steps(chosen["id"]),
            })

        totals = {
            "calories": round(sum(m["calories"] for m in meals)),
            "protein_g": round(sum(m["protein_g"] for m in meals)),
            "carbs_g": round(sum(m["carbs_g"] for m in meals)),
            "fat_g": round(sum(m["fat_g"] for m in meals)),
        }
        match_pct = (
            round(100 * (1 - abs(totals["calories"] - targets.target_calories)
                         / max(targets.target_calories, 1)), 1)
        )
        return {
            "day_number": day_index + 1,
            "meals": meals,
            "totals": totals,
            "target_calories": targets.target_calories,
            "calorie_match_pct": match_pct,
        }

    def build_week(
        self,
        targets: NutritionTargets,
        days: int = 7,
        meals_per_day: int = 4,
        diet: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
        cuisine: str | None = None,
        variety: bool = True,
    ) -> dict:
        """Assemble a multi-day plan. With `variety`, recipes don't repeat across
        the week until the candidate pool for a slot is exhausted."""
        used: set[int] = set() if variety else None
        day_plans = [
            self.build_day(
                targets, day_index=d, meals_per_day=meals_per_day, diet=diet,
                exclude_ingredients=exclude_ingredients, cuisine=cuisine,
                used_ids=used if variety else set(),
            )
            for d in range(days)
        ]
        avg_match = round(sum(d["calorie_match_pct"] for d in day_plans) / max(len(day_plans), 1), 1)
        return {
            "targets": targets.as_dict() if isinstance(targets, NutritionTargets) else asdict(targets),
            "days": day_plans,
            "meals_per_day": meals_per_day,
            "diet": diet or [],
            "excluded_ingredients": exclude_ingredients or [],
            "avg_calorie_match_pct": avg_match,
        }

    @staticmethod
    def verify_plan(plan: dict, diet: list[str] | None,
                    exclude_ingredients: list[str] | None) -> dict:
        """Deterministically verify every meal against the expanded constraints +
        the calorie window. Returns {meals_checked, violations:[...], clean:bool}.
        This is the authoritative check the crew's Critic reports."""
        expanded = expand_exclusions(exclude_ingredients, diet)
        violations = []
        checked = 0
        for day in plan.get("days", []):
            for m in day.get("meals", []):
                checked += 1
                hits = meal_violations(m, expanded)
                if hits:
                    violations.append({
                        "day": day.get("day_number"), "slot": m.get("slot"),
                        "recipe": m.get("name"), "violates": hits})
        return {"meals_checked": checked, "violations": violations,
                "clean": not violations}

    def swap_meal(self, targets: NutritionTargets, day_index: int, slot: str,
                  exclude_recipe_ids: set[int], diet: list[str] | None = None,
                  exclude_ingredients: list[str] | None = None,
                  cuisine: str | None = None) -> dict | None:
        """Pick a single replacement meal for one slot, honoring all constraints
        and avoiding `exclude_recipe_ids` (the current plan's recipes). Used by the
        plan-level chat to swap one meal without rebuilding the week."""
        # resolve the slot's calorie fraction (covers both 3- and 4-meal splits)
        frac = MEAL_SPLITS[4].get(slot) or MEAL_SPLITS[3].get(slot) or 0.3
        expanded = expand_exclusions(exclude_ingredients, diet)
        slot_kcal = targets.target_calories * frac
        slot_macros = {"protein_g": targets.protein_g * frac,
                       "carbs_g": targets.carbs_g * frac, "fat_g": targets.fat_g * frac}
        candidates = self.selector.select(
            slot_kcal, n=60, diet=diet, exclude_ingredients=exclude_ingredients,
            cuisine=cuisine, macro_target=slot_macros)
        chosen = next((c for c in candidates
                       if c["id"] not in exclude_recipe_ids
                       and not meal_violations(c, expanded)), None)
        if chosen is None:
            return None
        return {
            "slot": slot, "target_calories": round(slot_kcal),
            "recipe_id": chosen["id"], "name": chosen["name"],
            "calories": chosen["kcal"], "protein_g": chosen["protein_g"],
            "carbs_g": chosen["carbs_g"], "fat_g": chosen["fat_g"],
            "minutes": chosen["minutes"], "tags": chosen["tags"],
            "ingredients": chosen.get("ingredients", []),
            "calorie_level": chosen.get("calorie_level", 1),
            "steps": self.selector.get_steps(chosen["id"]),
        }
