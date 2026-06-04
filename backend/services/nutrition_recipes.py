"""
FitNova - Nutrition Layer 1b: Recipe Content Selector
=====================================================
Scores the cleaned Food.com recipe pool against a per-meal calorie target +
macro priority + dietary constraints, returning ranked candidate recipes.

Pure content-based scoring over the 166,998-recipe cleaned pool (N1). Mirrors
the role of content_filter.py in the fitness pipeline: it produces the candidate
set the meal-plan crew (N4) assembles. No LLM, no model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "nutrition")
_POOL_PATH = os.path.join(_DATA_DIR, "recipes_runtime.pkl")
_STEPS_PATH = os.path.join(_DATA_DIR, "recipe_steps.pkl")

# User-facing diet -> the Food.com tag that must be present
DIET_TAGS = {
    "vegetarian": "vegetarian", "vegan": "vegan",
    "gluten-free": "gluten-free", "gluten free": "gluten-free",
    "dairy-free": "dairy-free", "dairy free": "dairy-free",
    "low-carb": "low-carb", "keto": "low-carb",
    "low-fat": "low-fat", "low-sodium": "low-sodium",
    "low-cholesterol": "low-cholesterol",
}

# Default split of the daily target across meal slots
DEFAULT_MEAL_SPLIT = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10}


class NutritionRecipeSelector:
    def __init__(self, pool_path: str = _POOL_PATH):
        self.df = pd.read_pickle(pool_path).reset_index(drop=True)
        self.kcal = self.df["kcal"].to_numpy(dtype=np.float32)
        self.protein_g = self.df["protein_g"].to_numpy(dtype=np.float32)
        self.carbs_g = self.df["carbs_g"].to_numpy(dtype=np.float32)
        self.fat_g = self.df["fat_g"].to_numpy(dtype=np.float32)
        # Per-recipe macro CALORIE fractions (protein/carbs/fat shares of kcal),
        # used to match a target macro ratio. 4/4/9 kcal per gram.
        macro_kcal = (4.0 * self.protein_g + 4.0 * self.carbs_g + 9.0 * self.fat_g)
        macro_kcal = np.where(macro_kcal > 0, macro_kcal, 1.0)
        self.frac_p = (4.0 * self.protein_g / macro_kcal).astype(np.float32)
        self.frac_c = (4.0 * self.carbs_g / macro_kcal).astype(np.float32)
        self.frac_f = (9.0 * self.fat_g / macro_kcal).astype(np.float32)
        # protein per 100 kcal — density signal, goal-agnostic and scale-free
        self.protein_density = np.where(
            self.kcal > 0, self.protein_g / self.kcal * 100.0, 0.0
        ).astype(np.float32)
        # lowercased text blobs for constraint matching (built once)
        self._tags_text = self.df["tags"].apply(lambda xs: " ".join(xs).lower())
        self._ingr_text = self.df["ingredients"].apply(lambda xs: " ".join(xs).lower())
        # recipe_id -> list[str] of the ORIGINAL Food.com steps (dataset provenance).
        try:
            self._steps = pd.read_pickle(_STEPS_PATH)
        except Exception:
            self._steps = {}

    def get_steps(self, recipe_id: int) -> list[str]:
        """Return the recipe's ORIGINAL dataset steps (empty list if unavailable)."""
        return list(self._steps.get(int(recipe_id), []))

    def select(
        self,
        target_kcal: float,
        n: int = 20,
        diet: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
        cuisine: str | None = None,
        prioritize_protein: bool = True,
        macro_target: dict | None = None,
        tolerance: float = 0.30,
    ) -> list[dict]:
        """
        Rank recipes for one meal slot.

        Args:
            target_kcal: desired calories for this meal/slot.
            n: number of candidates to return.
            diet: required diet tags (e.g. ["vegetarian"]).
            exclude_ingredients: allergens/dislikes to exclude by substring.
            cuisine: optional cuisine keyword to bias toward (tag match bonus).
            prioritize_protein: weight protein density (only when macro_target is None).
            macro_target: optional {"protein_g","carbs_g","fat_g"} for this slot;
                recipes are scored by how closely their macro RATIO matches it
                (preferred over prioritize_protein — avoids over-stacking protein).
            tolerance: fractional calorie window around target for full credit.

        Returns:
            Up to n recipe dicts (id, name, kcal, macros, minutes, tags) ranked best-first.
        """
        mask = np.ones(len(self.df), dtype=bool)

        # Calorie window: hard-exclude beyond 2x tolerance, soft-score within.
        kcal_err = np.abs(self.kcal - target_kcal) / max(target_kcal, 1.0)
        mask &= kcal_err <= (2.0 * tolerance)

        # Dietary constraints — tag must be present
        for d in diet or []:
            tag = DIET_TAGS.get(str(d).lower().strip())
            if tag:
                mask &= self._tags_text.str.contains(tag, regex=False).to_numpy()

        # Allergen / dislike exclusion — SEMANTIC, word-boundary. Expands
        # categories ("fish"->tuna/salmon/...) and diet-implied exclusions
        # (vegetarian also drops meat/fish at the ingredient level), then matches
        # on word boundaries so "no fish" drops tuna but "egg" doesn't hit
        # "eggplant". This is what guarantees the exclusion actually holds.
        from backend.services.nutrition_constraints import (
            expand_exclusions, build_exclusion_regex)
        expanded = expand_exclusions(exclude_ingredients, diet)
        rx = build_exclusion_regex(expanded)
        if rx is not None:
            # match against ingredients + name (some recipes name the protein only in the title)
            hay = (self._ingr_text + " " + self.df["name"].str.lower())
            mask &= ~hay.str.contains(rx, regex=True).to_numpy()

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []

        # Score: calorie fit (primary) + macro-ratio fit / protein density + cuisine bonus
        cal_fit = np.clip(1.0 - kcal_err[idx] / (tolerance + 1e-9), 0.0, 1.0)
        score = cal_fit.copy()
        if macro_target:
            # Match the target macro RATIO (shares of calories), not absolute grams —
            # avoids the protein-maximizing failure mode. L1 distance over 3 fractions
            # is in [0,2]; map to a [0,1] fit.
            tp, tc, tf = (float(macro_target.get("protein_g", 0)),
                          float(macro_target.get("carbs_g", 0)),
                          float(macro_target.get("fat_g", 0)))
            tkcal = 4 * tp + 4 * tc + 9 * tf
            if tkcal > 0:
                tfp, tfc, tff = 4 * tp / tkcal, 4 * tc / tkcal, 9 * tf / tkcal
                l1 = (np.abs(self.frac_p[idx] - tfp)
                      + np.abs(self.frac_c[idx] - tfc)
                      + np.abs(self.frac_f[idx] - tff))
                macro_fit = np.clip(1.0 - 0.5 * l1, 0.0, 1.0)
                score = 0.6 * cal_fit + 0.4 * macro_fit
        elif prioritize_protein:
            pd_sub = self.protein_density[idx]
            pd_norm = pd_sub / (pd_sub.max() + 1e-9)
            score = 0.7 * cal_fit + 0.3 * pd_norm
        if cuisine:
            c = str(cuisine).lower().strip()
            cuisine_hit = self._tags_text.iloc[idx].str.contains(c, regex=False).to_numpy()
            score = score + 0.15 * cuisine_hit

        order = idx[np.argsort(-score)[:n]]
        cols = ["id", "name", "kcal", "protein_g", "carbs_g", "fat_g", "minutes", "calorie_level"]
        out = []
        for r in order:
            row = self.df.iloc[r]
            rec = {c: (int(row[c]) if c in ("id", "minutes", "calorie_level")
                       else round(float(row[c]), 1) if c in ("kcal", "protein_g", "carbs_g", "fat_g")
                       else row[c]) for c in cols}
            rec["tags"] = list(row["tags"])[:8]
            rec["ingredients"] = list(row["ingredients"])
            out.append(rec)
        return out
