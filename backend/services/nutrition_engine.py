"""
FitNova - Nutrition Layer 1a: Calorie & Macro Target Engine
===========================================================
Deterministic body-stats -> daily calorie/macro target.

Method (defensible, citable):
  BMI            = weight_kg / height_m^2                          (WHO categories)
  BMR            = Mifflin-St Jeor (1990)                          (most-validated predictive eq.)
  TDEE           = BMR * activity_factor
  target_kcal    = TDEE * (1 + goal_adjust)                        (clamped to a safe floor)
  protein/fat    = g-per-kg by goal (protein-priority); carbs = remaining kcal

No LLM, no model — pure formula. Mirrors the role of content_filter.py in the fitness pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# Mifflin-St Jeor sex constant
_SEX_CONSTANT = {"male": 5.0, "female": -161.0}

# Physical Activity Level multipliers (Harris-Benedict / WHO convention)
ACTIVITY_FACTORS = {
    "sedentary": 1.2,    # little/no exercise
    "light": 1.375,      # 1-3 days/week
    "moderate": 1.55,    # 3-5 days/week
    "active": 1.725,     # 6-7 days/week
    "athlete": 1.9,      # 2x/day, physical job
}

# Goal -> fraction of TDEE applied as surplus/deficit
GOAL_ADJUST = {"lose": -0.20, "maintain": 0.0, "gain": 0.12}

# Goal -> protein target (g per kg bodyweight); higher on a deficit to preserve lean mass
PROTEIN_G_PER_KG = {"lose": 2.2, "maintain": 1.8, "gain": 2.0}
_FAT_G_PER_KG = 0.9

# Safe minimum daily intake (never prescribe below this) by sex
_MIN_KCAL = {"male": 1500.0, "female": 1200.0}

# Aliases so chat/LLM-extracted values map onto the canonical keys
_GOAL_ALIASES = {
    "lose": "lose", "cut": "lose", "deficit": "lose", "weight loss": "lose",
    "fat loss": "lose", "lean": "lose",
    "maintain": "maintain", "maintenance": "maintain", "recomp": "maintain",
    "gain": "gain", "bulk": "gain", "surplus": "gain", "muscle": "gain",
    "build muscle": "gain", "powerbuilding": "gain", "mass": "gain",
}
_ACTIVITY_ALIASES = {
    "sedentary": "sedentary", "none": "sedentary",
    "light": "light", "lightly active": "light",
    "moderate": "moderate", "moderately active": "moderate",
    "active": "active", "very active": "active",
    "athlete": "athlete", "extra active": "athlete", "extremely active": "athlete",
}

_BMI_BANDS = [
    (18.5, "underweight"),
    (25.0, "normal"),
    (30.0, "overweight"),
    (float("inf"), "obese"),
]


@dataclass
class NutritionTargets:
    bmi: float
    bmi_category: str
    bmr: int
    tdee: int
    target_calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    goal: str
    activity_level: str

    def as_dict(self) -> dict:
        return asdict(self)


def _bmi_category(bmi: float) -> str:
    for upper, label in _BMI_BANDS:
        if bmi < upper:
            return label
    return "obese"


def compute_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 1), _bmi_category(bmi)


def compute_bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """Mifflin-St Jeor resting BMR (kcal/day)."""
    s = _SEX_CONSTANT.get(sex.lower(), -78.0)  # average of M/F when unknown
    return 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age + s


def compute_targets(
    weight_kg: float,
    height_cm: float,
    age: int,
    sex: str,
    activity_level: str = "moderate",
    goal: str = "maintain",
) -> NutritionTargets:
    """
    Body stats + goal -> daily calorie and macro targets.

    Args:
        weight_kg, height_cm, age: body stats.
        sex: "male" | "female".
        activity_level: see ACTIVITY_FACTORS (aliases accepted).
        goal: "lose" | "maintain" | "gain" (aliases like cut/bulk accepted).

    Returns:
        NutritionTargets with bmi, bmr, tdee, target_calories, and macro grams.
    """
    sex_k = sex.lower() if sex.lower() in _SEX_CONSTANT else "male"
    goal_k = _GOAL_ALIASES.get(str(goal).lower().strip(), "maintain")
    act_k = _ACTIVITY_ALIASES.get(str(activity_level).lower().strip(), "moderate")

    bmi, category = compute_bmi(weight_kg, height_cm)
    bmr = compute_bmr(weight_kg, height_cm, age, sex_k)
    tdee = bmr * ACTIVITY_FACTORS[act_k]
    target = tdee * (1.0 + GOAL_ADJUST[goal_k])
    target = max(target, _MIN_KCAL.get(sex_k, 1300.0))

    # Macro allocation: protein + fat by bodyweight, carbs fill the remainder.
    protein_g = PROTEIN_G_PER_KG[goal_k] * weight_kg
    fat_g = _FAT_G_PER_KG * weight_kg
    remaining = target - (4.0 * protein_g + 9.0 * fat_g)
    if remaining < 0:  # very low target: trim fat first, then protein, to stay >= 0 carbs
        fat_g = max(0.5 * weight_kg, (target - 4.0 * protein_g) / 9.0)
        remaining = target - (4.0 * protein_g + 9.0 * fat_g)
        if remaining < 0:
            protein_g = max(1.6 * weight_kg, target / 4.0 * 0.5)
            remaining = target - (4.0 * protein_g + 9.0 * fat_g)
    carbs_g = max(0.0, remaining / 4.0)

    return NutritionTargets(
        bmi=bmi,
        bmi_category=category,
        bmr=round(bmr),
        tdee=round(tdee),
        target_calories=round(target),
        protein_g=round(protein_g),
        carbs_g=round(carbs_g),
        fat_g=round(fat_g),
        goal=goal_k,
        activity_level=act_k,
    )
