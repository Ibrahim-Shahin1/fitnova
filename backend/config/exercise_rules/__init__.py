"""Per-exercise geometric rule configs consumed by form_geometry.

Each exercise file exports an EXERCISE_RULES dict matching the GeometricRules
schema in backend.services.form_geometry.

Currently supported: squat. Other exercises TBD pending demo feedback.
"""

from .squat import SQUAT_RULES

EXERCISE_RULES = {
    "squat":  SQUAT_RULES,
    "squats": SQUAT_RULES,   # QEVD plural — same rules
}


def get_rules(exercise_name: str | None):
    """Return the rule config for an exercise, or None if no rules defined."""
    if not exercise_name:
        return None
    key = exercise_name.lower().replace(" ", "_")
    return EXERCISE_RULES.get(key)
