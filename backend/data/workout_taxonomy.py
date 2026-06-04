"""
Shared workout taxonomy for FitNova data generation and inference.

The goal is to infer a program's training intent from its title and exercise list
without over-promoting incidental accessory movements into full program labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable


WORKOUT_TYPES = ("Cardio", "Strength", "Yoga", "HIIT")


@dataclass(frozen=True)
class KeywordGroup:
    strong: tuple[str, ...]
    weak: tuple[str, ...]


TITLE_KEYWORDS: dict[str, KeywordGroup] = {
    "Yoga": KeywordGroup(
        strong=(
            "mobility",
            "stretching",
            "stretch",
            "posture",
            "yoga",
            "yin",
            "vinyasa",
            "restorative",
            "flexibility",
        ),
        weak=(
            "recovery",
            "reset",
        ),
    ),
    "Strength": KeywordGroup(
        strong=(
            "strength",
            "powerlifting",
            "powerbuilding",
            "hypertrophy",
            "bodybuilding",
            "muscle",
            "mass",
            "strong",
            "upper/lower",
            "ppl",
            "full body",
            "full-body",
            "531",
            "5/3/1",
        ),
        weak=(
            "garage gym",
            "split",
            "bulk",
            "size",
            "bench",
            "squat",
        ),
    ),
    "Cardio": KeywordGroup(
        strong=(
            "cardio",
            "running",
            "run",
            "walking",
            "walk",
            "cycling",
            "cycling",
            "endurance",
            "aerobic",
            "liss",
            "marathon",
        ),
        weak=(
            "conditioning",
            "fat loss",
            "cut",
        ),
    ),
    "HIIT": KeywordGroup(
        strong=(
            "hiit",
            "tabata",
            "emom",
            "amrap",
            "metcon",
            "interval",
            "circuit",
        ),
        weak=(
            "conditioning",
            "explosive",
            "athletic",
            "hyrox",
        ),
    ),
}


EXERCISE_KEYWORDS: dict[str, KeywordGroup] = {
    "Yoga": KeywordGroup(
        strong=(
            "stretch",
            "mobility",
            "90/90",
            "pigeon",
            "downward",
            "sun salutation",
            "child pose",
            "cobra",
            "breathing",
            "thoracic rotation",
            "hip airplane",
            "ankle dorsiflexion",
            "banded ankle distraction",
            "shoulder dislocate",
            "hip opener",
            "cat cow",
            "world's greatest stretch",
        ),
        weak=(
            "dead bug",
            "bird dog",
            "wall slide",
            "t-spine",
        ),
    ),
    "Strength": KeywordGroup(
        strong=(
            "squat",
            "deadlift",
            "bench",
            "press",
            "row",
            "pulldown",
            "pull-up",
            "chin-up",
            "lunge",
            "extension",
            "curl",
            "fly",
            "raise",
            "barbell",
            "dumbbell",
            "machine",
            "cable",
            "smith",
            "shrug",
            "dip",
            "hip thrust",
            "leg press",
            "calf raise",
        ),
        weak=(
            "push up",
            "push-up",
            "sit up",
            "sit-up",
            "good morning",
        ),
    ),
    "Cardio": KeywordGroup(
        strong=(
            "run",
            "jog",
            "walk",
            "bike",
            "cycle",
            "swim",
            "treadmill",
            "elliptical",
            "stair master",
            "rower",
            "rowing machine",
            "cardio",
        ),
        weak=(
            "shadow boxing",
            "kick boxing",
            "sled pull",
            "sled row",
        ),
    ),
    "HIIT": KeywordGroup(
        strong=(
            "burpee",
            "tabata",
            "emom",
            "amrap",
            "interval",
            "circuit",
            "box jump",
            "battle rope",
            "jump rope",
            "mountain climber",
            "sled push",
            "clean and press",
        ),
        weak=(
            "explosive",
            "plyo",
            "sprint",
            "farmer's walk",
            "conditioning",
        ),
    ),
}


RELATED_TYPES = {
    "Cardio": {"HIIT"},
    "HIIT": {"Cardio", "Strength"},
    "Strength": {"HIIT"},
    "Yoga": set(),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _match_weight(text: str, keywords: KeywordGroup, strong_weight: float, weak_weight: float) -> float:
    score = 0.0
    for kw in keywords.strong:
        if kw in text:
            score += strong_weight
    for kw in keywords.weak:
        if kw in text:
            score += weak_weight
    return score


def _exercise_evidence(exercise_names: Iterable[str]) -> tuple[Counter, Counter]:
    strong_counts: Counter[str] = Counter()
    weak_counts: Counter[str] = Counter()

    for raw_name in {str(name).strip() for name in exercise_names if str(name).strip()}:
        name = _normalize(raw_name)
        for workout_type, groups in EXERCISE_KEYWORDS.items():
            matched_strong = any(kw in name for kw in groups.strong)
            matched_weak = any(kw in name for kw in groups.weak)
            if matched_strong:
                strong_counts[workout_type] += 1
            elif matched_weak:
                weak_counts[workout_type] += 1

    return strong_counts, weak_counts


def infer_program_type_profile(title: str, exercise_names: Iterable[str]) -> dict:
    normalized_title = _normalize(title)
    unique_exercises = [str(name).strip() for name in exercise_names if str(name).strip()]
    unique_count = max(1, len(set(unique_exercises)))

    strong_counts, weak_counts = _exercise_evidence(unique_exercises)
    title_scores = {
        workout_type: _match_weight(normalized_title, TITLE_KEYWORDS[workout_type], 6.0, 2.5)
        for workout_type in WORKOUT_TYPES
    }

    scores = {
        workout_type: title_scores[workout_type]
        + strong_counts[workout_type] * 1.4
        + weak_counts[workout_type] * 0.35
        for workout_type in WORKOUT_TYPES
    }
    shares = {
        workout_type: (strong_counts[workout_type] + weak_counts[workout_type]) / unique_count
        for workout_type in WORKOUT_TYPES
    }

    if (
        title_scores["Yoga"] >= 6.0
        and strong_counts["Yoga"] >= 5
        and shares["Yoga"] >= 0.15
    ):
        scores["Yoga"] += 14.0

    has_flags = {
        "Strength": (
            title_scores["Strength"] >= 6.0
            or strong_counts["Strength"] >= 4
            or shares["Strength"] >= 0.22
        ),
        "Cardio": (
            title_scores["Cardio"] >= 6.0
            or strong_counts["Cardio"] >= 2
            or shares["Cardio"] >= 0.12
        ),
        "HIIT": (
            title_scores["HIIT"] >= 6.0
            or strong_counts["HIIT"] >= 2
            or (
                title_scores["HIIT"] >= 2.5
                and (strong_counts["HIIT"] + weak_counts["HIIT"]) >= 3
            )
        ),
        "Yoga": (
            title_scores["Yoga"] >= 6.0
            or (
                strong_counts["Yoga"] >= 4
                and shares["Yoga"] >= 0.18
            )
            or (
                title_scores["Yoga"] >= 2.5
                and (strong_counts["Yoga"] + weak_counts["Yoga"]) >= 5
                and shares["Yoga"] >= 0.15
            )
        ),
    }

    # Guard against incidental mobility/core work marking whole strength plans as Yoga.
    if has_flags["Yoga"]:
        yoga_evidence = strong_counts["Yoga"] + weak_counts["Yoga"]
        strength_evidence = strong_counts["Strength"] + weak_counts["Strength"]
        if (
            title_scores["Yoga"] < 6.0
            and yoga_evidence < 6
            and strength_evidence >= max(8, yoga_evidence * 2)
        ):
            has_flags["Yoga"] = False

    active_types = [workout_type for workout_type in WORKOUT_TYPES if has_flags[workout_type]]
    if not active_types:
        active_types = ["Strength"]
        has_flags["Strength"] = True

    primary_type = max(active_types, key=lambda workout_type: (scores[workout_type], shares[workout_type]))
    if (
        has_flags["Yoga"]
        and title_scores["Yoga"] >= 6.0
        and title_scores["Strength"] < 6.0
        and strong_counts["Yoga"] >= 5
        and shares["Yoga"] >= 0.18
    ):
        primary_type = "Yoga"
    secondary_types = [workout_type for workout_type in active_types if workout_type != primary_type]

    return {
        "scores": scores,
        "shares": shares,
        "title_scores": title_scores,
        "strong_counts": dict(strong_counts),
        "weak_counts": dict(weak_counts),
        "primary_type": primary_type,
        "secondary_types": secondary_types,
        "has_cardio": int(has_flags["Cardio"]),
        "has_strength": int(has_flags["Strength"]),
        "has_yoga": int(has_flags["Yoga"]),
        "has_hiit": int(has_flags["HIIT"]),
    }
