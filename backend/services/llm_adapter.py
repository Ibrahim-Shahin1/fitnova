"""
FitNova - Layer 3: LLM Adaptation
=================================
Transforms a selected program into a personalized 7-day weekly plan.
Falls back to a deterministic template plan whenever the LLM call fails
or returns an invalid schema.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
import pickle
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI


_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_CATALOG_PATH = os.path.join(_DATA_DIR, "program_catalog.pkl")
_ENV_PATH = os.path.join(_BASE_DIR, ".env")


class LLMAdapter:
    SYSTEM_PROMPT = """You are FitNova's fitness coach.

Return ONLY valid JSON with this exact top-level structure:
{
  "program_title": "...",
  "personalization_notes": "...",
  "plan": {
    "day_1": {...},
    "day_2": {...},
    "day_3": {...},
    "day_4": {...},
    "day_5": {...},
    "day_6": {...},
    "day_7": {...}
  }
}

Rules:
- The plan must contain exactly 7 days named day_1 through day_7.
- Each day must include: day_number, focus, is_rest_day, exercises.
- Rest days must use: is_rest_day=true and exercises=[].
- Workout days must equal the user's workout_frequency exactly.
- For workout days, each exercise item must include:
  exercise_name, sets, reps, rest_seconds, coaching_cue.
- Adjust volume by level:
  beginner = 2-3 sets
  intermediate = 3-4 sets
  advanced = 4-5 sets
- Respect the supplied program structure and week 1 exercise list.
- Keep exercise names practical and do not invent medical advice.
"""

    def __init__(self, catalog_path: str = _CATALOG_PATH, env_path: str = _ENV_PATH, client: OpenAI | None = None):
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to backend/.env or your environment.")

        self.client = client or OpenAI(api_key=api_key)
        with open(catalog_path, "rb") as handle:
            self.catalog = pickle.load(handle)

    def generate_plan(self, program_id: int, user_profile: dict) -> dict:
        program = self._get_program(program_id)
        try:
            user_prompt = self._build_user_prompt(program, user_profile)
            llm_payload = self._call_llm(user_prompt)
            result = self._validate_response(llm_payload, program, user_profile)
            result["source"] = "llm"
            return result
        except Exception:
            result = self._build_template_plan(program, user_profile)
            result["source"] = "template_fallback"
            return result

    def _call_llm(self, user_prompt: str) -> dict:
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=3000,
                    timeout=30,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty content")
                return json.loads(content)
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                if attempt == 1:
                    raise
            except json.JSONDecodeError as exc:
                raise ValueError("LLM response was not valid JSON") from exc

        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed unexpectedly")

    def _build_user_prompt(self, program: dict, user_profile: dict) -> str:
        grouped = self._group_week1_exercises(program)
        lines = [
            "Build a personalized 7-day weekly workout plan from this program.",
            "",
            "User profile:",
            f"- experience_level: {self._experience_label(user_profile.get('experience_level', 2))}",
            f"- workout_type: {user_profile.get('workout_type', 'Strength')}",
            f"- session_duration_hours: {user_profile.get('session_duration_hours', 1.0)}",
            f"- workout_frequency: {user_profile.get('workout_frequency', 3)}",
            f"- age: {user_profile.get('age', 'unknown')}",
            f"- gender: {user_profile.get('gender', 'unknown')}",
            f"- bmi: {user_profile.get('bmi', 'unknown')}",
            "",
            "Program details:",
            f"- title: {program.get('title', 'Unknown Program')}",
            f"- primary_type: {program.get('primary_type', 'Strength')}",
            f"- goal: {program.get('goal', 'General Fitness')}",
            f"- equipment: {program.get('equipment', 'Unknown')}",
            f"- level_encoded: {program.get('level_encoded', 1)}",
            f"- time_per_workout_minutes: {program.get('time_per_workout_minutes', 60)}",
            "",
            "Week 1 exercises grouped by day:",
        ]

        for day_number in sorted(grouped):
            lines.append(f"Day {day_number}:")
            for exercise in grouped[day_number]:
                lines.append(
                    f"- {exercise['exercise_name']} | sets={exercise['sets']} | reps={exercise['reps']}"
                )

        return "\n".join(lines)

    def _build_template_plan(self, program: dict, user_profile: dict) -> dict:
        grouped = self._group_week1_exercises(program)
        workout_frequency = max(0, min(7, int(user_profile.get("workout_frequency", 3))))
        scheduled_days = self._select_workout_days(workout_frequency)
        source_days = sorted(grouped) or [1]
        level = int(user_profile.get("experience_level", 2))

        plan: dict[str, dict[str, Any]] = {}
        source_index = 0
        for day_number in range(1, 8):
            key = f"day_{day_number}"
            if day_number not in scheduled_days:
                plan[key] = {
                    "day_number": day_number,
                    "focus": "Rest & Recovery",
                    "is_rest_day": True,
                    "exercises": [],
                }
                continue

            source_day = source_days[source_index % len(source_days)]
            source_index += 1
            source_exercises = grouped.get(source_day, [])
            focus = self._infer_focus(program, source_day, source_exercises)
            plan[key] = {
                "day_number": day_number,
                "focus": focus,
                "is_rest_day": False,
                "exercises": [
                    self._normalize_exercise(exercise, level, program.get("primary_type", "Strength"))
                    for exercise in source_exercises
                ],
            }

        return {
            "program_title": program.get("title", f"Program {program.get('program_id', 'unknown')}"),
            "personalization_notes": (
                f"Template fallback generated for a {self._experience_label(level).lower()} "
                f"{user_profile.get('workout_type', 'Strength').lower()} user with "
                f"{workout_frequency} workout days per week."
            ),
            "plan": plan,
        }

    def _validate_response(self, payload: dict, program: dict, user_profile: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")

        plan = payload.get("plan")
        expected_keys = [f"day_{index}" for index in range(1, 8)]
        if not isinstance(plan, dict) or set(plan.keys()) != set(expected_keys):
            raise ValueError("Plan must contain exactly day_1 through day_7")

        level = int(user_profile.get("experience_level", 2))
        normalized_plan: dict[str, dict[str, Any]] = {}
        workout_day_count = 0
        for index, key in enumerate(expected_keys, start=1):
            day_payload = plan.get(key)
            if not isinstance(day_payload, dict):
                raise ValueError(f"{key} must be an object")

            is_rest_day = bool(day_payload.get("is_rest_day"))
            exercises = day_payload.get("exercises", [])
            if not isinstance(exercises, list):
                raise ValueError(f"{key}.exercises must be a list")
            if is_rest_day and exercises:
                raise ValueError(f"{key} is marked as rest but includes exercises")

            normalized_exercises = []
            if not is_rest_day:
                workout_day_count += 1
                for exercise in exercises:
                    if not isinstance(exercise, dict) or "exercise_name" not in exercise:
                        raise ValueError(f"{key} contains an invalid exercise entry")
                    normalized_exercises.append(
                        self._normalize_exercise(
                            exercise,
                            level,
                            program.get("primary_type", "Strength"),
                            source_is_llm=True,
                        )
                    )

            normalized_plan[key] = {
                "day_number": int(day_payload.get("day_number", index)),
                "focus": day_payload.get("focus") or ("Rest & Recovery" if is_rest_day else f"Workout Day {index}"),
                "is_rest_day": is_rest_day,
                "exercises": [] if is_rest_day else normalized_exercises,
            }

        if workout_day_count != int(user_profile.get("workout_frequency", 3)):
            raise ValueError("Workout day count does not match workout_frequency")

        return {
            "program_title": str(payload.get("program_title") or program.get("title", "Unknown Program")),
            "personalization_notes": str(
                payload.get("personalization_notes")
                or f"Adapted for {self._experience_label(level).lower()} level."
            ),
            "plan": normalized_plan,
        }

    def _get_program(self, program_id: int) -> dict:
        if isinstance(self.catalog, dict):
            if program_id not in self.catalog:
                raise KeyError(f"Unknown program_id: {program_id}")
            return self.catalog[program_id]
        return self.catalog[program_id]

    def _group_week1_exercises(self, program: dict) -> dict[int, list[dict]]:
        exercises = program.get("week1_exercises") or [
            exercise for exercise in program.get("exercises", []) if int(exercise.get("week", 1)) == 1
        ]
        grouped: dict[int, list[dict]] = defaultdict(list)
        for exercise in exercises:
            day = int(exercise.get("day", 1))
            grouped[day].append(exercise)
        return dict(grouped)

    def _select_workout_days(self, workout_frequency: int) -> list[int]:
        presets = {
            0: [],
            1: [1],
            2: [1, 4],
            3: [1, 3, 5],
            4: [1, 3, 5, 7],
            5: [1, 2, 4, 5, 7],
            6: [1, 2, 3, 5, 6, 7],
            7: [1, 2, 3, 4, 5, 6, 7],
        }
        return presets[max(0, min(7, workout_frequency))]

    def _infer_focus(self, program: dict, source_day: int, exercises: list[dict]) -> str:
        if exercises:
            names = " ".join(str(ex.get("exercise_name", "")).lower() for ex in exercises)
            if any(term in names for term in ("bench", "press", "row", "pull", "curl", "tricep", "shoulder")):
                return "Upper Body"
            if any(term in names for term in ("squat", "deadlift", "lunge", "calf", "leg", "glute", "hip")):
                return "Lower Body"
            if any(term in names for term in ("stretch", "mobility", "flow", "pose", "yoga")):
                return "Mobility & Recovery"
            if any(term in names for term in ("run", "bike", "sprint", "cardio", "interval")):
                return "Conditioning"
        return f"{program.get('primary_type', 'Workout')} Day {source_day}"

    def _normalize_exercise(
        self,
        exercise: dict,
        experience_level: int,
        primary_type: str,
        source_is_llm: bool = False,
    ) -> dict:
        default_sets = {1: 3, 2: 4, 3: 5}.get(experience_level, 4)
        raw_sets = exercise.get("sets", default_sets)
        try:
            sets = int(raw_sets)
        except (TypeError, ValueError):
            sets = default_sets

        if source_is_llm:
            min_sets, max_sets = {1: (2, 3), 2: (3, 4), 3: (4, 5)}.get(experience_level, (3, 4))
            sets = max(min_sets, min(max_sets, sets))

        reps = str(exercise.get("reps", "As prescribed"))
        if source_is_llm:
            rest_seconds = exercise.get("rest_seconds") or self._default_rest_seconds(primary_type, experience_level)
            coaching_cue = exercise.get("coaching_cue") or self._default_coaching_cue(
                primary_type,
                exercise.get("exercise_name", "exercise"),
            )
        else:
            rest_seconds = self._default_rest_seconds(primary_type, experience_level)
            coaching_cue = self._default_coaching_cue(primary_type, exercise.get("exercise_name", "exercise"))

        return {
            "exercise_name": str(exercise.get("exercise_name", "Unknown Exercise")),
            "sets": sets,
            "reps": reps,
            "rest_seconds": rest_seconds,
            "coaching_cue": coaching_cue,
        }

    @staticmethod
    def _default_rest_seconds(primary_type: str, experience_level: int) -> int:
        if primary_type == "Strength":
            return 90 if experience_level <= 2 else 120
        if primary_type == "Yoga":
            return 45
        if primary_type == "Cardio":
            return 60
        return 75

    @staticmethod
    def _default_coaching_cue(primary_type: str, exercise_name: str) -> str:
        if primary_type == "Yoga":
            return f"Move through {exercise_name} with control and steady breathing."
        if primary_type == "Cardio":
            return f"Keep a sustainable pace during {exercise_name} and monitor your breathing."
        if primary_type == "HIIT":
            return f"Attack {exercise_name} with intensity while maintaining clean form."
        return f"Use controlled form on {exercise_name} and stop each set before technique breaks down."

    @staticmethod
    def _experience_label(experience_level: int) -> str:
        return {1: "Beginner", 2: "Intermediate", 3: "Advanced"}.get(experience_level, "Intermediate")
