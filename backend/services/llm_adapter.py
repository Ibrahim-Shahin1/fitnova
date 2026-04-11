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
import logging
import os
import pickle
import re
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI

logger = logging.getLogger("fitnova.llm")

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_CATALOG_PATH = os.path.join(_DATA_DIR, "program_catalog.pkl")
_ENV_PATH = os.path.join(_BASE_DIR, ".env")

# Max exercises per day shown in the LLM prompt (keeps context bounded)
_MAX_PROMPT_EXERCISES_PER_DAY = 8

# Max exercises per day in the template fallback (keeps plans readable)
_MAX_TEMPLATE_EXERCISES = {1: 5, 2: 7, 3: 8}  # by experience level

# Reps threshold above which a time-based value is considered garbage for Strength
_MAX_REASONABLE_HOLD_SECONDS = 120


class LLMAdapter:
    SYSTEM_PROMPT = """You are FitNova's expert fitness coach. Your job is to create a structured, \
realistic weekly training plan that a real gym-goer would follow.

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

STRICT RULES:
- The plan must contain exactly 7 days named day_1 through day_7.
- Each day must include: day_number (int), focus (str), is_rest_day (bool), exercises (list).
- Rest days: is_rest_day=true, exercises=[].
- The NUMBER of workout days must equal workout_frequency exactly.
- For each workout day exercise include: exercise_name, sets (int), reps (str), rest_seconds (int), coaching_cue (str).
- Volume by experience level:
    beginner    = 2-3 sets, 10-15 reps, 60-90s rest
    intermediate = 3-4 sets, 8-12 reps, 90-120s rest
    advanced    = 4-5 sets, 5-8 reps, 120-180s rest
- program_title: Write a CLEAR, DESCRIPTIVE title like "Intermediate Push/Pull/Legs Strength" or \
"Beginner 3-Day Full Body". NEVER copy a gibberish or single-word raw program title.
- focus: Label each day precisely — "Push", "Pull", "Legs", "Upper Body", "Lower Body", "Full Body", \
"Conditioning", "Mobility & Recovery", etc. — based on the ACTUAL exercises in that day.
- coaching_cue: Write SPECIFIC technique cues for each exercise (2 sentences). \
DO NOT write the same generic cue for every exercise.
- Keep exercise names practical. Do not invent medical advice.
- Use the supplied week 1 exercise list as a reference for exercise selection and structure. \
You may add or substitute exercises to make the plan coherent.
- If training_focus is provided, adapt the plan accordingly:
  * "powerbuilding" — first 1-2 exercises per session are heavy compounds (bench/squat/deadlift) \
at 3-5 reps, followed by 3-5 hypertrophy accessories at 8-12 reps.
  * "powerlifting" — focus on big-3 + close variants at low reps (3-5), long rest (3-5 min).
  * "hypertrophy" — classic bodybuilding split, 8-15 reps, moderate rest (60-90s).
  * "general" — balanced mix of upper, lower, and core in every session.
- If available_equipment is listed, you MUST ONLY select exercises that can be performed \
with that equipment. Do NOT include any exercise requiring unlisted equipment. \
Cables are a separate equipment type (cable machine) — do not substitute with cables unless \
"Cables" is in the available equipment list.
"""

    def __init__(
        self,
        catalog_path: str = _CATALOG_PATH,
        env_path: str = _ENV_PATH,
        client: OpenAI | None = None,
    ):
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to backend/.env or your environment."
            )

        self.client = client or OpenAI(api_key=api_key)
        with open(catalog_path, "rb") as handle:
            self.catalog = pickle.load(handle)

        # ── Injury exercise blacklist ─────────────────────────────────────
        blacklist_path = os.path.join(_DATA_DIR, "injury_exercise_blacklist.json")
        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                self.injury_blacklist: dict = json.load(f)
        except FileNotFoundError:
            logger.warning("injury_exercise_blacklist.json not found — injury filtering disabled")
            self.injury_blacklist = {}

        # ── Exercise video lookup ─────────────────────────────────────────
        videos_path = os.path.join(_DATA_DIR, "exercise_videos.json")
        try:
            from rapidfuzz import process as _fuzz_process, fuzz as _fuzz
            with open(videos_path, "r", encoding="utf-8") as f:
                videos_data: dict = json.load(f)
            self.video_lookup: dict = {k: v for k, v in videos_data.items() if k != "_name_to_slug"}
            # Build a flat list of (normalised_alias, slug) for fuzzy matching
            self._video_choices: list[tuple[str, str]] = []
            for alias, slug in videos_data.get("_name_to_slug", {}).items():
                self._video_choices.append((alias, slug))
            # Also add slug display names as choices
            for slug, info in self.video_lookup.items():
                self._video_choices.append((slug.replace("_", " "), slug))
            self._fuzz_process = _fuzz_process
            self._fuzz = _fuzz
        except FileNotFoundError:
            logger.warning("exercise_videos.json not found — video demos disabled")
            self.video_lookup = {}
            self._video_choices = []
            self._fuzz_process = None
            self._fuzz = None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_plan(self, program_id: int, user_profile: dict) -> dict:
        program = self._get_program(program_id)
        try:
            user_prompt = self._build_user_prompt(program, user_profile)
            llm_payload = self._call_llm(user_prompt)
            result = self._validate_response(llm_payload, program, user_profile)
            result["source"] = "llm"
            return result
        except Exception as exc:
            logger.error("LLM call failed, using template fallback: %s", exc)
            result = self._build_template_plan(program, user_profile)
            result["source"] = "template_fallback"
            return result

    # ── LLM Call ──────────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> dict:
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=4000,
                    timeout=60,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty content")
                return json.loads(content)
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
                if attempt == 2:
                    raise
            except json.JSONDecodeError as exc:
                raise ValueError("LLM response was not valid JSON") from exc

        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed unexpectedly")

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def _build_user_prompt(self, program: dict, user_profile: dict) -> str:
        grouped = self._group_week1_exercises(program)
        level = int(user_profile.get("experience_level", 2))

        lines = [
            "Build a personalized 7-day weekly workout plan from this program.",
            "",
            "User profile:",
            f"- experience_level: {self._experience_label(level)}",
            f"- workout_type: {user_profile.get('workout_type', 'Strength')}",
            f"- session_duration_hours: {user_profile.get('session_duration_hours', 1.0)}",
            f"- workout_frequency: {user_profile.get('workout_frequency', 3)} days/week",
            f"- age: {user_profile.get('age', 'unknown')}",
            f"- gender: {user_profile.get('gender', 'unknown')}",
            f"- bmi: {user_profile.get('bmi', 'unknown')}",
        ]

        # Optional enrichment fields
        if user_profile.get("training_focus"):
            lines.append(f"- training_focus: {user_profile['training_focus']}")
        if user_profile.get("years_training") is not None:
            lines.append(f"- years_training: {user_profile['years_training']}")

        lines += [
            "",
            "Program details:",
            f"- title: {program.get('title', 'Unknown Program')}",
            f"- primary_type: {program.get('primary_type', 'Strength')}",
            f"- goal: {program.get('goal', 'General Fitness')}",
            f"- equipment: {program.get('equipment', 'Unknown')}",
            f"- level: {self._experience_label(int(program.get('level_encoded', 1)))}",
            f"- time_per_workout_minutes: {program.get('time_per_workout_minutes', 60)}",
            "",
            f"Week 1 exercises (up to {_MAX_PROMPT_EXERCISES_PER_DAY} per day shown):",
        ]

        avoid_kws = self._get_avoid_keywords(user_profile.get("injuries", []))

        for day_number in sorted(grouped):
            day_exercises = grouped[day_number][:_MAX_PROMPT_EXERCISES_PER_DAY]
            lines.append(f"Day {day_number}:")
            for exercise in day_exercises:
                ex_name_lower = str(exercise.get("exercise_name", "")).lower()
                if avoid_kws and any(kw in ex_name_lower for kw in avoid_kws):
                    continue  # skip exercises that conflict with user injuries
                clean_reps = self._sanitize_reps(
                    str(exercise.get("reps") or ""),
                    program.get("primary_type", "Strength"),
                    level,
                    str(exercise.get("exercise_name", "")),
                )
                lines.append(
                    f"  - {exercise['exercise_name']} | sets={exercise.get('sets', 3)} | reps={clean_reps}"
                )

        lines.append("")
        lines.append(
            "Note: Use the exercise list as a reference. "
            "Ensure the split and focus labels match the exercises you assign."
        )

        # ── Equipment constraint (hard rule) ─────────────────────────────
        equipment = user_profile.get("equipment", [])
        if equipment:
            eq_str = ", ".join(equipment)
            lines.append("")
            lines.append("## EQUIPMENT CONSTRAINT — STRICTLY ENFORCED")
            lines.append(f"The user ONLY has access to: {eq_str}")
            lines.append(
                "Do NOT include ANY exercise requiring equipment NOT in that list. "
                "Cables = cable machine (requires Cables). "
                "Barbell exercises require Barbell. "
                "Machine exercises require Machines. "
                "Only substitute exercises using equipment the user actually has."
            )

        # ── Injury safety constraints (appended to prompt) ───────────────
        injuries = user_profile.get("injuries", [])
        if injuries:
            nice_injuries = ", ".join(i.replace("_", " ") for i in injuries)
            lines.append("")
            lines.append("## SAFETY CONSTRAINTS — NEVER VIOLATE")
            lines.append(f"The user has these injuries: {nice_injuries}")
            lines.append("")
            lines.append(
                "NEVER include exercises whose name contains any of these substrings:"
            )
            for kw in avoid_kws:
                lines.append(f"  - {kw}")
            lines.append("")
            lines.append(
                "In personalization_notes, briefly mention the injury accommodations you made."
            )

        return "\n".join(lines)

    # ── Template Fallback ─────────────────────────────────────────────────────

    def _build_template_plan(self, program: dict, user_profile: dict) -> dict:
        grouped = self._group_week1_exercises(program)
        workout_frequency = max(0, min(7, int(user_profile.get("workout_frequency", 3))))
        scheduled_days = self._select_workout_days(workout_frequency)
        source_days = sorted(grouped) or [1]
        level = int(user_profile.get("experience_level", 2))
        max_ex = _MAX_TEMPLATE_EXERCISES.get(level, 7)

        plan: dict[str, dict[str, Any]] = {}
        source_index = 0
        day_focuses: list[str] = []

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
            raw_exercises = grouped.get(source_day, [])

            # Filter garbage exercise names AND injury-unsafe exercises, then cap count
            avoid_kws = self._get_avoid_keywords(user_profile.get("injuries", []))
            valid_exercises = [
                ex for ex in raw_exercises
                if self._is_valid_exercise_name(str(ex.get("exercise_name", "")))
                and not (
                    avoid_kws
                    and any(
                        kw in str(ex.get("exercise_name", "")).lower()
                        for kw in avoid_kws
                    )
                )
            ][:max_ex]

            focus = self._infer_focus(program, source_day, valid_exercises)
            day_focuses.append(focus)

            plan[key] = {
                "day_number": day_number,
                "focus": focus,
                "is_rest_day": False,
                "exercises": [
                    self._normalize_exercise(
                        ex, level, program.get("primary_type", "Strength")
                    )
                    for ex in valid_exercises
                ],
            }

        title = self._sanitize_title(
            program.get("title", ""),
            user_profile,
            level,
            workout_frequency,
            day_focuses,
        )

        return {
            "program_title": title,
            "personalization_notes": (
                f"Personalized for a {self._experience_label(level).lower()} "
                f"{user_profile.get('workout_type', 'Strength').lower()} user "
                f"training {workout_frequency}x per week."
            ),
            "plan": plan,
        }

    # ── Validate LLM Response ─────────────────────────────────────────────────

    def _validate_response(self, payload: dict, program: dict, user_profile: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")

        plan = payload.get("plan")
        expected_keys = [f"day_{i}" for i in range(1, 8)]
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
                "focus": day_payload.get("focus")
                or ("Rest & Recovery" if is_rest_day else f"Workout Day {index}"),
                "is_rest_day": is_rest_day,
                "exercises": [] if is_rest_day else normalized_exercises,
            }

        if workout_day_count != int(user_profile.get("workout_frequency", 3)):
            raise ValueError("Workout day count does not match workout_frequency")

        # ── Post-validation: remove injury-unsafe exercises ──────────────
        injuries = user_profile.get("injuries", [])
        if injuries:
            avoid_kws = self._get_avoid_keywords(injuries)
            for day_key, day_data in normalized_plan.items():
                if day_data.get("is_rest_day"):
                    continue
                original = day_data.get("exercises", [])
                safe = [
                    ex for ex in original
                    if not any(kw in ex["exercise_name"].lower() for kw in avoid_kws)
                ]
                if len(safe) != len(original):
                    removed = len(original) - len(safe)
                    logger.info(
                        "Removed %d unsafe exercises from %s for injuries %s",
                        removed, day_key, injuries,
                    )
                day_data["exercises"] = safe

        raw_title = str(payload.get("program_title") or "")
        clean_title = self._sanitize_title(
            raw_title,
            user_profile,
            level,
            int(user_profile.get("workout_frequency", 3)),
        )

        return {
            "program_title": clean_title,
            "personalization_notes": str(
                payload.get("personalization_notes")
                or f"Adapted for {self._experience_label(level).lower()} level."
            ),
            "plan": normalized_plan,
        }

    # ── Injury Helpers ─────────────────────────────────────────────────────────

    def _get_avoid_keywords(self, injuries: list[str]) -> list[str]:
        """Collect all avoid_keywords for the given injury IDs."""
        keywords: list[str] = []
        for injury_id in injuries:
            entry = self.injury_blacklist.get(injury_id, {})
            keywords.extend(entry.get("avoid_keywords", []))
        return [k.lower() for k in keywords]

    # ── Core Exercise Normalizer ───────────────────────────────────────────────

    def _normalize_exercise(
        self,
        exercise: dict,
        experience_level: int,
        primary_type: str,
        source_is_llm: bool = False,
    ) -> dict:
        # Sets: always clamp to experience-appropriate range
        default_sets = {1: 3, 2: 4, 3: 5}.get(experience_level, 4)
        raw_sets = exercise.get("sets", default_sets)
        try:
            sets = int(raw_sets)
        except (TypeError, ValueError):
            sets = default_sets

        min_sets, max_sets = {1: (2, 3), 2: (3, 4), 3: (4, 5)}.get(experience_level, (3, 4))
        sets = max(min_sets, min(max_sets, sets))

        # Reps: sanitize garbage time-based values from catalog
        raw_reps = str(exercise.get("reps") or "")
        reps = self._sanitize_reps(
            raw_reps,
            primary_type,
            experience_level,
            str(exercise.get("exercise_name", "")),
        )

        # Rest & coaching cue
        if source_is_llm:
            rest_seconds = exercise.get("rest_seconds") or self._default_rest_seconds(
                primary_type, experience_level
            )
            coaching_cue = exercise.get("coaching_cue") or self._default_coaching_cue(
                primary_type, exercise.get("exercise_name", "exercise")
            )
        else:
            rest_seconds = self._default_rest_seconds(primary_type, experience_level)
            coaching_cue = self._default_coaching_cue(
                primary_type, exercise.get("exercise_name", "exercise")
            )

        exercise_name = str(exercise.get("exercise_name", "Unknown Exercise"))
        result = {
            "exercise_name": exercise_name,
            "sets": sets,
            "reps": reps,
            "rest_seconds": int(rest_seconds),
            "coaching_cue": coaching_cue,
        }

        # Inject exercise demo video if available
        if self._video_choices and self._fuzz_process:
            # Normalise: strip leading "(MUSCLE) " and trailing " (qualifier)"
            norm = re.sub(r'^\([^)]+\)\s*', '', exercise_name)
            norm = re.sub(r'\s*\([^)]*\)\s*$', '', norm).lower().strip()

            match = self._fuzz_process.extractOne(
                norm,
                [alias for alias, _ in self._video_choices],
                scorer=self._fuzz.token_sort_ratio,
                score_cutoff=72,
            )
            if match:
                matched_alias = match[0]
                slug = next(s for a, s in self._video_choices if a == matched_alias)
                if slug in self.video_lookup:
                    demo = self.video_lookup[slug].get("demo")
                    if demo:
                        result["media_url"] = f"/static/exercise_videos/{slug}/{demo}"

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_reps(
        reps: str,
        primary_type: str,
        experience_level: int,
        exercise_name: str,
    ) -> str:
        """Replace nonsensical time-based reps values with appropriate rep ranges."""
        if not reps or reps.lower() in ("none", "as prescribed", ""):
            defaults = {1: "10", 2: "8-10", 3: "6-8"}
            return defaults.get(experience_level, "10")

        # Match "NNN sec" pattern
        match = re.match(r"^(\d+)\s*sec$", reps.strip(), re.IGNORECASE)
        if match:
            seconds = int(match.group(1))

            # Isometric / hold exercises — cap at reasonable duration
            isometric_terms = ("plank", "hold", "hang", "wall sit", "iso", "static")
            is_isometric = any(t in exercise_name.lower() for t in isometric_terms)
            if is_isometric:
                max_hold = {1: 30, 2: 45, 3: 60}.get(experience_level, 45)
                capped = min(seconds, max_hold)
                return f"{capped} sec"

            # Time-based cardio/yoga — keep if reasonable
            if primary_type in ("Cardio", "Yoga", "HIIT") and seconds <= 300:
                return reps

            # Strength or unreasonably long → replace with rep range
            if primary_type == "Strength" or seconds > _MAX_REASONABLE_HOLD_SECONDS:
                rep_defaults = {1: "10", 2: "8-10", 3: "6-8"}
                return rep_defaults.get(experience_level, "10")

        return reps

    @staticmethod
    def _sanitize_title(
        raw_title: str,
        user_profile: dict,
        level: int,
        workout_frequency: int,
        day_focuses: list[str] | None = None,
    ) -> str:
        """Return a clean, descriptive title. Replaces gibberish program titles."""
        level_label = LLMAdapter._experience_label(level)
        wtype = user_profile.get("workout_type", "Strength")

        # Detect gibberish: single word, non-alphabetic, or too short
        stripped = raw_title.replace(" ", "").replace("/", "").replace("-", "")
        is_gibberish = (
            not raw_title
            or len(raw_title.split()) < 2
            or not stripped.isalpha()
            or len(raw_title) < 5
        )

        if is_gibberish:
            split_label = LLMAdapter._infer_split_label(day_focuses or [])
            if split_label:
                return f"{level_label} {split_label} {wtype} Plan ({workout_frequency}x/week)"
            return f"{level_label} {wtype} Plan ({workout_frequency}x/week)"

        return raw_title

    @staticmethod
    def _infer_split_label(day_focuses: list[str]) -> str:
        """Infer a common split name from the focus labels of workout days."""
        workout_focuses = [f for f in day_focuses if f not in ("Rest & Recovery", "")]
        if not workout_focuses:
            return ""

        has_push = any("push" in f.lower() for f in workout_focuses)
        has_pull = any("pull" in f.lower() for f in workout_focuses)
        has_legs = any("leg" in f.lower() for f in workout_focuses)
        has_upper = any("upper" in f.lower() for f in workout_focuses)
        has_lower = any("lower" in f.lower() for f in workout_focuses)
        has_full = any("full" in f.lower() for f in workout_focuses)

        if has_push and has_pull and has_legs:
            return "Push/Pull/Legs"
        if has_upper and has_lower:
            return "Upper/Lower"
        if has_full:
            return "Full Body"
        return ""

    @staticmethod
    def _is_valid_exercise_name(name: str) -> bool:
        """Reject obviously garbage exercise names."""
        name = name.strip()
        if len(name) < 3:
            return False
        if name.isdigit():
            return False
        # Reject strings with no vowels (likely keyboard mash) if long enough
        if len(name) > 4 and not re.search(r"[aeiouAEIOU]", name):
            return False
        return True

    @staticmethod
    def _categorize_exercise(exercise_name: str) -> str:
        """Categorize an exercise by movement pattern for coaching cue selection."""
        name = exercise_name.lower()

        # Isometric holds first (before other checks)
        if any(t in name for t in ("plank", "hold", "hang", "wall sit", "iso", "static")):
            return "isometric"
        if any(t in name for t in ("crunch", "sit-up", "situp", "ab ", "oblique", "russian twist", "v-up")):
            return "core"

        # Compound lower
        if any(t in name for t in ("squat", "lunge", "leg press", "hip thrust", "romanian", "rdl", "good morning")):
            return "compound_lower"
        # Compound pull
        if any(t in name for t in ("deadlift", "row", "pull-up", "pullup", "chin-up", "chinup", "pulldown", "pull down")):
            return "compound_pull"
        # Compound push
        if any(t in name for t in ("bench press", "overhead press", "ohp", "dip", "push-up", "pushup", "chest press")):
            return "compound_push"
        # Isolation lower
        if any(t in name for t in ("leg curl", "leg extension", "calf raise", "hamstring curl", "glute")):
            return "isolation_lower"
        # Isolation upper
        if any(t in name for t in ("curl", "tricep", "triceps", "fly", "flye", "lateral raise", "face pull",
                                    "shrug", "rear delt", "pec deck", "cable cross")):
            return "isolation_upper"

        return "unknown"

    @staticmethod
    def _default_coaching_cue(primary_type: str, exercise_name: str) -> str:
        """Return an exercise-specific coaching cue based on movement category."""
        if primary_type == "Yoga":
            return f"Move through {exercise_name} with control and steady breathing. Hold each position for a full breath cycle."
        if primary_type == "Cardio":
            return f"Keep a sustainable pace during {exercise_name}. Monitor your breathing — you should be able to speak in short sentences."
        if primary_type == "HIIT":
            return f"Attack {exercise_name} with maximum effort. Maintain clean form even as fatigue builds."

        # Strength — category-specific cues
        category = LLMAdapter._categorize_exercise(exercise_name)

        cues = {
            "compound_push": (
                f"Retract your shoulder blades and brace your core before each rep of {exercise_name}. "
                "Drive through the full range of motion and squeeze at the top."
            ),
            "compound_pull": (
                f"Initiate {exercise_name} by engaging your lats, not your arms. "
                "Control the eccentric (lowering) phase — aim for 2-3 seconds down."
            ),
            "compound_lower": (
                f"Keep your chest tall and brace your core throughout {exercise_name}. "
                "Push your knees out in line with your toes and control the descent."
            ),
            "isolation_upper": (
                f"Focus on the mind-muscle connection during {exercise_name}. "
                "Use a slow, deliberate tempo — 2 seconds up, pause, 2 seconds down."
            ),
            "isolation_lower": (
                f"Perform {exercise_name} through the full range of motion. "
                "Squeeze hard at peak contraction and control the return."
            ),
            "core": (
                f"Maintain a neutral spine throughout {exercise_name}. "
                "Exhale on the exertion phase and keep tension on the abs — avoid using momentum."
            ),
            "isometric": (
                f"Hold {exercise_name} with your body in a straight line and breathe steadily. "
                "Focus on maintaining tension without compensating with other muscle groups."
            ),
            "unknown": (
                f"Maintain proper posture and a controlled tempo throughout {exercise_name}. "
                "Stop a rep or two before complete failure to preserve form."
            ),
        }
        return cues[category]

    @staticmethod
    def _default_rest_seconds(primary_type: str, experience_level: int) -> int:
        if primary_type == "Strength":
            return 90 if experience_level <= 2 else 120
        if primary_type == "Yoga":
            return 45
        if primary_type == "Cardio":
            return 60
        return 75

    def _get_program(self, program_id: int) -> dict:
        if isinstance(self.catalog, dict):
            if program_id not in self.catalog:
                raise KeyError(f"Unknown program_id: {program_id}")
            return self.catalog[program_id]
        return self.catalog[program_id]

    def _group_week1_exercises(self, program: dict) -> dict[int, list[dict]]:
        exercises = program.get("week1_exercises") or [
            ex for ex in program.get("exercises", []) if int(ex.get("week", 1)) == 1
        ]
        grouped: dict[int, list[dict]] = defaultdict(list)
        for ex in exercises:
            day = int(ex.get("day", 1))
            grouped[day].append(ex)
        return dict(grouped)

    def _infer_focus(self, program: dict, source_day: int, exercises: list[dict]) -> str:
        if not exercises:
            return f"{program.get('primary_type', 'Workout')} Day {source_day}"

        upper_terms = ("bench", "press", "row", "pull", "curl", "tricep", "shoulder", "fly", "delt")
        lower_terms = ("squat", "deadlift", "lunge", "calf", "leg", "glute", "hip", "hamstring", "quad")
        mobility_terms = ("stretch", "mobility", "flow", "pose", "yoga")
        cardio_terms = ("run", "bike", "sprint", "cardio", "interval")

        upper_count = 0
        lower_count = 0
        for ex in exercises:
            name = str(ex.get("exercise_name", "")).lower()
            if any(t in name for t in upper_terms):
                upper_count += 1
            if any(t in name for t in lower_terms):
                lower_count += 1

        if upper_count > 0 and lower_count > 0:
            if upper_count > lower_count:
                return "Upper Body"
            elif lower_count > upper_count:
                return "Lower Body"
            else:
                return "Full Body"
        if upper_count > 0:
            return "Upper Body"
        if lower_count > 0:
            return "Lower Body"

        names = " ".join(str(ex.get("exercise_name", "")).lower() for ex in exercises)
        if any(t in names for t in mobility_terms):
            return "Mobility & Recovery"
        if any(t in names for t in cardio_terms):
            return "Conditioning"

        return f"{program.get('primary_type', 'Workout')} Day {source_day}"

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

    @staticmethod
    def _experience_label(experience_level: int) -> str:
        return {1: "Beginner", 2: "Intermediate", 3: "Advanced"}.get(
            experience_level, "Intermediate"
        )
