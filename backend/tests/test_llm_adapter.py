import json
import os

import pytest
from dotenv import load_dotenv

from backend.services.llm_adapter import LLMAdapter


def _make_response(payload: dict):
    class _Message:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    return _Response(json.dumps(payload))


class FakeClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.error:
            raise self.error
        return _make_response(self.payload)


def _valid_plan_payload(workout_days: list[int] | None = None) -> dict:
    workout_days = workout_days or [1, 3, 5]
    plan = {}
    for day_number in range(1, 8):
        key = f"day_{day_number}"
        if day_number in workout_days:
            plan[key] = {
                "day_number": day_number,
                "focus": f"Workout Day {day_number}",
                "is_rest_day": False,
                "exercises": [
                    {
                        "exercise_name": "Bench Press",
                        "sets": 3,
                        "reps": "10",
                        "rest_seconds": 90,
                        "coaching_cue": "Press with control.",
                    }
                ],
            }
        else:
            plan[key] = {
                "day_number": day_number,
                "focus": "Rest & Recovery",
                "is_rest_day": True,
                "exercises": [],
            }

    return {
        "program_title": "Mock Program",
        "personalization_notes": "Mocked response.",
        "plan": plan,
    }


def _make_adapter(monkeypatch, client=None):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return LLMAdapter(client=client or FakeClient(payload=_valid_plan_payload()))


def test_generate_plan_returns_valid_structure(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    result = adapter.generate_plan(
        0,
        {
            "experience_level": 1,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "age": 24,
            "gender": "Male",
            "bmi": 24.0,
        },
    )

    assert "plan" in result
    assert len(result["plan"]) == 7
    assert list(result["plan"].keys()) == [f"day_{i}" for i in range(1, 8)]


def test_fallback_on_llm_failure(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    def _raise_timeout(_prompt):
        raise TimeoutError("mock timeout")

    adapter._call_llm = _raise_timeout
    result = adapter.generate_plan(
        0,
        {
            "experience_level": 2,
            "workout_type": "Yoga",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
        },
    )

    workout_days = sum(not day["is_rest_day"] for day in result["plan"].values())
    assert workout_days == 3
    assert result["program_title"]


def test_fallback_on_invalid_schema(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._call_llm = lambda _prompt: {"unexpected": "schema"}

    result = adapter.generate_plan(
        0,
        {
            "experience_level": 2,
            "workout_type": "Yoga",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
        },
    )

    assert len(result["plan"]) == 7
    assert sum(not day["is_rest_day"] for day in result["plan"].values()) == 3


def test_template_plan_respects_workout_frequency(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    program = adapter.catalog[0] if not isinstance(adapter.catalog, dict) else adapter.catalog[0]

    result = adapter._build_template_plan(
        program,
        {
            "experience_level": 1,
            "workout_type": "Yoga",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
        },
    )

    workout_days = sum(not day["is_rest_day"] for day in result["plan"].values())
    rest_days = sum(day["is_rest_day"] for day in result["plan"].values())
    assert workout_days == 3
    assert rest_days == 4


def test_user_prompt_includes_program_exercises(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    program = adapter.catalog[0] if not isinstance(adapter.catalog, dict) else adapter.catalog[0]

    prompt = adapter._build_user_prompt(
        program,
        {
            "experience_level": 1,
            "workout_type": "Yoga",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "age": 28,
            "gender": "Female",
            "bmi": 22.1,
        },
    )

    week1 = program.get("week1_exercises", program.get("exercises", []))
    assert any(exercise["exercise_name"] in prompt for exercise in week1[:5])
    assert program["title"] in prompt


def _has_api_key() -> bool:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path=env_path)
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="OPENAI_API_KEY not configured")
def test_live_llm_call():
    adapter = LLMAdapter()
    result = adapter.generate_plan(
        0,
        {
            "experience_level": 1,
            "workout_type": "Yoga",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "age": 28,
            "gender": "Female",
            "bmi": 22.1,
        },
    )

    assert "plan" in result
    assert len(result["plan"]) == 7


# ── New quality-fix tests ─────────────────────────────────────────────────────


def test_template_sets_are_clamped(monkeypatch):
    """Template fallback must clamp sets even when catalog has sets=1."""
    adapter = _make_adapter(monkeypatch)
    # Inject a fake program with sets=1 exercises
    fake_program = {
        "program_id": 9999,
        "title": "Test Program",
        "primary_type": "Strength",
        "level_encoded": 1,
        "week1_exercises": [
            {"week": 1, "day": 1, "exercise_name": "Bench Press (Barbell)", "sets": 1, "reps": "10"},
            {"week": 1, "day": 1, "exercise_name": "Squat (Barbell)", "sets": 1, "reps": "8"},
        ],
        "exercises": [],
    }
    adapter.catalog[9999] = fake_program

    result = adapter._build_template_plan(
        fake_program,
        {"experience_level": 2, "workout_type": "Strength", "session_duration_hours": 1.0, "workout_frequency": 3},
    )

    workout_days = [d for d in result["plan"].values() if not d["is_rest_day"]]
    for day in workout_days:
        for ex in day["exercises"]:
            assert ex["sets"] >= 3, f"Intermediate user got {ex['sets']} sets for {ex['exercise_name']}"
            assert ex["sets"] <= 4, f"Too many sets {ex['sets']} for intermediate"


def test_reps_sanitization_strength_garbage():
    """'1800 sec' on a Strength exercise must become a sensible rep range."""
    result = LLMAdapter._sanitize_reps("1800 sec", "Strength", 2, "Bench Press (Barbell)")
    assert result == "8-10", f"Expected '8-10', got {result!r}"


def test_reps_sanitization_isometric_capped():
    """Time-based holds are kept but capped at a reasonable duration."""
    # 300 sec plank = 5 minutes — absurd. Should be capped.
    result = LLMAdapter._sanitize_reps("300 sec", "Strength", 2, "Plank Hold")
    assert result.endswith("sec"), f"Expected a time value, got {result!r}"
    seconds = int(result.replace("sec", "").strip())
    assert seconds <= 60, f"Isometric hold too long: {seconds}s"


def test_reps_sanitization_reasonable_left_alone():
    """Reasonable rep values like '10' must not be changed."""
    result = LLMAdapter._sanitize_reps("10", "Strength", 2, "Bench Press")
    assert result == "10"


def test_coaching_cue_compound_push():
    """Bench Press should get a push-specific cue, not the generic fallback."""
    cue = LLMAdapter._default_coaching_cue("Strength", "Bench Press (Barbell)")
    assert "shoulder" in cue.lower() or "drive" in cue.lower() or "retract" in cue.lower(), (
        f"Expected push-specific cue, got: {cue!r}"
    )
    assert "technique breaks down" not in cue


def test_coaching_cue_isolation_upper():
    """Bicep Curl should get an isolation cue mentioning mind-muscle or tempo."""
    cue = LLMAdapter._default_coaching_cue("Strength", "Bicep Curl (Dumbbell)")
    assert "mind-muscle" in cue.lower() or "tempo" in cue.lower() or "deliberate" in cue.lower(), (
        f"Expected isolation cue, got: {cue!r}"
    )


def test_template_exercise_cap(monkeypatch):
    """Template fallback must cap exercises per day based on experience level."""
    adapter = _make_adapter(monkeypatch)
    # 15 exercises on day 1 — way too many
    fake_program = {
        "program_id": 9998,
        "title": "Overloaded Program",
        "primary_type": "Strength",
        "level_encoded": 1,
        "week1_exercises": [
            {"week": 1, "day": 1, "exercise_name": f"Exercise {i}", "sets": 3, "reps": "10"}
            for i in range(15)
        ],
        "exercises": [],
    }
    adapter.catalog[9998] = fake_program

    result = adapter._build_template_plan(
        fake_program,
        {"experience_level": 2, "workout_type": "Strength", "session_duration_hours": 1.0, "workout_frequency": 1},
    )

    workout_days = [d for d in result["plan"].values() if not d["is_rest_day"]]
    assert len(workout_days) == 1
    # Intermediate cap is 7
    assert len(workout_days[0]["exercises"]) <= 7, (
        f"Got {len(workout_days[0]['exercises'])} exercises — should be capped at 7"
    )


def test_gibberish_title_sanitized():
    """Single-word gibberish titles must be replaced with a clean descriptive title."""
    user_profile = {"workout_type": "Strength"}
    result = LLMAdapter._sanitize_title("Xjxioxijxxkx", user_profile, 2, 5)
    assert "Xjxioxijxxkx" not in result
    assert "Intermediate" in result or "Strength" in result, f"Got: {result!r}"


def test_prompt_exercise_limit(monkeypatch):
    """LLM prompt must cap exercises at 8 per day regardless of catalog size."""
    adapter = _make_adapter(monkeypatch)
    # 20 exercises across 3 days
    fake_program = {
        "program_id": 9997,
        "title": "Big Program",
        "primary_type": "Strength",
        "level_encoded": 1,
        "goal": "Hypertrophy",
        "equipment": "Barbell",
        "time_per_workout_minutes": 60,
        "week1_exercises": [
            {"week": 1, "day": (i % 3) + 1, "exercise_name": f"Exercise {i}", "sets": 3, "reps": "10"}
            for i in range(30)
        ],
        "exercises": [],
    }
    adapter.catalog[9997] = fake_program

    prompt = adapter._build_user_prompt(
        fake_program,
        {"experience_level": 2, "workout_type": "Strength", "session_duration_hours": 1.0, "workout_frequency": 3},
    )

    # Exercise lines are indented with "  -" (two spaces then dash)
    exercise_lines = [l for l in prompt.splitlines() if l.startswith("  -")]
    # Max 8 per day × 3 days = 24
    assert len(exercise_lines) <= 24, f"Too many exercise lines in prompt: {len(exercise_lines)}"


# ── Injury awareness tests ───────────────────────────────────────────────────


def test_template_fallback_filters_injury_keywords(monkeypatch):
    """Knee injury → no exercises matching any knee avoid keywords in template."""
    adapter = _make_adapter(monkeypatch)
    avoid_kws = adapter._get_avoid_keywords(["knees"])
    program = adapter._get_program(list(adapter.catalog.keys())[0])
    result = adapter._build_template_plan(
        program,
        {
            "experience_level": 2,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "injuries": ["knees"],
        },
    )
    for day_key, day_data in result["plan"].items():
        for ex in day_data.get("exercises", []):
            name_lower = ex["exercise_name"].lower()
            for kw in avoid_kws:
                assert kw not in name_lower, (
                    f"Unsafe exercise '{ex['exercise_name']}' matches "
                    f"avoid keyword '{kw}' for knees injury"
                )


def test_user_prompt_includes_safety_constraints(monkeypatch):
    """Lower back injury → SAFETY CONSTRAINTS block in user prompt."""
    adapter = _make_adapter(monkeypatch)
    program = adapter._get_program(list(adapter.catalog.keys())[0])
    prompt = adapter._build_user_prompt(
        program,
        {
            "experience_level": 2,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "injuries": ["lower_back"],
        },
    )
    assert "SAFETY CONSTRAINTS" in prompt
    assert "lower back" in prompt
    assert "deadlift" in prompt


def test_no_safety_block_without_injuries(monkeypatch):
    """No injuries → no SAFETY CONSTRAINTS block in prompt."""
    adapter = _make_adapter(monkeypatch)
    program = adapter._get_program(list(adapter.catalog.keys())[0])
    prompt = adapter._build_user_prompt(
        program,
        {
            "experience_level": 2,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 3,
            "injuries": [],
        },
    )
    assert "SAFETY CONSTRAINTS" not in prompt
