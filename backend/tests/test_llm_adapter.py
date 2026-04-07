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
