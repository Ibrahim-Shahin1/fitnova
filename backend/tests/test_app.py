"""
Tests for backend/app.py — FastAPI endpoints.
All heavy services (Recommender, LLMAdapter) are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.app import app

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_BODY = {
    "experience_level": 1,
    "workout_type": "Strength",
    "session_duration_hours": 1.0,
    "workout_frequency": 3,
}


def _sample_plan(workout_days=(1, 3, 5)):
    plan = {}
    for d in range(1, 8):
        key = f"day_{d}"
        if d in workout_days:
            plan[key] = {
                "day_number": d,
                "focus": f"Workout Day {d}",
                "is_rest_day": False,
                "exercises": [
                    {
                        "exercise_name": "Bench Press",
                        "sets": 3,
                        "reps": "10",
                        "rest_seconds": 90,
                        "coaching_cue": "Control the descent.",
                    }
                ],
            }
        else:
            plan[key] = {
                "day_number": d,
                "focus": "Rest & Recovery",
                "is_rest_day": True,
                "exercises": [],
            }
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_services():
    mock_rec = MagicMock()
    mock_rec.recommend.return_value = {
        "program_id": 42,
        "content_candidates": list(range(50)),
        "neumf_ranked": [(42, 0.95)],
        "similar_user_id": 7,
    }

    mock_llm = MagicMock()
    mock_llm.generate_plan.return_value = {
        "program_title": "Test Program",
        "personalization_notes": "Tailored for beginner.",
        "source": "llm",
        "plan": _sample_plan(),
    }

    app.state.recommender = mock_rec
    app.state.llm_adapter = mock_llm
    # Provide a bare mock for the new squat form service so /health and other
    # endpoints that read app.state.squat_form_service do not AttributeError.
    app.state.squat_form_service = MagicMock()
    yield mock_rec, mock_llm


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "squat_model_ready" in data


def test_generate_plan_valid_request(client):
    response = client.post("/generate-plan", json=VALID_BODY)
    assert response.status_code == 200
    data = response.json()
    assert "program_id" in data
    assert "program_title" in data
    assert "similar_user_id" in data
    assert "personalization_notes" in data
    assert "source" in data
    assert "weekly_plan" in data
    assert len(data["weekly_plan"]) == 7
    assert data["program_id"] == 42
    assert data["similar_user_id"] == 7


def test_generate_plan_missing_required_field(client):
    # Missing experience_level, session_duration_hours, workout_frequency
    response = client.post("/generate-plan", json={"workout_type": "Strength"})
    assert response.status_code == 422


def test_generate_plan_invalid_workout_type(client):
    body = {**VALID_BODY, "workout_type": "Swimming"}
    response = client.post("/generate-plan", json=body)
    assert response.status_code == 422


def test_generate_plan_invalid_experience_level(client):
    body = {**VALID_BODY, "experience_level": 5}
    response = client.post("/generate-plan", json=body)
    assert response.status_code == 422


def test_generate_plan_uses_defaults(client):
    # Only the 4 required fields — age/gender/bmi use defaults
    response = client.post("/generate-plan", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json()["program_id"] == 42


def test_generate_plan_recommender_failure(client, mock_services):
    mock_rec, _ = mock_services
    mock_rec.recommend.side_effect = RuntimeError("boom")
    response = client.post("/generate-plan", json=VALID_BODY)
    assert response.status_code == 500
    assert "Recommendation failed" in response.json()["detail"]


def test_generate_plan_llm_failure(client, mock_services):
    _, mock_llm = mock_services
    mock_llm.generate_plan.side_effect = RuntimeError("boom")
    response = client.post("/generate-plan", json=VALID_BODY)
    assert response.status_code == 500
    assert "Plan generation failed" in response.json()["detail"]
