"""
Tests for the POST /chat endpoint.
OpenAI is fully mocked — no real API calls.
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.app import app


VALID_CONTEXT = {
    "workout_type": "Strength",
    "age": 25,
    "gender": "Male",
    "bmi": 23.5,
}


def _mock_openai_response(text: str, tool_call_args: dict | None = None):
    """Build a fake OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = text

    if tool_call_args is not None:
        tc = MagicMock()
        tc.function.name = "extract_fitness_params"
        tc.function.arguments = json.dumps(tool_call_args)
        tc.id = "call_abc123"
        message.tool_calls = [tc]
        message.model_dump.return_value = {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "extract_fitness_params",
                        "arguments": json.dumps(tool_call_args),
                    },
                }
            ],
        }
    else:
        message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture(autouse=True)
def mock_services():
    """Mock all heavy services so tests don't load models."""
    mock_rec = MagicMock()
    mock_llm = MagicMock()
    mock_chat = MagicMock()
    app.state.recommender = mock_rec
    app.state.llm_adapter = mock_llm
    app.state.chat_service = mock_chat
    yield mock_chat


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_chat_initial_greeting(client, mock_services):
    """Empty conversation should return a greeting with status 'continue'."""
    mock_services.process_message.return_value = {
        "status": "continue",
        "message": "Hey! I'd love to help you build a strength plan. How long have you been working out?",
        "extracted": {},
    }

    response = client.post("/chat", json={
        "conversation": [],
        "user_context": VALID_CONTEXT,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "continue"
    assert len(data["message"]) > 0
    assert data["extracted"] == {}


def test_chat_partial_extraction(client, mock_services):
    """Message revealing experience should return partial extraction."""
    mock_services.process_message.return_value = {
        "status": "continue",
        "message": "A year of experience, nice! How much time can you dedicate to each workout?",
        "extracted": {"experience_level": 2},
    }

    response = client.post("/chat", json={
        "conversation": [
            {"role": "assistant", "content": "How long have you been working out?"},
            {"role": "user", "content": "About a year now"},
        ],
        "user_context": VALID_CONTEXT,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "continue"
    assert data["extracted"]["experience_level"] == 2


def test_chat_all_fields_ready(client, mock_services):
    """When all 3 fields are gathered, status should be 'ready'."""
    mock_services.process_message.return_value = {
        "status": "ready",
        "message": "Perfect! I have everything I need to create your plan.",
        "extracted": {
            "experience_level": 2,
            "session_duration_hours": 1.0,
            "workout_frequency": 4,
        },
    }

    response = client.post("/chat", json={
        "conversation": [
            {"role": "assistant", "content": "How many days a week can you train?"},
            {"role": "user", "content": "I can do 4 days, about an hour each"},
        ],
        "user_context": VALID_CONTEXT,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["extracted"]["experience_level"] == 2
    assert data["extracted"]["session_duration_hours"] == 1.0
    assert data["extracted"]["workout_frequency"] == 4


def test_chat_invalid_workout_type(client):
    """Invalid workout_type in user_context should return 422."""
    response = client.post("/chat", json={
        "conversation": [],
        "user_context": {"workout_type": "Swimming"},
    })
    assert response.status_code == 422


def test_chat_service_failure(client, mock_services):
    """Service exception should return 500."""
    mock_services.process_message.side_effect = RuntimeError("boom")
    response = client.post("/chat", json={
        "conversation": [],
        "user_context": VALID_CONTEXT,
    })
    assert response.status_code == 500
    assert "Chat failed" in response.json()["detail"]
