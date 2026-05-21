"""
Tests for Supabase token verification (backend/services/supabase_auth.py) and the
require_user FastAPI dependency (backend/deps/auth.py).

Verification is remote (GET /auth/v1/user), so the Supabase HTTP call is mocked —
no live project or network required. Covers the happy path plus every rejection
branch, and confirms auth never raises a 500 (always 401 on failure).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from backend.config import supabase as cfg
from backend.deps.auth import require_user
from backend.services import supabase_auth

_USER_ID = str(uuid.uuid4())


def _resp(status_code: int, json_data: dict | None = None):
    m = MagicMock(spec=httpx.Response)
    m.status_code = status_code
    m.json.return_value = json_data or {}
    m.text = str(json_data or "")
    return m


@pytest.fixture(autouse=True)
def _configure(monkeypatch):
    monkeypatch.setattr(cfg, "SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setattr(cfg, "SUPABASE_ANON_KEY", "test-anon-key")


# ── verify_token ────────────────────────────────────────────────────────────

@patch("backend.services.supabase_auth.httpx.get")
def test_valid_token_returns_claims(mock_get):
    mock_get.return_value = _resp(200, {"id": _USER_ID, "email": "test@fitnova.dev"})
    claims = supabase_auth.verify_token("any-valid-token")
    assert claims["sub"] == _USER_ID
    assert claims["email"] == "test@fitnova.dev"
    # apikey + bearer must both be sent
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["apikey"] == "test-anon-key"
    assert kwargs["headers"]["Authorization"] == "Bearer any-valid-token"


@patch("backend.services.supabase_auth.httpx.get")
def test_invalid_token_rejected(mock_get):
    mock_get.return_value = _resp(401, {"msg": "invalid token"})
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("bad-token")
    assert e.value.reason == "invalid_or_expired_token"


@patch("backend.services.supabase_auth.httpx.get")
def test_response_without_user_id_rejected(mock_get):
    mock_get.return_value = _resp(200, {"email": "x@y.z"})  # no id
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("weird-token")
    assert e.value.reason == "no_user_id_in_response"


@patch("backend.services.supabase_auth.httpx.get")
def test_unexpected_status_rejected(mock_get):
    mock_get.return_value = _resp(500, {"msg": "boom"})
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("token")
    assert e.value.reason.startswith("supabase_status:")


@patch("backend.services.supabase_auth.httpx.get", side_effect=httpx.ConnectError("down"))
def test_supabase_unreachable_rejected(mock_get):
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("token")
    assert e.value.reason.startswith("supabase_unreachable")


def test_missing_token_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("")
    assert e.value.reason == "missing_token"


def test_not_configured_rejected(monkeypatch):
    monkeypatch.setattr(cfg, "SUPABASE_URL", "")
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("token")
    assert e.value.reason == "supabase_not_configured"


# ── require_user dependency (must always 401 on failure, never 500) ───────────

@patch("backend.services.supabase_auth.httpx.get")
def test_require_user_valid_header(mock_get):
    mock_get.return_value = _resp(200, {"id": _USER_ID, "email": "test@fitnova.dev"})
    user = require_user(authorization="Bearer good-token")
    assert str(user.id) == _USER_ID
    assert user.email == "test@fitnova.dev"


def test_require_user_missing_header():
    with pytest.raises(HTTPException) as e:
        require_user(authorization=None)
    assert e.value.status_code == 401


def test_require_user_bad_scheme():
    with pytest.raises(HTTPException) as e:
        require_user(authorization="Basic abc123")
    assert e.value.status_code == 401


@patch("backend.services.supabase_auth.httpx.get")
def test_require_user_invalid_token(mock_get):
    mock_get.return_value = _resp(401)
    with pytest.raises(HTTPException) as e:
        require_user(authorization="Bearer bad")
    assert e.value.status_code == 401


@patch("backend.services.supabase_auth.httpx.get")
def test_require_user_non_uuid_sub(mock_get):
    mock_get.return_value = _resp(200, {"id": "not-a-uuid", "email": "x@y.z"})
    with pytest.raises(HTTPException) as e:
        require_user(authorization="Bearer good")
    assert e.value.status_code == 401
