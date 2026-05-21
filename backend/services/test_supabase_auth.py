"""
Tests for Supabase JWT verification (backend/services/supabase_auth.py) and the
require_user FastAPI dependency (backend/deps/auth.py).

These use locally-minted HS256 tokens against a test secret — no live Supabase
project required. They cover the happy path plus every rejection branch.
"""

from __future__ import annotations

import datetime as dt
import uuid

import jwt
import pytest
from fastapi import HTTPException

from backend.config import supabase as cfg
from backend.deps.auth import require_user
from backend.services import supabase_auth

_SECRET = "test-jwt-secret-do-not-use-in-prod"
_USER_ID = str(uuid.uuid4())


def _mint(secret: str = _SECRET, aud: str = "authenticated", sub: str = _USER_ID,
          exp_delta: int = 3600, **extra) -> str:
    now = dt.datetime.now(tz=dt.timezone.utc)
    payload = {
        "sub": sub,
        "aud": aud,
        "email": "test@fitnova.dev",
        "iat": now,
        "exp": now + dt.timedelta(seconds=exp_delta),
        **extra,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setattr(cfg, "SUPABASE_JWT_SECRET", _SECRET)
    monkeypatch.setattr(cfg, "JWT_AUDIENCE", "authenticated")


# ── verify_token ────────────────────────────────────────────────────────────

def test_valid_token_returns_claims():
    claims = supabase_auth.verify_token(_mint())
    assert claims["sub"] == _USER_ID
    assert claims["email"] == "test@fitnova.dev"


def test_expired_token_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token(_mint(exp_delta=-10))
    assert e.value.reason == "expired"


def test_wrong_audience_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token(_mint(aud="anon"))
    assert e.value.reason == "bad_audience"


def test_tampered_signature_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token(_mint(secret="a-different-secret"))
    assert e.value.reason.startswith("invalid_token")


def test_missing_token_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("")
    assert e.value.reason == "missing_token"


def test_malformed_token_rejected():
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token("not-a-jwt")
    assert e.value.reason == "malformed_token"


def test_secret_not_configured_rejected(monkeypatch):
    token = _mint()
    monkeypatch.setattr(cfg, "SUPABASE_JWT_SECRET", "")
    with pytest.raises(supabase_auth.AuthError) as e:
        supabase_auth.verify_token(token)
    assert e.value.reason == "jwt_secret_not_configured"


# ── require_user dependency ───────────────────────────────────────────────────

def test_require_user_valid_header():
    user = require_user(authorization=f"Bearer {_mint()}")
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


def test_require_user_invalid_token():
    with pytest.raises(HTTPException) as e:
        require_user(authorization="Bearer not-a-jwt")
    assert e.value.status_code == 401


def test_require_user_non_uuid_sub():
    with pytest.raises(HTTPException) as e:
        require_user(authorization=f"Bearer {_mint(sub='not-a-uuid')}")
    assert e.value.status_code == 401
