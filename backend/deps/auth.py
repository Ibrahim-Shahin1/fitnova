"""
FastAPI auth dependency — turns a verified Supabase token into an AuthUser.

Usage:
    @app.get("/something")
    async def handler(user: AuthUser = Depends(require_user)):
        ...

Endpoints NEVER trust a user_id supplied in the request body — identity comes
only from the verified bearer token resolved here. Any failure (including
unexpected ones) is converted to HTTP 401; auth must never surface a 500.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from fastapi import Header, HTTPException

from backend.services.supabase_auth import AuthError, verify_token

logger = logging.getLogger("fitnova.auth")


@dataclass(frozen=True)
class AuthUser:
    id: UUID
    email: str | None


def require_user(authorization: str | None = Header(default=None)) -> AuthUser:
    """Resolve and verify the bearer token; return the authenticated user.

    Raises HTTPException(401) on any auth failure.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = verify_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=f"Auth failed: {exc.reason}")
    except Exception as exc:  # defensive — auth must never leak a 500
        logger.exception("Unexpected auth error")
        raise HTTPException(status_code=401, detail=f"Auth error: {type(exc).__name__}")

    sub = claims.get("sub")
    try:
        user_id = UUID(str(sub))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Token missing valid 'sub' claim")

    return AuthUser(id=user_id, email=claims.get("email"))
