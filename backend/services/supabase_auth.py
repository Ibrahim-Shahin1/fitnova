"""
Supabase token verification (remote introspection).

Verifies a Supabase access token by calling the project's GET /auth/v1/user
endpoint with the token + anon apikey. This is ALGORITHM-AGNOSTIC: it works
whether the project signs access tokens with the legacy HS256 shared secret or
the newer asymmetric (ES256/RS256) signing keys, because Supabase itself does
the cryptographic check and returns the authenticated user.

Trade-off vs local JWT verification: one lightweight HTTPS call per authenticated
request. At this app's scale that is negligible, and it removes all secret/JWKS
handling plus the failure modes that come with it.

Used by backend/deps/auth.py (the require_user FastAPI dependency).
"""

from __future__ import annotations

import logging

import httpx

from backend.config import supabase as cfg

logger = logging.getLogger("fitnova.auth")


class AuthError(Exception):
    """Raised when a token fails verification. Carries a short reason code."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def verify_token(token: str) -> dict:
    """Verify a Supabase access token via /auth/v1/user; return a claims dict.

    Returns:
        dict with 'sub' (the user UUID), 'email', and 'raw' (the full user object).

    Raises:
        AuthError(reason): missing token, Supabase unreachable, invalid/expired
        token, or an unexpected response.
    """
    if not token:
        raise AuthError("missing_token")
    if not cfg.SUPABASE_URL or not cfg.SUPABASE_ANON_KEY:
        raise AuthError("supabase_not_configured")

    try:
        resp = httpx.get(
            f"{cfg.SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": cfg.SUPABASE_ANON_KEY,
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        logger.warning("Supabase auth endpoint unreachable: %s", exc)
        raise AuthError(f"supabase_unreachable:{type(exc).__name__}")

    if resp.status_code == 200:
        user = resp.json()
        sub = user.get("id")
        if not sub:
            raise AuthError("no_user_id_in_response")
        return {"sub": sub, "email": user.get("email"), "raw": user}

    if resp.status_code in (401, 403):
        raise AuthError("invalid_or_expired_token")

    logger.warning(
        "Supabase auth returned unexpected status %s: %s",
        resp.status_code, resp.text[:200],
    )
    raise AuthError(f"supabase_status:{resp.status_code}")
