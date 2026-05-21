"""
Supabase JWT verification.

Verifies the signature + standard claims of a Supabase access token and returns
its claims. Supports BOTH:
  * symmetric  (HS256/384/512) — the legacy / default shared-secret projects,
    verified against SUPABASE_JWT_SECRET.
  * asymmetric (RS256/ES256/…) — projects using Supabase "JWT Signing Keys",
    verified against the project's published JWKS public key.

The algorithm is read from the token header and the matching key source is
chosen automatically, so the backend works regardless of how the user's
Supabase project is configured. Restricting `algorithms` to the family that
matches the header alg also closes the classic HS/RS "alg confusion" hole.

Used by backend/deps/auth.py (the require_user FastAPI dependency).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import jwt
from jwt import PyJWKClient

from backend.config import supabase as cfg

logger = logging.getLogger("fitnova.auth")

_HS_ALGS = ("HS256", "HS384", "HS512")
_ASYM_ALGS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512")


class AuthError(Exception):
    """Raised when a token fails verification. Carries a short reason code."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@lru_cache(maxsize=1)
def _jwk_client() -> PyJWKClient:
    url = cfg.jwks_url()
    if not url:
        raise AuthError("no_jwks_url")
    # PyJWKClient caches fetched keys internally; lru_cache keeps one client.
    return PyJWKClient(url)


def verify_token(token: str) -> dict:
    """Verify a Supabase access token and return its claims dict.

    Raises:
        AuthError(reason): on any failure — missing/malformed token, bad
        signature, expired, wrong audience, unsupported alg, or missing config.
    """
    if not token:
        raise AuthError("missing_token")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise AuthError("malformed_token")

    alg = header.get("alg", "")
    options = {"require": ["exp", "sub"]}

    try:
        if alg in _HS_ALGS:
            if not cfg.SUPABASE_JWT_SECRET:
                raise AuthError("jwt_secret_not_configured")
            return jwt.decode(
                token,
                cfg.SUPABASE_JWT_SECRET,
                algorithms=list(_HS_ALGS),
                audience=cfg.JWT_AUDIENCE,
                options=options,
            )
        if alg in _ASYM_ALGS:
            signing_key = _jwk_client().get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=list(_ASYM_ALGS),
                audience=cfg.JWT_AUDIENCE,
                options=options,
            )
        raise AuthError(f"unsupported_alg:{alg or 'none'}")
    except jwt.ExpiredSignatureError:
        raise AuthError("expired")
    except jwt.InvalidAudienceError:
        raise AuthError("bad_audience")
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"invalid_token:{type(exc).__name__}")
