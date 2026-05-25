"""
Supabase configuration — single source of truth for auth + Postgres env vars.

Loads backend/.env (same pattern as chat_service / llm_adapter) and exposes:
  SUPABASE_URL          public project URL (also shipped to the Flutter client)
  SUPABASE_ANON_KEY     public anon key (client-side)
  SUPABASE_JWT_SECRET   backend-only — verifies HS256 access tokens
  SUPABASE_DB_URL       backend-only — service-role Postgres connection string

Imported by backend/services/supabase_auth.py (JWT verification) and, in a later
unit, the psycopg pool. Values default to "" when unset so the module imports
cleanly before the user has created backend/.env; consumers raise a clear error
if a required value is missing at use time.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")

# Supabase sets `aud` = "authenticated" on access tokens for logged-in users.
JWT_AUDIENCE: str = "authenticated"


def jwks_url() -> str:
    """JWKS endpoint for asymmetric (RS256/ES256) projects. Empty if URL unset."""
    if not SUPABASE_URL:
        return ""
    return f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
