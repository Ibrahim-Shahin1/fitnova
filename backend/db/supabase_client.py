"""
psycopg connection pool against Supabase Postgres (service-role).

The backend connects with the service-role connection string (SUPABASE_DB_URL),
which BYPASSES row-level security. Every repository function is therefore
responsible for scoping queries to the user_id verified upstream by
require_user (backend/deps/auth.py) — never trust a client-supplied id.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from psycopg_pool import ConnectionPool

from backend.config import supabase as cfg

logger = logging.getLogger("fitnova.db")


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    """Lazily create and open a singleton connection pool.

    Raises RuntimeError if SUPABASE_DB_URL is not configured.
    """
    if not cfg.SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not configured — add it to backend/.env")
    logger.info("Opening Supabase Postgres connection pool")
    return ConnectionPool(
        conninfo=cfg.SUPABASE_DB_URL,
        min_size=1,
        max_size=5,
        open=True,
        timeout=10.0,
    )


def healthcheck() -> bool:
    """Return True if a trivial `select 1` round-trips through the pool.

    Raises on connection failure (so callers/tests see the real error).
    """
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select 1")
        row = cur.fetchone()
    return bool(row and row[0] == 1)
