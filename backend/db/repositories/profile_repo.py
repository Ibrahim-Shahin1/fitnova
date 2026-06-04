"""
Profile repository — reads/writes public.profiles via the service-role pool.

Scaffolding for the server-side per-user data layer. The Flutter client does most
profile CRUD directly against Supabase (RLS-scoped); these backend helpers exist
for flows that run server-side (e.g. the coach LLM's get_user_profile tool in a
later unit). More methods are added as features need them.
"""

from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row

from backend.db.supabase_client import pool


def get_profile(user_id: UUID) -> dict | None:
    """Return the profile row for a user as a dict, or None if it doesn't exist."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("select * from public.profiles where id = %s", (str(user_id),))
        return cur.fetchone()
