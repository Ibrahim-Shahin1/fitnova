"""
Conversation repository — the coach chat thread + its messages.

v1 keeps a single conversation per user ('Fitness Coach'). Messages store the
full multi-turn transcript including tool calls/results so the coach has memory
and the UI can render what happened. Service-role pool; every query is scoped to
the user_id verified upstream by require_user.
"""

from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Json

from backend.db.supabase_client import pool


def get_or_create_conversation(user_id: UUID) -> dict:
    """Return the user's coach conversation, creating it on first use."""
    with pool().connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "select * from public.conversations where user_id = %s "
                    "order by last_message_at desc limit 1",
                    (str(user_id),),
                )
                conv = cur.fetchone()
                if conv is not None:
                    return conv
                cur.execute(
                    "insert into public.conversations (user_id) values (%s) "
                    "returning *",
                    (str(user_id),),
                )
                return cur.fetchone()


def fetch_messages(user_id: UUID, conversation_id: UUID, limit: int = 200) -> list[dict]:
    """All messages in a conversation, oldest first."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            select * from public.conversation_messages
            where conversation_id = %s and user_id = %s
            order by created_at asc
            limit %s
            """,
            (str(conversation_id), str(user_id), limit),
        )
        return cur.fetchall()


def append_message(
    user_id: UUID,
    conversation_id: UUID,
    role: str,
    *,
    content: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: dict | None = None,
) -> dict:
    """Insert one message and bump the conversation's last_message_at."""
    with pool().connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    insert into public.conversation_messages
                        (conversation_id, user_id, role, content,
                         tool_call_id, tool_name, tool_args, tool_result)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    returning *
                    """,
                    (
                        str(conversation_id),
                        str(user_id),
                        role,
                        content,
                        tool_call_id,
                        tool_name,
                        Json(tool_args) if tool_args is not None else None,
                        Json(tool_result) if tool_result is not None else None,
                    ),
                )
                msg = cur.fetchone()
                cur.execute(
                    "update public.conversations set last_message_at = now() "
                    "where id = %s",
                    (str(conversation_id),),
                )
                return msg
