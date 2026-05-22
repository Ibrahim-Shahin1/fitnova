"""
Workout-log repository — per-set logs in public.workout_logs.

Writes use the service-role pool (RLS bypassed); every query is scoped to the
user_id verified upstream by require_user. If a plan_exercise_id is supplied it
is verified to belong to the user (join plan_days→plans) before being stored;
a foreign id is dropped rather than trusted, but the set is still logged by
exercise_name (the durable snapshot).
"""

from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row

from backend.db.supabase_client import pool


def insert_log(user_id: UUID, data: dict) -> dict:
    """Insert one logged set and return the stored row."""
    pe_id = data.get("plan_exercise_id")
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if pe_id:
            cur.execute(
                """
                select 1 from public.plan_exercises pe
                join public.plan_days d on d.id = pe.plan_day_id
                join public.plans p on p.id = d.plan_id
                where pe.id = %s and p.user_id = %s
                """,
                (str(pe_id), str(user_id)),
            )
            if cur.fetchone() is None:
                pe_id = None  # not the user's exercise — log by name only

        cur.execute(
            """
            insert into public.workout_logs
                (user_id, plan_exercise_id, exercise_name, set_number,
                 reps_completed, weight_kg, duration_seconds, rpe, notes)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning *
            """,
            (
                str(user_id),
                str(pe_id) if pe_id else None,
                str(data["exercise_name"]),
                int(data["set_number"]),
                data.get("reps_completed"),
                data.get("weight_kg"),
                data.get("duration_seconds"),
                data.get("rpe"),
                data.get("notes"),
            ),
        )
        return cur.fetchone()


def list_logs(
    user_id: UUID,
    days: int = 90,
    exercise_name: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Return the user's logs within the window, newest first."""
    clauses = ["user_id = %s", "performed_at >= now() - make_interval(days => %s)"]
    params: list = [str(user_id), max(1, min(365, int(days)))]
    if exercise_name:
        clauses.append("exercise_name ilike %s")
        params.append(f"%{exercise_name}%")
    sql = (
        "select * from public.workout_logs where "
        + " and ".join(clauses)
        + " order by performed_at desc limit %s"
    )
    params.append(max(1, min(1000, int(limit))))
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def delete_log(user_id: UUID, log_id: UUID) -> bool:
    """Delete one of the user's logs. Returns True if a row was removed."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "delete from public.workout_logs where id = %s and user_id = %s",
            (str(log_id), str(user_id)),
        )
        return cur.rowcount > 0
