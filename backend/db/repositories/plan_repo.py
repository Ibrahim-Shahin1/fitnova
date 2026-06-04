"""
Plan repository — persists generated plans into public.plans / plan_days /
plan_exercises and reads back the active plan.

Writes use the service-role pool (RLS bypassed); every query is scoped to the
user_id verified upstream by require_user. One active plan per user is enforced
by deactivating any prior active plan inside the same transaction (the partial
unique index on plans(user_id) WHERE is_active is the backstop).
"""

from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row

from backend.db.supabase_client import pool


def insert_plan(user_id: UUID, rec: dict, plan_result: dict) -> str:
    """Insert a freshly generated plan + its days + exercises atomically and
    mark it the active plan. Returns the new plan id (str).

    Args:
        rec:         recommender output — needs 'program_id', 'similar_user_id'.
        plan_result: llm_adapter.generate_plan output — 'program_title',
                     'personalization_notes', 'source', and 'plan' (a dict whose
                     values each have day_number/focus/is_rest_day/exercises).
    """
    with pool().connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "update public.plans set is_active = false "
                    "where user_id = %s and is_active",
                    (str(user_id),),
                )
                cur.execute(
                    """
                    insert into public.plans
                        (user_id, program_id, similar_user_id, program_title,
                         personalization_notes, source, is_active)
                    values (%s, %s, %s, %s, %s, %s, true)
                    returning id
                    """,
                    (
                        str(user_id),
                        rec.get("program_id"),
                        rec.get("similar_user_id"),
                        plan_result.get("program_title", "Training plan"),
                        plan_result.get("personalization_notes"),
                        plan_result.get("source", "llm"),
                    ),
                )
                plan_id = cur.fetchone()["id"]

                for day in plan_result.get("plan", {}).values():
                    cur.execute(
                        """
                        insert into public.plan_days
                            (plan_id, day_number, focus, is_rest_day)
                        values (%s, %s, %s, %s)
                        returning id
                        """,
                        (
                            plan_id,
                            int(day["day_number"]),
                            str(day.get("focus", "")),
                            bool(day.get("is_rest_day", False)),
                        ),
                    )
                    day_id = cur.fetchone()["id"]

                    for pos, ex in enumerate(day.get("exercises", [])):
                        cur.execute(
                            """
                            insert into public.plan_exercises
                                (plan_day_id, position, exercise_name, sets, reps,
                                 rest_seconds, coaching_cue, media_url)
                            values (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                day_id,
                                pos,
                                str(ex.get("exercise_name", "")),
                                int(ex.get("sets", 0) or 0),
                                str(ex.get("reps", "")),
                                int(ex.get("rest_seconds", 0) or 0),
                                str(ex.get("coaching_cue", "")),
                                ex.get("media_url"),
                            ),
                        )
    return str(plan_id)


def fetch_active(user_id: UUID) -> dict | None:
    """Return the user's active plan with nested days + exercises, or None."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select * from public.plans where user_id = %s and is_active limit 1",
            (str(user_id),),
        )
        plan = cur.fetchone()
        if plan is None:
            return None

        cur.execute(
            "select * from public.plan_days where plan_id = %s order by day_number",
            (plan["id"],),
        )
        days = cur.fetchall()
        for d in days:
            cur.execute(
                "select * from public.plan_exercises where plan_day_id = %s "
                "order by position",
                (d["id"],),
            )
            d["exercises"] = cur.fetchall()
        plan["days"] = days
        return plan


def set_exercise_completed(
    user_id: UUID, exercise_id: UUID, completed: bool
) -> dict | None:
    """Flip a plan exercise's completion. Ownership is enforced in the WHERE via
    join plan_days→plans, so a cross-user id updates nothing → returns None."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            update public.plan_exercises pe
            set is_completed = %s,
                completed_at = case when %s then now() else null end
            from public.plan_days d
            join public.plans p on p.id = d.plan_id
            where pe.id = %s and pe.plan_day_id = d.id and p.user_id = %s
            returning pe.id, pe.is_completed, pe.completed_at
            """,
            (completed, completed, str(exercise_id), str(user_id)),
        )
        return cur.fetchone()


def set_day_completed(user_id: UUID, day_id: UUID, completed: bool) -> dict | None:
    """Flip a plan day's completion (ownership enforced via join to plans)."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            update public.plan_days d
            set is_completed = %s,
                completed_at = case when %s then now() else null end
            from public.plans p
            where d.id = %s and d.plan_id = p.id and p.user_id = %s
            returning d.id, d.is_completed, d.completed_at
            """,
            (completed, completed, str(day_id), str(user_id)),
        )
        return cur.fetchone()
