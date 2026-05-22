"""
Logs router — per-set workout logging for progress tracking.

All endpoints are scoped to the authenticated user (require_user); the user_id
comes from the verified JWT, never the request body.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.db.repositories import log_repo
from backend.deps.auth import AuthUser, require_user

logger = logging.getLogger("fitnova.logs")

router = APIRouter(prefix="/api/logs", tags=["Logs"])


class LogSetRequest(BaseModel):
    exercise_name: str = Field(..., min_length=1, max_length=120)
    set_number: int = Field(..., ge=1, le=50)
    reps_completed: int | None = Field(default=None, ge=0, le=1000)
    weight_kg: float | None = Field(default=None, ge=0, le=999)
    duration_seconds: int | None = Field(default=None, ge=0, le=86400)
    rpe: float | None = Field(default=None, ge=1, le=10)
    notes: str | None = Field(default=None, max_length=500)
    plan_exercise_id: UUID | None = None


@router.post("")
async def create_log(req: LogSetRequest, user: AuthUser = Depends(require_user)):
    """Log one set."""
    try:
        row = log_repo.insert_log(user.id, req.model_dump())
    except Exception as exc:
        logger.exception("Log insert failed")
        raise HTTPException(status_code=500, detail=f"Log failed: {exc}")
    return {"log": row}


@router.get("")
async def list_logs(
    user: AuthUser = Depends(require_user),
    days: int = 90,
    exercise_name: str | None = None,
):
    """List the user's logs within `days`, newest first."""
    try:
        rows = log_repo.list_logs(user.id, days=days, exercise_name=exercise_name)
    except Exception as exc:
        logger.exception("Log list failed")
        raise HTTPException(status_code=500, detail=f"Log list failed: {exc}")
    return {"logs": rows}


@router.delete("/{log_id}")
async def delete_log(log_id: UUID, user: AuthUser = Depends(require_user)):
    """Delete one of the user's logs (undo a mistaken entry)."""
    try:
        ok = log_repo.delete_log(user.id, log_id)
    except Exception as exc:
        logger.exception("Log delete failed")
        raise HTTPException(status_code=500, detail=f"Log delete failed: {exc}")
    if not ok:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"deleted": True}
