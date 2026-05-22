"""
Coach router — the persistent AI coach for the Fitness Planning feature.

This unit ships conversation persistence + read endpoints. The POST
/api/chat/messages tool-loop (where the coach gathers nuance, confirms,
generates/edits the plan, and answers questions) is added in the next unit.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.db.repositories import conversation_repo
from backend.deps.auth import AuthUser, require_user

logger = logging.getLogger("fitnova.coach")

router = APIRouter(prefix="/api/chat", tags=["Coach"])


@router.get("/conversation")
async def get_conversation(user: AuthUser = Depends(require_user)):
    """Return (creating if needed) the user's coach conversation."""
    try:
        conv = conversation_repo.get_or_create_conversation(user.id)
    except Exception as exc:
        logger.exception("Loading conversation failed")
        raise HTTPException(status_code=500, detail=f"Conversation load failed: {exc}")
    return {"conversation": conv}


@router.get("/messages")
async def get_messages(user: AuthUser = Depends(require_user)):
    """Return the full transcript of the user's coach conversation."""
    try:
        conv = conversation_repo.get_or_create_conversation(user.id)
        msgs = conversation_repo.fetch_messages(user.id, conv["id"])
    except Exception as exc:
        logger.exception("Loading messages failed")
        raise HTTPException(status_code=500, detail=f"Messages load failed: {exc}")
    return {"conversation_id": str(conv["id"]), "messages": msgs}


@router.post("/start")
async def start_conversation(
    request: Request,
    user: AuthUser = Depends(require_user),
):
    """Have the coach speak first: returns (and persists) an opening message for
    a fresh conversation, or opening_message=null if dialogue already exists."""
    svc = getattr(request.app.state, "coach_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Coach service unavailable")
    try:
        return svc.start(user.id)
    except Exception as exc:
        logger.exception("Coach start failed")
        raise HTTPException(status_code=500, detail=f"Coach start failed: {exc}")


class CoachMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


@router.post("/messages")
async def post_message(
    req: CoachMessageRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
):
    """Send a message to the coach; runs the tool-loop and returns its reply."""
    svc = getattr(request.app.state, "coach_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Coach service unavailable")
    try:
        return svc.send(
            user.id,
            req.content,
            request.app.state.recommender,
            request.app.state.llm_adapter,
        )
    except Exception as exc:
        logger.exception("Coach send failed")
        raise HTTPException(status_code=500, detail=f"Coach failed: {exc}")
