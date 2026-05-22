"""
Coach router — the persistent AI coach for the Fitness Planning feature.

This unit ships conversation persistence + read endpoints. The POST
/api/chat/messages tool-loop (where the coach gathers nuance, confirms,
generates/edits the plan, and answers questions) is added in the next unit.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

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
