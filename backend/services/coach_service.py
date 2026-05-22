"""
CoachChatService — the persistent AI coach's tool-using chat loop.

Loads conversation history, runs an OpenAI tool-calling loop (gather → confirm →
generate / discuss / edit), persists every turn (incl. tool calls + results), and
returns the final assistant text plus the tool invocations for the UI.

Tool handlers live in coach_tools.py; the recommender + llm_adapter are passed in
from app.state per request so the generate tool can run the real pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI

from backend.db.repositories import conversation_repo
from backend.services import coach_tools

logger = logging.getLogger("fitnova.coach")

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

_MAX_HOPS = 5

SYSTEM_PROMPT = (
    "You are FitNova's AI strength & conditioning coach. You help the user build "
    "and manage their training plan, and answer questions about it.\n\n"
    "How you work:\n"
    "- The user already has a profile (experience 1-3, training focus, weekly "
    "frequency, session length, body stats, injuries, equipment). Call "
    "get_user_profile before giving plan advice so you know their baseline.\n"
    "- When they want a plan (or have none): check for anything to work around "
    "that isn't already in their profile — injuries, available equipment, "
    "preferences. Ask at most one or two focused questions, confirm the key "
    "choices in one short line, then call generate_workout_plan.\n"
    "- generate_workout_plan REPLACES their current plan, so confirm before "
    "calling it. After it runs, summarise the plan in a few lines (the split + "
    "key lifts) and invite questions or changes.\n"
    "- To discuss the current plan, call get_active_plan first so you're accurate.\n"
    "- If a tool returns an 'error', explain it plainly and suggest a next step.\n\n"
    "Style: encouraging, concise, concrete — a knowledgeable coach, not a chatbot. "
    "No filler. Use the user's first name if you know it. Never invent exercises "
    "that aren't in the generated plan when describing it."
)


class CoachChatService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set — coach unavailable")
        self.client = OpenAI(api_key=api_key)

    def send(self, user_id: UUID, user_text: str, recommender, llm_adapter) -> dict:
        conv = conversation_repo.get_or_create_conversation(user_id)
        conv_id = conv["id"]

        # Replay prior dialogue (text only — tool mechanics aren't re-fed).
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in conversation_repo.fetch_messages(user_id, conv_id):
            if m["role"] in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_text})
        conversation_repo.append_message(user_id, conv_id, "user", content=user_text)

        ctx = coach_tools.ToolContext(
            user_id=user_id, recommender=recommender, llm_adapter=llm_adapter
        )
        invocations: list[dict] = []

        for _ in range(_MAX_HOPS):
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=coach_tools.TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.4,
                timeout=90,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = coach_tools.dispatch(ctx, name, args)
                    invocations.append({"tool": name, "args": args, "result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                    conversation_repo.append_message(
                        user_id, conv_id, "assistant",
                        content=msg.content, tool_call_id=tc.id,
                        tool_name=name, tool_args=args,
                    )
                    conversation_repo.append_message(
                        user_id, conv_id, "tool",
                        tool_call_id=tc.id, tool_name=name, tool_result=result,
                    )
                continue

            final = msg.content or ""
            conversation_repo.append_message(user_id, conv_id, "assistant", content=final)
            return {
                "conversation_id": str(conv_id),
                "assistant_message": final,
                "tool_invocations": invocations,
            }

        fallback = ("I got a little tangled working through that — "
                    "could you rephrase or try again?")
        conversation_repo.append_message(user_id, conv_id, "assistant", content=fallback)
        return {
            "conversation_id": str(conv_id),
            "assistant_message": fallback,
            "tool_invocations": invocations,
        }
