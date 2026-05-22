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

from backend.db.repositories import conversation_repo, plan_repo, profile_repo
from backend.services import coach_tools

logger = logging.getLogger("fitnova.coach")

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

_MAX_HOPS = 5

# Action-promise phrases that signal the model is stalling (promising to act
# without calling a tool). A tool-less turn is treated as final, so an
# unbacked promise would do nothing — we nudge once when we see one.
_STALL_PHRASES = (
    "one moment", "give me a sec", "give me a second", "hold on", "hang on",
    "stand by", "working on it", "coming right up", "right away", "on it,",
    "i'll get", "i'll put", "i'll build", "i'll generate", "i'll create",
    "i'll update", "i will build", "i will generate", "i will create",
    "let me build", "let me generate", "let me create", "let me update",
    "let me put together", "generating it", "regenerating",
)


def _looks_like_stall(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) > 220:
        return False
    return any(p in t for p in _STALL_PHRASES)

SYSTEM_PROMPT = (
    "You are FitNova's AI strength & conditioning coach. You help the user build, "
    "change, and understand their training plan.\n\n"
    "TOOLS:\n"
    "- get_user_profile — their saved baseline (experience 1-3, training focus, "
    "weekly frequency, session length, body stats, injuries, equipment).\n"
    "- get_active_plan — their current 7-day plan.\n"
    "- generate_workout_plan — build and SAVE a new 7-day plan (this REPLACES the "
    "current one).\n\n"
    "HOW TO ACT — read this carefully:\n"
    "- When the user gives a clear instruction to build or change the plan (e.g. "
    "'build me a plan', 'make it 6 days', 'change to a 4-day split', 'use dumbbells "
    "only'), treat it as the go-ahead and CALL generate_workout_plan in the SAME "
    "turn, passing the change as an override (e.g. workout_frequency=6). NEVER reply "
    "'one moment', 'let me do that', 'give me a sec', or any promise to act later — "
    "if you intend to do it, do it now by calling the tool. A turn with no tool call "
    "is treated as your final answer, so a promise alone does nothing.\n"
    "- Treat a short confirmation right after you asked to proceed ('yes', 'go "
    "ahead', 'do it') as the go-ahead — call generate_workout_plan immediately.\n"
    "- Only ask a question when you genuinely need something that's missing or "
    "ambiguous (e.g. an injury to work around that isn't on file). Ask at most one "
    "short question, then act. For a first plan, confirm the key choices in ONE "
    "short line and, unless the user objects, generate.\n"
    "- Always pull defaults from get_user_profile; pass overrides only for what the "
    "user states in chat.\n\n"
    "AFTER generate_workout_plan SUCCEEDS:\n"
    "- The app shows the user the full schedule automatically, so DO NOT list every "
    "day and exercise and DO NOT draw tables. Reply with one or two short, "
    "encouraging lines (e.g. 'Done — here's your new 6-day powerbuilding split. "
    "Want to adjust anything?').\n\n"
    "DISCUSSING the plan: call get_active_plan first so you're accurate; never "
    "invent exercises that aren't in it. If a tool returns an 'error', explain it "
    "plainly and suggest a next step.\n\n"
    "Style: a knowledgeable coach — encouraging, concise, concrete. No filler. Use "
    "the user's first name if you know it."
)


GREETING_INSTRUCTION = (
    "Open the conversation now — you speak first. Greet the user warmly in 1-2 "
    "short sentences as their coach, using their first name if known. If they "
    "already have an active plan, acknowledge it and ask whether they'd like to "
    "review or adjust it. If they have NO plan yet, offer to build one and ask "
    "their goal (or say you can use their profile). Concise and friendly — no "
    "lists, no markdown, no tool calls."
)


class CoachChatService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set — coach unavailable")
        self.client = OpenAI(api_key=api_key)

    def start(self, user_id: UUID) -> dict:
        """Ensure the coach speaks first. For a fresh (empty) conversation,
        generate a personalized opener, persist it, and return it. If the
        conversation already has dialogue, returns opening_message=None."""
        conv = conversation_repo.get_or_create_conversation(user_id)
        conv_id = conv["id"]
        existing = conversation_repo.fetch_messages(user_id, conv_id)
        if any(m["role"] in ("user", "assistant") and m.get("content") for m in existing):
            return {"conversation_id": str(conv_id), "opening_message": None}

        profile = profile_repo.get_profile(user_id) or {}
        name = str(profile.get("display_name") or "").strip()
        first = name.split()[0] if name else ""
        has_plan = bool(plan_repo.fetch_active(user_id))
        ctx = (
            f"User first name: {first or 'unknown'}\n"
            f"Training focus: {profile.get('training_focus') or 'not set'}\n"
            f"Weekly frequency: {profile.get('workout_frequency') or 'not set'}\n"
            f"Has an active plan: {'yes' if has_plan else 'no'}"
        )
        greeting = ""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": "Context:\n" + ctx},
                    {"role": "user", "content": GREETING_INSTRUCTION},
                ],
                temperature=0.6,
                max_tokens=160,
                timeout=30,
            )
            greeting = (resp.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("Greeting generation failed; using fallback")

        if not greeting:
            hi = f"Hey {first}! " if first else "Hey! "
            greeting = hi + (
                "Want to review or tweak your current plan, or build something new?"
                if has_plan
                else "I'm your FitNova coach. Tell me your goal and I'll build your "
                "plan — and adjust it whenever you ask."
            )

        conversation_repo.append_message(user_id, conv_id, "assistant", content=greeting)
        return {"conversation_id": str(conv_id), "opening_message": greeting}

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
        nudged = False

        def _plan_was_generated() -> bool:
            return any(
                inv.get("tool") == "generate_workout_plan"
                and not (
                    isinstance(inv.get("result"), dict)
                    and inv["result"].get("error")
                )
                for inv in invocations
            )

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

            # Backstop: if the model stalled (promised to act but called no
            # tool) and hasn't actually generated anything, nudge it once to
            # act now rather than returning a do-nothing reply.
            if not nudged and not _plan_was_generated() and _looks_like_stall(final):
                nudged = True
                messages.append({"role": "assistant", "content": final})
                messages.append({
                    "role": "user",
                    "content": "Don't just say you'll do it — do it now. If this "
                    "needs a tool (e.g. generating or changing the plan), call the "
                    "tool in this turn. Otherwise give your actual answer.",
                })
                continue

            conversation_repo.append_message(user_id, conv_id, "assistant", content=final)
            return {
                "conversation_id": str(conv_id),
                "assistant_message": final,
                "tool_invocations": invocations,
                "plan_generated": _plan_was_generated(),
            }

        fallback = ("I got a little tangled working through that — "
                    "could you rephrase or try again?")
        conversation_repo.append_message(user_id, conv_id, "assistant", content=fallback)
        return {
            "conversation_id": str(conv_id),
            "assistant_message": fallback,
            "tool_invocations": invocations,
            "plan_generated": _plan_was_generated(),
        }
