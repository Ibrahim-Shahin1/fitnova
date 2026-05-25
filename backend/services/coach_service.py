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
    "You are FitNova's AI strength & conditioning coach. You help the user shape "
    "and understand their training plan.\n\n"
    "TOOLS:\n"
    "- get_user_profile — their saved baseline (experience 1-3, training focus, "
    "weekly frequency, session length, body stats, injuries, equipment).\n"
    "- get_active_plan — their current 7-day plan.\n"
    "- prepare_plan — readies a plan to be built. It does NOT build the plan; it "
    "surfaces a 'Generate Plan' button the user taps to watch the multi-agent crew "
    "build it live.\n\n"
    "HOW TO ACT:\n"
    "- You ALREADY have the user's profile: goal/training focus, experience level, "
    "weekly frequency, injuries, and equipment. Treat all of it as known — NEVER "
    "re-ask for anything already in the profile; CONFIRM it in one short line "
    "instead. NEVER ask how long they want to train per session.\n"
    "- Only ask a question if something is genuinely missing from the profile, or "
    "the user wants to change it.\n"
    "- The MOMENT the user is ready — 'yes', 'build it', 'give me the plan', or any "
    "go-ahead — CALL prepare_plan immediately. Do NOT reply with 'how can I help', "
    "do NOT re-ask anything. Pass ONLY the overrides they stated (e.g. "
    "workout_frequency=6, equipment=['Dumbbells']); everything else comes from "
    "their profile. Do NOT build or list the plan yourself — prepare_plan hands "
    "off to the live builder.\n"
    "- After calling prepare_plan, tell the user in ONE short line that their plan "
    "is ready and to tap 'Generate Plan'. NEVER claim you already built it, and do "
    "not draw tables or list exercises.\n"
    "- If they're still deciding or haven't given enough, keep the conversation "
    "going — don't call prepare_plan until it's time.\n\n"
    "DISCUSSING an existing plan: call get_active_plan first; never invent exercises "
    "that aren't in it. If a tool returns an 'error', explain it plainly.\n\n"
    "Style: a knowledgeable coach — encouraging, concise, concrete. No filler. Use "
    "the user's first name if you know it."
)


GREETING_INSTRUCTION = (
    "Open the conversation — you speak first. Introduce yourself as the FitNova AI "
    "coach and welcome the user by their first name. Briefly CONFIRM what you "
    "already know from their profile (their goal/training focus, experience level, "
    "weekly frequency, and any injuries or equipment) — do NOT ask for any of that "
    "again. Then, if they have NO active plan, ask if they're ready to build their "
    "plan; if they already have one, ask whether they'd like to review or adjust "
    "it. 2-3 short, friendly sentences — no lists, no markdown, no tool calls."
)


class CoachChatService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set — coach unavailable")
        self.client = OpenAI(api_key=api_key)

    def start(self, user_id: UUID) -> dict:
        """The coach always speaks first when the screen opens — a fresh,
        context-aware greeting from the current profile/plan state. It is
        TRANSIENT (not persisted), so reopening regenerates it instead of
        cluttering the saved thread."""
        conv = conversation_repo.get_or_create_conversation(user_id)
        conv_id = conv["id"]

        profile = profile_repo.get_profile(user_id) or {}
        name = str(profile.get("display_name") or "").strip()
        first = name.split()[0] if name else ""
        has_plan = bool(plan_repo.fetch_active(user_id))
        inj = ", ".join(profile.get("injuries") or []) or "none"
        equip = ", ".join(profile.get("equipment") or []) or "not set"
        ctx = (
            f"User first name: {first or 'unknown'}\n"
            f"Experience level (1-3): {profile.get('experience_level') or 'not set'}\n"
            f"Goal / training focus: {profile.get('training_focus') or 'not set'}\n"
            f"Weekly frequency: {profile.get('workout_frequency') or 'not set'}\n"
            f"Injuries: {inj}\n"
            f"Equipment: {equip}\n"
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

        return {"conversation_id": str(conv_id), "opening_message": greeting}

    def send(self, user_id: UUID, user_text: str, recommender, llm_adapter) -> dict:
        conv = conversation_repo.get_or_create_conversation(user_id)
        conv_id = conv["id"]

        # Inject the KNOWN profile every turn so the coach actually has the data
        # (not just told it does) and confirms instead of re-asking. The greeting
        # is transient and not in the replayed history, so we also remind the
        # model it already greeted + offered to build.
        prof = profile_repo.get_profile(user_id) or {}
        has_plan = bool(plan_repo.fetch_active(user_id))
        profile_ctx = (
            "KNOWN USER PROFILE — already collected; treat as given, CONFIRM, never "
            "re-ask:\n"
            f"- goal / training focus: {prof.get('training_focus') or 'not set'}\n"
            f"- experience level (1-3): {prof.get('experience_level') or 'not set'}\n"
            f"- weekly frequency: {prof.get('workout_frequency') or 'not set'}\n"
            f"- injuries: {', '.join(prof.get('injuries') or []) or 'none'}\n"
            f"- equipment: {', '.join(prof.get('equipment') or []) or 'not set'}\n"
            f"- has active plan: {'yes' if has_plan else 'no'}\n"
            "You have ALREADY greeted the user and offered to build their plan. If "
            "they confirm readiness or ask for the plan ('yes', 'give me the plan', "
            "'build it'), call prepare_plan NOW — do not deflect or re-ask."
        )
        # Replay prior dialogue (text only — tool mechanics aren't re-fed).
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": profile_ctx},
        ]
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

        def _prepared() -> dict | None:
            """If the coach readied a plan this turn, return the override params
            (possibly empty) so the app can surface the Generate Plan button."""
            for inv in invocations:
                if (inv.get("tool") == "prepare_plan"
                        and isinstance(inv.get("result"), dict)
                        and inv["result"].get("ready")):
                    return inv.get("args") or {}
            return None

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
            if not nudged and not _prepared() and _looks_like_stall(final):
                nudged = True
                messages.append({"role": "assistant", "content": final})
                messages.append({
                    "role": "user",
                    "content": "Don't just say you'll do it — act now. If you have "
                    "enough to build the plan, call prepare_plan in this turn. "
                    "Otherwise give your actual answer.",
                })
                continue

            prep = _prepared()
            conversation_repo.append_message(user_id, conv_id, "assistant", content=final)
            return {
                "conversation_id": str(conv_id),
                "assistant_message": final,
                "tool_invocations": invocations,
                "ready_to_generate": prep is not None,
                "plan_params": prep or {},
            }

        fallback = ("I got a little tangled working through that — "
                    "could you rephrase or try again?")
        conversation_repo.append_message(user_id, conv_id, "assistant", content=fallback)
        prep = _prepared()
        return {
            "conversation_id": str(conv_id),
            "assistant_message": fallback,
            "tool_invocations": invocations,
            "ready_to_generate": prep is not None,
            "plan_params": prep or {},
        }
