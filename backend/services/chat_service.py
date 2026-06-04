"""
FitNova — Conversational Chat Service
======================================
Uses GPT-4o-mini with function calling to extract fitness parameters
from a natural conversation. Returns ``status: "ready"`` once all three
fields (experience_level, session_duration_hours, workout_frequency)
have been gathered.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_ENV_PATH = os.path.join(_BASE_DIR, ".env")

logger = logging.getLogger("fitnova.chat")

# ── OpenAI function-calling tool definition ───────────────────────────────────

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_fitness_params",
        "description": (
            "Extract fitness parameters that the user has revealed. "
            "Call this after every user message with whatever fields "
            "you can determine from the conversation so far."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experience_level": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": (
                        "1 = Beginner (less than 6 months), "
                        "2 = Intermediate (6 months to 2 years), "
                        "3 = Advanced (2+ years)"
                    ),
                },
                "session_duration_hours": {
                    "type": "number",
                    "description": "Hours per workout session, e.g. 0.5, 1.0, 1.5",
                },
                "workout_frequency": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5, 6, 7],
                    "description": "How many days per week the user can train",
                },
            },
            "required": [],
        },
    },
}

_REQUIRED_FIELDS = {"experience_level", "session_duration_hours", "workout_frequency"}

_WORKOUT_DESCRIPTIONS = {
    "Strength": "build muscle and get stronger",
    "Cardio": "improve cardiovascular fitness and lose weight",
    "Yoga": "improve flexibility and mindfulness",
    "HIIT": "get fit fast with high-intensity training",
}


def _build_system_prompt(workout_type: str, injuries: list[str] | None = None) -> str:
    goal = _WORKOUT_DESCRIPTIONS.get(workout_type, workout_type.lower())
    base = f"""You are FitNova's friendly fitness assistant. The user wants to {goal}.

Your job is to have a brief, natural conversation to learn three things:
1. Their fitness experience level (beginner / intermediate / advanced)
2. How long they can work out per session
3. How many days per week they can commit to training

Guidelines:
- Be warm, encouraging, and conversational.
- Ask ONE question at a time. Do not overwhelm the user.
- After each user reply, call the extract_fitness_params tool with whatever you've gathered.
- Once you have all three pieces of information, confirm what you learned and let them know you're ready to create their personalised plan.
- Keep responses concise (2-3 sentences max).
- Do NOT ask for information you already have."""

    if injuries:
        nice = ", ".join(i.replace("_", " ") for i in injuries)
        base += f"""

IMPORTANT: This user has reported the following injuries: {nice}.
During the conversation, briefly acknowledge these injuries and reassure
them that their plan will avoid triggering movements for those areas.
Do this once, naturally — don't dwell on it."""

    return base


class ChatService:
    """Stateless chat processor — conversation history is sent by the client."""

    def __init__(self, env_path: str = _ENV_PATH, client: OpenAI | None = None):
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to backend/.env or your environment."
            )
        self.client = client or OpenAI(api_key=api_key)

    def process_message(
        self,
        conversation: list[dict],
        user_context: dict,
    ) -> dict:
        """
        Process one round of conversation.

        Parameters
        ----------
        conversation : list[dict]
            Chat history so far — each item has ``role`` and ``content``.
        user_context : dict
            Contains ``workout_type`` and optionally ``age``, ``gender``, ``bmi``.

        Returns
        -------
        dict with keys ``status`` ("continue" | "ready"), ``message`` (str),
        and ``extracted`` (dict of gathered fields).
        """
        workout_type = user_context.get("workout_type", "Strength")
        injuries = user_context.get("injuries", [])
        system_msg = {"role": "system", "content": _build_system_prompt(workout_type, injuries=injuries)}
        messages = [system_msg] + [
            {"role": m["role"], "content": m["content"]} for m in conversation
        ]

        # First API call — model may respond with text, tool call, or both
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[EXTRACT_TOOL],
            tool_choice="auto",
            temperature=0.7,
            max_tokens=300,
        )

        choice = response.choices[0]
        assistant_text = choice.message.content or ""
        extracted = {}

        # Parse tool calls if present
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == "extract_fitness_params":
                    try:
                        extracted = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool call arguments")

            # If the model only made a tool call without text, do a follow-up
            if not assistant_text.strip():
                messages.append(choice.message.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": choice.message.tool_calls[0].id,
                    "content": json.dumps({"status": "ok", "extracted": extracted}),
                })
                followup = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=[EXTRACT_TOOL],
                    tool_choice="none",
                    temperature=0.7,
                    max_tokens=300,
                )
                assistant_text = followup.choices[0].message.content or ""

        # Clean extracted — only keep valid fields
        clean = {}
        if "experience_level" in extracted and extracted["experience_level"] in (1, 2, 3):
            clean["experience_level"] = extracted["experience_level"]
        if "session_duration_hours" in extracted:
            val = float(extracted["session_duration_hours"])
            if 0 < val <= 4.0:
                clean["session_duration_hours"] = val
        if "workout_frequency" in extracted:
            val = int(extracted["workout_frequency"])
            if 1 <= val <= 7:
                clean["workout_frequency"] = val

        status = "ready" if _REQUIRED_FIELDS <= set(clean.keys()) else "continue"

        return {
            "status": status,
            "message": assistant_text.strip(),
            "extracted": clean,
        }
