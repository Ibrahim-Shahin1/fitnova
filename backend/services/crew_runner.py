"""
crew_runner — the CrewAI 4-agent plan pipeline. Runs ONLY inside the isolated
.crewenv (so CrewAI's deps never touch the backend's TF/mediapipe env).

Contract: read one JSON object from stdin, write one JSON object to stdout.

  stdin:  {pool:[{exercise_name,group,sets,reps}], profile:{...}, split:[...],
           frequency:int, min_per_day:int, max_per_day:int,
           avoid_keywords:[...], equipment:[...]}
  stdout: {ok:true, program_title:str, plan:{day_1:{...}...day_7}, quality:{score,issues,rounds}}
          or {ok:false, error:str}

The four agents (Profiler → Generator → Critic → Optimizer) are real CrewAI LLM
agents; the deterministic constraint check (plan_validators.check_plan) is run in
Python and its findings are fed to the Critic so its scoring is evidence-based.
"""

from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # import the pure-Python validators living next door
import plan_validators as pv  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_HERE, "..", ".env"))

# Keep CrewAI/OpenTelemetry quiet so stdout carries only our result JSON.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402

_OUT = sys.stdout  # real stdout; library chatter is later redirected to stderr


def emit(event: dict) -> None:
    """Write one progress event to the result channel (real stdout),
    sentinel-prefixed so the bridge can pick it out of any stray output."""
    print("FNEV " + json.dumps(event), file=_OUT, flush=True)


# Max optimizer↔critic rounds. The loop stops as soon as the plan is objectively
# constraint-clean (deterministic hard_violations == 0); this is just the ceiling.
_MAX_ROUNDS = 3


def _extract_json(raw: str) -> dict:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    m = re.search(r"\{.*\}", s, re.S)
    candidate = m.group(0) if m else s
    try:
        return json.loads(candidate)
    except Exception:
        from json_repair import repair_json
        return json.loads(repair_json(candidate))


def _run(agent: Agent, description: str, expected: str) -> str:
    task = Task(description=description, expected_output=expected, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    out = crew.kickoff()
    return str(getattr(out, "raw", out))


_PLAN_SCHEMA = (
    'A JSON object exactly: {"program_title": str, "plan": {"day_1": {...}, ... '
    '"day_7": {...}}}. Each day: {"day_number": int, "focus": str, '
    '"is_rest_day": bool, "exercises": [{"exercise_name": str, "sets": int, '
    '"reps": str}]}. Rest days have is_rest_day=true and exercises=[]. Output '
    "ONLY the JSON, no prose, no markdown fences."
)


def main() -> None:
    payload = json.load(sys.stdin)
    pool = payload["pool"]
    profile = payload.get("profile", {})
    frequency = int(payload["frequency"])
    split = payload["split"]
    avoid = payload.get("avoid_keywords", [])
    equipment = payload.get("equipment", [])
    minpd = int(payload.get("min_per_day", 4))
    maxpd = int(payload.get("max_per_day", 8))

    llm = LLM(model="gpt-4o-mini", temperature=0.3)  # OPENAI_API_KEY from .env

    pool_txt = "\n".join(
        f"- {e['exercise_name']} [{e.get('group','?')}] {e.get('sets','')}x{e.get('reps','')}"
        for e in pool
    )
    split_txt = ", ".join(f"Day{i+1}={f}" for i, f in enumerate(split))
    goal = profile.get("training_focus") or "general strength"
    exp = {1: "beginner", 2: "intermediate", 3: "advanced"}.get(
        int(profile.get("experience_level", 2)), "intermediate")
    inj = ", ".join(profile.get("injuries", [])) or "none"
    equip = ", ".join(equipment) or "full gym"

    common = (
        f"USER: {exp}, goal={goal}, {frequency} training days/week, "
        f"injuries=[{inj}], equipment=[{equip}].\n"
        f"EXERCISE POOL (the ONLY exercises you may use — they come from the "
        f"recommended program; [group]=movement group):\n{pool_txt}\n"
    )

    profiler = Agent(
        role="Fitness Profiler",
        goal="Turn the user's constraints into a clean training brief and the best weekly split.",
        backstory="A strength coach who structures training around the athlete's goal, recovery and schedule.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    generator = Agent(
        role="Plan Generator",
        goal="Assign pool exercises to each training day so every day matches its focus.",
        backstory="A programming specialist who builds coherent splits ONLY from the given pool.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    critic = Agent(
        role="Plan Critic",
        goal="Find every flaw: wrong-day exercises, imbalance, poor ordering; score the plan out of 10.",
        backstory="A meticulous reviewer who trusts the deterministic constraint report and adds expert judgement.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    optimizer = Agent(
        role="Plan Optimizer",
        goal="Apply the critic's fixes using ONLY pool exercises and return a polished plan.",
        backstory="An editor who corrects splits without inventing exercises.",
        llm=llm, verbose=False, allow_delegation=False,
    )

    # 1) PROFILER → confirm/refine the split.
    emit({"event": "agent", "name": "Profiler", "status": "running"})
    prof_raw = _run(
        profiler,
        common + f"Suggested split: {split_txt}. Confirm or refine the per-day "
        f"focus for a {goal} athlete. Keep exactly {frequency} training days.",
        'JSON: {"split": [<one focus label per training day>], "emphasis": str}. ONLY JSON.',
    )
    try:
        brief = _extract_json(prof_raw)
        ref_split = brief.get("split") or split
        if len(ref_split) != frequency:
            ref_split = split
    except Exception:
        ref_split = split
    ref_split_txt = ", ".join(f"Day{i+1}={f}" for i, f in enumerate(ref_split))
    emit({"event": "agent", "name": "Profiler", "status": "done",
          "detail": {"split": ref_split}})

    # 2) GENERATOR → first draft.
    emit({"event": "agent", "name": "Generator", "status": "running"})
    gen_raw = _run(
        generator,
        common + f"Split: {ref_split_txt}. Build the full 7-day plan. Place each "
        f"exercise on a day whose focus matches its [group] (push→Push/Upper, "
        f"pull→Pull/Upper, legs→Legs/Lower, core anywhere, etc.). {minpd}-{maxpd} "
        f"exercises per training day. Use ONLY pool exercises. Compounds first.",
        _PLAN_SCHEMA,
    )
    plan = _extract_json(gen_raw)
    program_title = plan.get("program_title") or f"{exp.title()} {goal.title()} Plan"
    plan = plan.get("plan", plan)
    emit({"event": "agent", "name": "Generator", "status": "done", "plan": plan})

    def critique(p: dict, round_no: int) -> dict:
        emit({"event": "agent", "name": "Critic", "status": "running", "round": round_no})
        findings = pv.check_plan(
            p, frequency=frequency, avoid_keywords=avoid, equipment=equipment,
            min_per_day=minpd, max_per_day=maxpd)
        crit_raw = _run(
            critic,
            common + "DETERMINISTIC CONSTRAINT REPORT (authoritative):\n"
            + json.dumps(findings) + "\nPLAN:\n" + json.dumps(p)
            + "\nUsing the report plus your judgement (balance, ordering, goal "
            "fit), list concrete fixes and score the plan 1-10.",
            'JSON: {"score": int(1-10), "issues": [str], "fixes": [str]}. ONLY JSON.',
        )
        try:
            c = _extract_json(crit_raw)
        except Exception:
            c = {}
        # The deterministic hard-violation count caps the score (LLM can't inflate).
        hard = findings.get("hard_violations", 0)
        llm_score = int(c.get("score", findings.get("score", 5)) or 5)
        c["score"] = min(llm_score, 10 - 2 * hard) if hard else llm_score
        c["hard_violations"] = hard
        det = findings.get("issues", [])
        llm_fixes = c.get("fixes") or c.get("issues") or []
        c["issues"] = det if det else (c.get("issues") or [])
        # Optimizer acts on the concrete deterministic violations first.
        c["fixes"] = list(dict.fromkeys([*det, *[f for f in llm_fixes if f]]))
        emit({"event": "agent", "name": "Critic", "status": "done",
              "round": round_no, "score": c["score"],
              "hard_violations": hard, "issues": c["issues"][:6]})
        return c

    crit = critique(plan, 0)
    rounds = 0
    # Iterate until the plan is objectively constraint-clean (no wrong-day /
    # equipment / injury / volume violations), capped to bound cost.
    while crit.get("hard_violations", 1) > 0 and rounds < _MAX_ROUNDS:
        emit({"event": "agent", "name": "Optimizer", "status": "running",
              "round": rounds + 1})
        opt_raw = _run(
            optimizer,
            common + f"Split: {ref_split_txt}.\nCURRENT PLAN:\n" + json.dumps(plan)
            + "\nPROBLEMS TO FIX (fix ONLY these; leave every other exercise "
            "exactly as it is):\n" + json.dumps(crit.get("fixes", []))
            + "\nFor each problem, either move that exercise to a day whose focus "
            "matches its movement, or swap it for a POOL exercise of that day's "
            "group. Do not touch unlisted exercises. Return the full 7-day plan.",
            _PLAN_SCHEMA,
        )
        try:
            revised = _extract_json(opt_raw)
            plan = revised.get("plan", revised)
        except Exception:
            break
        rounds += 1
        emit({"event": "agent", "name": "Optimizer", "status": "done",
              "round": rounds, "plan": plan})
        crit = critique(plan, rounds)

    return {
        "ok": True,
        "program_title": program_title,
        "plan": plan,
        "quality": {
            "score": crit.get("score"),
            "issues": crit.get("issues", []),
            "rounds": rounds,
        },
    }


if __name__ == "__main__":
    _OUT = sys.stdout  # capture real stdout for emit() before redirecting
    sys.stdout = sys.stderr  # quarantine library chatter off the event channel
    try:
        _result = main()
    except Exception as exc:  # never crash silently — the bridge reads this
        _result = {"ok": False, "error": str(exc)[:400]}
    sys.stdout = _OUT
    emit({"event": "result", **_result})
