"""
plan_crew — backend-side bridge to the isolated CrewAI pipeline.

Runs in the MAIN backend env (no CrewAI import here). Builds a dataset-grounded
exercise pool + brief, runs the 4-agent crew in the .crewenv subprocess
(crew_runner.py), then applies the deterministic hard gate (repair_plan) and
normalization so the result is always constraint-clean and matches the shape
plan_repo.insert_plan consumes.

CrewAI's heavy dependency tree lives only in backend/.crewenv — set
FITNOVA_CREW_PYTHON to override the interpreter path.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess

from backend.services import plan_validators as pv

logger = logging.getLogger("fitnova.crew")

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_HERE, "..")
_RUNNER = os.path.join(_HERE, "crew_runner.py")
_CREW_PY = os.environ.get("FITNOVA_CREW_PYTHON") or os.path.join(
    _BACKEND_DIR, ".crewenv", "Scripts", "python.exe")

_MIN_PER_DAY = {1: 4, 2: 5, 3: 5}
_MAX_PER_DAY = 8
_POOL_CAP = 70  # keep the crew prompt bounded
_MIN_PER_GROUP = 4  # augment from candidates until each needed group has this many


def _primary_groups(focus: str) -> set[str]:
    return pv.allowed_groups_for_focus(focus) - {"core", "mobility", "unknown"}


def _build_pool(adapter, program: dict, candidates, split: list[str]) -> list[dict]:
    """Group-tagged, dataset-only pool from the recommended program, augmented
    from the recommender's candidate programs only for movement groups the split
    needs but the top program lacks. Never fabricates exercises."""
    grouped = adapter._group_week1_exercises(program)
    pool = pv.build_pool_from_grouped(grouped)

    needed: set[str] = set()
    for f in split:
        needed |= _primary_groups(f)
    have = {g: sum(1 for e in pool if e["group"] == g) for g in needed}
    names = {e["exercise_name"].lower() for e in pool}

    catalog = getattr(adapter, "catalog", None)
    if candidates and isinstance(catalog, dict):
        for cid in candidates:
            if all(have.get(g, 0) >= _MIN_PER_GROUP for g in needed):
                break
            prog = catalog.get(cid)
            if not prog:
                continue
            for cand in pv.build_pool_from_grouped(adapter._group_week1_exercises(prog)):
                g = cand["group"]
                if (g in needed and have.get(g, 0) < _MIN_PER_GROUP
                        and cand["exercise_name"].lower() not in names):
                    pool.append(cand)
                    names.add(cand["exercise_name"].lower())
                    have[g] = have.get(g, 0) + 1
    return pool[:_POOL_CAP]


def build_plan(adapter, program: dict, profile: dict, candidates=None) -> dict:
    """Run the CrewAI pipeline (subprocess) then hard-gate + normalize. Returns
    the same dict shape as LLMAdapter.generate_plan."""
    level = int(profile.get("experience_level", 2))
    primary_type = program.get("primary_type", "Strength")
    frequency = max(1, min(7, int(profile.get("workout_frequency", 3))))
    split = pv.default_split(frequency, profile.get("training_focus"))
    avoid = adapter._get_avoid_keywords(profile.get("injuries", []))
    equipment = profile.get("equipment", []) or []
    min_pd = _MIN_PER_DAY.get(level, 4)
    pool = _build_pool(adapter, program, candidates, split)

    payload = {
        "pool": pool,
        "profile": {
            "experience_level": level,
            "training_focus": profile.get("training_focus"),
            "workout_frequency": frequency,
            "injuries": profile.get("injuries", []),
        },
        "split": split,
        "frequency": frequency,
        "min_per_day": min_pd,
        "max_per_day": _MAX_PER_DAY,
        "avoid_keywords": avoid,
        "equipment": equipment,
    }

    if not os.path.exists(_CREW_PY):
        raise RuntimeError(f"Crew interpreter not found at {_CREW_PY} "
                           "(create backend/.crewenv or set FITNOVA_CREW_PYTHON)")

    proc = subprocess.run(
        [_CREW_PY, _RUNNER],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=240, cwd=_BACKEND_DIR,
    )
    if proc.returncode != 0:
        logger.error("crew_runner exited %s: %s", proc.returncode, proc.stderr[-500:])
        raise RuntimeError(f"Crew pipeline failed (exit {proc.returncode})")
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        logger.error("crew_runner bad stdout: %r | stderr: %s",
                     proc.stdout[-300:], proc.stderr[-300:])
        raise RuntimeError(f"Crew output unparseable: {exc}")
    if not result.get("ok"):
        raise RuntimeError(f"Crew error: {result.get('error')}")

    plan = result.get("plan", {})
    # Guarantee all 7 days exist with the right shape.
    for i in range(1, 8):
        plan.setdefault(f"day_{i}", {
            "day_number": i, "focus": "Rest & Recovery",
            "is_rest_day": True, "exercises": []})
        plan[f"day_{i}"].setdefault("day_number", i)
        plan[f"day_{i}"].setdefault("is_rest_day", False)
        plan[f"day_{i}"].setdefault("focus", f"Training Day {i}")
        plan[f"day_{i}"].setdefault("exercises", [])

    # Deterministic hard gate — grounded in the pool, never invents.
    plan = pv.repair_plan(
        plan, frequency=frequency, pool=pool, avoid_keywords=avoid,
        equipment=equipment, min_per_day=min_pd,
        select_days=adapter._select_workout_days(frequency))

    # Normalize exercises (rest, cues, media, sanitized reps/sets).
    for key in sorted(plan.keys()):
        day = plan[key]
        if day.get("is_rest_day"):
            day["exercises"] = []
            continue
        day["exercises"] = [
            adapter._normalize_exercise(ex, level, primary_type)
            for ex in day.get("exercises", [])
        ]

    # Final score on the repaired plan (authoritative).
    final = pv.check_plan(plan, frequency=frequency, avoid_keywords=avoid,
                          equipment=equipment, min_per_day=min_pd,
                          max_per_day=_MAX_PER_DAY)
    q = result.get("quality", {})
    score = final["score"] if final["hard_violations"] else max(
        int(q.get("score", final["score"]) or final["score"]), final["score"])

    title = adapter._sanitize_title(
        result.get("program_title", ""), profile, level, frequency)
    note = (f"Built by the FitNova coaching crew (Profiler → Generator → "
            f"Critic → Optimizer) and validated against split, injury and "
            f"equipment constraints — quality {score}/10.")
    if profile.get("injuries"):
        note += " Adjusted around your noted injuries."

    return {
        "program_title": title,
        "personalization_notes": note,
        # DB constraint allows only 'llm'|'template_fallback'; the crew IS an LLM
        # pipeline, and its provenance is carried in the notes + quality_report.
        "source": "llm",
        "plan": plan,
        "quality_report": {
            "score": score,
            "hard_violations": final["hard_violations"],
            "rounds": q.get("rounds", 0),
            "issues": final["issues"][:8],
        },
    }
