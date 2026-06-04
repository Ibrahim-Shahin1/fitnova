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

# User-facing training focus → the dataset program goals it should pull from, so
# the chosen goal actually steers which catalog program is used.
FOCUS_TO_GOALS = {
    "bodybuilding": {"Bodybuilding", "Muscle & Sculpting"},
    "powerbuilding": {"Powerbuilding"},
    "powerlifting": {"Powerlifting"},
    "cardio": {"Athletics", "Athletic Performance", "Weight Loss"},
    "general": {"General Fitness", "Toning", "Bodyweight Fitness",
                "Endurance", "Flexibility", "Cardio Health"},
}


def select_program_id(adapter, rec: dict, training_focus: str | None):
    """Prefer the highest-ranked recommended candidate whose dataset goal matches
    the user's chosen focus, so the goal actually influences the program (the
    content filter alone doesn't key on goal). Falls back to the recommender's
    top pick — always stays within the dataset."""
    goals = FOCUS_TO_GOALS.get((training_focus or "").lower())
    catalog = getattr(adapter, "catalog", None)
    if goals and isinstance(catalog, dict):
        for cid in (rec.get("content_candidates") or []):
            prog = catalog.get(cid)
            if prog and str(prog.get("goal")) in goals:
                return cid
    return rec.get("program_id")


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


_SENTINEL = "FNEV "


def _prepare(adapter, program: dict, profile: dict, candidates) -> dict:
    """Build the crew subprocess payload + the context the finalizer needs."""
    level = int(profile.get("experience_level", 2))
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
    return {
        "payload": payload, "pool": pool, "level": level,
        "primary_type": program.get("primary_type", "Strength"),
        "frequency": frequency, "avoid": avoid, "equipment": equipment,
        "min_pd": min_pd,
        # Exact program name + id from the trained dataset (no rewriting).
        "program_title": str(program.get("title") or "").strip(),
        "program_id": program.get("program_id"),
    }


def _finalize(adapter, result: dict, profile: dict, ctx: dict) -> dict:
    """Hard-gate + normalize the crew's plan into the generate_plan shape."""
    pool, level = ctx["pool"], ctx["level"]
    primary_type, frequency = ctx["primary_type"], ctx["frequency"]
    avoid, equipment, min_pd = ctx["avoid"], ctx["equipment"], ctx["min_pd"]

    plan = result.get("plan", {}) or {}
    for i in range(1, 8):
        plan.setdefault(f"day_{i}", {
            "day_number": i, "focus": "Rest & Recovery",
            "is_rest_day": True, "exercises": []})
        plan[f"day_{i}"].setdefault("day_number", i)
        plan[f"day_{i}"].setdefault("is_rest_day", False)
        plan[f"day_{i}"].setdefault("focus", f"Training Day {i}")
        plan[f"day_{i}"].setdefault("exercises", [])

    plan = pv.repair_plan(
        plan, frequency=frequency, pool=pool, avoid_keywords=avoid,
        equipment=equipment, min_per_day=min_pd,
        select_days=adapter._select_workout_days(frequency))

    for key in sorted(plan.keys()):
        day = plan[key]
        if day.get("is_rest_day"):
            day["exercises"] = []
            continue
        day["exercises"] = [
            adapter._normalize_exercise(ex, level, primary_type)
            for ex in day.get("exercises", [])
        ]

    final = pv.check_plan(plan, frequency=frequency, avoid_keywords=avoid,
                          equipment=equipment, min_per_day=min_pd,
                          max_per_day=_MAX_PER_DAY)
    q = result.get("quality", {})
    score = final["score"] if final["hard_violations"] else max(
        int(q.get("score", final["score"]) or final["score"]), final["score"])
    # Use the EXACT dataset program name (fall back only if the catalog lacks one).
    title = ctx.get("program_title") or adapter._sanitize_title(
        result.get("program_title", ""), profile, level, frequency)
    pid = ctx.get("program_id")
    note = (f"Program '{title}'"
            + (f" (#{pid})" if pid is not None else "")
            + " — selected from the trained dataset by the recommender "
            "(content-based filter + NCF). Every exercise is drawn from this "
            f"program's dataset entry; the 4-agent crew organized and validated "
            f"it (quality {score}/10).")
    if profile.get("injuries"):
        note += " Adjusted around your noted injuries."
    return {
        "program_title": title,
        "personalization_notes": note,
        "source": "llm",  # DB constraint allows only 'llm'|'template_fallback'
        "plan": plan,
        "quality_report": {
            "score": score,
            "hard_violations": final["hard_violations"],
            "rounds": q.get("rounds", 0),
            "issues": final["issues"][:8],
        },
    }


def _events(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(_SENTINEL):
            try:
                yield json.loads(line[len(_SENTINEL):])
            except Exception:
                continue


def build_plan(adapter, program: dict, profile: dict, candidates=None) -> dict:
    """Run the CrewAI pipeline (subprocess) then hard-gate + normalize. Returns
    the same dict shape as LLMAdapter.generate_plan."""
    ctx = _prepare(adapter, program, profile, candidates)
    if not os.path.exists(_CREW_PY):
        raise RuntimeError(f"Crew interpreter not found at {_CREW_PY} "
                           "(create backend/.crewenv or set FITNOVA_CREW_PYTHON)")
    proc = subprocess.run(
        [_CREW_PY, _RUNNER],
        input=json.dumps(ctx["payload"]),
        capture_output=True, text=True, timeout=240, cwd=_BACKEND_DIR,
    )
    if proc.returncode != 0:
        logger.error("crew_runner exited %s: %s", proc.returncode, proc.stderr[-500:])
        raise RuntimeError(f"Crew pipeline failed (exit {proc.returncode})")
    result = next((e for e in _events(proc.stdout)
                   if e.get("event") == "result"), None)
    if result is None:
        logger.error("crew_runner no result | stderr: %s", proc.stderr[-400:])
        raise RuntimeError("Crew produced no result event")
    if not result.get("ok"):
        raise RuntimeError(f"Crew error: {result.get('error')}")
    return _finalize(adapter, result, profile, ctx)


def build_plan_streamed(adapter, program: dict, profile: dict, candidates=None):
    """Generator: yields agent progress events as the crew works, then a final
    {'event':'plan', ...} with the hard-gated, normalized plan. Drives the SSE
    endpoint behind the live 4-agent screen."""
    ctx = _prepare(adapter, program, profile, candidates)
    if not os.path.exists(_CREW_PY):
        yield {"event": "error", "error": "Crew environment not set up"}
        return
    proc = subprocess.Popen(
        [_CREW_PY, _RUNNER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, cwd=_BACKEND_DIR, bufsize=1,
    )
    proc.stdin.write(json.dumps(ctx["payload"]))
    proc.stdin.close()
    result = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith(_SENTINEL):
                continue
            try:
                ev = json.loads(line[len(_SENTINEL):])
            except Exception:
                continue
            if ev.get("event") == "result":
                result = ev
                break
            yield ev  # agent progress
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if not result or not result.get("ok"):
        yield {"event": "error", "error": (result or {}).get("error", "crew failed")}
        return
    yield {"event": "plan", **_finalize(adapter, result, profile, ctx)}
