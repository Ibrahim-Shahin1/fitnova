"""
Deterministic plan validators — the constraint backbone the AI agents lean on.

Pure Python, no LLM, no network: classify each exercise into a movement group,
know which groups belong on which day-focus, and flag/repair violations. This is
what reliably catches "a chest exercise on a Pull day" — the agents add judgment
on top, but correctness here is guaranteed, not left to the model.

Everything is grounded in the recommended program's exercise pool; repair never
invents exercises, it only drops violators and backfills from the pool.
"""

from __future__ import annotations

import re

# ── Movement-group classification ────────────────────────────────────────────
# Keyword → group. Order matters: more specific buckets are tested first so
# "Leg Curl" lands in legs, not pull, and "Leg Press" in legs, not push.

_MOBILITY = ("stretch", "mobility", "foam roll", "yoga", "cat-cow", "cat cow",
             "child's pose", "childs pose", "pigeon", "downward", "thoracic",
             "hip opener", "band pull-apart", "world's greatest", "couch stretch")
_CARDIO = ("run", "jog", "sprint", "treadmill", "bike", "cycling", "cycle",
           "elliptical", "rowing machine", "row erg", "erg", "jump rope",
           "skipping", "stairmaster", "sled", "prowler", "assault", "battle rope",
           "burpee", "mountain climber", "shuttle")
_CORE = ("plank", "crunch", "sit-up", "situp", "sit up", "oblique",
         "russian twist", "twist", "v-up", "leg raise", "knee raise",
         "hanging knee", "hollow", "dead bug", "ab wheel", "ab-wheel",
         "ab rollout", "cable crunch", "pallof", "woodchop", "wood chop",
         "hyperextension", "back extension", "carry", "farmer", "toe touch",
         "side bend")
_LEGS = ("squat", "lunge", "leg press", "leg extension", "leg ext", "leg curl",
         "hamstring curl", "calf", "glute", "hip thrust", "romanian", "rdl",
         "good morning", "split squat", "step-up", "step up", "hack squat",
         "stiff leg", "stiff-leg", "adduction", "abduction", "nordic", "sissy")
_PULL = ("deadlift", "row", "pull-up", "pullup", "pull up", "chin-up", "chinup",
         "chin up", "pulldown", "pull-down", "pull down", "lat ", "lat pull",
         "face pull", "rear delt", "reverse fly", "reverse flye", "shrug",
         "bicep", "biceps", "curl", "hammer curl", "preacher", "high pull",
         "upright row", "pull through", "pullover", "rack pull")
_PUSH = ("bench", "press", "push-up", "pushup", "push up", "dip", "fly", "flye",
         "chest", "incline", "decline", "overhead", "ohp", "shoulder press",
         "military", "lateral raise", "front raise", "tricep", "triceps",
         "pushdown", "push down", "skull crusher", "skullcrusher", "jm press",
         "close grip", "close-grip", "pec deck", "kickback", "crossover")


def _compile(words: tuple[str, ...]) -> re.Pattern:
    # Match at a word start (\b) so short keywords don't match inside others
    # (e.g. 'run' must not match 'crunch'); allows suffixes (press→presses).
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")")


_RE_MOBILITY = _compile(_MOBILITY)
_RE_CARDIO = _compile(_CARDIO)
_RE_CORE = _compile(_CORE)
_RE_LEGS = _compile(_LEGS)
_RE_PULL = _compile(_PULL)
_RE_PUSH = _compile(_PUSH)


def movement_group(name: str) -> str:
    """Best-effort movement group: push|pull|legs|core|cardio|mobility|unknown.
    'unknown' is intentional and is never treated as a violation."""
    n = (name or "").lower()
    if _RE_MOBILITY.search(n):
        return "mobility"
    if _RE_CARDIO.search(n):
        return "cardio"
    if _RE_CORE.search(n):
        return "core"
    if _RE_LEGS.search(n):
        return "legs"
    if _RE_PULL.search(n):
        return "pull"
    if _RE_PUSH.search(n):
        return "push"
    return "unknown"


_ALL_GROUPS = {"push", "pull", "legs", "core", "cardio", "mobility", "unknown"}


def allowed_groups_for_focus(focus: str) -> set[str]:
    """Which movement groups legitimately belong on a day with this focus label.
    Ambiguous/unknown focus → allow everything (never false-flag)."""
    f = (focus or "").lower()
    if "rest" in f or "recovery" in f or "mobility" in f:
        return {"mobility", "core"}
    if "full body" in f or "full-body" in f or "total body" in f:
        return {"push", "pull", "legs", "core", "mobility", "cardio", "unknown"}
    if "upper" in f:
        return {"push", "pull", "core", "mobility", "unknown"}
    if "lower" in f or "leg" in f:
        return {"legs", "core", "mobility", "unknown"}
    if "push" in f:
        return {"push", "core", "mobility", "unknown"}
    if "pull" in f:
        return {"pull", "core", "mobility", "unknown"}
    if "chest" in f or "shoulder" in f or "tricep" in f:
        return {"push", "core", "mobility", "unknown"}
    if "back" in f or "bicep" in f:
        return {"pull", "core", "mobility", "unknown"}
    if "arm" in f:
        return {"push", "pull", "core", "mobility", "unknown"}
    if "condition" in f or "cardio" in f or "hiit" in f or "metcon" in f:
        return {"cardio", "core", "legs", "unknown"}
    return set(_ALL_GROUPS)  # unknown focus → don't flag anything


# ── Equipment inference (self-contained mirror of llm_adapter's logic) ────────

_EQUIPMENT_HINTS = {
    "barbell": "barbell", "dumbbell": "dumbbell", "dumbbells": "dumbbell",
    "cable": "cable", "cables": "cable", "machine": "machine", "smith": "machine",
    "leverage": "machine", "lever": "machine", "kettlebell": "kettlebell",
    "band": "band", "bands": "band", "bodyweight": "bodyweight",
}


def required_equipment(name: str) -> str | None:
    for token in re.findall(r"\(([^)]+)\)", (name or "").lower()):
        for hint, canon in _EQUIPMENT_HINTS.items():
            if hint in token:
                return canon
    return None


def equipment_ok(name: str, equipment: list[str]) -> bool:
    if not equipment:
        return True
    eq = [str(e).lower() for e in equipment]
    if any(("full gym" in e) or (e == "gym") or ("all equipment" in e) for e in eq):
        return True
    req = required_equipment(name)
    if req is None or req == "bodyweight":
        return True
    return any(req in e for e in eq)


def injury_ok(name: str, avoid_keywords: list[str]) -> bool:
    if not avoid_keywords:
        return True
    nl = (name or "").lower()
    return not any(kw in nl for kw in avoid_keywords)


def _norm(name: str) -> str:
    """Normalize an exercise name for catalog matching (case/whitespace-insensitive)."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# ── Validation ────────────────────────────────────────────────────────────────

def check_plan(
    plan: dict,
    *,
    frequency: int,
    avoid_keywords: list[str],
    equipment: list[str],
    min_per_day: int = 4,
    max_per_day: int = 8,
) -> dict:
    """Run every hard constraint over a plan dict ({day_1:{...}, ...}).
    Returns {score:int(1-10), issues:[str], hard_violations:int}."""
    issues: list[str] = []
    hard = 0

    train_days = [d for d in plan.values() if not d.get("is_rest_day")]
    if len(train_days) != frequency:
        issues.append(
            f"Training-day count {len(train_days)} != requested {frequency}.")
        hard += 1

    for key in sorted(plan.keys()):
        day = plan[key]
        if day.get("is_rest_day"):
            continue
        focus = day.get("focus", "")
        allowed = allowed_groups_for_focus(focus)
        exs = day.get("exercises", [])
        seen: set[str] = set()

        for ex in exs:
            nm = str(ex.get("exercise_name", ""))
            grp = movement_group(nm)
            if grp not in allowed:
                issues.append(
                    f"{key} ({focus}): '{nm}' is a {grp} movement — wrong day.")
                hard += 1
            if not equipment_ok(nm, equipment):
                issues.append(
                    f"{key}: '{nm}' needs equipment the user lacks.")
                hard += 1
            if not injury_ok(nm, avoid_keywords):
                issues.append(f"{key}: '{nm}' conflicts with an injury.")
                hard += 1
            low = nm.lower()
            if low in seen:
                issues.append(f"{key}: '{nm}' is duplicated.")
            seen.add(low)

        if len(exs) < min_per_day:
            issues.append(
                f"{key} ({focus}): only {len(exs)} exercises (min {min_per_day}).")
        elif len(exs) > max_per_day:
            issues.append(
                f"{key} ({focus}): {len(exs)} exercises (max {max_per_day}).")

    # Soft check: compounds should generally precede isolations within a day.
    for key in sorted(plan.keys()):
        day = plan[key]
        if day.get("is_rest_day"):
            continue
        groups = [movement_group(str(e.get("exercise_name", "")))
                  for e in day.get("exercises", [])]
        # (informational only — not scored hard)
        del groups

    score = max(1, 10 - 2 * hard - max(0, len(issues) - hard))
    return {"score": score, "issues": issues, "hard_violations": hard}


# ── Repair (grounded, never invents) ──────────────────────────────────────────

def repair_plan(
    plan: dict,
    *,
    frequency: int,
    pool: list[dict],
    avoid_keywords: list[str],
    equipment: list[str],
    min_per_day: int = 4,
    select_days: list[int] | None = None,
) -> dict:
    """Force the plan to satisfy hard constraints using ONLY pool exercises:
    drop wrong-day/unsafe/duplicate exercises, snap to exactly `frequency`
    training days, and backfill each short day with on-group pool exercises.

    `pool` items: {"exercise_name","sets","reps","group", ...}. `select_days` is
    the canonical training-day positions (1..7) when the schedule must be rebuilt.
    """
    # Safe, group-tagged candidates from the program pool.
    safe_pool: list[dict] = []
    seen_pool: set[str] = set()
    for ex in pool:
        nm = str(ex.get("exercise_name", "")).strip()
        key = nm.lower()
        if not nm or key in seen_pool:
            continue
        if not equipment_ok(nm, equipment) or not injury_ok(nm, avoid_keywords):
            continue
        seen_pool.add(key)
        safe_pool.append({**ex, "group": ex.get("group") or movement_group(nm)})

    # Canonical name map → every kept exercise must be a real catalog exercise
    # (100% dataset provenance); the LLM's name is replaced by the exact pool
    # name it matches, and anything not in the pool is dropped.
    canon = {_norm(e["exercise_name"]): e["exercise_name"] for e in safe_pool}

    def pool_for(groups: set[str]) -> list[dict]:
        return [e for e in safe_pool if e["group"] in groups]

    # 1) Snap schedule to exactly `frequency` training days if needed.
    train_keys_now = [k for k in sorted(plan.keys())
                      if not plan[k].get("is_rest_day")]
    if len(train_keys_now) != frequency and select_days:
        positions = set(select_days)
        llm_days = [plan[f"day_{i}"] for i in range(1, 8)
                    if not plan.get(f"day_{i}", {}).get("is_rest_day")]
        rebuilt: dict = {}
        ti = 0
        for i in range(1, 8):
            k = f"day_{i}"
            if i in positions:
                src = llm_days[ti] if ti < len(llm_days) else None
                ti += 1 if src else 0
                rebuilt[k] = {
                    "day_number": i,
                    "focus": (src or {}).get("focus") or f"Training Day {i}",
                    "is_rest_day": False,
                    "exercises": list((src or {}).get("exercises", [])),
                }
            else:
                rebuilt[k] = {"day_number": i, "focus": "Rest & Recovery",
                              "is_rest_day": True, "exercises": []}
        plan = rebuilt

    # 2) Per training day: drop violators, then backfill on-group from the pool.
    for key in sorted(plan.keys()):
        day = plan[key]
        if day.get("is_rest_day"):
            day["exercises"] = []
            continue
        allowed = allowed_groups_for_focus(day.get("focus", ""))
        kept: list[dict] = []
        used: set[str] = set()
        for ex in day.get("exercises", []):
            canonical = canon.get(_norm(str(ex.get("exercise_name", ""))))
            if canonical is None:
                continue  # not in the catalog pool — drop (dataset-only guarantee)
            if movement_group(canonical) not in allowed:
                continue
            if canonical.lower() in used:
                continue
            used.add(canonical.lower())
            kept.append({**ex, "exercise_name": canonical})

        if len(kept) < min_per_day:
            # backfill: prefer the day's primary group(s), fall back to allowed
            primary = allowed - {"core", "mobility", "unknown"} or allowed
            for grp_set in (primary, allowed):
                for cand in pool_for(grp_set):
                    if len(kept) >= min_per_day:
                        break
                    if cand["exercise_name"].lower() in used:
                        continue
                    used.add(cand["exercise_name"].lower())
                    kept.append(dict(cand))
                if len(kept) >= min_per_day:
                    break
        day["exercises"] = kept

    return plan


def default_split(frequency: int, training_focus: str | None = None) -> list[str]:
    """A sensible per-training-day focus template for the frequency. The crew's
    Profiler may refine it, but this guarantees a coherent starting structure."""
    f = max(1, min(7, int(frequency)))
    goal = (training_focus or "").lower()
    body_part_goal = goal in ("general", "")
    table = {
        1: ["Full Body"],
        2: ["Upper Body", "Lower Body"],
        3: (["Full Body", "Full Body", "Full Body"] if body_part_goal
            else ["Push", "Pull", "Legs"]),
        4: ["Upper Body", "Lower Body", "Upper Body", "Lower Body"],
        5: ["Push", "Pull", "Legs", "Upper Body", "Lower Body"],
        6: ["Push", "Pull", "Legs", "Push", "Pull", "Legs"],
        7: ["Push", "Pull", "Legs", "Push", "Pull", "Legs", "Conditioning"],
    }
    return table[f]


def build_pool_from_grouped(grouped_by_day: dict[int, list[dict]]) -> list[dict]:
    """Flatten {day: [exercise dicts]} into a group-tagged, de-duplicated pool."""
    pool: list[dict] = []
    seen: set[str] = set()
    for day in sorted(grouped_by_day):
        for ex in grouped_by_day[day]:
            nm = str(ex.get("exercise_name", "")).strip()
            if not nm or nm.lower() in seen:
                continue
            seen.add(nm.lower())
            pool.append({
                "exercise_name": nm,
                "sets": ex.get("sets"),
                "reps": ex.get("reps"),
                "group": movement_group(nm),
            })
    return pool
