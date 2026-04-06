"""
FitNova — Program Catalog Builder
==================================
Replays the exact groupby logic from generate_interactions.py to reconstruct
the program_id -> title -> exercise schedule mapping.

Output: program_catalog.pkl
  A dict keyed by program_id (0-2597), each value containing:
    - title, description
    - level_encoded, goal, goal_encoded, equipment, equipment_encoded
    - program_length_weeks, time_per_workout_minutes (raw, un-normalized)
    - has_cardio, has_strength, has_yoga, has_hiit
    - exercises: full list [{week, day, exercise_name, sets, reps}]
    - week1_exercises: only week 1 (for LLM prompt, smaller context)

Normalization min/max values are also saved to norm_stats.pkl so the
content filter and FastAPI service can normalize new user inputs correctly.
"""

import os
import sys
import pickle
import time
from ast import literal_eval

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# All paths default to repo-local locations.
# Override with env vars if your datasets live elsewhere:
#   FITNOVA_PROGRAM_CSV  — path to programs_detailed_boostcamp_kaggle.csv
#   FITNOVA_GYM_CSV      — path to gym_members_exercise_tracking.csv
#   FITNOVA_DATA_DIR     — directory containing generated_data files

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.environ.get("FITNOVA_DATA_DIR", _THIS_DIR)
_DATASETS_DIR = os.path.join(_THIS_DIR, "datasets")

PROGRAM_CSV   = os.environ.get(
    "FITNOVA_PROGRAM_CSV",
    os.path.join(_DATASETS_DIR, "programs_detailed_boostcamp_kaggle.csv")
)
PROG_FEAT_CSV = os.path.join(_DATA_DIR, "program_features.csv")
USER_FEAT_CSV = os.path.join(_DATA_DIR, "user_features.csv")

OUTPUT_DIR    = _THIS_DIR
CATALOG_PATH  = os.path.join(OUTPUT_DIR, "program_catalog.pkl")
NORM_PATH     = os.path.join(OUTPUT_DIR, "norm_stats.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (copied exactly from generate_interactions.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_list_field(val):
    """Parse string representation of a Python list."""
    if pd.isna(val) or str(val).strip() in ('[]', ''):
        return []
    try:
        result = literal_eval(str(val))
        if isinstance(result, list):
            return result
        return [str(result)]
    except Exception:
        return [str(val).strip()]


def classify_exercise(name):
    """Classify an exercise name into workout type categories.
    Exact copy of the function in generate_interactions.py."""
    if pd.isna(name):
        return set()
    name_lower = str(name).lower()
    types = set()

    hiit_kw = ['hiit', 'circuit', 'burpee', 'sprint', 'tabata', 'explosive',
               'plyo', 'box jump', 'battle rope', 'kettlebell swing', 'emom',
               'amrap', 'interval', 'metabolic']
    cardio_kw = ['run', 'jog', 'walk', 'cycle', 'bike', 'swim', 'cardio',
                 'treadmill', 'elliptical', 'stair', 'skip', 'jumping jack',
                 'mountain climber', 'jump rope', 'rowing machine', 'step']
    yoga_kw = ['yoga', 'stretch', 'flexibility', 'pigeon', 'warrior',
               'downward', 'sun salutation', 'meditation', 'breathing',
               'mobility', 'hip opener', 'cobra', 'child pose', 'flow',
               'vinyasa', 'yin', 'restorative']
    strength_kw = ['squat', 'bench', 'deadlift', 'press', 'curl', 'row',
                   'pull-up', 'push-up', 'lunge', 'extension', 'fly', 'raise',
                   'shrug', 'dip', 'chin', 'pulldown', 'pushdown', 'cable',
                   'dumbbell', 'barbell', 'smith', 'machine', 'weight', 'lat',
                   'tricep', 'bicep', 'incline', 'decline', 'overhead',
                   'hammer', 'leg press', 'calf', 'hip thrust', 'glute',
                   'romanian', 'clean', 'snatch', 'jerk', 'power clean',
                   'military', 'skull crusher', 'preacher', 'pec']

    for kw in hiit_kw:
        if kw in name_lower:
            types.add('HIIT')
            break
    for kw in cardio_kw:
        if kw in name_lower:
            types.add('Cardio')
            break
    for kw in yoga_kw:
        if kw in name_lower:
            types.add('Yoga')
            break
    for kw in strength_kw:
        if kw in name_lower:
            types.add('Strength')
            break

    if not types:
        if any(x in name_lower for x in ['hold', 'plank', 'rotation', 'twist']):
            types.add('Yoga')
        elif any(x in name_lower for x in ['push', 'pull', 'lift']):
            types.add('Strength')
        else:
            types.add('Strength')

    return types


def format_reps(reps_val):
    """
    Format the reps field for display.
    Negative values are time-based (e.g., -30 = 30 seconds hold).
    """
    if pd.isna(reps_val):
        return None
    val = float(reps_val)
    if val < 0:
        return f"{int(abs(val))} sec"
    return str(int(val))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_catalog():
    print("=" * 60)
    print("FitNova -- Program Catalog Builder")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Load raw program data ────────────────────────────────────────
    print("\n[1/5] Loading raw program data (large file)...")
    t1 = time.time()
    programs_raw = pd.read_csv(PROGRAM_CSV, low_memory=False)
    print(f"  Loaded {len(programs_raw):,} exercise rows in {time.time()-t1:.1f}s")
    print(f"  Columns: {programs_raw.columns.tolist()}")

    # ── Step 2: Replay groupby to assign program_ids ──────────────────────────
    print("\n[2/5] Replaying groupby('title', sort=False) to assign program_ids...")
    t2 = time.time()

    program_groups = programs_raw.groupby('title', sort=False)
    n_programs = len(program_groups)
    print(f"  Found {n_programs} unique program titles")

    programs_list = []  # intermediate list (mirrors generate_interactions.py)

    for title, group in program_groups:
        first_row = group.iloc[0]

        # Level (same logic as generate_interactions.py)
        levels_raw = parse_list_field(first_row['level'])
        level_map = {'beginner': 0, 'novice': 0, 'intermediate': 1, 'advanced': 2}
        if levels_raw:
            level_values = [level_map.get(l.lower().strip(), 1)
                            for l in levels_raw if isinstance(l, str)]
            primary_level = min(level_values) if level_values else 1
        else:
            primary_level = 1

        # Goal (primary = first item in list)
        goals_raw = parse_list_field(first_row['goal'])
        primary_goal = goals_raw[0] if goals_raw else 'General Fitness'

        # Equipment
        equipment = (str(first_row['equipment']).strip()
                     if pd.notna(first_row['equipment']) else 'Unknown')

        # Description
        description = (str(first_row['description']).strip()
                       if pd.notna(first_row.get('description', float('nan'))) else '')

        # Program length and time per workout (raw, un-normalized)
        prog_len = float(first_row['program_length']) if pd.notna(first_row['program_length']) else 4.0
        tpw      = float(first_row['time_per_workout']) if pd.notna(first_row['time_per_workout']) else 60.0

        # Workout type flags
        all_types = set()
        for ex_name in group['exercise_name'].dropna().unique():
            all_types.update(classify_exercise(ex_name))

        # Full exercise schedule (all weeks/days)
        exercise_rows = []
        for _, ex_row in group.iterrows():
            if pd.isna(ex_row['exercise_name']):
                continue
            exercise_rows.append({
                'week':          int(ex_row['week']) if pd.notna(ex_row['week']) else 1,
                'day':           int(ex_row['day'])  if pd.notna(ex_row['day'])  else 1,
                'exercise_name': str(ex_row['exercise_name']).strip(),
                'sets':          int(ex_row['sets']) if pd.notna(ex_row['sets']) else None,
                'reps':          format_reps(ex_row['reps']),
            })

        # Sort by week then day
        exercise_rows.sort(key=lambda r: (r['week'], r['day']))

        # Week 1 only (for LLM prompt to keep context size manageable)
        week1 = [r for r in exercise_rows if r['week'] == 1]
        if not week1:
            week1 = exercise_rows[:20]  # fallback: first 20 exercises

        programs_list.append({
            'program_id':              len(programs_list),
            'title':                   title,
            'description':             description,
            'level_encoded':           primary_level,
            'goal':                    primary_goal,
            'equipment':               equipment,
            'program_length_weeks':    prog_len,
            'time_per_workout_minutes': tpw,
            'has_cardio':  1 if 'Cardio'   in all_types else 0,
            'has_strength': 1 if 'Strength' in all_types else 0,
            'has_yoga':    1 if 'Yoga'     in all_types else 0,
            'has_hiit':    1 if 'HIIT'     in all_types else 0,
            'exercises':   exercise_rows,
            'week1_exercises': week1,
        })

    programs_df = pd.DataFrame([
        {k: v for k, v in p.items() if k not in ('exercises', 'week1_exercises')}
        for p in programs_list
    ])
    print(f"  Built {len(programs_list)} program records in {time.time()-t2:.1f}s")

    # ── Step 3: Compute goal_encoded and equipment_encoded ────────────────────
    # Must match generate_interactions.py lines 654-657 EXACTLY:
    #   unique_goals     = sorted(programs_df['goal'].unique())
    #   unique_equipment = sorted(programs_df['equipment'].unique())
    print("\n[3/5] Computing goal_encoded and equipment_encoded...")

    unique_goals     = sorted(programs_df['goal'].unique())
    unique_equipment = sorted(programs_df['equipment'].unique())
    goal_to_int  = {g: i for i, g in enumerate(unique_goals)}
    equip_to_int = {e: i for i, e in enumerate(unique_equipment)}

    print(f"  Goals  ({len(unique_goals)}): {goal_to_int}")
    print(f"  Equipment ({len(unique_equipment)}): {equip_to_int}")

    for p in programs_list:
        p['goal_encoded']      = goal_to_int.get(p['goal'], -1)
        p['equipment_encoded'] = equip_to_int.get(p['equipment'], -1)

    # ── Step 4: Cross-verify against program_features.csv ────────────────────
    print("\n[4/5] Cross-verifying against program_features.csv...")
    pf = pd.read_csv(PROG_FEAT_CSV)

    mismatches = 0
    for p in programs_list:
        pid = p['program_id']
        ref = pf[pf['program_id'] == pid]
        if ref.empty:
            print(f"  WARNING: program_id {pid} not found in program_features.csv")
            mismatches += 1
            continue
        ref = ref.iloc[0]

        # Check level, goal, equipment, binary flags
        for field in ['level_encoded', 'goal_encoded', 'equipment_encoded',
                      'has_cardio', 'has_strength', 'has_yoga', 'has_hiit']:
            if p[field] != int(ref[field]):
                print(f"  MISMATCH pid={pid} title='{p['title'][:40]}' "
                      f"field={field}: catalog={p[field]} vs csv={int(ref[field])}")
                mismatches += 1

    if mismatches == 0:
        print("  All 2598 programs verified against program_features.csv. OK")
    else:
        print(f"  {mismatches} mismatches found (see above)")

    # ── Step 5: Save catalog and normalization stats ──────────────────────────
    print("\n[5/5] Saving catalog and normalization statistics...")

    # Build final dict keyed by program_id
    catalog = {p['program_id']: p for p in programs_list}

    with open(CATALOG_PATH, 'wb') as f:
        pickle.dump(catalog, f)
    print(f"  Catalog saved -> {CATALOG_PATH}")
    print(f"  Catalog size: {os.path.getsize(CATALOG_PATH) / 1e6:.1f} MB")

    # Normalization stats needed at inference time:
    # For program features (from program_features.csv, which is already normalized)
    # We need the raw min/max values to normalize new user inputs for content filter
    prog_len_min   = programs_df['program_length_weeks'].min()
    prog_len_max   = programs_df['program_length_weeks'].max()
    tpw_min        = programs_df['time_per_workout_minutes'].min()
    tpw_max        = programs_df['time_per_workout_minutes'].max()

    # For user features (from user_features.csv raw values, need the original data)
    # We read user_features.csv to get the normalization ranges used there
    uf = pd.read_csv(USER_FEAT_CSV)

    # user_features.csv already normalized to [0,1]. The original ranges came
    # from the gym_members dataset. We compute the effective min/max from the
    # normalized values alongside the original dataset.
    # For new user inference, we apply the same normalization using the saved ranges.
    # Since we can't recover exact original min/max from normalized values alone,
    # we derive them from the raw gym member data directly.

    gym_csv = os.environ.get(
        "FITNOVA_GYM_CSV",
        os.path.join(_DATASETS_DIR, "gym_members_exercise_tracking.csv")
    )
    gym_raw = pd.read_csv(gym_csv)

    # Clean column names (same as generate_interactions.py)
    col_renames = {}
    for c in gym_raw.columns:
        clean = c.strip().replace(' ', '_').replace('(', '').replace(')', '')
        col_renames[c] = clean
    gym_raw.rename(columns=col_renames, inplace=True)

    # Compute BMI
    gym_raw['bmi'] = gym_raw['Weight_kg'] / (gym_raw['Height_m'] ** 2)

    # Frequency column
    freq_col = [c for c in gym_raw.columns if 'frequency' in c.lower() or 'Frequency' in c.lower()]
    if freq_col:
        gym_raw.rename(columns={freq_col[0]: 'workout_frequency'}, inplace=True)
    if 'Session_Duration_hours' not in gym_raw.columns:
        dur_col = [c for c in gym_raw.columns if 'duration' in c.lower()]
        if dur_col:
            gym_raw.rename(columns={dur_col[0]: 'Session_Duration_hours'}, inplace=True)

    user_norm_stats = {}
    for col, src_col in [
        ('session_duration_hours', 'Session_Duration_hours'),
        ('workout_frequency',      'workout_frequency'),
        ('bmi',                    'bmi'),
        ('age',                    'Age'),
    ]:
        if src_col in gym_raw.columns:
            user_norm_stats[col] = {
                'min': float(gym_raw[src_col].min()),
                'max': float(gym_raw[src_col].max()),
            }

    norm_stats = {
        # For normalizing program continuous features (matching program_features.csv)
        'program_length_weeks':     {'min': prog_len_min, 'max': prog_len_max},
        'time_per_workout_minutes': {'min': tpw_min,      'max': tpw_max},
        # For normalizing user continuous features (matching user_features.csv)
        'user': user_norm_stats,
        # Encoding maps (needed by content filter and API encoding)
        'goal_to_int':  goal_to_int,
        'equip_to_int': equip_to_int,
    }

    with open(NORM_PATH, 'wb') as f:
        pickle.dump(norm_stats, f)
    print(f"  Norm stats saved -> {NORM_PATH}")

    # ── Verification summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"  Total programs in catalog : {len(catalog)}")
    print(f"  catalog[0]['title']       : {catalog[0]['title']}")
    print(f"  catalog[0] exercises      : {len(catalog[0]['exercises'])} rows")
    print(f"  catalog[0] week1          : {len(catalog[0]['week1_exercises'])} exercises")
    print(f"  catalog[2597]['title']    : {catalog[2597]['title']}")
    print()
    print("  Sample program (id=1):")
    p = catalog[1]
    print(f"    title       : {p['title']}")
    print(f"    goal        : {p['goal']} (encoded={p['goal_encoded']})")
    print(f"    equipment   : {p['equipment']} (encoded={p['equipment_encoded']})")
    print(f"    level       : {p['level_encoded']}")
    print(f"    types       : cardio={p['has_cardio']} strength={p['has_strength']} "
          f"yoga={p['has_yoga']} hiit={p['has_hiit']}")
    print(f"    weeks/tpw   : {p['program_length_weeks']}w / {p['time_per_workout_minutes']}min")
    print(f"    exercises   : {len(p['exercises'])} total, "
          f"{len(p['week1_exercises'])} in week 1")
    if p['week1_exercises']:
        ex = p['week1_exercises'][0]
        print(f"    first ex    : {ex['exercise_name']} - "
              f"{ex['sets']}x{ex['reps']}")
    print()
    print("  Normalization stats:")
    for k, v in norm_stats.items():
        if k not in ('goal_to_int', 'equip_to_int', 'user'):
            print(f"    {k}: min={v['min']:.2f}, max={v['max']:.2f}")
    print("  User normalization ranges:")
    for k, v in norm_stats['user'].items():
        print(f"    {k}: min={v['min']:.2f}, max={v['max']:.2f}")

    print(f"\nDone in {time.time()-t_start:.1f}s")
    print("=" * 60)

    return catalog, norm_stats


if __name__ == '__main__':
    build_catalog()
