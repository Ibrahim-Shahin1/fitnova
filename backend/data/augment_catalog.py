"""
FitNova - Synthetic Program Augmentation
==========================================
The real Kaggle dataset is 99.2% Strength programs (2,577/2,598).
This makes NeuMF almost blind to Yoga/Cardio/HIIT preferences.

This script generates 150 synthetic programs per minority type and saves
them to synthetic_programs.pkl. Both build_catalog.py and
generate_interactions.py load this file automatically if it exists.

Run first, then rebuild data and retrain:
    python backend/data/augment_catalog.py
    python backend/data/build_catalog.py
    python backend/data/generate_interactions.py
    cd backend && python training/train_neumf.py
"""

from __future__ import annotations

import os
import pickle
import random
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("FITNOVA_DATA_DIR", THIS_DIR)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "synthetic_programs.pkl")

RANDOM_SEED = 42
N_PER_TYPE = 150  # programs generated per minority type (Yoga, Cardio, HIIT)

# ─────────────────────────────────────────────────────────────────────────────
# Exercise pools  (chosen to match workout_taxonomy.py keyword lists)
# ─────────────────────────────────────────────────────────────────────────────

YOGA_EXERCISES = [
    "Sun Salutation A", "Sun Salutation B", "Downward Dog", "Child's Pose",
    "Cobra Stretch", "Cat Cow Stretch", "Pigeon Pose", "Hip Flexor Stretch",
    "World's Greatest Stretch", "Thoracic Rotation Stretch", "90/90 Hip Stretch",
    "Ankle Dorsiflexion Stretch", "Chest Opener Stretch",
    "Thread the Needle Stretch", "Seated Forward Fold", "Standing Quad Stretch",
    "Hip Airplane", "Deep Breathing Exercise", "Dead Bug", "Bird Dog",
    "Shoulder Dislocate", "Wall Slide", "Banded Ankle Distraction",
    "Lateral Hip Stretch", "Spinal Twist Stretch", "Hamstring Stretch",
    "Calf Stretch", "Hip Opener Stretch", "Mobility Flow",
    "Restorative Stretch", "Neck Stretch", "Shoulder Mobility Drill",
    "Wrist Mobility Exercise", "Ankle Circles", "Kneeling Hip Flexor Stretch",
]

CARDIO_EXERCISES = [
    "Treadmill Run", "Stationary Cycling", "Rowing Machine", "Elliptical Trainer",
    "Stair Master", "Jump Rope", "Jogging", "Walking",
    "Incline Treadmill Walk", "Rower Intervals", "Brisk Walk",
    "Easy Bike Ride", "Cardio Cooldown Walk", "Moderate Jog",
    "Low-Intensity Cardio", "Steady-State Cardio", "Cross Trainer",
    "Air Bike", "Recumbent Bike", "Swimming Laps",
]

HIIT_EXERCISES = [
    "Burpees", "Box Jumps", "Battle Ropes", "Mountain Climbers",
    "Jump Rope", "Sled Push", "Tabata Intervals", "Explosive Push-Up",
    "Plyo Lunge", "Sprint Intervals", "Farmer's Walk", "Kettlebell Swing",
    "Jump Squat", "Medicine Ball Slam", "Bear Crawl", "Lateral Shuffle",
    "High Knees", "Broad Jump", "Tuck Jump", "AMRAP Circuit",
    "Conditioning Circuit", "Agility Ladder Drill", "Sandbag Carry",
    "Power Clean", "Push Press",
]

# ─────────────────────────────────────────────────────────────────────────────
# Title templates  (use {level}, {weeks}, {days} as placeholders)
# ─────────────────────────────────────────────────────────────────────────────

YOGA_TITLES = [
    "{level} Yoga & Mobility Program",
    "{weeks}-Week Flexibility & Mobility Plan",
    "Daily Stretching & Mobility Routine",
    "{level} Yoga Flow Program",
    "Total Body Flexibility Program",
    "Posture & Mobility Improvement Plan",
    "{weeks}-Week Yoga & Stretching Plan",
    "Restorative Yoga Program",
    "Full Body Mobility & Flexibility Program",
    "Morning Yoga & Stretch Routine",
    "Yin Yoga Recovery Program",
    "Vinyasa Flow {level} Program",
    "Flexibility & Movement Quality Plan",
    "Mobility-First Training Program",
    "{days}x Per Week Mobility Program",
    "Active Recovery & Stretching Plan",
    "{level} Stretch & Mobility Series",
    "Yoga for Athletic Performance",
    "{weeks}-Week Posture Correction Program",
    "Hip & Shoulder Mobility Program",
]

CARDIO_TITLES = [
    "{weeks}-Week Cardio Endurance Plan",
    "{level} Cardio & Conditioning Program",
    "Fat Loss Cardio Program",
    "{weeks}-Week Running Program",
    "Endurance & Aerobic Fitness Plan",
    "LISS Cardio Program",
    "{level} Aerobic Base Builder",
    "Cardio for Weight Loss",
    "Low-Intensity Steady-State Program",
    "Heart Health & Endurance Plan",
    "{days}x Per Week Cardio Plan",
    "{weeks}-Week Aerobic Conditioning",
    "Rowing Machine Endurance Program",
    "Cycling Fitness Program",
    "{level} Running & Cardio Plan",
    "Mixed Cardio Training Program",
    "Treadmill Endurance Program",
    "Cardio Health & Fitness Plan",
    "{weeks}-Week Cardiovascular Fitness",
    "{level} Endurance Training Plan",
]

HIIT_TITLES = [
    "{weeks}-Week HIIT Program",
    "{level} HIIT & Conditioning Plan",
    "High Intensity Interval Training Program",
    "{weeks}-Week Metabolic Conditioning",
    "HIIT Circuit Training Program",
    "Explosive HIIT Program",
    "{level} MetCon Training Plan",
    "AMRAP & EMOM Training Program",
    "Athletic HIIT Conditioning",
    "Tabata & HIIT Fusion Program",
    "Circuit Training & HIIT Plan",
    "{days}x Per Week HIIT Plan",
    "Interval Training & Conditioning",
    "HIIT for Fat Loss",
    "Power & Conditioning Program",
    "{weeks}-Week Athletic Conditioning",
    "{level} Explosive Training Plan",
    "Full Body HIIT Program",
    "Metabolic HIIT Training Plan",
    "{level} Interval Conditioning Program",
]

LEVEL_NAMES = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}

# Per-type configuration
TYPE_CONFIG: dict[str, dict] = {
    "Yoga": {
        "titles":    YOGA_TITLES,
        "goals":     ["Flexibility", "General Fitness", "Weight Loss", "Toning"],
        "equipment": ["Yoga Mat", "Bodyweight", "No Equipment Required"],
        "exercises": YOGA_EXERCISES,
        "days":      [3, 4, 5, 6],
        "durations": [30, 45, 60],
        "weeks":     [4, 6, 8, 12],
        "sets_range": (1, 3),
        "reps_fn":   lambda rng: f"{rng.randint(30, 60)} sec",
        "has_cardio": 0, "has_strength": 0, "has_yoga": 1, "has_hiit": 0,
    },
    "Cardio": {
        "titles":    CARDIO_TITLES,
        "goals":     ["Weight Loss", "Endurance", "General Fitness", "Toning",
                      "Cardio Health"],
        "equipment": ["Cardio Equipment", "Treadmill", "Bodyweight",
                      "Stationary Bike"],
        "exercises": CARDIO_EXERCISES,
        "days":      [3, 4, 5],
        "durations": [30, 45, 60, 75],
        "weeks":     [4, 6, 8, 12],
        "sets_range": (1, 1),
        "reps_fn":   lambda rng: f"{rng.randint(20, 45)} min",
        "has_cardio": 1, "has_strength": 0, "has_yoga": 0, "has_hiit": 0,
    },
    "HIIT": {
        "titles":    HIIT_TITLES,
        "goals":     ["Weight Loss", "Athletic Performance", "General Fitness",
                      "Toning"],
        "equipment": ["Full Gym", "Bodyweight", "Minimal Equipment"],
        "exercises": HIIT_EXERCISES,
        "days":      [3, 4, 5],
        "durations": [20, 30, 45, 60],
        "weeks":     [4, 6, 8, 12],
        "sets_range": (3, 5),
        "reps_fn":   lambda rng: str(rng.choice([10, 12, 15, 20])),
        "has_cardio": 0, "has_strength": 0, "has_yoga": 0, "has_hiit": 1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unique_title(template: str, level: int, weeks: int, days: int,
                  used: set[str], rng: random.Random) -> str:
    base = template.format(level=LEVEL_NAMES[level], weeks=weeks, days=days)
    if base not in used:
        return base
    for n in range(2, 30):
        candidate = f"{base} {n}"
        if candidate not in used:
            return candidate
    return f"{base} {rng.randint(100, 999)}"


def _make_schedule(pool: list[str], days_per_week: int, weeks: int,
                   sets_range: tuple[int, int], reps_fn,
                   rng: random.Random) -> tuple[list, list]:
    """Build a week-by-week exercise schedule."""
    schedule = []
    ex_per_day = rng.randint(4, 8)
    for week in range(1, weeks + 1):
        for day in range(1, days_per_week + 1):
            day_exs = rng.sample(pool, min(ex_per_day, len(pool)))
            for ex_name in day_exs:
                schedule.append({
                    "week":          week,
                    "day":           day,
                    "exercise_name": ex_name,
                    "sets":          rng.randint(*sets_range),
                    "reps":          reps_fn(rng),
                })
    week1 = [e for e in schedule if e["week"] == 1]
    return schedule, week1


# ─────────────────────────────────────────────────────────────────────────────
# Main generator (importable for tests)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_programs(n_per_type: int = N_PER_TYPE) -> list[dict]:
    """
    Generate n_per_type programs each for Yoga, Cardio, and HIIT.

    Returns a list of program dicts compatible with both build_catalog.py
    and generate_interactions.py.  No program_id is assigned here — the
    callers append these after the real programs and assign IDs sequentially.
    """
    rng = random.Random(RANDOM_SEED)
    programs: list[dict] = []
    used_titles: set[str] = set()

    for workout_type, cfg in TYPE_CONFIG.items():
        titles = cfg["titles"]
        for i in range(n_per_type):
            level   = i % 3          # cycles 0/1/2 → 50 of each per type
            weeks   = rng.choice(cfg["weeks"])
            days    = rng.choice(cfg["days"])
            dur     = rng.choice(cfg["durations"])
            goal    = rng.choice(cfg["goals"])
            equip   = rng.choice(cfg["equipment"])
            tmpl    = titles[i % len(titles)]

            title = _unique_title(tmpl, level, weeks, days, used_titles, rng)
            used_titles.add(title)

            description = (
                f"A {LEVEL_NAMES[level].lower()}-level {workout_type.lower()} "
                f"program designed for {goal.lower()}. "
                f"{weeks} weeks, {days} days per week, {dur} minutes per session."
            )

            schedule, week1 = _make_schedule(
                cfg["exercises"], days, weeks,
                cfg["sets_range"], cfg["reps_fn"], rng
            )

            programs.append({
                # Shared fields (generate_interactions.py + build_catalog.py)
                "title":                   title,
                "level_encoded":           level,
                "goal":                    goal,
                "equipment":               equip,
                "program_length_weeks":    float(weeks),
                "time_per_workout_minutes": float(dur),
                "weekly_frequency":        days,
                "primary_type":            workout_type,
                "secondary_types":         [],
                "has_cardio":              cfg["has_cardio"],
                "has_strength":            cfg["has_strength"],
                "has_yoga":                cfg["has_yoga"],
                "has_hiit":                cfg["has_hiit"],
                # Build-catalog-only fields
                "description":             description,
                "exercises":               schedule,
                "week1_exercises":         week1,
            })

    return programs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("FitNova -- Synthetic Program Augmentation")
    print("=" * 60)

    programs = generate_synthetic_programs()

    counts = Counter(p["primary_type"] for p in programs)
    print(f"\nGenerated {len(programs)} synthetic programs:")
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(programs, f)

    print(f"\nSaved -> {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  python backend/data/build_catalog.py")
    print("  python backend/data/generate_interactions.py")
    print("  cd backend && python training/train_neumf.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
