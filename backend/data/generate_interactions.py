"""
FitNova synthetic interaction generator.

This version is repo-local and uses the shared workout taxonomy so the
generated features, interaction matrix, and catalog stay aligned.
"""

from __future__ import annotations

from ast import literal_eval
from collections import Counter
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from backend.data.workout_taxonomy import RELATED_TYPES, infer_program_type_profile
except ImportError:
    from workout_taxonomy import RELATED_TYPES, infer_program_type_profile


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(THIS_DIR, "datasets")
USER_DATA_PATH = os.environ.get(
    "FITNOVA_GYM_CSV",
    os.path.join(DATASETS_DIR, "gym_members_exercise_tracking.csv"),
)
PROGRAM_DATA_PATH = os.environ.get(
    "FITNOVA_PROGRAM_CSV",
    os.path.join(DATASETS_DIR, "programs_detailed_boostcamp_kaggle.csv"),
)
OUTPUT_DIR = os.environ.get("FITNOVA_DATA_DIR", THIS_DIR)

RANDOM_SEED = 42
NOISE_STD = 0.08
NEGATIVE_RATIO = 4
MIN_POSITIVES_PER_USER = 3
TARGET_SPARSITY_MIN = 0.94
TARGET_SPARSITY_MAX = 0.96
EXPERT_BOOST_FACTOR = 1.35

W_EXPERIENCE = 0.20
W_WORKOUT_TYPE = 0.55
W_DURATION = 0.15
W_FREQUENCY = 0.10

SECONDARY_TYPE_MATCH = {
    "Cardio": 0.24,
    "Strength": 0.32,
    "Yoga": 0.16,
    "HIIT": 0.22,
}
RELATED_PRIMARY_MATCH = {
    "Cardio": 0.20,
    "Strength": 0.18,
    "Yoga": 0.0,
    "HIIT": 0.24,
}
RELATED_SECONDARY_MATCH = {
    "Cardio": 0.08,
    "Strength": 0.06,
    "Yoga": 0.0,
    "HIIT": 0.10,
}


def parse_list_field(val):
    if pd.isna(val) or str(val).strip() in ("[]", ""):
        return []
    try:
        result = literal_eval(str(val))
        if isinstance(result, list):
            return result
        return [str(result)]
    except Exception:
        return [str(val).strip()]


def normalize_minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val <= min_val:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - min_val) / (max_val - min_val)).astype(np.float32)


def workout_match_score(
    user_type: str,
    primary_type: str,
    secondary_types: set[str],
    active_types: set[str],
) -> float:
    if user_type == primary_type:
        return 1.0
    if user_type in secondary_types:
        return SECONDARY_TYPE_MATCH.get(user_type, 0.2)

    related_types = RELATED_TYPES.get(user_type, set())
    if primary_type in related_types:
        return RELATED_PRIMARY_MATCH.get(user_type, 0.1)
    if related_types & secondary_types:
        return RELATED_SECONDARY_MATCH.get(user_type, 0.05)
    return 0.0


def build_program_dataframe(programs_raw: pd.DataFrame) -> pd.DataFrame:
    program_groups = programs_raw.groupby("title", sort=False)
    programs_list = []

    for title, group in program_groups:
        first_row = group.iloc[0]

        levels_raw = parse_list_field(first_row["level"])
        level_map = {"beginner": 0, "novice": 0, "intermediate": 1, "advanced": 2}
        if levels_raw:
            level_values = [level_map.get(str(level).lower().strip(), 1) for level in levels_raw]
            primary_level = min(level_values) if level_values else 1
        else:
            primary_level = 1

        goals_raw = parse_list_field(first_row["goal"])
        primary_goal = goals_raw[0] if goals_raw else "General Fitness"
        equipment = str(first_row["equipment"]).strip() if pd.notna(first_row["equipment"]) else "Unknown"
        program_length = float(first_row["program_length"]) if pd.notna(first_row["program_length"]) else 4.0
        time_per_workout = float(first_row["time_per_workout"]) if pd.notna(first_row["time_per_workout"]) else 60.0

        exercises = group["exercise_name"].dropna().astype(str).tolist()
        type_profile = infer_program_type_profile(title, exercises)

        week_day = group[["week", "day"]].dropna().drop_duplicates()
        week_count = group["week"].dropna().nunique()
        weekly_frequency = round(len(week_day) / week_count) if week_count > 0 else 3
        weekly_frequency = max(1, min(7, weekly_frequency))

        programs_list.append(
            {
                "program_id": len(programs_list),
                "title": title,
                "level_encoded": primary_level,
                "goal": primary_goal,
                "equipment": equipment,
                "program_length_weeks": program_length,
                "time_per_workout_minutes": time_per_workout,
                "weekly_frequency": weekly_frequency,
                "primary_type": type_profile["primary_type"],
                "secondary_types": type_profile["secondary_types"],
                "has_cardio": type_profile["has_cardio"],
                "has_strength": type_profile["has_strength"],
                "has_yoga": type_profile["has_yoga"],
                "has_hiit": type_profile["has_hiit"],
            }
        )

    return pd.DataFrame(programs_list)


def load_user_dataframe() -> pd.DataFrame:
    users_raw = pd.read_csv(USER_DATA_PATH)
    col_renames = {}
    for column in users_raw.columns:
        cleaned = column.strip().replace(" ", "_").replace("(", "").replace(")", "")
        col_renames[column] = cleaned
    users_raw.rename(columns=col_renames, inplace=True)
    users_raw.rename(
        columns={
            "Weight_kg": "weight_kg",
            "Height_m": "height_m",
            "Session_Duration_hours": "session_duration_hours",
            "Calories_Burned": "calories_burned",
            "Workout_Type": "workout_type",
            "Fat_Percentage": "fat_percentage",
            "Water_Intake_liters": "water_intake_liters",
            "Workout_Frequency_days/week": "workout_frequency",
            "Experience_Level": "experience_level",
            "Age": "age",
            "Gender": "gender",
        },
        inplace=True,
    )

    freq_col = [column for column in users_raw.columns if "frequency" in column.lower()]
    if freq_col and "workout_frequency" not in users_raw.columns:
        users_raw.rename(columns={freq_col[0]: "workout_frequency"}, inplace=True)

    users_raw["user_id"] = range(len(users_raw))
    users_raw["workout_type_encoded"] = users_raw["workout_type"].map(
        {"Cardio": 0, "Strength": 1, "Yoga": 2, "HIIT": 3}
    )
    users_raw["gender_encoded"] = users_raw["gender"].map({"Male": 0, "Female": 1})
    users_raw["bmi"] = users_raw["weight_kg"] / (users_raw["height_m"] ** 2)
    return users_raw


def build_program_features(programs_df: pd.DataFrame) -> pd.DataFrame:
    unique_goals = sorted(programs_df["goal"].unique())
    unique_equipment = sorted(programs_df["equipment"].unique())
    goal_to_int = {goal: index for index, goal in enumerate(unique_goals)}
    equip_to_int = {equipment: index for index, equipment in enumerate(unique_equipment)}

    features = pd.DataFrame()
    features["program_id"] = programs_df["program_id"]
    features["level_encoded"] = programs_df["level_encoded"]
    features["goal_encoded"] = programs_df["goal"].map(goal_to_int)
    features["equipment_encoded"] = programs_df["equipment"].map(equip_to_int)
    features["program_length_weeks"] = normalize_minmax(programs_df["program_length_weeks"])
    features["time_per_workout_minutes"] = normalize_minmax(programs_df["time_per_workout_minutes"])
    features["has_cardio"] = programs_df["has_cardio"]
    features["has_strength"] = programs_df["has_strength"]
    features["has_yoga"] = programs_df["has_yoga"]
    features["has_hiit"] = programs_df["has_hiit"]
    return features


def build_user_features(users_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    features["user_id"] = users_df["user_id"]
    features["experience_level"] = users_df["experience_level"]
    features["workout_type_encoded"] = users_df["workout_type_encoded"]
    features["session_duration_hours"] = normalize_minmax(users_df["session_duration_hours"])
    features["workout_frequency"] = normalize_minmax(users_df["workout_frequency"])
    features["bmi"] = normalize_minmax(users_df["bmi"])
    features["age"] = normalize_minmax(users_df["age"])
    features["gender_encoded"] = users_df["gender_encoded"]
    return features


def build_score_matrix(users_df: pd.DataFrame, programs_df: pd.DataFrame) -> np.ndarray:
    n_users = len(users_df)
    n_programs = len(programs_df)
    all_scores = np.zeros((n_users, n_programs), dtype=np.float32)

    exp_score_table = {
        1: {0: 1.0, 1: 0.4, 2: 0.0},
        2: {0: 0.5, 1: 1.0, 2: 0.5},
        3: {0: 0.1, 1: 0.6, 2: 1.0},
    }

    primary_types = programs_df["primary_type"].tolist()
    active_types = [
        {
            workout_type
            for workout_type, has_value in (
                ("Cardio", row.has_cardio),
                ("Strength", row.has_strength),
                ("Yoga", row.has_yoga),
                ("HIIT", row.has_hiit),
            )
            if has_value
        }
        for row in programs_df.itertuples()
    ]
    secondary_types = [set(types) for types in programs_df["secondary_types"].tolist()]

    for user_index, user_row in enumerate(users_df.itertuples()):
        for program_index, program_row in enumerate(programs_df.itertuples()):
            experience_score = exp_score_table[user_row.experience_level][program_row.level_encoded]
            type_score = workout_match_score(
                user_row.workout_type,
                primary_types[program_index],
                secondary_types[program_index],
                active_types[program_index],
            )

            duration_diff = abs(user_row.session_duration_hours * 60 - program_row.time_per_workout_minutes)
            if duration_diff <= 10:
                duration_score = 1.0
            elif duration_diff <= 20:
                duration_score = 0.6
            elif duration_diff <= 30:
                duration_score = 0.3
            else:
                duration_score = 0.0

            frequency_diff = abs(user_row.workout_frequency - program_row.weekly_frequency)
            if frequency_diff == 0:
                frequency_score = 1.0
            elif frequency_diff == 1:
                frequency_score = 0.7
            elif frequency_diff == 2:
                frequency_score = 0.3
            else:
                frequency_score = 0.0

            score = (
                W_EXPERIENCE * experience_score
                + W_WORKOUT_TYPE * type_score
                + W_DURATION * duration_score
                + W_FREQUENCY * frequency_score
            )

            all_scores[user_index, program_index] = min(score, 1.0)

    return all_scores


def threshold_scores(noisy_scores: np.ndarray, user_experience: np.ndarray, program_levels: np.ndarray) -> np.ndarray:
    total_cells = noisy_scores.shape[0] * noisy_scores.shape[1]
    lo, hi = 0.82, 0.98
    best_matrix = None

    for _ in range(50):
        threshold = (lo + hi) / 2
        matrix = (noisy_scores >= threshold).astype(np.int8)
        sparsity = 1.0 - (matrix.sum() / total_cells)
        if sparsity < TARGET_SPARSITY_MIN:
            lo = threshold
        elif sparsity > TARGET_SPARSITY_MAX:
            hi = threshold
        else:
            best_matrix = matrix
            break

    if best_matrix is None:
        threshold = (lo + hi) / 2
        best_matrix = (noisy_scores >= threshold).astype(np.int8)

    expert_users = np.where(user_experience == 3)[0]
    advanced_programs = np.where(program_levels == 2)[0]
    for user_index in expert_users:
        for program_index in advanced_programs:
            noisy_scores[user_index, program_index] = min(
                noisy_scores[user_index, program_index] * EXPERT_BOOST_FACTOR,
                1.0,
            )

    final_threshold = (lo + hi) / 2
    matrix = (noisy_scores >= final_threshold).astype(np.int8)
    return matrix


def enforce_quality(matrix: np.ndarray) -> np.ndarray:
    user_positive_counts = matrix.sum(axis=1)
    for user_index in np.where(user_positive_counts < MIN_POSITIVES_PER_USER)[0]:
        deficit = MIN_POSITIVES_PER_USER - int(user_positive_counts[user_index])
        if deficit <= 0:
            continue
        negatives = np.where(matrix[user_index] == 0)[0][:deficit]
        matrix[user_index, negatives] = 1

    program_positive_counts = matrix.sum(axis=0)
    for program_index in np.where(program_positive_counts == 0)[0]:
        matrix[0:3, program_index] = 1

    return matrix


def build_interactions(matrix: np.ndarray) -> pd.DataFrame:
    rows: list[tuple[int, int, int]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for user_index in range(matrix.shape[0]):
        positives = np.where(matrix[user_index] == 1)[0]
        negatives = np.where(matrix[user_index] == 0)[0]
        negative_count = min(len(negatives), len(positives) * NEGATIVE_RATIO)
        sampled_negatives = (
            rng.choice(negatives, size=negative_count, replace=False) if negative_count else np.array([], dtype=int)
        )

        rows.extend((user_index, int(program_index), 1) for program_index in positives)
        rows.extend((user_index, int(program_index), 0) for program_index in sampled_negatives)

    return pd.DataFrame(rows, columns=["user_id", "program_id", "interaction"])


def write_report(programs_df: pd.DataFrame, interactions_df: pd.DataFrame, output_path: str) -> None:
    positives = int((interactions_df["interaction"] == 1).sum())
    negatives = int((interactions_df["interaction"] == 0).sum())
    counts = Counter(programs_df["primary_type"])
    report = [
        "FITNOVA GENERATION REPORT",
        "=" * 40,
        f"Programs: {len(programs_df)}",
        f"Interactions: {len(interactions_df)}",
        f"Positives: {positives}",
        f"Negatives: {negatives}",
        "",
        "Primary type distribution:",
    ]
    for workout_type, count in counts.items():
        report.append(f"  {workout_type}: {count}")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))


def main():
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    users_df = load_user_dataframe()
    programs_raw = pd.read_csv(
        PROGRAM_DATA_PATH,
        usecols=[
            "title",
            "level",
            "goal",
            "equipment",
            "program_length",
            "time_per_workout",
            "week",
            "day",
            "exercise_name",
        ],
        low_memory=False,
    )

    programs_df = build_program_dataframe(programs_raw)
    program_features = build_program_features(programs_df)
    user_features = build_user_features(users_df)

    base_scores = build_score_matrix(users_df, programs_df)
    noise = np.random.default_rng(RANDOM_SEED).normal(0, NOISE_STD, size=base_scores.shape).astype(np.float32)
    noisy_scores = np.clip(base_scores + noise, 0.0, 1.0)
    interaction_matrix = threshold_scores(
        noisy_scores,
        users_df["experience_level"].values,
        programs_df["level_encoded"].values,
    )
    interaction_matrix = enforce_quality(interaction_matrix)
    interactions_df = build_interactions(interaction_matrix)

    interactions_df.to_csv(os.path.join(OUTPUT_DIR, "interactions.csv"), index=False)
    user_features.to_csv(os.path.join(OUTPUT_DIR, "user_features.csv"), index=False)
    program_features.to_csv(os.path.join(OUTPUT_DIR, "program_features.csv"), index=False)
    write_report(programs_df, interactions_df, os.path.join(OUTPUT_DIR, "generation_report.txt"))

    sparsity = 1.0 - (interaction_matrix.sum() / (interaction_matrix.shape[0] * interaction_matrix.shape[1]))
    print("=" * 70)
    print("FitNova Synthetic Interaction Matrix Generator")
    print("=" * 70)
    print(f"Users: {len(users_df)} | Programs: {len(programs_df)}")
    print(f"Positives: {(interactions_df['interaction'] == 1).sum():,}")
    print(f"Negatives: {(interactions_df['interaction'] == 0).sum():,}")
    print(f"Sparsity: {sparsity * 100:.2f}%")
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
