"""
Tests for the synthetic program augmentation module.
"""

from collections import Counter

from backend.data.augment_catalog import generate_synthetic_programs, N_PER_TYPE


def test_correct_count_per_type():
    programs = generate_synthetic_programs()
    counts = Counter(p["primary_type"] for p in programs)
    assert counts["Yoga"] == N_PER_TYPE, (
        f"Expected {N_PER_TYPE} Yoga programs, got {counts['Yoga']}"
    )
    assert counts["Cardio"] == N_PER_TYPE, (
        f"Expected {N_PER_TYPE} Cardio programs, got {counts['Cardio']}"
    )
    assert counts["HIIT"] == N_PER_TYPE, (
        f"Expected {N_PER_TYPE} HIIT programs, got {counts['HIIT']}"
    )
    assert counts.get("Strength", 0) == 0, "Strength programs should not be generated"


def test_titles_are_unique():
    programs = generate_synthetic_programs()
    titles = [p["title"] for p in programs]
    assert len(titles) == len(set(titles)), "All synthetic program titles must be unique"


def test_type_flags_are_correct():
    programs = generate_synthetic_programs()
    for p in programs:
        t = p["primary_type"]
        if t == "Yoga":
            assert p["has_yoga"] == 1 and p["has_cardio"] == 0
            assert p["has_strength"] == 0 and p["has_hiit"] == 0
        elif t == "Cardio":
            assert p["has_cardio"] == 1 and p["has_yoga"] == 0
            assert p["has_strength"] == 0 and p["has_hiit"] == 0
        elif t == "HIIT":
            assert p["has_hiit"] == 1 and p["has_yoga"] == 0
            assert p["has_strength"] == 0 and p["has_cardio"] == 0


def test_required_fields_present():
    required = [
        "title", "level_encoded", "goal", "equipment",
        "program_length_weeks", "time_per_workout_minutes",
        "weekly_frequency", "primary_type", "secondary_types",
        "has_cardio", "has_strength", "has_yoga", "has_hiit",
        "description", "exercises", "week1_exercises",
    ]
    programs = generate_synthetic_programs()
    for p in programs:
        for field in required:
            assert field in p, f"Missing field '{field}' in program: {p.get('title')}"


def test_level_distribution_is_balanced():
    programs = generate_synthetic_programs()
    level_counts = Counter(p["level_encoded"] for p in programs)
    # Each type generates 50 Beginner / 50 Intermediate / 50 Advanced
    total = N_PER_TYPE * 3  # 3 types
    assert level_counts[0] == total // 3, f"Expected {total//3} Beginner, got {level_counts[0]}"
    assert level_counts[1] == total // 3
    assert level_counts[2] == total // 3


def test_exercises_schedule_has_correct_weeks():
    programs = generate_synthetic_programs()
    for p in programs:
        exercises = p["exercises"]
        week1 = p["week1_exercises"]
        assert len(exercises) > 0, f"No exercises in {p['title']}"
        assert len(week1) > 0, f"No week1 exercises in {p['title']}"
        # All week1_exercises should have week == 1
        assert all(e["week"] == 1 for e in week1), (
            f"week1_exercises contains non-week-1 entries in {p['title']}"
        )
        # The number of weeks should match program_length_weeks
        weeks_in_schedule = {e["week"] for e in exercises}
        assert len(weeks_in_schedule) == int(p["program_length_weeks"]), (
            f"Week count mismatch in {p['title']}: "
            f"expected {int(p['program_length_weeks'])}, got {len(weeks_in_schedule)}"
        )
