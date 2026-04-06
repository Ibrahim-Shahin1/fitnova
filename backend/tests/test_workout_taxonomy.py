from backend.data.workout_taxonomy import infer_program_type_profile


def test_mobility_program_is_yoga_primary():
    profile = infer_program_type_profile(
        "Weightlifting Mobility program",
        [
            "Knee-to-wall ankle dorsiflexion test",
            "Slant board calf stretch",
            "90/90 Hip Rotations",
            "Pigeon Stretch with Thoracic Rotation",
            "Hip Airplanes",
            "Quadruped T-Spine Rotations",
        ],
    )

    assert profile["primary_type"] == "Yoga"
    assert profile["has_yoga"] == 1


def test_strength_program_with_one_mobility_drill_is_not_yoga():
    profile = infer_program_type_profile(
        "Powerlifting upper lower garage gym",
        [
            "Bench Press (Barbell)",
            "Squat (Barbell)",
            "Deadlift (Barbell)",
            "Lat Pulldown",
            "Lateral Raise (Dumbbell)",
            "Pigeon Stretch with Thoracic Rotation",
        ],
    )

    assert profile["primary_type"] == "Strength"
    assert profile["has_strength"] == 1
    assert profile["has_yoga"] == 0


def test_hiit_program_is_detected_without_strength_takeover():
    profile = infer_program_type_profile(
        "Body Fat Loss - HIIT",
        [
            "Burpee",
            "Mountain Climber",
            "Box Jump",
            "Battle Rope",
            "Jump Rope",
            "Sprint",
        ],
    )

    assert profile["primary_type"] == "HIIT"
    assert profile["has_hiit"] == 1

