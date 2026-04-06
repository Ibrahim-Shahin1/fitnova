from backend.services.recommender import Recommender


def test_recommender_returns_sorted_rankings_and_valid_ids():
    recommender = Recommender()
    result = recommender.recommend(
        {
            "experience_level": 1,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 4,
            "age": 22,
            "gender": "Male",
            "bmi": 24.0,
        }
    )

    assert 0 <= result["similar_user_id"] < 973
    assert len(result["content_candidates"]) == 50
    assert result["program_id"] in result["content_candidates"]

    scores = [score for _, score in result["neumf_ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_recommender_keeps_yoga_intent_at_top():
    recommender = Recommender()
    result = recommender.recommend(
        {
            "experience_level": 3,
            "workout_type": "Yoga",
            "session_duration_hours": 1.5,
            "workout_frequency": 5,
            "age": 35,
            "gender": "Female",
            "bmi": 22.0,
        }
    )

    top_program_id = result["program_id"]
    program = recommender.content_filter.catalog[top_program_id]
    top5 = [recommender.content_filter.catalog[pid] for pid, _ in result["neumf_ranked"][:5]]

    assert program["has_yoga"] == 1
    assert program.get("primary_type") == "Yoga"
    assert sum(candidate.get("primary_type") == "Yoga" for candidate in top5) >= 3
