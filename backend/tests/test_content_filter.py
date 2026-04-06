import pickle
import time

from backend.services.content_filter import ContentBasedFilter


def _load_catalog():
    with open("backend/data/program_catalog.pkl", "rb") as handle:
        return pickle.load(handle)


def test_strength_query_returns_strength_dominant_results():
    content_filter = ContentBasedFilter()
    catalog = _load_catalog()

    start = time.time()
    results = content_filter.get_top_n(
        {
            "experience_level": 1,
            "workout_type": "Strength",
            "session_duration_hours": 1.0,
            "workout_frequency": 4,
        }
    )
    elapsed_ms = (time.time() - start) * 1000

    top5 = [catalog[pid] for pid in results[:5]]
    assert elapsed_ms < 100
    assert all(program["has_strength"] == 1 for program in top5)
    assert all(program["level_encoded"] == 0 for program in top5)


def test_yoga_query_prefers_true_yoga_primary_programs():
    content_filter = ContentBasedFilter()
    catalog = _load_catalog()

    results = content_filter.get_top_n(
        {
            "experience_level": 3,
            "workout_type": "Yoga",
            "session_duration_hours": 1.5,
            "workout_frequency": 5,
        }
    )

    top5 = [catalog[pid] for pid in results[:5]]
    yoga_primary_count = sum(program.get("primary_type") == "Yoga" for program in top5)
    yoga_flag_count = sum(program["has_yoga"] == 1 for program in top5)

    assert yoga_primary_count >= 3, top5
    assert yoga_flag_count == 5, top5

