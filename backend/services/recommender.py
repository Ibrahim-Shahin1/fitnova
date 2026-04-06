"""
FitNova - Recommendation Pipeline Orchestrator
================================================
Chains Layer 1 (content filter) and Layer 2 (NeuMF re-ranker)
to produce the final recommended program for a user profile.

Pipeline:
  user_profile -> ContentBasedFilter (top 50) -> NeuMF re-rank -> top-1 program
"""

from backend.services.content_filter import ContentBasedFilter
from backend.services.neumf_ranker import NeuMFRanker


class Recommender:
    def __init__(self):
        self.content_filter = ContentBasedFilter()
        self.neumf_ranker = NeuMFRanker()

    def recommend(self, user_profile: dict, top_n_content: int = 50) -> dict:
        """
        Full recommendation pipeline.

        Args:
            user_profile: dict with keys:
                - experience_level (int 1-3)
                - workout_type (str)
                - session_duration_hours (float)
                - workout_frequency (int)
                - age (int, optional)
                - gender (str, optional)
                - bmi (float, optional)
                - goal (str, optional)
                - equipment (str, optional)

        Returns:
            dict with:
                - program_id: int (top recommended program)
                - content_candidates: list[int] (top-50 from Layer 1)
                - neumf_ranked: list[tuple[int, float]] (re-ranked by Layer 2)
                - similar_user_id: int (cold-start mapped user)
        """
        # Layer 1: Content-based filtering -> top N candidates with scores
        scored_candidates = self.content_filter.score_programs(user_profile)[:top_n_content]
        content_candidates = [program_id for program_id, _ in scored_candidates]
        content_score_map = {program_id: score for program_id, score in scored_candidates}

        # Cold-start: map new user to most similar trained user
        similar_user_id = self.neumf_ranker.find_similar_user(user_profile)

        # Layer 2: NeuMF re-ranking of candidates
        neumf_ranked = self.neumf_ranker.score_candidates(
            similar_user_id,
            content_candidates,
            user_profile=user_profile,
            content_scores=content_score_map,
        )

        return {
            'program_id': neumf_ranked[0][0],
            'content_candidates': content_candidates,
            'neumf_ranked': neumf_ranked,
            'similar_user_id': similar_user_id,
        }
