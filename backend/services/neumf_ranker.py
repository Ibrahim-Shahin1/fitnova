"""
FitNova - Layer 2: NeuMF Re-Ranker
====================================
Loads the trained NeuMF model and scores candidate programs for a given user_id.
Handles cold-start by mapping new user profiles to the most similar trained user.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
_MODELS_DIR = os.path.join(_BASE_DIR, 'models')
_DATA_DIR = os.path.join(_BASE_DIR, 'data')

INTENT_ALIGNMENT_SCORE = {
    'primary': 1.0,
    'secondary': 0.45,
    'related': 0.18,
    'none': 0.0,
}
INTENT_ALIGNMENT_BUCKET = {
    'primary': 3,
    'secondary': 2,
    'related': 1,
    'none': 0,
}


class NeuMFRanker:
    def __init__(self):
        # Lazy import to avoid TF startup cost if not needed
        import tensorflow as tf
        self.model = tf.keras.models.load_model(
            os.path.join(_MODELS_DIR, 'neumf_final.keras')
        )

        with open(os.path.join(_MODELS_DIR, 'neumf_metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_users = self.metadata['num_users']
        self.num_items = self.metadata['num_items']

        # Load user features for cold-start nearest-neighbor mapping
        self.user_features_df = pd.read_csv(
            os.path.join(_DATA_DIR, 'user_features.csv')
        )
        # Build feature matrix for similarity (exclude user_id column)
        feature_cols = ['experience_level', 'workout_type_encoded',
                        'session_duration_hours', 'workout_frequency',
                        'bmi', 'age', 'gender_encoded']
        self.user_feature_matrix = self.user_features_df[feature_cols].values.astype(np.float64)
        # Normalize experience_level (1-3) and workout_type_encoded (0-3) to [0,1]
        # so they don't dominate cosine similarity over the already-normalized features
        self.user_feature_matrix[:, 0] = (self.user_feature_matrix[:, 0] - 1) / 2.0
        self.user_feature_matrix[:, 1] = self.user_feature_matrix[:, 1] / 3.0

        with open(os.path.join(_DATA_DIR, 'program_catalog.pkl'), 'rb') as f:
            self.catalog = pickle.load(f)

        with open(os.path.join(_DATA_DIR, 'norm_stats.pkl'), 'rb') as f:
            self.norm_stats = pickle.load(f)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score - min_score < 1e-9:
            return np.ones_like(scores, dtype=np.float32)
        return ((scores - min_score) / (max_score - min_score)).astype(np.float32)

    def _intent_alignment(self, program_id: int, workout_type: str) -> tuple[float, int]:
        program = self.catalog[program_id]
        primary_type = program.get('primary_type')
        secondary_types = set(program.get('secondary_types', []))

        if workout_type == primary_type:
            return INTENT_ALIGNMENT_SCORE['primary'], INTENT_ALIGNMENT_BUCKET['primary']
        if workout_type in secondary_types:
            return INTENT_ALIGNMENT_SCORE['secondary'], INTENT_ALIGNMENT_BUCKET['secondary']

        related_types = set()
        if workout_type == 'Cardio':
            related_types = {'HIIT'}
        elif workout_type == 'HIIT':
            related_types = {'Cardio', 'Strength'}
        elif workout_type == 'Strength':
            related_types = {'HIIT'}

        if primary_type in related_types or related_types & secondary_types:
            return INTENT_ALIGNMENT_SCORE['related'], INTENT_ALIGNMENT_BUCKET['related']
        return INTENT_ALIGNMENT_SCORE['none'], INTENT_ALIGNMENT_BUCKET['none']

    def find_similar_user(self, user_profile: dict) -> int:
        """
        Map a new user profile to the most similar trained user via cosine similarity.

        Args:
            user_profile: dict with keys matching the API input format.

        Returns:
            user_id (int) of the most similar trained user (0-972).
        """
        norm = self.norm_stats
        user_norm = norm.get('user', {})

        # Normalize experience_level to [0,1] (matching the normalized matrix)
        exp_norm = (user_profile['experience_level'] - 1) / 2.0

        # Encode and normalize workout_type to [0,1]
        wt_map = {'Cardio': 0, 'Strength': 1, 'Yoga': 2, 'HIIT': 3}
        wt_norm = wt_map.get(user_profile['workout_type'], 1) / 3.0

        # Normalize session_duration_hours
        dur = user_profile['session_duration_hours']
        dur_stats = user_norm.get('session_duration_hours', {'min': 0.5, 'max': 2.0})
        dur_norm = (dur - dur_stats['min']) / (dur_stats['max'] - dur_stats['min'] + 1e-9)
        dur_norm = max(0.0, min(1.0, dur_norm))

        # Normalize workout_frequency (days/week)
        freq = user_profile['workout_frequency']
        freq_stats = user_norm.get('workout_frequency', {'min': 2, 'max': 5})
        freq_norm = (freq - freq_stats['min']) / (freq_stats['max'] - freq_stats['min'] + 1e-9)
        freq_norm = max(0.0, min(1.0, freq_norm))

        # Normalize BMI
        bmi = user_profile.get('bmi', 25.0)
        bmi_stats = user_norm.get('bmi', {'min': 15, 'max': 45})
        bmi_norm = (bmi - bmi_stats['min']) / (bmi_stats['max'] - bmi_stats['min'] + 1e-9)
        bmi_norm = max(0.0, min(1.0, bmi_norm))

        # Normalize age
        age = user_profile.get('age', 30)
        age_stats = user_norm.get('age', {'min': 18, 'max': 60})
        age_norm = (age - age_stats['min']) / (age_stats['max'] - age_stats['min'] + 1e-9)
        age_norm = max(0.0, min(1.0, age_norm))

        # Gender
        gender = 0 if user_profile.get('gender', 'Male') == 'Male' else 1

        # Build feature vector (same order as user_feature_matrix columns)
        new_user_vec = np.array([[exp_norm, wt_norm, dur_norm, freq_norm, bmi_norm, age_norm, gender]])

        similarities = cosine_similarity(new_user_vec, self.user_feature_matrix)[0]
        return int(np.argmax(similarities))

    def score_candidates(
        self,
        user_id: int,
        candidate_ids: list[int],
        user_profile: dict | None = None,
        content_scores: dict[int, float] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Score candidate programs for a given user_id using NeuMF.

        Args:
            user_id: trained user_id (0-972)
            candidate_ids: list of program_ids to score
            user_profile: optional request profile for intent-aware blending
            content_scores: optional content-filter score map for these candidates

        Returns:
            List of (program_id, score) sorted by score descending.
        """
        n = len(candidate_ids)
        user_arr = np.full(n, user_id, dtype=np.int32)
        item_arr = np.array(candidate_ids, dtype=np.int32)

        scores = self.model.predict(
            [user_arr, item_arr], batch_size=n, verbose=0
        ).flatten()

        final_scores = scores.astype(np.float32)
        if user_profile:
            neumf_norm = self._normalize_scores(scores)
            if content_scores:
                content_arr = np.array(
                    [content_scores.get(candidate_id, 0.0) for candidate_id in candidate_ids],
                    dtype=np.float32,
                )
                content_norm = self._normalize_scores(content_arr)
            else:
                content_norm = np.zeros(n, dtype=np.float32)

            intent_profiles = [
                self._intent_alignment(candidate_id, user_profile['workout_type'])
                for candidate_id in candidate_ids
            ]
            intent_arr = np.array([score for score, _ in intent_profiles], dtype=np.float32)
            intent_buckets = np.array([bucket for _, bucket in intent_profiles], dtype=np.int32)
            final_scores = (
                0.45 * neumf_norm
                + 0.25 * content_norm
                + 0.30 * intent_arr
            )
        else:
            intent_buckets = np.zeros(n, dtype=np.int32)

        ranked = sorted(
            zip(candidate_ids, final_scores, intent_buckets),
            key=lambda x: (-x[2], -x[1], x[0]),
        )
        return [(int(pid), float(s)) for pid, s, _ in ranked]
