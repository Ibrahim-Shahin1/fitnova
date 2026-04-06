"""
FitNova - Layer 1: Content-Based Filtering
============================================
Scores all 2,598 programs against a user profile using the 4-rule
weighted compatibility model from Gupta et al. (2024).

Returns top-N candidate program_ids for downstream NeuMF re-ranking.
"""

import os
import pickle

import numpy as np

# Paths (relative to this file -> backend/data/)
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_CATALOG_PATH = os.path.join(_DATA_DIR, 'program_catalog.pkl')
_NORM_PATH = os.path.join(_DATA_DIR, 'norm_stats.pkl')

# Gupta et al. (2024) feature weights
W_EXPERIENCE = 0.35
W_WORKOUT_TYPE = 0.30
W_DURATION = 0.20
W_FREQUENCY = 0.15

# Rule 1: Experience level match lookup
# user_experience_level (1-3) -> program_level_encoded (0-2) -> score
EXP_SCORE_TABLE = {
    1: {0: 1.0, 1: 0.4, 2: 0.0},  # Beginner
    2: {0: 0.5, 1: 1.0, 2: 0.5},  # Intermediate
    3: {0: 0.1, 1: 0.6, 2: 1.0},  # Expert
}

# Rule 2: Related workout types
RELATED_TYPES = {
    'Cardio': {'HIIT'},
    'HIIT': {'Cardio', 'Strength'},
    'Strength': {'HIIT'},
    'Yoga': set(),
}


class ContentBasedFilter:
    def __init__(self, catalog_path=_CATALOG_PATH, norm_stats_path=_NORM_PATH):
        with open(catalog_path, 'rb') as f:
            self.catalog = pickle.load(f)
        with open(norm_stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)

        self.n_programs = len(self.catalog)
        self._precompute_program_arrays()

    def _precompute_program_arrays(self):
        """Extract program attributes into numpy arrays for fast scoring."""
        n = self.n_programs
        self.prog_level = np.zeros(n, dtype=np.int32)
        self.prog_tpw = np.zeros(n, dtype=np.float32)
        self.prog_freq = np.zeros(n, dtype=np.int32)
        self.prog_type_sets = []

        for pid in range(n):
            p = self.catalog[pid]
            self.prog_level[pid] = p['level_encoded']
            self.prog_tpw[pid] = p['time_per_workout_minutes']

            # Derive weekly frequency from exercises (same as generate_interactions.py)
            exercises = p['exercises']
            if exercises:
                week_days = set()
                weeks = set()
                for ex in exercises:
                    w, d = ex['week'], ex['day']
                    week_days.add((w, d))
                    weeks.add(w)
                n_weeks = len(weeks)
                weekly_freq = round(len(week_days) / n_weeks) if n_weeks > 0 else 3
                weekly_freq = max(1, min(7, weekly_freq))
            else:
                weekly_freq = 3
            self.prog_freq[pid] = weekly_freq

            # Workout type set
            types = set()
            if p['has_cardio']:
                types.add('Cardio')
            if p['has_strength']:
                types.add('Strength')
            if p['has_yoga']:
                types.add('Yoga')
            if p['has_hiit']:
                types.add('HIIT')
            self.prog_type_sets.append(types)

    def score_programs(self, user_profile: dict) -> list[tuple[int, float]]:
        """
        Score all programs against a user profile.

        Args:
            user_profile: dict with keys:
                - experience_level: int (1=Beginner, 2=Intermediate, 3=Expert)
                - workout_type: str (Cardio/Strength/Yoga/HIIT)
                - session_duration_hours: float (hours per session)
                - workout_frequency: int (days per week)

        Returns:
            List of (program_id, score) sorted by score descending.
        """
        u_exp = user_profile['experience_level']
        u_wt = user_profile['workout_type']
        u_dur_min = user_profile['session_duration_hours'] * 60
        u_freq = user_profile['workout_frequency']
        u_related = RELATED_TYPES.get(u_wt, set())

        exp_lookup = EXP_SCORE_TABLE[u_exp]
        scores = np.zeros(self.n_programs, dtype=np.float32)

        for p_idx in range(self.n_programs):
            # Rule 1: Experience level match
            r1 = exp_lookup[self.prog_level[p_idx]]

            # Rule 2: Workout type match
            p_types = self.prog_type_sets[p_idx]
            if u_wt in p_types:
                r2 = 1.0
            elif u_related & p_types:
                r2 = 0.5
            else:
                r2 = 0.0

            # Rule 3: Session duration compatibility
            dur_diff = abs(u_dur_min - self.prog_tpw[p_idx])
            if dur_diff <= 10:
                r3 = 1.0
            elif dur_diff <= 20:
                r3 = 0.6
            elif dur_diff <= 30:
                r3 = 0.3
            else:
                r3 = 0.0

            # Rule 4: Workout frequency compatibility
            freq_diff = abs(u_freq - self.prog_freq[p_idx])
            if freq_diff == 0:
                r4 = 1.0
            elif freq_diff == 1:
                r4 = 0.7
            elif freq_diff == 2:
                r4 = 0.3
            else:
                r4 = 0.0

            scores[p_idx] = (W_EXPERIENCE * r1 + W_WORKOUT_TYPE * r2 +
                             W_DURATION * r3 + W_FREQUENCY * r4)

        ranked = np.argsort(-scores)
        return [(int(pid), float(scores[pid])) for pid in ranked]

    def get_top_n(self, user_profile: dict, n: int = 50) -> list[int]:
        """Return top-N program_ids for a user profile."""
        scored = self.score_programs(user_profile)
        return [pid for pid, _ in scored[:n]]
