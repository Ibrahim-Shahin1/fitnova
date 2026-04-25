"""
Load Fit3D dataset: joints3d_25 JSONs and rep_ann.json files.
Segments full exercise sequences into individual repetitions.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

# Exercises that have rep annotations (walk_the_box excluded)
SUPPORTED_EXERCISES = [
    "squat", "deadlift", "pushup", "diamond_pushup",
    "barbell_row", "barbell_dead_row", "barbell_shrug", "one_arm_row",
    "dumbbell_overhead_shoulder_press", "neutral_overhead_shoulder_press",
    "dumbbell_biceps_curls", "dumbbell_hammer_curls", "dumbbell_curl_trifecta",
    "drag_curl", "dumbbell_reverse_lunge", "dumbbell_high_pulls",
    "dumbbell_scaptions", "side_lateral_raise", "overhead_trap_raises",
    "w_raise", "band_pull_apart", "burpees", "clean_and_press",
    "man_maker", "mule_kick", "overhead_extension_thruster", "standing_ab_twists",
]

EXERCISE_TO_IDX = {ex: i for i, ex in enumerate(SUPPORTED_EXERCISES)}
N_EXERCISES = len(SUPPORTED_EXERCISES)  # 27

ALL_SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
INSTRUCTOR_SUBJECT = "s03"


def load_joints(subject_dir: str, exercise: str) -> Optional[np.ndarray]:
    """Load joints3d_25 for one subject+exercise. Returns (N_frames, 25, 3) float32."""
    path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    arr = np.array(data["joints3d_25"], dtype=np.float32)  # (N, 25, 3)
    return arr


def load_rep_boundaries(subject_dir: str) -> Dict[str, List[int]]:
    """Load rep_ann.json. Returns {exercise: [b0, b1, ..., bn]} boundary frame indices."""
    path = os.path.join(subject_dir, "rep_ann.json")
    with open(path, "r") as f:
        return json.load(f)


def segment_reps(
    joints: np.ndarray,
    boundaries: List[int],
) -> List[np.ndarray]:
    """
    Segment a full exercise sequence into individual reps.

    boundaries = [b0, b1, b2, ..., bn]
    Rep i spans frames [b_i, b_{i+1}).
    Returns list of (n_frames_i, 25, 3) arrays — one per rep.
    """
    reps = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        rep = joints[start:end]  # (rep_len, 25, 3)
        if len(rep) < 10:        # skip degenerate reps
            continue
        reps.append(rep)
    return reps


def load_subject(
    dataset_root: str,
    subject_id: str,
    exercises: Optional[List[str]] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Load all reps for one subject.

    Returns:
        {exercise_name: [rep_0, rep_1, ...]}
        Each rep is (n_frames, 25, 3) float32.
    """
    if exercises is None:
        exercises = SUPPORTED_EXERCISES

    subject_dir = os.path.join(dataset_root, subject_id)
    rep_boundaries = load_rep_boundaries(subject_dir)

    result: Dict[str, List[np.ndarray]] = {}
    for ex in exercises:
        if ex not in rep_boundaries:
            continue
        joints = load_joints(subject_dir, ex)
        if joints is None:
            continue
        boundaries = rep_boundaries[ex]
        reps = segment_reps(joints, boundaries)
        if reps:
            result[ex] = reps

    return result


def load_all_subjects(
    dataset_root: str,
    subjects: Optional[List[str]] = None,
    exercises: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Load all reps for all subjects.

    Returns:
        {subject_id: {exercise_name: [rep_0, rep_1, ...]}}
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    all_data = {}
    for subj in subjects:
        subj_data = load_subject(dataset_root, subj, exercises)
        if subj_data:
            all_data[subj] = subj_data
            n_reps = sum(len(v) for v in subj_data.values())
            print(f"  Loaded {subj}: {len(subj_data)} exercises, {n_reps} reps")

    return all_data


def count_reps(data: Dict[str, Dict[str, List[np.ndarray]]]) -> int:
    """Count total number of reps across all subjects and exercises."""
    return sum(
        len(reps)
        for subj_data in data.values()
        for reps in subj_data.values()
    )


def get_rep_lengths(data: Dict[str, Dict[str, List[np.ndarray]]]) -> List[int]:
    """Return list of all rep lengths (in frames) for analysis."""
    lengths = []
    for subj_data in data.values():
        for reps in subj_data.values():
            lengths.extend(len(r) for r in reps)
    return lengths
