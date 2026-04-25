"""
Load MediaPipe-extracted Fit3D landmarks as training reps.

Produced by `mediapipe_extractor.py` at
    <out_root>/<subject>/<exercise>.npy   shape (T, 33, 4) float32
                                          channels = [x, y, z, visibility]

We reuse Fit3D's `rep_ann.json` frame indices to segment sequences into
individual reps — they describe the same video the MediaPipe extractor ran
over, so the indices transfer 1-to-1.

NaN handling
------------
Frames where MediaPipe didn't detect a pose are saved as NaN (33, 3). If
more than `MAX_NAN_FRAC` of a rep's frames are NaN we drop that rep;
otherwise we linearly interpolate the NaN frames forwards-and-backwards.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

# Reuse the master rep-boundary loader + supported-exercise list from the
# Fit3D loader so there's one source of truth for "what is a rep" and
# "which exercises do we train on".
from .fit3d_loader import (
    SUPPORTED_EXERCISES, load_rep_boundaries, segment_reps,
)

DEFAULT_MP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d")
)
DEFAULT_FIT3D_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"

MAX_NAN_FRAC = 0.25   # reject reps where more than a quarter of frames are NaN


def _interpolate_nans_3d(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN frames in a (T, 33, 3) array.

    NaN frames are identified by any-NaN-in-frame. Edges are filled by
    nearest-frame propagation (ffill / bfill).
    """
    T = arr.shape[0]
    bad = np.isnan(arr).any(axis=(1, 2))    # (T,)
    if not bad.any():
        return arr
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return arr  # all-bad; caller should drop

    out = arr.copy()
    # ffill then bfill to cover edges
    for t in range(T):
        if bad[t]:
            # nearest good frame
            left  = good_idx[good_idx <= t].max(initial=-1)
            right = good_idx[good_idx >= t].min(initial=-1)
            if left == -1:
                out[t] = arr[right]
            elif right == -1:
                out[t] = arr[left]
            else:
                alpha = (t - left) / max(1, (right - left))
                out[t] = (1.0 - alpha) * arr[left] + alpha * arr[right]
    return out


def load_mediapipe_joints(
    mediapipe_root: str, subject_id: str, exercise: str,
) -> Optional[np.ndarray]:
    """Load the saved MediaPipe landmark tensor for (subject, exercise).

    Returns (T, 33, 3) float32 (visibility channel dropped) or None if missing.
    """
    path = os.path.join(mediapipe_root, subject_id, f"{exercise}.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path)     # (T, 33, 4)
    if arr.ndim != 3 or arr.shape[1] != 33:
        return None
    return arr[..., :3].astype(np.float32)


def load_mediapipe_subject(
    mediapipe_root: str,
    fit3d_root: str,
    subject_id: str,
    exercises: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """Load every MediaPipe rep for one subject, segmented by rep_ann.json.

    Returns {exercise: [rep_0, rep_1, ...]} where each rep is (n_frames, 33, 3).
    """
    if exercises is None:
        exercises = SUPPORTED_EXERCISES

    subject_dir = os.path.join(fit3d_root, subject_id)
    if not os.path.isdir(subject_dir):
        if verbose:
            print(f"  [mp-loader] {subject_id}: missing fit3d subject dir {subject_dir}")
        return {}
    try:
        rep_boundaries = load_rep_boundaries(subject_dir)
    except FileNotFoundError:
        if verbose:
            print(f"  [mp-loader] {subject_id}: missing rep_ann.json")
        return {}

    result: Dict[str, List[np.ndarray]] = {}
    for ex in exercises:
        if ex not in rep_boundaries:
            continue
        arr = load_mediapipe_joints(mediapipe_root, subject_id, ex)
        if arr is None:
            continue

        # Interpolate NaN frames before segmenting
        nan_frac = float(np.isnan(arr).any(axis=(1, 2)).mean())
        if nan_frac > MAX_NAN_FRAC:
            if verbose:
                print(f"  [mp-loader] {subject_id}/{ex}: dropping "
                      f"(nan_frac={nan_frac:.2f} > {MAX_NAN_FRAC})")
            continue
        arr = _interpolate_nans_3d(arr)

        # Sanity: Fit3D boundaries index into the video's frames. The MediaPipe
        # extractor iterated every frame of the video, so indices align.
        # But if the .npy has fewer frames than rep_ann assumes, clamp.
        T = arr.shape[0]
        boundaries = [b for b in rep_boundaries[ex] if b < T]
        if len(boundaries) < 2:
            continue

        reps = segment_reps(arr, boundaries)
        if reps:
            result[ex] = reps
    return result


def load_mediapipe_all_subjects(
    mediapipe_root: str,
    fit3d_root: str,
    subjects: Optional[List[str]] = None,
    exercises: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """{subject_id: {exercise: [reps...]}} loaded from MediaPipe npys."""
    from .fit3d_loader import ALL_SUBJECTS
    subjects = subjects or ALL_SUBJECTS
    all_data: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for subj in subjects:
        sd = load_mediapipe_subject(mediapipe_root, fit3d_root, subj,
                                    exercises=exercises, verbose=verbose)
        if sd:
            all_data[subj] = sd
            if verbose:
                n_reps = sum(len(v) for v in sd.values())
                print(f"  [mp-loader] {subj}: {len(sd)} exercises, {n_reps} reps")
    return all_data
