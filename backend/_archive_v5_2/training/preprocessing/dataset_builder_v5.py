"""
v5 dataset builder — MediaPipe-native, AIFit-supervised.

Differences from v4 (``dataset_builder.py``):
  • Model input is **MediaPipe joints** extracted from Fit3D videos (not raw
    MoCap). This eliminates the train/inference distribution mismatch that
    was the documented cause of the §5 phone-deployment failure.
  • Quality + per-joint-group labels come from **AIFit's analytical
    signature** (see ``aifit_features.py``) — instructor s03 motion-energy
    clustering + per-rep deviation. No more synthetic corruption.
  • Multi-view: every (subject, exercise, rep) shows up 4 times — once per
    train camera. Test subjects (s02, s12, s13) appear once each (single
    cam in the dataset).

Output schema
-------------
Saved to ``backend/data/v5_dataset/{split}.npz``:

    pose         : (N, 64, 15, 4) float32  — MediaPipe canonical joints + visibility
    angles       : (N, 64, 22)    float32  — AIFit angular features
    exercise_idx : (N,)           int32    — 0..14 for v5 supported exercises
    quality      : (N,)           float32  — in (0, 1]
    joint_err    : (N, 64, 5)     float32  — per-frame, per-group binary in {0,1}
    boundary     : (N, 64)        float32  — per-frame Gaussian-smoothed boundary
    rep_count    : (N,)           float32  — always 1.0 for single-rep samples
    subject_idx  : (N,)           int32
    camera_id    : (N,)           int32    — -1 for test (single-camera)
    rep_idx      : (N,)           int32    — index of this rep within (subj, ex)

Plus:
    backend/data/v5_dataset/signatures.npz  — per-exercise signature bundles
    backend/data/v5_dataset/exercise_labels.json  — idx → name mapping
    backend/data/v5_dataset/dataset_info.json     — split sizes, params

Train-time multi-view sampling
------------------------------
For the multi-view consistency loss in §5 of the v5 plan, two camera views
of the same rep must end up in the same batch. This is the dataloader's
job — the dataset just exposes ``(subject_idx, exercise_idx, rep_idx)`` as
a key. The training script samples positive pairs on the fly.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aifit_features import (
    GROUP_TO_IDX_V5,
    JOINT_GROUPS_V5,
    N_JOINT_GROUPS_V5,
    build_exercise_signature_bundle,
    per_frame_joint_errors,
    quality_scalar,
)
from .angular_features import N_ANGULAR, compute_sequence_angles
from .fit3d_loader import load_joints, load_rep_boundaries, segment_reps
from .joint_mapping import N_CANONICAL
from .normalize import (
    TARGET_FRAMES,
    extract_canonical_from_fit3d,
    extract_canonical_from_mediapipe,
    normalize_skeleton,
    resample_sequence,
)

# ── v5 supervised exercise list (15 weight movements) ────────────────────────
SUPPORTED_EXERCISES_V5: List[str] = [
    "squat",
    "deadlift",
    "pushup",
    "diamond_pushup",
    "barbell_row",
    "barbell_dead_row",
    "dumbbell_overhead_shoulder_press",
    "dumbbell_biceps_curls",
    "dumbbell_hammer_curls",
    "side_lateral_raise",
    "dumbbell_reverse_lunge",
    "burpees",
    "clean_and_press",
    "mule_kick",
    "standing_ab_twists",
]
EXERCISE_TO_IDX_V5 = {ex: i for i, ex in enumerate(SUPPORTED_EXERCISES_V5)}
N_EXERCISES_V5 = len(SUPPORTED_EXERCISES_V5)

# Subject splits (see plan §3)
INSTRUCTOR_SUBJECT_V5 = "s03"
TRAIN_SUBJECTS_V5 = ["s03", "s04", "s05", "s07", "s08", "s10"]
VAL_SUBJECTS_V5 = ["s09"]
TEST_SUBJECTS_V5 = ["s11"]               # in-domain (still trainees)
OOD_SUBJECTS_V5 = ["s02", "s12", "s13"]  # held-out test split (single camera)

TRAIN_CAMERAS_V5 = ["50591643", "58860488", "60457274", "65906101"]

DEFAULT_FIT3D_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Drafts2\Form Correction"
DEFAULT_MEDIAPIPE_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d_v5"
    )
)
DEFAULT_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "v5_dataset")
)

# Boundary Gaussian σ (frames)
BOUNDARY_SIGMA = 2.0


# ── Loaders ───────────────────────────────────────────────────────────────────


def _load_mediapipe_npy(
    mediapipe_root: str,
    split: str,
    subject: str,
    camera: str,
    exercise: str,
) -> Optional[np.ndarray]:
    """Load (T, 33, 4) MediaPipe extraction for one (subject, camera, exercise).

    Returns ``None`` if the file is missing.
    """
    path = os.path.join(mediapipe_root, split, subject, camera, f"{exercise}.npy")
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _mocap_rep_to_angles(rep_joints25: np.ndarray) -> np.ndarray:
    """Convert one (T, 25, 3) MoCap rep → (T, 22) angular features.

    No resampling: returns the native-length angle sequence (variable T).
    Used for instructor-signature building, where keeping native rep length
    preserves motion-energy magnitudes.
    """
    canon = extract_canonical_from_fit3d(rep_joints25)  # (T, 15, 3)
    canon = normalize_skeleton(canon)                    # (T, 15, 3)
    return compute_sequence_angles(canon)                # (T, 22)


def _mocap_rep_to_label_angles(
    fit3d_root: str,
    split_subdir: str,
    subject: str,
    exercise: str,
    start: int,
    end: int,
    target_frames: int = TARGET_FRAMES,
) -> Optional[np.ndarray]:
    """Load + slice + resample MoCap angles for one rep — for LABEL use only.

    Mirrors ``_mediapipe_rep_to_inputs`` but operates on Fit3D's clean
    ``joints3d_25.json`` MoCap stream rather than the noisy MediaPipe
    extraction. Used to derive AIFit-signature labels (quality, joint_err)
    that are computed in MoCap space, where the instructor signature
    actually lives.

    Returns:
        angles64 : (target_frames, 22) float32 — angular features in
            **MoCap** space, ready for ``quality_scalar`` and
            ``per_frame_joint_errors`` against a MoCap-built signature.
        Or ``None`` if the MoCap file is missing / rep is too short.
    """
    path = os.path.join(
        fit3d_root, split_subdir, subject, "joints3d_25", f"{exercise}.json"
    )
    if not os.path.isfile(path):
        return None
    try:
        data = json.load(open(path))
        joints25 = np.array(data["joints3d_25"], dtype=np.float32)  # (T, 25, 3)
    except Exception:
        return None
    rep = joints25[start:end]
    if len(rep) < 4:
        return None
    canon = extract_canonical_from_fit3d(rep)               # (rep_T, 15, 3)
    canon = normalize_skeleton(canon)                        # (rep_T, 15, 3)
    canon64 = resample_sequence(canon, target_frames)        # (64, 15, 3)
    return compute_sequence_angles(canon64).astype(np.float32)  # (64, 22)


def _mediapipe_rep_to_inputs(
    mp_clip: np.ndarray,
    start: int,
    end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice + normalise + resample one rep's MediaPipe joints.

    Args:
        mp_clip: (T_full, 33, 4) — full clip's MediaPipe extraction
        start:   rep start frame index (inclusive)
        end:     rep end frame index (exclusive)

    Returns:
        pose64   : (64, 15, 4) — canonical joints + visibility, hip-centred,
                   torso-scaled, resampled to 64 frames
        angles64 : (64, 22)    — angular features computed on canonical pose
    """
    rep = mp_clip[start:end]  # (rep_T, 33, 4)
    if len(rep) < 4:
        raise ValueError(f"rep too short: {len(rep)} frames")

    xyz = rep[..., :3]                      # (rep_T, 33, 3)
    vis = rep[..., 3:4]                     # (rep_T, 33, 1)

    # Canonical 15-joint mapping (xyz only — visibility is per-MediaPipe-joint)
    canon_xyz = extract_canonical_from_mediapipe(xyz)   # (rep_T, 15, 3)
    canon_xyz = normalize_skeleton(canon_xyz)            # (rep_T, 15, 3)

    # Visibility for the 15 canonical joints: average visibilities of the
    # MediaPipe joints that map to each canonical index. We approximate this
    # via mean(vis) for midpoints; for direct mappings we use the source
    # landmark's visibility. This keeps the model's C=4 channel meaningful.
    from .joint_mapping import MEDIAPIPE_JOINTS, MEDIAPIPE_TO_CANONICAL

    canon_vis = np.zeros((rep.shape[0], N_CANONICAL, 1), dtype=np.float32)
    for can_idx, mp_idx in MEDIAPIPE_TO_CANONICAL.items():
        if can_idx == 14:  # spine_mid: average of pelvis + neck after the loop
            continue
        if mp_idx is None:
            continue
        if mp_idx == "neck_weighted":
            l = MEDIAPIPE_JOINTS["l_shoulder"]
            r = MEDIAPIPE_JOINTS["r_shoulder"]
            le = MEDIAPIPE_JOINTS["l_ear"]
            re = MEDIAPIPE_JOINTS["r_ear"]
            canon_vis[:, can_idx, :] = 0.25 * (
                vis[:, l] + vis[:, r] + vis[:, le] + vis[:, re]
            )
        elif isinstance(mp_idx, tuple):
            canon_vis[:, can_idx, :] = np.mean(
                [vis[:, i] for i in mp_idx], axis=0
            )
        else:
            canon_vis[:, can_idx, :] = vis[:, mp_idx]
    # spine_mid visibility = mean(pelvis, neck)
    canon_vis[:, 14, :] = 0.5 * (canon_vis[:, 12, :] + canon_vis[:, 13, :])

    # Stack pose (xyz + vis) → (rep_T, 15, 4)
    pose = np.concatenate([canon_xyz, canon_vis], axis=-1)

    # Resample to 64 frames
    pose64 = resample_sequence(pose, TARGET_FRAMES)              # (64, 15, 4)
    canon_xyz_64 = resample_sequence(canon_xyz, TARGET_FRAMES)   # (64, 15, 3)
    angles64 = compute_sequence_angles(canon_xyz_64)             # (64, 22)
    return pose64.astype(np.float32), angles64.astype(np.float32)


# ── Boundary label ────────────────────────────────────────────────────────────


def _gaussian_boundary_label(boundary_positions: List[float]) -> np.ndarray:
    """Per-frame boundary label = sum of Gaussians at given positions.

    For a single-rep sample, ``boundary_positions = [0.0, 63.0]`` —
    rep transitions at the start and end of the window.

    Args:
        boundary_positions: list of frame indices (can be float for sub-frame
            boundaries, but here always ints).

    Returns:
        (64,) float32, max value capped at 1.0.
    """
    out = np.zeros(TARGET_FRAMES, dtype=np.float32)
    if not boundary_positions:
        return out
    t = np.arange(TARGET_FRAMES, dtype=np.float32)
    for pos in boundary_positions:
        out += np.exp(-0.5 * ((t - pos) / BOUNDARY_SIGMA) ** 2)
    return np.clip(out, 0.0, 1.0)


# ── Phase 1: Build per-exercise signature bundles ─────────────────────────────


def build_signatures(
    fit3d_root: str = DEFAULT_FIT3D_ROOT,
    instructor_subject: str = INSTRUCTOR_SUBJECT_V5,
    activeness_threshold: float = 0.5,
    target_instructor_quality: float = 0.95,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build the AIFit signature for every v5 exercise.

    Reads MoCap joints (`joints3d_25/<exercise>.json`) from train/<instructor>
    in the Fit3D folder. Returns ``{exercise: bundle}`` where each bundle
    has the keys produced by ``build_exercise_signature_bundle``.
    """
    train_root = os.path.join(fit3d_root, "train")
    inst_dir = os.path.join(train_root, instructor_subject)
    rep_boundaries = load_rep_boundaries(inst_dir)

    bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for ex in SUPPORTED_EXERCISES_V5:
        if ex not in rep_boundaries:
            if verbose:
                print(f"  [signatures] skip {ex}: no rep_ann entry for {instructor_subject}")
            continue
        joints = load_joints(inst_dir, ex)
        if joints is None:
            if verbose:
                print(f"  [signatures] skip {ex}: no joints3d_25")
            continue
        reps_joints = segment_reps(joints, rep_boundaries[ex])
        # Convert each rep to angle sequence for signature building
        reps_angles: List[np.ndarray] = []
        for rep_j25 in reps_joints:
            try:
                reps_angles.append(_mocap_rep_to_angles(rep_j25))
            except Exception:
                continue
        if not reps_angles:
            if verbose:
                print(f"  [signatures] skip {ex}: no usable reps")
            continue

        bundle = build_exercise_signature_bundle(
            angles_per_subject={instructor_subject: reps_angles},
            instructor_subject=instructor_subject,
            activeness_threshold=activeness_threshold,
            target_instructor_quality=target_instructor_quality,
        )
        bundles[ex] = bundle
        if verbose:
            n_active = int(bundle["active_mask"].sum())
            print(
                f"  [signatures] {ex:<35s} "
                f"reps={len(reps_angles):3d}  "
                f"active={n_active:2d}/22  "
                f"tau={float(bundle['tau']):.4f}  "
                f"p75={bundle['p75_per_group']}"
            )
    return bundles


# ── Phase 2: Build labelled samples ───────────────────────────────────────────


def _build_split_samples(
    split_name: str,
    subjects: List[str],
    cameras: List[str],
    fit3d_root: str,
    mediapipe_root: str,
    signatures: Dict[str, Dict[str, np.ndarray]],
    fit3d_split_subdir: str = "train",  # most splits live under fit3d/train/
    test_camera_label: str = "single",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Build a single split's npz contents.

    For each (subject, camera, exercise, rep):
      • Load MediaPipe joints (.npy from Phase 1)
      • Use rep boundaries from rep_ann.json to slice the rep
      • Resample to T=64; compute angular features
      • Use exercise signature to compute quality + joint-error labels
    """
    pose_arrs: List[np.ndarray] = []
    angles_arrs: List[np.ndarray] = []
    exercise_idxs: List[int] = []
    qualities: List[float] = []
    joint_errs: List[np.ndarray] = []
    boundaries: List[np.ndarray] = []
    rep_counts: List[float] = []
    subject_idxs: List[int] = []
    camera_ids: List[int] = []
    rep_idxs: List[int] = []

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    camera_to_idx = (
        {c: i for i, c in enumerate(cameras)} if cameras else {test_camera_label: -1}
    )

    n_total = 0
    n_skipped_missing = 0
    n_skipped_short = 0
    n_skipped_bad = 0

    for subj in subjects:
        subj_dir = os.path.join(fit3d_root, fit3d_split_subdir, subj)
        if not os.path.isdir(subj_dir):
            if verbose:
                print(f"  [{split_name}] skip subject {subj}: folder missing")
            continue
        try:
            rep_boundaries = load_rep_boundaries(subj_dir)
        except FileNotFoundError:
            if verbose:
                print(f"  [{split_name}] skip subject {subj}: rep_ann.json missing")
            continue

        cams_this_subj = cameras if cameras else [test_camera_label]
        for cam in cams_this_subj:
            for ex in SUPPORTED_EXERCISES_V5:
                if ex not in rep_boundaries or ex not in signatures:
                    continue
                bounds = rep_boundaries[ex]
                if len(bounds) < 2:
                    continue
                mp_clip = _load_mediapipe_npy(
                    mediapipe_root, fit3d_split_subdir, subj, cam, ex
                )
                if mp_clip is None:
                    n_skipped_missing += 1
                    continue
                signature = signatures[ex]
                tau = float(signature["tau"])
                p75 = signature["p75_per_group"]

                for rep_idx in range(len(bounds) - 1):
                    start, end = int(bounds[rep_idx]), int(bounds[rep_idx + 1])
                    if end - start < 10:
                        n_skipped_short += 1
                        continue
                    if end > len(mp_clip):
                        n_skipped_short += 1
                        continue
                    try:
                        pose64, angles64_mp = _mediapipe_rep_to_inputs(
                            mp_clip, start, end
                        )
                    except Exception:
                        n_skipped_bad += 1
                        continue
                    if not np.all(np.isfinite(pose64)):
                        n_skipped_bad += 1
                        continue

                    # MoCap-derived angles for label generation (teacher signal).
                    # The signature lives in MoCap space (built from s03 MoCap
                    # joints), so quality + joint-error must also be computed
                    # in MoCap space. The MediaPipe-derived angles_mp go in as
                    # model INPUT — student learns to predict the MoCap label
                    # from the MediaPipe input. This is the teacher-student
                    # setup the v5 plan §4 actually describes.
                    angles64_label = _mocap_rep_to_label_angles(
                        fit3d_root, fit3d_split_subdir,
                        subj, ex, start, end,
                    )
                    if angles64_label is None:
                        n_skipped_bad += 1
                        continue

                    # Quality + joint errors from the AIFit signature
                    q = quality_scalar(angles64_label, signature, tau)
                    je = per_frame_joint_errors(
                        angles64_label, signature, p75
                    ).astype(np.float32)  # (64, 5)
                    bd = _gaussian_boundary_label([0.0, float(TARGET_FRAMES - 1)])

                    pose_arrs.append(pose64)
                    angles_arrs.append(angles64_mp)
                    exercise_idxs.append(EXERCISE_TO_IDX_V5[ex])
                    qualities.append(q)
                    joint_errs.append(je)
                    boundaries.append(bd)
                    rep_counts.append(1.0)
                    subject_idxs.append(subject_to_idx[subj])
                    camera_ids.append(camera_to_idx[cam] if cam in camera_to_idx else -1)
                    rep_idxs.append(rep_idx)
                    n_total += 1

    if verbose:
        print(
            f"  [{split_name}] built {n_total} samples "
            f"(missing_npy={n_skipped_missing}, short={n_skipped_short}, "
            f"bad={n_skipped_bad})"
        )

    if n_total == 0:
        return {}

    return {
        "pose":         np.stack(pose_arrs).astype(np.float32),
        "angles":       np.stack(angles_arrs).astype(np.float32),
        "exercise_idx": np.array(exercise_idxs, dtype=np.int32),
        "quality":      np.array(qualities, dtype=np.float32),
        "joint_err":    np.stack(joint_errs).astype(np.float32),
        "boundary":     np.stack(boundaries).astype(np.float32),
        "rep_count":    np.array(rep_counts, dtype=np.float32),
        "subject_idx":  np.array(subject_idxs, dtype=np.int32),
        "camera_id":    np.array(camera_ids, dtype=np.int32),
        "rep_idx":      np.array(rep_idxs, dtype=np.int32),
    }


# ── Top-level driver ──────────────────────────────────────────────────────────


def build_v5_dataset(
    fit3d_root: str = DEFAULT_FIT3D_ROOT,
    mediapipe_root: str = DEFAULT_MEDIAPIPE_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Build the full v5 dataset (train, val, test, ood) + signatures.

    Writes everything to ``out_root``. Idempotent — overwrites existing files.
    """
    os.makedirs(out_root, exist_ok=True)

    if verbose:
        print(f"[v5-dataset] fit3d_root      : {fit3d_root}")
        print(f"[v5-dataset] mediapipe_root  : {mediapipe_root}")
        print(f"[v5-dataset] out_root        : {out_root}")
        print(f"[v5-dataset] # exercises     : {N_EXERCISES_V5}")
        print(f"[v5-dataset] # train subj    : {len(TRAIN_SUBJECTS_V5)}")
        print(f"[v5-dataset] # train cams    : {len(TRAIN_CAMERAS_V5)}")
        print(f"[v5-dataset] # val subj      : {len(VAL_SUBJECTS_V5)}")
        print(f"[v5-dataset] # test subj     : {len(TEST_SUBJECTS_V5)}")
        print(f"[v5-dataset] # ood subj      : {len(OOD_SUBJECTS_V5)}")
        print()

    # Phase 1: build signatures (s03 instructor)
    if verbose:
        print("[v5-dataset] Building per-exercise AIFit signatures...")
    signatures = build_signatures(
        fit3d_root=fit3d_root,
        instructor_subject=INSTRUCTOR_SUBJECT_V5,
        verbose=verbose,
    )
    if not signatures:
        raise RuntimeError("No signatures built — check instructor data.")

    # Persist signatures for inference-time use
    sig_path = os.path.join(out_root, "signatures.npz")
    sig_to_save: Dict[str, np.ndarray] = {}
    sig_meta: Dict[str, Dict] = {}
    for ex, b in signatures.items():
        for k, v in b.items():
            sig_to_save[f"{ex}__{k}"] = np.asarray(v)
        sig_meta[ex] = {
            "tau": float(b["tau"]),
            "n_active": int(b["active_mask"].sum()),
            "n_instructor_reps": int(b["n_instructor_reps"]),
            "p75_per_group": [float(x) for x in b["p75_per_group"]],
        }
    np.savez(sig_path, **sig_to_save)
    with open(os.path.join(out_root, "signatures_meta.json"), "w") as f:
        json.dump(sig_meta, f, indent=2)
    if verbose:
        print(f"[v5-dataset] saved signatures -> {sig_path}")
        print()

    # Phase 2: per-split samples
    splits_def = [
        ("train", TRAIN_SUBJECTS_V5, TRAIN_CAMERAS_V5, "train"),
        ("val",   VAL_SUBJECTS_V5,   TRAIN_CAMERAS_V5, "train"),
        ("test",  TEST_SUBJECTS_V5,  TRAIN_CAMERAS_V5, "train"),
        ("ood",   OOD_SUBJECTS_V5,   [],               "test"),
    ]

    splits: Dict[str, dict] = {}
    for split_name, subjects, cameras, mp_split in splits_def:
        if verbose:
            print(f"[v5-dataset] Building {split_name} (subjects={subjects}, cams={cameras or 'single'})...")
        contents = _build_split_samples(
            split_name=split_name,
            subjects=subjects,
            cameras=cameras,
            fit3d_root=fit3d_root,
            mediapipe_root=mediapipe_root,
            signatures=signatures,
            fit3d_split_subdir=mp_split,
            verbose=verbose,
        )
        if not contents:
            if verbose:
                print(f"  [{split_name}] empty — nothing to save")
            splits[split_name] = {"n_samples": 0}
            continue
        out_path = os.path.join(out_root, f"{split_name}.npz")
        np.savez_compressed(out_path, **contents)
        if verbose:
            print(f"  [{split_name}] saved -> {out_path}")
        splits[split_name] = {
            "n_samples": int(contents["pose"].shape[0]),
            "path": out_path,
            "quality_mean": float(contents["quality"].mean()),
            "quality_std": float(contents["quality"].std()),
            "joint_err_rate": float(contents["joint_err"].mean()),
        }
        if verbose:
            print(
                f"  [{split_name}] quality μ={splits[split_name]['quality_mean']:.3f} "
                f"σ={splits[split_name]['quality_std']:.3f}  "
                f"joint-err rate={splits[split_name]['joint_err_rate']:.3f}"
            )
        print()

    # Save metadata
    info = {
        "n_exercises": N_EXERCISES_V5,
        "exercises": SUPPORTED_EXERCISES_V5,
        "n_joint_groups": N_JOINT_GROUPS_V5,
        "joint_groups": JOINT_GROUPS_V5,
        "target_frames": TARGET_FRAMES,
        "n_canonical_joints": N_CANONICAL,
        "n_angular": N_ANGULAR,
        "instructor_subject": INSTRUCTOR_SUBJECT_V5,
        "train_subjects": TRAIN_SUBJECTS_V5,
        "val_subjects": VAL_SUBJECTS_V5,
        "test_subjects": TEST_SUBJECTS_V5,
        "ood_subjects": OOD_SUBJECTS_V5,
        "train_cameras": TRAIN_CAMERAS_V5,
        "splits": {k: v for k, v in splits.items()},
    }
    with open(os.path.join(out_root, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_root, "exercise_labels.json"), "w") as f:
        json.dump(EXERCISE_TO_IDX_V5, f, indent=2)

    if verbose:
        print("[v5-dataset] DONE")
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--fit3d-root", default=DEFAULT_FIT3D_ROOT)
    p.add_argument("--mediapipe-root", default=DEFAULT_MEDIAPIPE_ROOT)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--signatures-only",
        action="store_true",
        help="Only build per-exercise signatures, skip dataset assembly",
    )
    args = p.parse_args()

    if args.signatures_only:
        bundles = build_signatures(fit3d_root=args.fit3d_root)
        os.makedirs(args.out_root, exist_ok=True)
        out = os.path.join(args.out_root, "signatures.npz")
        flat: Dict[str, np.ndarray] = {}
        for ex, b in bundles.items():
            for k, v in b.items():
                flat[f"{ex}__{k}"] = np.asarray(v)
        np.savez(out, **flat)
        print(f"saved {out}")
        return

    build_v5_dataset(
        fit3d_root=args.fit3d_root,
        mediapipe_root=args.mediapipe_root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    _cli()
