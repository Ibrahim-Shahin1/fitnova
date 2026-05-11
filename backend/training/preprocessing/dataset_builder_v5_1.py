"""
v5.1 dataset builder — synthetic-perturbation-supervised, AIFit-signature-free.

Why a v5.1 builder
------------------
The v5.0 ``dataset_builder_v5.py`` derived quality and per-joint-error labels
from the AIFit angular signature math. That math depends on the angular
features being biomechanically meaningful — i.e. on ``l_knee_flex`` actually
measuring the knee angle. Verification on real Fit3D data revealed that the
joint indices in ``joint_mapping.py`` don't match the Fit3D 25-joint
convention (joints 17-20 are nearly static across a 28-second 5-rep squat
sequence, joints 0/1/4 form a tight cluster equidistant by 0.15-0.29 m).
So the AIFit signature was scoring noise on mis-named joints.

v5.1 sidesteps the issue by replacing the canonical-AIFit-signature label
with a **synthetic-perturbation severity** label, in the spirit of
Fitness-AQA (Parmar et al., ECCV 2022). Crucially:

  * The MODEL INPUT is still real MediaPipe-from-Fit3D-video poses — no
    synthetic input distribution shift, unlike v4.
  * The PERTURBATION is applied in canonical 15-joint space as a real
    geometric joint rotation — anatomically interpretable even if our
    indexing is off, because the perturbation is consistent end-to-end.
  * The QUALITY LABEL maps directly from severity: clean=1.0,
    mild=0.7, moderate=0.4, severe=0.1.
  * The JOINT_ERR LABEL is a per-frame mask of (perturbed_group ×
    envelope_above_threshold). For clean reps it's all zeros.

The resulting model is supervised to predict "how perturbed is this pose
relative to clean Fit3D reference, and which joint group was perturbed".

Output schema (same as v5.0 for downstream model code compatibility):

    pose         : (N, 64, 15, 4) float32  — MediaPipe canonical + visibility
    angles       : (N, 64, 22)    float32  — angles on the (possibly perturbed) pose
    exercise_idx : (N,)           int32
    quality      : (N,)           float32  — severity-mapped, (0, 1]
    joint_err    : (N, 64, 5)     float32  — per-frame, per-group, in {0,1}
    boundary     : (N, 64)        float32  — Gaussian peaks at frames 0 and 63
    rep_count    : (N,)           float32
    subject_idx  : (N,)           int32
    camera_id    : (N,)           int32
    rep_idx      : (N,)           int32
    perturb_group: (N,)           int32    — -1 if clean, else 0..4 (joint group)
    perturb_severity_idx : (N,)   int32    — 0=clean, 1=mild, 2=moderate, 3=severe
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .aifit_features import GROUP_TO_IDX_V5, JOINT_GROUPS_V5, N_JOINT_GROUPS_V5
from .angular_features import N_ANGULAR, compute_sequence_angles
from .fit3d_loader import load_rep_boundaries
from .joint_mapping import N_CANONICAL
from .normalize import (
    TARGET_FRAMES,
    extract_canonical_from_mediapipe,
    normalize_skeleton,
    resample_sequence,
)
from .synthetic_perturbation import (
    JOINT_GROUPS,
    SEVERITY_DEG,
    apply_paired_perturbation,
    sample_paired_perturbation,
    _envelope,
)

# ── v5.1 supervised exercise list (same 15 as v5.0) ──────────────────────────
SUPPORTED_EXERCISES_V5 = [
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

INSTRUCTOR_SUBJECT_V5 = "s03"
TRAIN_SUBJECTS_V5 = ["s03", "s04", "s05", "s07", "s08", "s10"]
VAL_SUBJECTS_V5 = ["s09"]
TEST_SUBJECTS_V5 = ["s11"]
OOD_SUBJECTS_V5 = ["s02", "s12", "s13"]

TRAIN_CAMERAS_V5 = ["50591643", "58860488", "60457274", "65906101"]

DEFAULT_FIT3D_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Drafts2\Form Correction"
DEFAULT_MEDIAPIPE_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d_v5"
    )
)
DEFAULT_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "v5_1_dataset")
)

# ── Severity → quality scalar ────────────────────────────────────────────────
# Fitness-AQA tier convention (Parmar et al. 2022, Table 2):
#   clean → 1.0, mild → 0.7, moderate → 0.4, severe → 0.1
SEVERITY_TO_QUALITY = {
    "clean":    1.0,
    "mild":     0.7,
    "moderate": 0.4,
    "severe":   0.1,
}
SEVERITY_INDEX = {"clean": 0, "mild": 1, "moderate": 2, "severe": 3}

# Variants per clean rep at training time. 3 per rep means train inflates 4×
# (1 clean + 3 perturbed). Across 5 joint groups × 3 severities = 15 possible
# variants we randomly sample 3 — stratified so each rep covers diverse forms.
N_TRAIN_PERTURBED_VARIANTS = 3

# A frame is flagged "erroring" for a perturbed group when the temporal
# envelope exceeds this threshold. Mid-rep is always flagged; rep edges are
# clean (no error fires at frame 0 or frame 63).
ENVELOPE_ERROR_THRESHOLD = 0.3

# Boundary Gaussian sigma for the (frames 0 and 63) rep boundaries.
BOUNDARY_SIGMA = 2.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_mediapipe_npy(
    mediapipe_root: str,
    split: str,
    subject: str,
    camera: str,
    exercise: str,
) -> Optional[np.ndarray]:
    """Return (T, 33, 4) MediaPipe joints, or None if missing."""
    path = os.path.join(mediapipe_root, split, subject, camera, f"{exercise}.npy")
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _slice_canonicalize(
    mp_clip: np.ndarray, start: int, end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice + canonicalize + normalize ONE rep's MediaPipe joints.

    Returns:
        canon_xyz : (T, 15, 3) — canonical joints, hip-centred + torso-scaled,
                     pre-resample. T is rep length in original frames.
        canon_vis : (T, 15, 1) — per-joint visibility (mean of contributing
                     MediaPipe landmarks; canonical-15 mapping mirrors what
                     ``dataset_builder_v5._mediapipe_rep_to_inputs`` does).
    """
    rep = mp_clip[start:end]  # (rep_T, 33, 4)
    if len(rep) < 4:
        raise ValueError(f"rep too short: {len(rep)} frames")

    xyz = rep[..., :3]
    vis = rep[..., 3:4]
    canon_xyz = extract_canonical_from_mediapipe(xyz)
    canon_xyz = normalize_skeleton(canon_xyz)

    from .joint_mapping import MEDIAPIPE_JOINTS, MEDIAPIPE_TO_CANONICAL

    T = xyz.shape[0]
    canon_vis = np.zeros((T, N_CANONICAL, 1), dtype=np.float32)
    for can_idx, mp_idx in MEDIAPIPE_TO_CANONICAL.items():
        if can_idx == 14:
            continue
        if mp_idx is None:
            continue
        if mp_idx == "neck_weighted":
            l = MEDIAPIPE_JOINTS["l_shoulder"]
            r = MEDIAPIPE_JOINTS["r_shoulder"]
            le = MEDIAPIPE_JOINTS["l_ear"]
            re = MEDIAPIPE_JOINTS["r_ear"]
            canon_vis[:, can_idx] = 0.25 * (
                vis[:, l] + vis[:, r] + vis[:, le] + vis[:, re]
            )
        elif isinstance(mp_idx, tuple):
            canon_vis[:, can_idx] = np.mean(
                [vis[:, i] for i in mp_idx], axis=0
            )
        else:
            canon_vis[:, can_idx] = vis[:, mp_idx]
    canon_vis[:, 14] = 0.5 * (canon_vis[:, 12] + canon_vis[:, 13])
    return canon_xyz.astype(np.float32), canon_vis.astype(np.float32)


def _build_sample_from_canon(
    canon_xyz: np.ndarray,
    canon_vis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (T, 15, 3+1) to 64 frames, return pose64 (T, 15, 4) + angles64 (T, 22)."""
    canon_xyz_64 = resample_sequence(canon_xyz, TARGET_FRAMES)  # (64, 15, 3)
    canon_vis_64 = resample_sequence(canon_vis, TARGET_FRAMES)  # (64, 15, 1)
    pose64 = np.concatenate([canon_xyz_64, canon_vis_64], axis=-1)
    angles64 = compute_sequence_angles(canon_xyz_64).astype(np.float32)
    return pose64.astype(np.float32), angles64


def _gaussian_boundary_label(positions: List[float]) -> np.ndarray:
    """Per-frame boundary = sum of Gaussians at given frame indices, capped at 1.0."""
    out = np.zeros(TARGET_FRAMES, dtype=np.float32)
    if not positions:
        return out
    t = np.arange(TARGET_FRAMES, dtype=np.float32)
    for pos in positions:
        out += np.exp(-0.5 * ((t - pos) / BOUNDARY_SIGMA) ** 2)
    return np.clip(out, 0.0, 1.0)


def _joint_err_label_for_perturbation(
    perturb_group: str,
) -> np.ndarray:
    """(T=64, 5) binary mask. The perturbed group fires on mid-rep frames."""
    t_norm = np.linspace(0.0, 1.0, TARGET_FRAMES, dtype=np.float32)
    env = _envelope(t_norm, kind="midpeak")  # (T,)
    fires = (env > ENVELOPE_ERROR_THRESHOLD).astype(np.float32)
    out = np.zeros((TARGET_FRAMES, N_JOINT_GROUPS_V5), dtype=np.float32)
    out[:, GROUP_TO_IDX_V5[perturb_group]] = fires
    return out


# ── Per-rep variant generation ──────────────────────────────────────────────


def _sample_perturbation_plans(
    rng: np.random.Generator,
    n_variants: int,
) -> List[Tuple[str, str]]:
    """Pick (joint_group, severity) pairs stratified across groups.

    With n_variants=3 we cover 3 distinct joint groups, randomly chosen
    severities. Caller handles the "1 clean + n_variants perturbed" split.
    """
    groups_shuffled = list(JOINT_GROUPS)
    rng.shuffle(groups_shuffled)
    severities = ("mild", "moderate", "severe")
    plan: List[Tuple[str, str]] = []
    for g in groups_shuffled[:n_variants]:
        s = str(rng.choice(severities))
        plan.append((g, s))
    while len(plan) < n_variants:
        plan.append((str(rng.choice(JOINT_GROUPS)), str(rng.choice(severities))))
    return plan


# ── Per-split builder ────────────────────────────────────────────────────────


def _build_split_samples(
    split_name: str,
    subjects: List[str],
    cameras: List[str],
    fit3d_root: str,
    mediapipe_root: str,
    fit3d_split_subdir: str = "train",
    test_camera_label: str = "single",
    n_perturbed_per_rep: int = N_TRAIN_PERTURBED_VARIANTS,
    rng_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Produce one split's samples as a dict of stacked arrays.

    For each (subject, exercise, rep_idx):
      1. Load all camera views' MediaPipe joints, slice the rep, canonicalize.
      2. Emit one CLEAN sample per camera (quality=1.0, joint_err=zeros).
      3. If ``n_perturbed_per_rep > 0``, sample N perturbation plans
         (joint_group, severity). Apply each plan IDENTICALLY across all
         camera views (paired perturbation, same RNG seed → same rotations)
         so multi-view loss still gets matched pairs.
      4. Emit one perturbed sample per (camera, plan) combination.
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
    perturb_groups: List[int] = []
    perturb_sev_idxs: List[int] = []

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    camera_to_idx = (
        {c: i for i, c in enumerate(cameras)} if cameras else {test_camera_label: -1}
    )
    cams_used = cameras if cameras else [test_camera_label]

    rng = np.random.default_rng(rng_seed)
    n_clean = 0
    n_perturbed = 0
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

        for ex in SUPPORTED_EXERCISES_V5:
            if ex not in rep_boundaries:
                continue
            bounds = rep_boundaries[ex]
            if len(bounds) < 2:
                continue

            # Pre-load MediaPipe clips for every camera once
            clips: Dict[str, np.ndarray] = {}
            for cam in cams_used:
                c = _load_mediapipe_npy(
                    mediapipe_root, fit3d_split_subdir, subj, cam, ex
                )
                if c is not None:
                    clips[cam] = c
            if not clips:
                n_skipped_missing += len(bounds) - 1
                continue

            for rep_idx in range(len(bounds) - 1):
                start, end = int(bounds[rep_idx]), int(bounds[rep_idx + 1])
                if end - start < 10:
                    n_skipped_short += 1
                    continue

                # Slice + canonicalize for every camera with data
                per_cam_canon: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for cam, clip in clips.items():
                    if end > len(clip):
                        continue
                    try:
                        canon_xyz, canon_vis = _slice_canonicalize(clip, start, end)
                    except Exception:
                        continue
                    if not np.all(np.isfinite(canon_xyz)):
                        continue
                    per_cam_canon[cam] = (canon_xyz, canon_vis)
                if not per_cam_canon:
                    n_skipped_bad += 1
                    continue

                # ── Sample perturbation plans (shared across cameras) ────────
                # We want IDENTICAL rotation parameters across all cameras of
                # this rep so multi-view consistency loss has matched pairs.
                # We pre-sample n_perturbed_per_rep plans once per rep.
                plans: List[Tuple[str, str, dict]] = []
                if n_perturbed_per_rep > 0:
                    plan_picks = _sample_perturbation_plans(rng, n_perturbed_per_rep)
                    for (g, sev) in plan_picks:
                        # Pre-sample sign + side once for this plan; we'll reuse
                        # across cameras with identical jitter RNG seed.
                        plan_kwargs = sample_paired_perturbation(rng, g, sev)
                        plans.append((g, sev, plan_kwargs))

                # ── Emit clean samples (one per camera) ──────────────────────
                bd_clean = _gaussian_boundary_label([0.0, float(TARGET_FRAMES - 1)])
                je_clean = np.zeros((TARGET_FRAMES, N_JOINT_GROUPS_V5), dtype=np.float32)

                for cam, (canon_xyz, canon_vis) in per_cam_canon.items():
                    pose64, angles64 = _build_sample_from_canon(canon_xyz, canon_vis)
                    pose_arrs.append(pose64)
                    angles_arrs.append(angles64)
                    exercise_idxs.append(EXERCISE_TO_IDX_V5[ex])
                    qualities.append(SEVERITY_TO_QUALITY["clean"])
                    joint_errs.append(je_clean.copy())
                    boundaries.append(bd_clean.copy())
                    rep_counts.append(1.0)
                    subject_idxs.append(subject_to_idx[subj])
                    camera_ids.append(camera_to_idx.get(cam, -1))
                    rep_idxs.append(rep_idx)
                    perturb_groups.append(-1)
                    perturb_sev_idxs.append(SEVERITY_INDEX["clean"])
                    n_clean += 1

                # ── Emit perturbed samples (one per (camera, plan)) ─────────
                for plan_i, (g, sev, plan_kwargs) in enumerate(plans):
                    je_pert = _joint_err_label_for_perturbation(g)
                    # Per-rep deterministic jitter seed so MoCap-and-MediaPipe
                    # paths (if both exist) get the same jitter. We only have
                    # MediaPipe here, but we still want each (rep, plan) pair
                    # to have a stable seed for reproducibility.
                    jitter_seed = int(
                        (subject_to_idx[subj] * 1000003
                         + EXERCISE_TO_IDX_V5[ex] * 991
                         + rep_idx * 41
                         + plan_i)
                        & 0x7FFFFFFF
                    )

                    for cam, (canon_xyz, canon_vis) in per_cam_canon.items():
                        # Apply the SAME plan with the SAME jitter seed across
                        # all cameras of this (rep, plan) so multi-view loss
                        # has matched perturbed pairs.
                        jitter_rng = np.random.default_rng(jitter_seed)
                        canon_pert = apply_paired_perturbation(
                            canon_xyz, plan_kwargs, jitter_rng,
                        )
                        if not np.all(np.isfinite(canon_pert)):
                            n_skipped_bad += 1
                            continue
                        pose64, angles64 = _build_sample_from_canon(
                            canon_pert, canon_vis,
                        )
                        pose_arrs.append(pose64)
                        angles_arrs.append(angles64)
                        exercise_idxs.append(EXERCISE_TO_IDX_V5[ex])
                        qualities.append(SEVERITY_TO_QUALITY[sev])
                        joint_errs.append(je_pert.copy())
                        boundaries.append(bd_clean.copy())
                        rep_counts.append(1.0)
                        subject_idxs.append(subject_to_idx[subj])
                        camera_ids.append(camera_to_idx.get(cam, -1))
                        rep_idxs.append(rep_idx)
                        perturb_groups.append(GROUP_TO_IDX_V5[g])
                        perturb_sev_idxs.append(SEVERITY_INDEX[sev])
                        n_perturbed += 1

    if verbose:
        print(
            f"  [{split_name}] clean={n_clean}  perturbed={n_perturbed}  "
            f"(missing_npy={n_skipped_missing}, short={n_skipped_short}, "
            f"bad={n_skipped_bad})"
        )

    n_total = n_clean + n_perturbed
    if n_total == 0:
        return {}

    return {
        "pose":                 np.stack(pose_arrs).astype(np.float32),
        "angles":               np.stack(angles_arrs).astype(np.float32),
        "exercise_idx":         np.array(exercise_idxs,        dtype=np.int32),
        "quality":              np.array(qualities,            dtype=np.float32),
        "joint_err":            np.stack(joint_errs).astype(np.float32),
        "boundary":             np.stack(boundaries).astype(np.float32),
        "rep_count":            np.array(rep_counts,           dtype=np.float32),
        "subject_idx":          np.array(subject_idxs,         dtype=np.int32),
        "camera_id":            np.array(camera_ids,           dtype=np.int32),
        "rep_idx":              np.array(rep_idxs,             dtype=np.int32),
        "perturb_group":        np.array(perturb_groups,       dtype=np.int32),
        "perturb_severity_idx": np.array(perturb_sev_idxs,     dtype=np.int32),
    }


# ── Top-level driver ────────────────────────────────────────────────────────


def build_v5_1_dataset(
    fit3d_root: str = DEFAULT_FIT3D_ROOT,
    mediapipe_root: str = DEFAULT_MEDIAPIPE_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    n_perturbed_per_rep: int = N_TRAIN_PERTURBED_VARIANTS,
    rng_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, dict]:
    os.makedirs(out_root, exist_ok=True)

    if verbose:
        print(f"[v5.1-dataset] fit3d_root          : {fit3d_root}")
        print(f"[v5.1-dataset] mediapipe_root      : {mediapipe_root}")
        print(f"[v5.1-dataset] out_root            : {out_root}")
        print(f"[v5.1-dataset] perturbed/rep (train): {n_perturbed_per_rep}")
        print(f"[v5.1-dataset] severity-quality    : {SEVERITY_TO_QUALITY}")
        print()

    # Per-split: train+val+test get full train-cameras; ood is single-cam.
    # Train-only gets perturbations; val/test are clean-only so we can
    # measure how the model interpolates on real reps.
    splits_def = [
        ("train", TRAIN_SUBJECTS_V5, TRAIN_CAMERAS_V5, "train", n_perturbed_per_rep),
        ("val",   VAL_SUBJECTS_V5,   TRAIN_CAMERAS_V5, "train", 0),
        ("test",  TEST_SUBJECTS_V5,  TRAIN_CAMERAS_V5, "train", 0),
        ("ood",   OOD_SUBJECTS_V5,   [],               "test",  0),
    ]
    # ALSO emit a perturbed val for tracking quality-gradient learning.
    splits_def.insert(2, ("val_perturbed", VAL_SUBJECTS_V5, TRAIN_CAMERAS_V5, "train",
                          max(2, n_perturbed_per_rep)))

    splits: Dict[str, dict] = {}
    for split_name, subjects, cameras, mp_split, n_pert in splits_def:
        if verbose:
            print(f"[v5.1-dataset] Building {split_name}: subjects={subjects}, "
                  f"cams={cameras or 'single'}, perturb={n_pert}/rep")
        contents = _build_split_samples(
            split_name=split_name,
            subjects=subjects,
            cameras=cameras,
            fit3d_root=fit3d_root,
            mediapipe_root=mediapipe_root,
            fit3d_split_subdir=mp_split,
            n_perturbed_per_rep=n_pert,
            rng_seed=rng_seed + hash(split_name) % 1000,
            verbose=verbose,
        )
        if not contents:
            if verbose:
                print(f"  [{split_name}] empty; nothing to save")
            splits[split_name] = {"n_samples": 0}
            continue
        out_path = os.path.join(out_root, f"{split_name}.npz")
        np.savez_compressed(out_path, **contents)
        n = int(contents["pose"].shape[0])
        clean_mask = contents["perturb_severity_idx"] == 0
        n_c = int(clean_mask.sum())
        splits[split_name] = {
            "n_samples": n,
            "n_clean": n_c,
            "n_perturbed": n - n_c,
            "path": out_path,
            "quality_mean": float(contents["quality"].mean()),
            "quality_std":  float(contents["quality"].std()),
            "joint_err_rate": float(contents["joint_err"].mean()),
        }
        if verbose:
            print(f"  [{split_name}] saved -> {out_path}")
            print(f"  [{split_name}] N={n}  clean={n_c}  perturbed={n - n_c}  "
                  f"q_mean={splits[split_name]['quality_mean']:.3f}  "
                  f"q_std={splits[split_name]['quality_std']:.3f}  "
                  f"je_rate={splits[split_name]['joint_err_rate']:.3f}")
        print()

    info = {
        "version": "v5.1",
        "label_strategy": "synthetic_perturbation_severity",
        "n_exercises": N_EXERCISES_V5,
        "exercises": SUPPORTED_EXERCISES_V5,
        "n_joint_groups": N_JOINT_GROUPS_V5,
        "joint_groups": JOINT_GROUPS_V5,
        "target_frames": TARGET_FRAMES,
        "n_canonical_joints": N_CANONICAL,
        "n_angular": N_ANGULAR,
        "instructor_subject": INSTRUCTOR_SUBJECT_V5,
        "train_subjects":  TRAIN_SUBJECTS_V5,
        "val_subjects":    VAL_SUBJECTS_V5,
        "test_subjects":   TEST_SUBJECTS_V5,
        "ood_subjects":    OOD_SUBJECTS_V5,
        "train_cameras":   TRAIN_CAMERAS_V5,
        "perturbations_per_rep_train": n_perturbed_per_rep,
        "severity_to_quality": SEVERITY_TO_QUALITY,
        "splits": {k: v for k, v in splits.items()},
    }
    with open(os.path.join(out_root, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_root, "exercise_labels.json"), "w") as f:
        json.dump(EXERCISE_TO_IDX_V5, f, indent=2)
    if verbose:
        print("[v5.1-dataset] DONE")
    return splits


# ── CLI ─────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--fit3d-root", default=DEFAULT_FIT3D_ROOT)
    p.add_argument("--mediapipe-root", default=DEFAULT_MEDIAPIPE_ROOT)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--n-perturbed", type=int, default=N_TRAIN_PERTURBED_VARIANTS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_v5_1_dataset(
        fit3d_root=args.fit3d_root,
        mediapipe_root=args.mediapipe_root,
        out_root=args.out_root,
        n_perturbed_per_rep=args.n_perturbed,
        rng_seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
