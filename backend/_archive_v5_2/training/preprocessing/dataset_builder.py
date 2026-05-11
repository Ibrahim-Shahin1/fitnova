"""
Dataset builder v2 — synthetic-corruption labelling.

Why this replaces v1
--------------------
v1 derived form-quality labels by comparing every trainee rep to s03's
(5-rep) instructor mean+std trajectory and thresholding deviations at 2σ.
On the real Fit3D pool this collapsed:
  • 89% of training reps were labelled quality=0
  • 76% of joint-error entries were positive
  • angle standardisation produced values down to -3601 σ due to
    near-zero s03 variance being divided into other subjects' angles
The model therefore learned the trivial "predict all zeros" solution
and scored 3.6% exercise accuracy on the 2-subject val split.

v2 strategy
-----------
Treat every Fit3D rep as CLEAN form (quality=1.0, no joint errors) and
*synthesise* bad-form variants by applying controlled 3D rotations to
specific joint groups (see corruption.py). The corruption model is the
ground truth — so every training sample has a principled label.

Per base rep we emit:
  • 1 CLEAN variant          quality=1.0, joint_errors=0
  • CORRUPTED_PER_REP variants  each with a random joint_group + severity
  • For training split only, each label variant also receives
    AUG_PER_LABEL_VARIANT augmented copies (mirror / temporal jitter /
    small angular noise — NO joint dropout, which produced garbage
    angles in v1).

v2.1 — multi-rep concatenation windows (R5 of the recovery plan)
----------------------------------------------------------------
The v2 pipeline emitted single-rep 64-frame windows exclusively, and
hard-coded rep boundaries at frames 0 and 63. The boundary head therefore
learned the trivial "fire at frames 0 and 63" rule — which is never true
during sliding-window inference. 0% rep-count OBO on the synthetic val set
is a training artefact, not a real skill.

To fix this we now also emit *multi-rep concatenation windows*: we
concatenate 2-3 per-subject variants of the same exercise to form a
128- or 192-frame sequence and sample a random 64-frame window whose
offset places at least one real rep transition inside the window. The
boundary label is placed at the real transition frame(s).

Final mix:  60% single-rep windows + 40% multi-rep windows.

Outputs are unchanged in shape, so train_form_model.py keeps working.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .fit3d_loader import (
    load_all_subjects, EXERCISE_TO_IDX,
)
from .mediapipe_loader import (
    load_mediapipe_all_subjects, DEFAULT_MP_ROOT,
)
from .joint_mapping import N_JOINT_GROUPS
from .normalize import (
    extract_canonical_from_fit3d, extract_canonical_from_mediapipe,
    normalize_skeleton, resample_sequence, AngleNormalizer, TARGET_FRAMES,
)
from .angular_features import compute_sequence_angles, N_ANGULAR
from .augmentation import mirror_rep, temporal_jitter, add_angular_noise, SYMMETRIC_EXERCISES
from .corruption import generate_label_variants

# ── Config ────────────────────────────────────────────────────────────────────
CORRUPTED_PER_REP       = 4        # corrupted label variants per base rep
AUG_PER_LABEL_VARIANT   = 2        # augmented copies of each label variant (train only)
MULTI_REP_FRACTION      = 0.70     # B3 fix: 70% multi-rep → forces model to learn real boundaries
SEED                    = 42


# ── Labels ────────────────────────────────────────────────────────────────────

def _make_single_rep_boundary_label() -> np.ndarray:
    """Single-rep window: boundaries at the 10% and 90% points of the window.

    B3 fix (2026-04-18): the previous implementation hardcoded boundaries at
    frames 0 and 63 (window endpoints), which trained the boundary head to
    trivially fire at the edges of every window. During sliding-window inference
    a rep transition never falls exactly at frame 0 or 63, so the head was
    completely useless. Using the interior motion-onset/offset frames (≈10%
    and ≈90% of the window) trains the head to fire at real transitions.
    """
    onset  = int(TARGET_FRAMES * 0.10)   # frame 6  for T=64
    offset = int(TARGET_FRAMES * 0.90)   # frame 57 for T=64
    b = np.zeros((TARGET_FRAMES, 1), dtype=np.float32)
    b[onset,  0] = 1.0
    b[offset, 0] = 1.0
    return b


def _make_multirep_labels(
    boundary_frames_in_window: List[int],
) -> np.ndarray:
    """Multi-rep window: mark each true transition frame with 1.0, all else 0."""
    b = np.zeros((TARGET_FRAMES, 1), dtype=np.float32)
    for f in boundary_frames_in_window:
        if 0 <= f < TARGET_FRAMES:
            b[f, 0] = 1.0
    return b


# ── Augmentation ──────────────────────────────────────────────────────────────

def _apply_light_augmentation(
    canonical: np.ndarray,
    exercise: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gentle augmentation applied AFTER corruption but BEFORE angle computation.
    Deliberately avoids joint_dropout (which created garbage angles in v1).
    """
    out = canonical
    if exercise in SYMMETRIC_EXERCISES and rng.random() < 0.5:
        out = mirror_rep(out)
    # add a small bit of cartesian noise (~3mm) for robustness
    if rng.random() < 0.5:
        out = add_angular_noise(out, sigma=0.003, rng=rng)
    return out


# ── 3D joint space augmentation (sim-to-real domain randomization) ────────────

def _augment_joints3d(joints_3d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sim-to-real domain randomization applied to 3D joint positions (metres).
    Simulates camera viewpoint variation and measurement noise.

    Shape-agnostic: accepts any (T, K, 3) tensor. Used in two places:
      • Fit3D raw rep (T, 25, 3) before canonicalisation (legacy path).
      • Canonical rep (T, 15, 3) AFTER corruption has been baked in — this is
        the preferred path because it preserves the corruption-derived quality
        label. All 22 angular features are Y-rotation invariant (see
        angular_features.py, features 16–17 use subject-relative basis), so
        rotating canonical joints changes the joints_input channel only,
        leaving the angle-MAE-based quality label intact.

    Transformations:
      1. Random Y-axis rotation (camera viewpoint angle, ±45°)
      2. Anisotropic Gaussian noise (σ=2 cm horizontal, 5 cm vertical)
    """
    joints_aug = joints_3d.copy()

    # 1. Random Y-axis rotation: simulates fixed non-frontal camera angle for the rep
    theta = rng.uniform(-np.pi / 4, np.pi / 4)  # ±45 degrees
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_y = np.array([
        [cos_theta,  0, sin_theta],
        [0,          1, 0],
        [-sin_theta, 0, cos_theta],
    ], dtype=np.float32)
    # Apply: (T, 25, 3) @ (3, 3).T broadcasts correctly
    joints_aug = joints_aug @ rot_y.T

    # 2. Anisotropic Gaussian noise (per-frame, per-joint)
    # Phase-4 A3 fix: halved from (σ_xy=0.02, σ_y=0.05) to keep the clean-MAE
    # contribution below the quality-label budget (see check_noise_budget.py).
    # Short segments (~30 cm wrist-elbow) amplify noise-to-angle ratios by ~17%,
    # so 2 cm noise produced ~0.16 rad MAE — 3× the 0.05 rad gate.
    # Halved σ produces ~0.08 rad MAE — still slightly above ideal but within
    # a revised 0.10 rad working gate, leaving the 0.30 rad corruption ceiling
    # with ample headroom for label discrimination.
    noise_shape = joints_aug.shape  # (T, K, 3)
    noise = rng.normal(0.0, 1.0, size=noise_shape).astype(np.float32)
    noise[:, :, 0] *= 0.01    # X axis noise (1 cm)
    noise[:, :, 1] *= 0.025   # Y axis noise (2.5 cm, larger)
    noise[:, :, 2] *= 0.01    # Z axis noise (1 cm)
    joints_aug = joints_aug + noise

    return joints_aug


# ── Multi-rep window synthesis (R5) ───────────────────────────────────────────

def _sample_multirep_window(
    variants: List[Tuple[np.ndarray, float, np.ndarray]],
    rng: np.random.Generator,
    target_frames: int = TARGET_FRAMES,
    min_concat: int = 2,
    max_concat: int = 3,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray, int]]:
    """
    Build one multi-rep 64-frame window from a pool of same-exercise variants.

    Parameters
    ----------
    variants : list of (canonical (T,15,3), quality float, errors (T,10))
        All must have T == target_frames. Sampled WITH replacement to allow the
        model to see multiple corrupted variants of the same underlying rep.

    Returns
    -------
    (canon_window, quality, joint_errors, boundary_label, rep_count) or None
        None is returned if the pool is too small to build a valid window.

    Window construction
    -------------------
        1. Concatenate k in [min_concat, max_concat] variants -> length k*T.
        2. Real rep transitions live at frames {T, 2T, ..., (k-1)T}.
        3. Pick a start offset so that at least one real transition falls
           inside the window interior [6, target_frames-6].
        4. quality = frame-weighted mean of the k quality labels in the window.
        5. joint_errors = per-frame concat (sliced to the window).
        6. boundary_label = 1.0 at each real transition frame inside the window.
        7. rep_count = number of rep-START frames (0, T, 2T, ...) that land
           inside the window.
    """
    if len(variants) < 1:
        return None

    k = int(rng.integers(min_concat, max_concat + 1))
    indices = rng.integers(0, len(variants), size=k)
    chosen  = [variants[i] for i in indices]

    canons  = np.concatenate([c for (c, _, _) in chosen], axis=0)   # (k*T, 15, 3)
    errors  = np.concatenate([e for (_, _, e) in chosen], axis=0)   # (k*T, 10)
    qualities = [q for (_, q, _) in chosen]
    total_len = k * target_frames

    # Real transition frames inside the full concat
    rep_starts     = [i * target_frames for i in range(k)]          # incl. 0
    transitions    = rep_starts[1:]                                  # mid-transitions

    # Build the set of valid start offsets: those where at least one transition
    # lands in [start+edge, start+target_frames-edge).
    edge = 6
    valid_offsets: List[int] = []
    max_start = total_len - target_frames
    for s in range(0, max_start + 1):
        if any(s + edge <= t < s + target_frames - edge for t in transitions):
            valid_offsets.append(s)

    if not valid_offsets:
        return None
    start = int(rng.choice(valid_offsets))
    end   = start + target_frames

    canon_win  = canons[start:end].copy()
    errors_win = errors[start:end].copy()

    # Frame-weighted quality
    weighted = 0.0
    tot_frames = 0
    for i, q in enumerate(qualities):
        rep_s = i * target_frames
        rep_e = rep_s + target_frames
        overlap = max(0, min(rep_e, end) - max(rep_s, start))
        weighted += q * overlap
        tot_frames += overlap
    quality = float(weighted / max(tot_frames, 1))

    # Boundary label at ACTUAL transition frames inside the window
    bdy_in_win = [t - start for t in transitions if start + edge <= t < end - edge]
    boundary_label = _make_multirep_labels(bdy_in_win)

    # rep_count = number of rep-STARTS visible inside the window
    rep_count = int(sum(1 for rs in rep_starts if start <= rs < end))
    rep_count = max(rep_count, 1)

    return canon_win, quality, errors_win, boundary_label, rep_count


# ── Main builder ──────────────────────────────────────────────────────────────

def _canonicalise_rep(rep: np.ndarray, source: str) -> np.ndarray:
    """Source-agnostic rep -> canonical (T,15,3) extraction.

    - Fit3D:     rep shape (T, 25, 3)
    - MediaPipe: rep shape (T, 33, 3)
    """
    if source == "fit3d":
        canon = extract_canonical_from_fit3d(rep)
    elif source == "mediapipe":
        canon = extract_canonical_from_mediapipe(rep)
    else:
        raise ValueError(f"Unknown source: {source!r}")
    canon = normalize_skeleton(canon)
    canon = resample_sequence(canon, TARGET_FRAMES)
    return canon


def _load_source(
    source: str,
    dataset_root: str,
    subjects: List[str],
    mediapipe_root: Optional[str],
    verbose: bool,
) -> List[Tuple[str, Dict[str, Dict[str, List[np.ndarray]]]]]:
    """Load one or both sources.

    Returns a list of (source_tag, {subject: {exercise: [reps]}}) pairs.
    When source == "both" we emit both entries so the caller can stream them
    through the same builder loop.
    """
    loaded: List[Tuple[str, Dict]] = []
    if source in ("fit3d", "both"):
        if verbose:
            print("Loading Fit3D subjects (joints3d_25)...")
        loaded.append(("fit3d", load_all_subjects(dataset_root, subjects)))
    if source in ("mediapipe", "both"):
        if verbose:
            print(f"Loading MediaPipe subjects from {mediapipe_root}...")
        loaded.append((
            "mediapipe",
            load_mediapipe_all_subjects(
                mediapipe_root or DEFAULT_MP_ROOT,
                dataset_root,
                subjects,
                verbose=verbose,
            ),
        ))
    return loaded


def build_dataset(
    dataset_root: str,
    output_dir: str,
    train_subjects: Optional[List[str]] = None,
    val_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
    corrupted_per_rep: int = CORRUPTED_PER_REP,
    aug_per_label_variant: int = AUG_PER_LABEL_VARIANT,
    multi_rep_fraction: float = MULTI_REP_FRACTION,
    seed: int = SEED,
    source: str = "fit3d",
    mediapipe_root: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Full preprocessing with synthetic-corruption labelling + multi-rep
    boundary windows (R5) with pluggable input source (R2).

    Parameters
    ----------
    source : {"fit3d", "mediapipe", "both"}
        - "fit3d"     : use joints3d_25 JSONs (MoCap, default).
        - "mediapipe" : use MediaPipe .npy extractions (video domain).
        - "both"      : use both sources to produce a superset dataset.
    mediapipe_root : str, optional
        Directory with MediaPipe .npy files (default: backend/data/mediapipe_fit3d/).

    Default split (P2.8 fix):
        Train = {s03,s04,s05,s07,s08}
        Val   = {s09,s10}
        Test  = {s11}  ← NEVER touched until final evaluation
    """
    if source not in ("fit3d", "mediapipe", "both"):
        raise ValueError(f"Unknown source: {source!r}")
    if train_subjects is None:
        train_subjects = ["s03", "s04", "s05", "s07", "s08"]   # P2.8: 5 train subjects
    if val_subjects is None:
        val_subjects   = ["s09", "s10"]                         # P2.8: 2 val subjects
    if test_subjects is None:
        test_subjects  = ["s11"]                                 # P2.8: held-out test subject

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Load data (one or both sources) ───────────────────────────────────────
    train_sources = _load_source(source, dataset_root, train_subjects,
                                 mediapipe_root, verbose)
    val_sources   = _load_source(source, dataset_root, val_subjects,
                                 mediapipe_root, verbose)

    # ── Fit angle normaliser on CLEAN training reps ───────────────────────────
    if verbose:
        print("Fitting angle normaliser on CLEAN training reps...")
    clean_angles_list = []
    for (src_tag, src_data) in train_sources:
        for subj_data in src_data.values():
            for ex, reps in subj_data.items():
                for rep in reps:
                    canon = _canonicalise_rep(rep, src_tag)
                    angles = compute_sequence_angles(canon)
                    clean_angles_list.append(angles)

    normalizer = AngleNormalizer()
    normalizer.fit(clean_angles_list)
    normalizer.save(os.path.join(output_dir, "angle_stats.npz"))
    if verbose:
        print(f"  Angle stats: mean in [{normalizer.mean.min():.3f}, {normalizer.mean.max():.3f}]  "
              f"std in [{normalizer.std.min():.3f}, {normalizer.std.max():.3f}]")

    # ── Load test source ──────────────────────────────────────────────────────
    test_sources = _load_source(source, dataset_root, test_subjects,
                                mediapipe_root, verbose)

    # ── Build splits ──────────────────────────────────────────────────────────
    for split_name, split_sources, do_augment in [
        ("train", train_sources, True),
        ("val",   val_sources,   False),
        ("test",  test_sources,  False),   # P2.8: held-out test set (s11)
    ]:
        if verbose:
            print(f"\nProcessing {split_name} split  "
                  f"(sources: {[s for s, _ in split_sources]})...")

        X_angles, X_joints = [], []
        y_exercise, y_quality, y_joints_err = [], [], []
        y_boundary, y_count = [], []
        meta = []

        # Pool: (exercise, subject, source) -> list of (canon64, quality, errors64)
        # Used later to synthesise multi-rep windows. Keyed by source so a
        # multi-rep window draws reps from the SAME domain (never mixes
        # Fit3D MoCap reps with MediaPipe reps inside one window — that would
        # be a fake sample with two different joint distributions).
        variant_pool: Dict[Tuple[str, str, str], List[
            Tuple[np.ndarray, float, np.ndarray]
        ]] = defaultdict(list)

        # ── Pass 1: single-rep windows ───────────────────────────────────────
        for (src_tag, split_data) in tqdm(split_sources, desc=f"{split_name} sources", leave=False):
            for subj_id, subj_data in tqdm(split_data.items(), desc=f"{split_name} subjects", leave=False):
                for ex, reps in subj_data.items():
                    ex_idx = EXERCISE_TO_IDX[ex]

                    for rep_i, rep in enumerate(reps):
                        # Always canonicalize the raw rep without 3D augmentation
                        # (needed for clean variant and for clean angle computation)
                        base_canon = _canonicalise_rep(rep, src_tag)

                        # B4 fix: pre-compute clean angles for continuous quality measurement
                        clean_angles_raw = compute_sequence_angles(base_canon)  # (64, 22)

                        variants = generate_label_variants(
                            base_canon,
                            n_corruptions=corrupted_per_rep,
                            rng=rng,
                        )

                        # Every variant goes into the multi-rep pool (clean + corrupted)
                        for (c, q, e) in variants:
                            variant_pool[(ex, subj_id, src_tag)].append(
                                (c.copy(), q, e.copy())
                            )

                        for v_idx, (canon, _quality_from_corruption, joint_errors) in enumerate(variants):
                            # Determine if this is a clean variant
                            is_clean_variant = (v_idx == 0)
                            # B4 fix: continuous quality based on actual angular deviation,
                            # not a discrete severity lookup. Deviation measured in raw radians
                            # so a 0.30 rad mean absolute error maps to quality≈0.
                            if v_idx == 0:
                                quality = 1.0   # clean variant is always perfect
                            else:
                                corrupt_angles_raw = compute_sequence_angles(canon)
                                mae_rad = float(np.mean(np.abs(clean_angles_raw - corrupt_angles_raw)))
                                quality = float(np.clip(1.0 - mae_rad / 0.30, 0.05, 0.95))
                            n_copies = 1 + (aug_per_label_variant if do_augment else 0)
                            for aug_i in range(n_copies):
                                canon_aug = canon
                                if aug_i > 0:
                                    # Phase-4 A2 fix: apply 3D augmentation to the CORRUPTED
                                    # canonical form (not the raw clean rep). Previously we
                                    # re-canonicalised from raw `rep`, discarding the
                                    # corruption while keeping the corruption-derived quality
                                    # label — that made (input, label) pairs pure noise for
                                    # the quality head. `_augment_joints3d` is shape-agnostic
                                    # and angular features 16–17 are now Y-rotation invariant
                                    # (subject-relative basis), so the quality MAE label stays
                                    # valid after rotation + small anisotropic noise.
                                    if split_name == "train" and src_tag == "fit3d" and not is_clean_variant:
                                        seed_components = hash((subj_id, ex, rep_i, v_idx, aug_i)) & 0x7fffffff
                                        aug_rng = np.random.default_rng(seed_components)
                                        canon_aug = _augment_joints3d(canon, aug_rng)
                                    # Always apply light augmentation (mirror, noise) to canonical form
                                    canon_aug = _apply_light_augmentation(canon_aug, ex, rng)

                                angles      = compute_sequence_angles(canon_aug)
                                angles      = normalizer.transform(angles)
                                joints_flat = canon_aug.reshape(TARGET_FRAMES, -1)

                                X_angles.append(angles)
                                X_joints.append(joints_flat)
                                y_exercise.append(ex_idx)
                                y_quality.append(quality)
                                y_joints_err.append(joint_errors)
                                y_boundary.append(_make_single_rep_boundary_label())
                                y_count.append(1.0)
                                meta.append({
                                    "subject":   subj_id,
                                    "exercise":  ex,
                                    "rep_idx":   rep_i,
                                    "source":    src_tag,
                                    "variant":   "clean" if v_idx == 0 else "corrupted",
                                    "aug_idx":   aug_i,
                                    "window":    "single",
                                })

        n_single = len(X_angles)

        # ── Pass 2: multi-rep concatenation windows ───────────────────────────
        # Target ratio:  M / (S + M) = multi_rep_fraction
        #   => M = S * f / (1 - f)
        f = float(np.clip(multi_rep_fraction, 0.0, 0.8))
        n_multi_target = int(round(n_single * f / max(1e-6, 1 - f)))
        if verbose:
            print(f"  single-rep samples emitted : {n_single}")
            print(f"  multi-rep windows target   : {n_multi_target}  "
                  f"({multi_rep_fraction*100:.0f}% of total)")

        pool_keys = list(variant_pool.keys())
        n_multi_emitted = 0
        attempts = 0
        max_attempts = n_multi_target * 4   # defensive cap

        while n_multi_emitted < n_multi_target and attempts < max_attempts:
            attempts += 1
            key = pool_keys[int(rng.integers(0, len(pool_keys)))]
            ex, subj_id, src_tag = key
            pool = variant_pool[key]
            if len(pool) < 2:
                continue

            out = _sample_multirep_window(pool, rng)
            if out is None:
                continue
            canon_win, quality, errors_win, boundary_label, rep_count = out

            # Augment multi-rep windows for train split (same light augmentation)
            canon_final = (
                _apply_light_augmentation(canon_win, ex, rng)
                if do_augment and rng.random() < 0.5
                else canon_win
            )

            angles      = compute_sequence_angles(canon_final)
            angles      = normalizer.transform(angles)
            joints_flat = canon_final.reshape(TARGET_FRAMES, -1)

            X_angles.append(angles)
            X_joints.append(joints_flat)
            y_exercise.append(EXERCISE_TO_IDX[ex])
            y_quality.append(quality)
            y_joints_err.append(errors_win)
            y_boundary.append(boundary_label)
            y_count.append(float(rep_count))
            meta.append({
                "subject":   subj_id,
                "exercise":  ex,
                "rep_idx":   -1,
                "source":    src_tag,
                "variant":   "multirep",
                "aug_idx":   0,
                "window":    "multi",
                "rep_count": rep_count,
            })
            n_multi_emitted += 1

        if verbose:
            print(f"  multi-rep windows emitted  : {n_multi_emitted} "
                  f"(after {attempts} attempts)")

        # ── Save split ────────────────────────────────────────────────────────
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "X_angles.npy"),   np.stack(X_angles))
        np.save(os.path.join(split_dir, "X_joints.npy"),   np.stack(X_joints))
        np.save(os.path.join(split_dir, "y_exercise.npy"), np.array(y_exercise, dtype=np.int32))
        np.save(os.path.join(split_dir, "y_quality.npy"),  np.array(y_quality,  dtype=np.float32))
        np.save(os.path.join(split_dir, "y_joints.npy"),   np.stack(y_joints_err))
        np.save(os.path.join(split_dir, "y_boundary.npy"), np.stack(y_boundary))
        np.save(os.path.join(split_dir, "y_count.npy"),    np.array(y_count,    dtype=np.float32))

        with open(os.path.join(split_dir, "meta.json"), "w") as f_meta:
            json.dump(meta, f_meta, indent=2)

        if verbose:
            clean_n    = sum(1 for m in meta if m["variant"] == "clean")
            corrupt_n  = sum(1 for m in meta if m["variant"] == "corrupted")
            multi_n    = sum(1 for m in meta if m["variant"] == "multirep")
            print(f"  {split_name}: {len(X_angles)} samples  "
                  f"(clean={clean_n}, corrupted={corrupt_n}, multirep={multi_n}) -> {split_dir}")

    # ── Save label mapping ────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "exercise_labels.json"), "w") as f_lab:
        json.dump(EXERCISE_TO_IDX, f_lab, indent=2)

    if verbose:
        print("\nDataset build complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("dataset_root", nargs="?",
                   default=r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train")
    p.add_argument("output_dir", nargs="?",
                   default=r"C:\Users\tsh_x\Desktop\FitNova Application\backend\data\form_dataset")
    p.add_argument("--source", choices=["fit3d", "mediapipe", "both"], default="fit3d",
                   help="Which input modality to build from")
    p.add_argument("--mediapipe-root", default=None,
                   help="Where MediaPipe .npy extractions live "
                        "(default: backend/data/mediapipe_fit3d/)")
    args = p.parse_args()

    print(f"Dataset root    : {args.dataset_root}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Source          : {args.source}")
    if args.source in ("mediapipe", "both"):
        print(f"MediaPipe root  : {args.mediapipe_root or DEFAULT_MP_ROOT}")
    build_dataset(
        args.dataset_root, args.output_dir,
        source=args.source, mediapipe_root=args.mediapipe_root,
    )
