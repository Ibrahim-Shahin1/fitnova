"""Unit tests for backend.training.preprocessing.qevd_dataset.

Run from repo root:
    pytest backend/training/preprocessing/test_qevd_dataset.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.preprocessing.qevd_dataset import (
    assign_clip_to_split,
    split_subjects,
    stratified_sample_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# split_subjects
# ─────────────────────────────────────────────────────────────────────────────


def test_split_subjects_disjoint_and_sized_correctly():
    ids = [f"h_{i:04d}" for i in range(100)]
    train, val = split_subjects(ids, val_frac=0.09, seed=42)
    # Disjoint
    assert train.isdisjoint(val)
    # Cover the input set
    assert train | val == set(ids)
    # Size: ~9% val, allowing rounding
    assert 8 <= len(val) <= 10
    assert len(train) + len(val) == 100


def test_split_subjects_is_deterministic():
    ids = [f"h_{i:04d}" for i in range(200)]
    a = split_subjects(ids, val_frac=0.10, seed=123)
    b = split_subjects(ids, val_frac=0.10, seed=123)
    assert a == b
    # Different seed -> different partition (with high probability for N=200)
    c = split_subjects(ids, val_frac=0.10, seed=999)
    assert a != c


def test_split_subjects_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_subjects(["h_1"], val_frac=0.0)
    with pytest.raises(ValueError):
        split_subjects(["h_1"], val_frac=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# assign_clip_to_split
# ─────────────────────────────────────────────────────────────────────────────


def test_assign_clip_to_split_routes_all_four_outcomes():
    train_humans = {"h_001", "h_002"}
    val_humans = {"h_900"}

    cases = [
        ({"source": "fit300k_train", "human_id": "h_001"}, "train"),
        ({"source": "fit300k_train", "human_id": "h_900"}, "val"),
        ({"source": "fit300k_test",  "human_id": "h_test_5"}, "in_domain_test"),
        ({"source": "fitcoach_test", "human_id": "h_coach_1"}, "ood_test"),
        # Edge: human in neither set within fit300k_train -> unused
        ({"source": "fit300k_train", "human_id": "h_orphan"}, "unused"),
        # FIT-COACH train is unused per master plan section II.4.5
        ({"source": "fitcoach_train", "human_id": "h_coach_train"}, "unused"),
        # Unknown source -> unused
        ({"source": "mystery_source", "human_id": "h_x"}, "unused"),
    ]
    for meta, expected in cases:
        got = assign_clip_to_split(meta, train_humans, val_humans)
        assert got == expected, (meta, expected, got)


# ─────────────────────────────────────────────────────────────────────────────
# stratified_sample_weights
# ─────────────────────────────────────────────────────────────────────────────


def test_stratified_sample_weights_inverse_frequency_and_mean_one():
    # Two exercises (0, 1) and a clear class imbalance + bucket imbalance.
    # 80 of class 0 with quality 0.9 (bucket 3) -- common
    # 10 of class 0 with quality 0.1 (bucket 0) -- rare
    # 10 of class 1 with quality 0.5 (bucket 2) -- rare
    labels = np.array([0]*80 + [0]*10 + [1]*10)
    qualities = np.array([0.9]*80 + [0.1]*10 + [0.5]*10)

    weights = stratified_sample_weights(labels, qualities, n_buckets=4)

    assert weights.shape == labels.shape
    assert weights.dtype == np.float32
    # Mean should be ~1.0 by construction
    assert abs(float(weights.mean()) - 1.0) < 1e-5
    # Common class gets the smallest weight; rare classes the largest
    common_w = float(weights[0])             # one of the 80 common samples
    rare_w_class0 = float(weights[80])       # one of the rare class-0 low-q samples
    rare_w_class1 = float(weights[90])       # one of the rare class-1 mid-q samples
    assert rare_w_class0 > common_w
    assert rare_w_class1 > common_w
    # The two rare strata have equal counts, so equal weights
    assert abs(rare_w_class0 - rare_w_class1) < 1e-5


def test_stratified_sample_weights_handles_empty_input():
    weights = stratified_sample_weights(
        np.array([], dtype=int), np.array([], dtype=float), n_buckets=4,
    )
    assert weights.shape == (0,)
    assert weights.dtype == np.float32


def test_stratified_sample_weights_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        stratified_sample_weights(
            np.array([0, 1, 2]),
            np.array([0.1, 0.2]),     # wrong shape
            n_buckets=4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# build_qevd_splits — subject-disjoint train/val carve-out using worker_ids
# ─────────────────────────────────────────────────────────────────────────────


from backend.training.preprocessing.qevd_dataset import build_qevd_splits


def test_build_qevd_splits_routes_to_four_buckets():
    """Train clips with known worker_ids split into train/val
    (participant-disjoint); test clips go to in_domain_test."""
    # 100 train participants -> ~9 in val
    clip_to_split = {}
    clip_to_human = {}
    for i in range(100):
        for j in range(3):     # 3 clips per participant
            cid = f"{i:08d}{j}"
            clip_to_split[cid] = "train"
            clip_to_human[cid] = f"w_{i}"
    # 5 test participants
    for i in range(5):
        for j in range(2):
            cid = f"t{i:07d}{j}"
            clip_to_split[cid] = "test"
            clip_to_human[cid] = f"w_test_{i}"

    splits = build_qevd_splits(clip_to_split, clip_to_human,
                                val_frac=0.09, seed=42)

    # All 4 buckets present
    assert set(splits.keys()) == {"train", "val", "in_domain_test", "unused"}

    # Sets are pairwise disjoint
    for a in splits:
        for b in splits:
            if a < b:
                assert splits[a].isdisjoint(splits[b]), (a, b)

    # Union covers all clips
    all_assigned = set().union(*splits.values())
    assert all_assigned == set(clip_to_split.keys())

    # in_domain_test == every test clip
    assert splits["in_domain_test"] == {
        c for c, s in clip_to_split.items() if s == "test"
    }

    # val carve-out is participant-disjoint w.r.t. train
    train_humans = {clip_to_human[c] for c in splits["train"]}
    val_humans   = {clip_to_human[c] for c in splits["val"]}
    assert train_humans.isdisjoint(val_humans), (
        f"val and train share humans: {train_humans & val_humans}"
    )

    # ~9 of 100 train participants in val
    assert 8 <= len(val_humans) <= 11


def test_build_qevd_splits_deterministic_on_seed():
    clip_to_split = {f"{i:08d}": "train" for i in range(50)}
    clip_to_human = {f"{i:08d}": f"w_{i // 5}" for i in range(50)}  # 10 humans
    a = build_qevd_splits(clip_to_split, clip_to_human, seed=7)
    b = build_qevd_splits(clip_to_split, clip_to_human, seed=7)
    assert a == b
    c = build_qevd_splits(clip_to_split, clip_to_human, seed=99)
    # Different seed should change the partition (with high probability for n=10 humans)
    assert a != c


def test_build_qevd_splits_degrades_gracefully_without_worker_ids():
    """No worker_ids: train clips all land in 'train', val stays empty,
    no exception."""
    clip_to_split = {f"{i:08d}": "train" for i in range(10)}
    clip_to_split.update({f"t{i:07d}": "test" for i in range(3)})
    splits = build_qevd_splits(clip_to_split, clip_to_human={},
                                val_frac=0.09, seed=42)
    assert len(splits["train"]) == 10
    assert len(splits["val"]) == 0
    assert len(splits["in_domain_test"]) == 3
    assert len(splits["unused"]) == 0


def test_build_qevd_splits_unknown_split_value_routes_to_unused():
    clip_to_split = {"a": "train", "b": "test", "c": "validation"}
    clip_to_human = {"a": "w1", "b": "w2", "c": "w3"}
    splits = build_qevd_splits(clip_to_split, clip_to_human)
    assert "c" in splits["unused"]


# ─────────────────────────────────────────────────────────────────────────────
# Section: load_clip_npz / .npz round-trip
# ─────────────────────────────────────────────────────────────────────────────


from pathlib import Path

from backend.training.preprocessing.qevd_dataset import (
    DEFAULT_N_ANGULAR, DEFAULT_N_EXERCISES, DEFAULT_N_JOINT_GROUPS,
    DEFAULT_N_JOINTS, DEFAULT_N_POSE_CHANNELS, DEFAULT_TARGET_FRAMES,
    OTHER_KEY,
    build_exercise_map, build_v6_sample, derive_v6_targets,
    load_clip_npz, load_exercise_map, make_clip_dataset,
    map_exercise_to_idx, save_exercise_map,
)


def _write_synthetic_npz(path: Path, T: int = 150) -> None:
    """Write a synthetic .npz that mimics qevd_extractor output."""
    rng = np.random.default_rng(0)
    pose_canon = rng.standard_normal((T, DEFAULT_N_JOINTS,
                                       DEFAULT_N_POSE_CHANNELS)).astype(np.float32)
    pose_canon[..., 3] = rng.uniform(0.5, 1.0,
                                     size=pose_canon[..., 3].shape)
    angles_raw = rng.uniform(0, 3.0,
                             (T, DEFAULT_N_ANGULAR)).astype(np.float32)
    status = np.zeros(T, dtype=np.uint8)
    np.savez_compressed(
        path,
        pose_canon=pose_canon,
        angles_raw=angles_raw,
        fps_native=np.float32(30.0),
        status_per_frame=status,
        no_pose_fraction=np.float32(0.0),
    )


def test_load_clip_npz_roundtrip(tmp_path):
    p = tmp_path / "00000000.npz"
    _write_synthetic_npz(p, T=120)
    out = load_clip_npz(p)
    assert out["pose_canon"].shape == (120, DEFAULT_N_JOINTS,
                                        DEFAULT_N_POSE_CHANNELS)
    assert out["angles_raw"].shape == (120, DEFAULT_N_ANGULAR)
    assert out["fps_native"] == 30.0
    assert out["status_per_frame"].shape == (120,)
    assert out["no_pose_fraction"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section: exercise_map round-trip
# ─────────────────────────────────────────────────────────────────────────────


class _FakeLabels:
    def __init__(self, fines):
        self.clip_to_fine = fines


def test_build_exercise_map_top_n_plus_other():
    """Top-N most-frequent exercises get dedicated indices, rest -> 'other'."""
    fines = {
        f"clip_{i:04d}": {"exercise": ex}
        for i, ex in enumerate(
            ["squat"] * 100 +
            ["pushup"] * 80 +
            ["plank"] * 60 +
            ["lunge"] * 30
        )
    }
    labels = _FakeLabels(fines)
    m = build_exercise_map(labels, n_named=3)
    assert m["squat"] == 0
    assert m["pushup"] == 1
    assert m["plank"] == 2
    assert m[OTHER_KEY] == 3
    # 'lunge' isn't named; map_exercise_to_idx returns 'other'.
    assert map_exercise_to_idx("lunge", m) == 3


def test_build_exercise_map_default_size():
    """Default n_named = DEFAULT_N_EXERCISES - 1, so output has DEFAULT_N_EXERCISES keys."""
    fines = {f"c{i}": {"exercise": f"ex{i}"} for i in range(40)}
    labels = _FakeLabels(fines)
    m = build_exercise_map(labels)
    assert OTHER_KEY in m
    assert max(m.values()) == DEFAULT_N_EXERCISES - 1
    # n_named names + 'other'
    assert len(m) == DEFAULT_N_EXERCISES


def test_build_exercise_map_fixed_exercises():
    fines = {f"c{i}": {"exercise": "lots_of_squats"} for i in range(100)}
    labels = _FakeLabels(fines)
    m = build_exercise_map(labels, fixed_exercises=["plank", "burpee"], n_named=2)
    assert m["plank"]  == 0
    assert m["burpee"] == 1
    assert m[OTHER_KEY] == 2
    # Even though 'lots_of_squats' is most common, fixed_exercises wins.
    assert "lots_of_squats" not in m


def test_save_load_exercise_map(tmp_path):
    m = {"squat": 0, "pushup": 1, OTHER_KEY: 2}
    p = tmp_path / "ex_map.json"
    save_exercise_map(m, p)
    m2 = load_exercise_map(p)
    assert m == m2


def test_map_exercise_to_idx_unknown_routes_to_other():
    m = {"squat": 0, "pushup": 1, OTHER_KEY: 2}
    assert map_exercise_to_idx("squat",   m) == 0
    assert map_exercise_to_idx("unknown", m) == 2
    assert map_exercise_to_idx(None,      m) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Section: derive_v6_targets
# ─────────────────────────────────────────────────────────────────────────────


def test_derive_v6_targets_shapes():
    ex_map = {"squat": 0, OTHER_KEY: 1}
    feedbacks = ["watch your knees"]
    labels_list = ["squat - rounded back"]
    out = derive_v6_targets(
        feedbacks, labels_list, "squat", ex_map,
        target_frames=64, n_joint_groups=10,
    )
    assert out["quality"].shape == (1,)
    assert out["action"].shape == ()
    assert out["rep_count"].shape == (1,)
    assert out["boundary"].shape == (64, 1)
    assert out["joint_err"].shape == (64, 10)
    assert int(out["action"]) == 0     # 'squat' -> 0


def test_derive_v6_targets_unknown_exercise_routes_to_other():
    ex_map = {"squat": 0, OTHER_KEY: 1}
    out = derive_v6_targets(
        [], [], "novel_exercise_not_in_map", ex_map,
        target_frames=64, n_joint_groups=10,
    )
    assert int(out["action"]) == 1


def test_derive_v6_targets_quality_in_unit_range():
    ex_map = {"squat": 0, OTHER_KEY: 1}
    out = derive_v6_targets(
        ["keep your back straight"],
        ["squat - rounded back"],
        "squat", ex_map,
    )
    q = float(out["quality"][0])
    assert 0.0 <= q <= 1.0


def test_derive_v6_targets_joint_err_tiled_over_time():
    ex_map = {"squat": 0, OTHER_KEY: 1}
    out = derive_v6_targets(
        [], ["squat - rounded back"], "squat", ex_map,
        target_frames=8, n_joint_groups=10,
    )
    je = out["joint_err"]
    # All T frames should be identical (clip-level label tiled)
    for t in range(1, je.shape[0]):
        np.testing.assert_array_equal(je[0], je[t])


# ─────────────────────────────────────────────────────────────────────────────
# Section: build_v6_sample resamples + assembles inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_build_v6_sample_resamples_to_target_frames(tmp_path):
    p = tmp_path / "00000001.npz"
    _write_synthetic_npz(p, T=150)
    clip_data = load_clip_npz(p)
    targets = {
        "quality":   np.array([0.7], dtype=np.float32),
        "action":    np.int32(3),
        "rep_count": np.array([1.0], dtype=np.float32),
        "boundary":  np.zeros((64, 1), dtype=np.float32),
        "joint_err": np.zeros((64, 10), dtype=np.float32),
    }
    inputs, tgts = build_v6_sample(clip_data, targets, target_frames=64)
    assert inputs["pose"].shape   == (64, DEFAULT_N_JOINTS,
                                       DEFAULT_N_POSE_CHANNELS)
    assert inputs["angles"].shape == (64, DEFAULT_N_ANGULAR)
    assert int(inputs["exercise_id"]) == 3   # mirrors action target
    assert tgts["quality"].shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# Section: make_clip_dataset end-to-end (synthetic .npz + fake QEVDLabels)
# ─────────────────────────────────────────────────────────────────────────────


def test_make_clip_dataset_yields_correct_shapes(tmp_path):
    """Spin up a synthetic Drive-like dir, build a tf.data.Dataset, pull
    one batch and verify every shape matches the model's input contract."""
    # Make 3 synthetic .npz files
    npz_dir = tmp_path / "qevd_extracted"
    npz_dir.mkdir()
    clip_ids = ["00000001", "00000002", "00000003"]
    for cid in clip_ids:
        _write_synthetic_npz(npz_dir / f"{cid}.npz", T=120)

    # Fake a QEVDLabels object with just enough surface area
    class _FakeQEVD:
        def lookup(self, cid):
            return {
                "feedbacks":   ["watch your knees"],
                "labels":      ["squat - rounded back"],
                "exercise":    "squat",
                "human_id":    "h1",
                "split":       "train",
            }
    labels = _FakeQEVD()
    ex_map = {"squat": 0, "pushup": 1, OTHER_KEY: 2}

    ds = make_clip_dataset(
        [npz_dir], clip_ids, labels, ex_map,
        target_frames=64, n_joint_groups=10,
        batch_size=2, shuffle=False,
    )
    for inputs, targets in ds.take(1):
        # Inputs
        assert tuple(inputs["pose"].shape)   == (2, 64,
                                                  DEFAULT_N_JOINTS,
                                                  DEFAULT_N_POSE_CHANNELS)
        assert tuple(inputs["angles"].shape) == (2, 64, DEFAULT_N_ANGULAR)
        assert tuple(inputs["exercise_id"].shape) == (2,)
        # Targets
        assert tuple(targets["quality"].shape)   == (2, 1)
        assert tuple(targets["action"].shape)    == (2,)
        assert tuple(targets["rep_count"].shape) == (2, 1)
        assert tuple(targets["boundary"].shape)  == (2, 64, 1)
        assert tuple(targets["joint_err"].shape) == (2, 64, 10)
        break


def test_make_clip_dataset_routes_unknown_exercise_to_other(tmp_path):
    """An exercise name absent from the map should yield action == OTHER idx."""
    npz_dir = tmp_path / "extracted"
    npz_dir.mkdir()
    cid = "00009999"
    _write_synthetic_npz(npz_dir / f"{cid}.npz")

    class _FakeQEVD:
        def lookup(self, cid):
            return {
                "feedbacks": [], "labels": [],
                "exercise":  "weird_unmapped_exercise",
                "human_id":  "h", "split": "train",
            }
    labels = _FakeQEVD()
    ex_map = {"squat": 0, OTHER_KEY: 1}

    ds = make_clip_dataset([npz_dir], [cid], labels, ex_map,
                            batch_size=1, shuffle=False)
    for inputs, targets in ds.take(1):
        # Use .numpy().flat[0] so the test works regardless of whether TF
        # batched scalars to shape (1,) or kept them at shape ().
        action_val = int(np.atleast_1d(targets["action"].numpy()).flat[0])
        ex_id_val  = int(np.atleast_1d(inputs["exercise_id"].numpy()).flat[0])
        assert action_val == ex_map[OTHER_KEY] == 1
        assert ex_id_val  == 1
        break


def test_make_clip_dataset_skips_missing_npz(tmp_path):
    """A clip_id with no .npz on disk should be silently dropped, not crash."""
    npz_dir = tmp_path / "extracted"
    npz_dir.mkdir()
    real_cid = "00000005"
    _write_synthetic_npz(npz_dir / f"{real_cid}.npz")

    class _FakeQEVD:
        def lookup(self, cid):
            return {"feedbacks": [], "labels": [], "exercise": None,
                    "human_id": "h", "split": "train"}
    labels = _FakeQEVD()
    ex_map = {OTHER_KEY: 0}

    # Mix one real id with two fakes that don't exist on disk
    ds = make_clip_dataset(
        [npz_dir],
        clip_ids=[real_cid, "99999998", "99999997"],
        labels=labels, exercise_map=ex_map,
        batch_size=8, shuffle=False,
    )
    n_seen = 0
    for inputs, _ in ds:
        n_seen += int(inputs["pose"].shape[0])
    assert n_seen == 1   # only real_cid produced a sample
