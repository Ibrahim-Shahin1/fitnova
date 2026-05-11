"""Unit tests for backend.training.train_form_model_v6.

Uses synthetic .npz + fake QEVDLabels to validate the training loop runs
end-to-end (build -> compile -> 2 fit epochs -> save) without needing
real QEVD data. Fast (~30 s).

Run from repo root:
    pytest backend/training/test_train_form_model_v6.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: synthetic dataset on disk
# ─────────────────────────────────────────────────────────────────────────────


def _write_synthetic_npz(path: Path, T: int = 120) -> None:
    rng = np.random.default_rng(int(path.stem) if path.stem.isdigit() else 0)
    pose_canon = rng.standard_normal((T, 15, 4)).astype(np.float32)
    pose_canon[..., 3] = rng.uniform(0.5, 1.0,
                                     size=pose_canon[..., 3].shape).astype(np.float32)
    angles_raw = rng.uniform(0, 3.0, (T, 22)).astype(np.float32)
    np.savez_compressed(
        path,
        pose_canon=pose_canon,
        angles_raw=angles_raw,
        fps_native=np.float32(30.0),
        status_per_frame=np.zeros(T, dtype=np.uint8),
        no_pose_fraction=np.float32(0.0),
    )


def _write_synthetic_labels(out_dir: Path, n_clips: int) -> None:
    """Create minimal feedbacks + fine_grained + worker_ids JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exercises = ["squat", "pushup", "plank", "lunge", "burpee"]
    feedbacks_data = []
    fine_data = []
    worker_data = []
    for i in range(n_clips):
        cid = f"{i:08d}"
        ex = exercises[i % len(exercises)]
        feedbacks_data.append({
            "video_path": f"./{cid}.mp4",
            "feedbacks":  ["watch your form"] if i % 3 == 0 else [],
            "split":      "train" if i % 10 != 0 else "test",
        })
        fine_data.append({
            "video_path": f"./{cid}.mp4",
            "labels":     [f"{ex} - average"],
            "labels_descriptive": [f"{ex} described"],
            "split":      "train" if i % 10 != 0 else "test",
        })
        # Each "human" gets ~5 clips so we have enough humans for split logic
        worker_data.append({
            "video_path": f"./{cid}.mp4",
            "worker_id":  i // 5,
        })
    (out_dir / "feedbacks_short_clips.json").write_text(json.dumps(feedbacks_data))
    (out_dir / "fine_grained_labels.json").write_text(json.dumps(fine_data))
    (out_dir / "fine_grained_labels_with_worker_ids.json").write_text(
        json.dumps(worker_data)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit-level: loss function returns sensible numbers
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_task_loss_shapes_and_finite():
    """Each per-head loss is a finite scalar; total is positive."""
    import tensorflow as tf
    from backend.training.train_form_model_v6 import _multi_task_loss

    B, T = 2, 64
    outputs = {
        "quality":   tf.constant(np.random.uniform(0, 1, (B, 1)).astype(np.float32)),
        "joint_err": tf.constant(np.random.uniform(0, 1, (B, T, 10)).astype(np.float32)),
        "boundary":  tf.constant(np.random.uniform(0, 1, (B, T, 1)).astype(np.float32)),
        "rep_count": tf.constant(np.random.uniform(0, 5, (B, 1)).astype(np.float32)),
        "action":    tf.constant(np.random.uniform(0, 1, (B, 25)).astype(np.float32)),
    }
    targets = {
        "quality":   tf.constant(np.array([[0.7], [0.4]], dtype=np.float32)),
        "joint_err": tf.constant(np.zeros((B, T, 10), dtype=np.float32)),
        "boundary":  tf.constant(np.zeros((B, T, 1), dtype=np.float32)),
        "rep_count": tf.constant(np.array([[2.0], [3.0]], dtype=np.float32)),
        "action":    tf.constant(np.array([0, 5], dtype=np.int32)),
    }
    total, parts = _multi_task_loss(outputs, targets)
    total = float(total.numpy())
    assert np.isfinite(total)
    assert total > 0
    for name, val in parts.items():
        v = float(val.numpy())
        assert np.isfinite(v), f"{name} = {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: launch the training script via CLI on synthetic data
# ─────────────────────────────────────────────────────────────────────────────


def test_train_form_model_v6_runs_smoke(tmp_path):
    """Fire `python train_form_model_v6.py` on synthetic data for 2 epochs.
    Verifies: script runs to completion, weights saved, history written, all
    five head losses appear in history, training loss tracked across epochs.
    """
    npz_dir   = tmp_path / "npz"
    npz_dir.mkdir()
    n_clips = 30
    for i in range(n_clips):
        _write_synthetic_npz(npz_dir / f"{i:08d}.npz", T=80)

    labels_dir = tmp_path / "labels"
    _write_synthetic_labels(labels_dir, n_clips)

    out_dir = tmp_path / "models"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend/training/train_form_model_v6.py"),
        "--npz-dirs",   str(npz_dir),
        "--labels-dir", str(labels_dir),
        "--out-dir",    str(out_dir),
        "--epochs",     "2",
        "--batch-size", "4",
        "--target-frames", "32",   # smaller for speed
        "--warmup-epochs", "1",
        "--val-frac",   "0.2",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logspam

    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=300)
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        pytest.fail(f"train script failed with exit code {result.returncode}")

    # ── Verify outputs ────────────────────────────────────────────────────
    final_weights = out_dir / "v6_supervised.weights.h5"
    history_path  = out_dir / "v6_history.json"
    config_path   = out_dir / "model_config.json"
    ex_map_path   = out_dir / "qevd_exercise_map.json"

    assert final_weights.exists(), f"weights missing: {final_weights}"
    assert history_path.exists(),  f"history missing: {history_path}"
    assert config_path.exists(),   f"config missing: {config_path}"
    assert ex_map_path.exists(),   f"exercise map missing: {ex_map_path}"

    # ── History contains all 5 head losses + total + val_* mirrors ────────
    history = json.loads(history_path.read_text())
    expected_keys = {"loss", "l_quality", "l_joint_err",
                     "l_boundary", "l_rep_count", "l_action"}
    assert expected_keys.issubset(set(history.keys())), (
        f"missing keys in history: {expected_keys - set(history.keys())}"
    )
    # 2 epochs => each list has length 2
    for k in expected_keys:
        assert len(history[k]) == 2, (k, history[k])
        for v in history[k]:
            assert np.isfinite(v), (k, history[k])

    # ── Model config has the v6 contract ──────────────────────────────────
    cfg = json.loads(config_path.read_text())
    assert cfg["target_frames"]    == 32
    assert cfg["n_exercises"]      == 25
    assert cfg["n_joint_groups"]   == 10
    assert cfg["version"]          == "v6"

    # ── Exercise map has 25 entries ───────────────────────────────────────
    ex_map = json.loads(ex_map_path.read_text())
    assert "__other__" in ex_map
    # Synthetic data only has 5 exercises so the rest are 'other'.
    assert ex_map["__other__"] == 24


def test_train_form_model_v6_resumes_from_checkpoint(tmp_path):
    """Run 1 epoch, then re-run with epochs=2; second run should pick up
    from the saved checkpoint and only train 1 more epoch."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    for i in range(20):
        _write_synthetic_npz(npz_dir / f"{i:08d}.npz", T=80)
    labels_dir = tmp_path / "labels"
    _write_synthetic_labels(labels_dir, 20)
    out_dir = tmp_path / "models"

    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "backend/training/train_form_model_v6.py"),
        "--npz-dirs", str(npz_dir),
        "--labels-dir", str(labels_dir),
        "--out-dir", str(out_dir),
        "--batch-size", "4",
        "--target-frames", "32",
        "--warmup-epochs", "1",
        "--val-frac", "0.2",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Round 1: epochs=1
    r1 = subprocess.run(
        base_cmd + ["--epochs", "1"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert r1.returncode == 0, r1.stderr
    h1 = json.loads((out_dir / "v6_history.json").read_text())
    assert len(h1["loss"]) == 1

    # Round 2: epochs=2 — should resume from epoch_01.weights.h5 and append 1 row
    r2 = subprocess.run(
        base_cmd + ["--epochs", "2"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert r2.returncode == 0, r2.stderr
    assert "resuming from epoch 1" in r2.stdout, r2.stdout
    h2 = json.loads((out_dir / "v6_history.json").read_text())
    assert len(h2["loss"]) == 2, h2
