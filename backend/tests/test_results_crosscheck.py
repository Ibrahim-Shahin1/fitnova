"""results.pkl cross-check — the no-fabrication proof.

Covers SRV-05 — live serving confidence reproduces the committed per-clip
results.pkl score for each exercise within decode-parity tolerance
(video ~0.02 cv2-vs-torchvision; Shallow ~0.001 image). Requires staged weights +
extracted benchmark media, so all tests are @pytest.mark.slow and skip cleanly when
weights/media are absent.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from backend.services.clip_decode import decode_clip_cv2, get_frame_count_and_fps
from backend.services.benchmark_catalog import BenchmarkCatalog
from backend.training.aqa.datasets.transforms import uniform_sample_indices

# ─────────────────────────────────────────────────────────────────────────────
# Path constants
# ─────────────────────────────────────────────────────────────────────────────

_AQA_ROOT = Path(
    "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
    "/Fitness-AQA_dataset_release"
)
_RESULTS_DIR = Path("backend/data/benchmark")

_SQUAT_MODEL_DIR = Path("backend/models/form_model_squat_md")
_OHP_MODEL_DIR = Path("backend/models/form_model_ohp_md")
_SHALLOW_MODEL_DIR = Path("backend/models/form_model_shallow_cvcspc")

_SQUAT_PKL = _RESULTS_DIR / "squat/results.pkl"
_OHP_PKL = _RESULTS_DIR / "ohp/results.pkl"
_SHALLOW_PKL = _RESULTS_DIR / "shallow/results.pkl"

_SQUAT_SPLIT = _AQA_ROOT / "Squat/Labeled_Dataset/Splits/test_keys.json"
_SHALLOW_SPLIT = _AQA_ROOT / "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/splits/test_ids.json"
_SQUAT_VIDEOS = _AQA_ROOT / "Squat/Labeled_Dataset/videos_extracted/videos"
_OHP_VIDEOS = _AQA_ROOT / "OHP/Labeled_Dataset/videos_extracted/videos"
_SHALLOW_CROPS = _AQA_ROOT / "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/crops_unaligned"

# ─────────────────────────────────────────────────────────────────────────────
# Skip helpers
# ─────────────────────────────────────────────────────────────────────────────


def _squat_weights_present() -> bool:
    return all(
        (_SQUAT_MODEL_DIR / f"seed{s}/best.pt").exists()
        for s in (42, 1337, 7)
    )


def _ohp_weights_present() -> bool:
    return all(
        (_OHP_MODEL_DIR / f"seed{s}/best.pt").exists()
        for s in (42, 1337)
    )


def _shallow_weights_present() -> bool:
    return all(
        (_SHALLOW_MODEL_DIR / f"seed{s}/best.pt").exists()
        for s in (42, 1337, 7)
    )


def _squat_media_present() -> bool:
    return _SQUAT_VIDEOS.is_dir() and any(_SQUAT_VIDEOS.iterdir())


def _ohp_media_present() -> bool:
    return _OHP_VIDEOS.is_dir() and any(_OHP_VIDEOS.iterdir())


def _shallow_media_present() -> bool:
    # crops_unaligned contains a .gitkeep; need at least one .jpg
    return _SHALLOW_CROPS.is_dir() and any(_SHALLOW_CROPS.glob("*.jpg"))


# ─────────────────────────────────────────────────────────────────────────────
# Video live-infer helper
# ─────────────────────────────────────────────────────────────────────────────


def _infer_video_clip(svc, video_path: Path) -> dict:
    """Decode a clip via the serving path and classify it.

    Reproduces the exact decode recipe used in the upload endpoint:
    get_frame_count_and_fps -> uniform_sample_indices(jitter=0) -> decode_clip_cv2.
    """
    total_frames, _fps = get_frame_count_and_fps(str(video_path))
    idx32 = uniform_sample_indices(total_frames, 32, jitter=0).clamp(0, max(total_frames - 1, 0))
    frames = decode_clip_cv2(str(video_path), idx32)
    return svc.classify_clip(frames.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check tests
# ─────────────────────────────────────────────────────────────────────────────

TOL_VIDEO = 0.02
TOL_IMAGE = 0.001
N_SAMPLE = 5


@pytest.mark.slow
@pytest.mark.skipif(
    not (_squat_weights_present() and _squat_media_present()),
    reason="Squat staged weights or extracted media absent",
)
def test_results_pkl_crosscheck_squat() -> None:
    """Live Squat serving confidence matches results.pkl per-clip score within 0.02."""
    from backend.services.squat_form_service import SquatFormService

    svc = SquatFormService(model_dir=str(_SQUAT_MODEL_DIR))
    assert svc.model_ready, "Squat model did not load — check staged weights"

    split_list: list[str] = json.loads(_SQUAT_SPLIT.read_text())
    pkl_data = pickle.loads(_SQUAT_PKL.read_bytes())
    scores = pkl_data["raw"]["ens_test_scores"]  # (244, 2) float64

    idx_map = {clip_id: i for i, clip_id in enumerate(split_list)}

    sample_ids = split_list[:N_SAMPLE]
    failures: list[str] = []

    for clip_id in sample_ids:
        video_path = _AQA_ROOT / "Squat/Labeled_Dataset/videos_extracted/videos" / f"{clip_id}.mp4"
        result = _infer_video_clip(svc, video_path)

        errors_by_type = {e["type"]: e for e in result["errors"]}
        i = idx_map[clip_id]

        for err_type, col in (("KIE", 0), ("KFE", 1)):
            served = errors_by_type[err_type]["confidence"]
            pkl_score = float(scores[i, col])
            delta = abs(served - pkl_score)
            if delta > TOL_VIDEO:
                failures.append(
                    f"Squat clip={clip_id} {err_type}: served={served:.4f} pkl={pkl_score:.4f} delta={delta:.4f} > {TOL_VIDEO}"
                )

    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_ohp_weights_present() and _ohp_media_present()),
    reason="OHP staged weights or extracted media absent",
)
def test_results_pkl_crosscheck_ohp() -> None:
    """Live OHP serving confidence matches results.pkl per-clip score within 0.02."""
    from backend.services.ohp_form_service import OHPFormService

    svc = OHPFormService(model_dir=str(_OHP_MODEL_DIR))
    assert svc.model_ready, "OHP model did not load — check staged weights"

    pkl_data = pickle.loads(_OHP_PKL.read_bytes())
    # OHP authoritative ordering: test_clip_ids from results.pkl (not test_keys.json)
    split_list: list[str] = pkl_data["test_clip_ids"]
    scores = pkl_data["ensemble_test_scores"]  # (339, 2) float32

    idx_map = {clip_id: i for i, clip_id in enumerate(split_list)}

    sample_ids = split_list[:N_SAMPLE]
    failures: list[str] = []

    for clip_id in sample_ids:
        video_path = _AQA_ROOT / "OHP/Labeled_Dataset/videos_extracted/videos" / f"{clip_id}.mp4"
        result = _infer_video_clip(svc, video_path)

        errors_by_type = {e["type"]: e for e in result["errors"]}
        i = idx_map[clip_id]

        for err_type, col in (("ELBOWS", 0), ("KNEES", 1)):
            served = errors_by_type[err_type]["confidence"]
            pkl_score = float(scores[i, col])
            delta = abs(served - pkl_score)
            if delta > TOL_VIDEO:
                failures.append(
                    f"OHP clip={clip_id} {err_type}: served={served:.4f} pkl={pkl_score:.4f} delta={delta:.4f} > {TOL_VIDEO}"
                )

    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_shallow_weights_present() and _shallow_media_present()),
    reason="Shallow staged weights or extracted crops absent",
)
def test_results_pkl_crosscheck_shallow() -> None:
    """Live Shallow serving confidence matches results.pkl per-clip DEPTH score within 0.001."""
    from backend.services.shallow_squat_form_service import ShallowSquatFormService

    svc = ShallowSquatFormService(model_dir=str(_SHALLOW_MODEL_DIR))
    assert svc.model_ready, "Shallow model did not load — check staged weights"

    split_list: list[str] = json.loads(_SHALLOW_SPLIT.read_text())
    pkl_data = pickle.loads(_SHALLOW_PKL.read_bytes())
    scores = pkl_data["ensemble_test_scores"]  # (540,) float32

    idx_map = {clip_id: i for i, clip_id in enumerate(split_list)}

    sample_ids = split_list[:N_SAMPLE]
    failures: list[str] = []

    for clip_id in sample_ids:
        crop_path = _SHALLOW_CROPS / f"{clip_id}.jpg"
        result = svc.classify_image(str(crop_path))

        depth_err = result["errors"][0]
        assert depth_err["type"] == "DEPTH"
        served = depth_err["confidence"]
        pkl_score = float(scores[idx_map[clip_id]])
        delta = abs(served - pkl_score)
        if delta > TOL_IMAGE:
            failures.append(
                f"Shallow clip={clip_id} DEPTH: served={served:.6f} pkl={pkl_score:.6f} delta={delta:.6f} > {TOL_IMAGE}"
            )

    assert not failures, "\n".join(failures)
