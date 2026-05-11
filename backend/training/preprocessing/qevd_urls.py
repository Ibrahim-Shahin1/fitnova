"""All Qualcomm-hosted download URLs for the QEVD dataset.

Verified publicly fetchable (HTTP 206 + correct content type) on
2026-05-06. No authentication, no signed query parameters, no expiry —
these URLs are stable as long as Qualcomm keeps the dataset published.

Used by:
- D2 (MediaPipe extraction): all FIT-300K Parts 1-4
- D3+ (label sanity & dataset assembly): FIT-COACH + worker_ids
- D8 (OOD evaluation): FIT-COACH-Benchmark

Single source of truth — every notebook imports from here.
No URL substitution / pattern-matching anywhere; every URL is explicit.
"""

from __future__ import annotations

# ─── FIT-300K (4 parts × 3 multivolume files = 12 URLs) ────────────────────
QEVD_FIT_300K_URLS: dict[int, dict[str, str]] = {
    1: {
        "z01": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-1/QEVD-FIT-300k-Part-1.z01",
        "z02": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-1/QEVD-FIT-300k-Part-1.z02",
        "zip": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-1/QEVD-FIT-300k-Part-1.zip",
    },
    2: {
        "z01": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-2/QEVD-FIT-300k-Part-2.z01",
        "z02": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-2/QEVD-FIT-300k-Part-2.z02",
        "zip": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-2/QEVD-FIT-300k-Part-2.zip",
    },
    3: {
        "z01": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-3/QEVD-FIT-300k-Part-3.z01",
        "z02": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-3/QEVD-FIT-300k-Part-3.z02",
        "zip": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-3/QEVD-FIT-300k-Part-3.zip",
    },
    4: {
        "z01": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-4/QEVD-FIT-300k-Part-4.z01",
        "z02": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-4/QEVD-FIT-300k-Part-4.z02",
        "zip": "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Part-4/QEVD-FIT-300k-Part-4.zip",
    },
}

# ─── FIT-COACH labels & long-range training videos (1 zip, ~4.6 GB) ─────────
# Contains: fine_grained_labels.json, feedbacks_short_clips.json,
#           feedbacks_long_range.json, questions.json,
#           long_range_videos/ (149 train MP4s + their *_timestamps.npy)
QEVD_FIT_COACH_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
    "AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-COACH/"
    "QEVD-FIT-COACH.zip"
)

# ─── FIT-COACH BENCHMARK (sealed test set, 1 zip, ~2 GB) ────────────────────
# Contains: feedbacks_long_range.json for the 74 BENCHMARK long-range videos
# (the official QEVD test set used in the paper's OOD evaluation).
# This is paper Table 2's "QEVD-FIT-COACH Test" column (7 subjects, 74 vids).
# Required for D8 (OOD evaluation per master plan §II.4.5).
QEVD_FIT_COACH_BENCHMARK_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
    "AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-COACH-Benchmark/"
    "QEVD-FIT-COACH-Benchmark.zip"
)

# ─── Additional Labels (1 JSON, ~20 MB) ─────────────────────────────────────
# Maps clip_id -> worker_id (= participant ID, verified 2026-05-06).
# Required for subject-disjoint val splits.
QEVD_WORKER_IDS_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
    "AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-300k-Additional-Labels/"
    "fine_grained_labels_with_worker_ids.json"
)


def get_part_urls(part_number: int) -> list[str]:
    """Return the 3 URLs for Part N in [.z01, .z02, .zip] order."""
    if part_number not in QEVD_FIT_300K_URLS:
        raise ValueError(
            f"part_number must be 1, 2, 3, or 4 — got {part_number!r}"
        )
    p = QEVD_FIT_300K_URLS[part_number]
    return [p["z01"], p["z02"], p["zip"]]


def all_urls() -> dict[str, str]:
    """Flat dict of every URL (for sanity / documentation)."""
    out: dict[str, str] = {}
    for n, urls in QEVD_FIT_300K_URLS.items():
        for ext, url in urls.items():
            out[f"FIT-300K_Part-{n}.{ext}"] = url
    out["FIT-COACH.zip"]                    = QEVD_FIT_COACH_URL
    out["FIT-COACH-Benchmark.zip"]          = QEVD_FIT_COACH_BENCHMARK_URL
    out["fine_grained_labels_with_worker_ids.json"] = QEVD_WORKER_IDS_URL
    return out


__all__ = [
    "QEVD_FIT_300K_URLS",
    "QEVD_FIT_COACH_URL",
    "QEVD_FIT_COACH_BENCHMARK_URL",
    "QEVD_WORKER_IDS_URL",
    "get_part_urls",
    "all_urls",
]
