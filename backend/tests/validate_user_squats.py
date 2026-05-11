"""Phase-F user-recorded validation.

Runs two user-recorded videos through the full FormSession pipeline (same
path as the `/analyze-form-video` endpoint) and reports the quality delta
and joint-error breakdown.

Pass criterion (from plan GF):
    mean(quality_bad) <= mean(quality_good) - 0.15
    AND joint-error breakdown highlights knees and/or back for the bad reps.

Usage:
    python backend/tests/validate_user_squats.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Dummy key — we stub _generate_llm_feedback, so no call is ever made,
# but the OpenAI() constructor still raises if no key is present.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-validation")

from backend.services.form_analyzer import FormAnalyzer  # noqa: E402
from backend.services.form_session  import FormSession, JOINT_GROUP_NAMES  # noqa: E402
from backend.config.exercises       import relevant_joint_indices  # noqa: E402


MODEL_DIR = str(REPO_ROOT / "backend" / "models" / "form_model_v5_2")
VIDEOS = {
    "good_side":   r"C:\Users\tsh_x\Downloads\Good_Squats.mp4",
    "bad_side":    r"C:\Users\tsh_x\Downloads\Bad_Squats.mp4",
    "good_angled": r"C:\Users\tsh_x\Downloads\Good_Squats2.mp4",
    "bad_angled":  r"C:\Users\tsh_x\Downloads\Bad_Squats2.mp4",
}
SELECTED_EXERCISE = "squat"
RELEVANT_IDX = relevant_joint_indices(SELECTED_EXERCISE)  # [4,5,6,7,8]
RELEVANT_NAMES = [JOINT_GROUP_NAMES[i] for i in RELEVANT_IDX]
REPORT_PATH = REPO_ROOT / "backend" / "models" / "form_model" / "reality_check" / "user_squat_validation.json"


def _run_video(analyzer: FormAnalyzer, video_path: str, label: str) -> dict:
    """Stream one video through a FormSession and return the session summary."""
    session = FormSession(analyzer=analyzer, selected_exercise=SELECTED_EXERCISE)
    # Stub out the LLM — we only need numbers for validation
    session._generate_llm_feedback = lambda *a, **k: "(skipped for validation)"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    skip = max(1, int(fps / 10))  # sample ~10 fps — matches /analyze-form-video

    per_frame_quality: list[float] = []
    per_frame_joint_err: list[list[float]] = []
    per_frame_mismatch_fired = False

    frame_idx = 0
    started = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            ts_ms = int((frame_idx / fps) * 1000)
            out = session.add_frame(jpeg.tobytes(), ts_ms)
            if out.get("status") in ("ok", "pose_interpolated"):
                per_frame_quality.append(float(out["quality_score"]))
                per_frame_joint_err.append(list(out["joint_errors"]))
            if out.get("mismatch_warning") is not None:
                per_frame_mismatch_fired = True
        frame_idx += 1
    cap.release()

    elapsed = time.time() - started
    summary = session.end_session()

    # Attach raw per-frame stats
    arr_q = np.array(per_frame_quality, dtype=np.float32) if per_frame_quality else np.array([])
    arr_j = np.array(per_frame_joint_err, dtype=np.float32) if per_frame_joint_err else np.zeros((0, 10))

    return {
        "label":                label,
        "video":                os.path.basename(video_path),
        "frames_total":         total,
        "frames_sampled":       frame_idx // skip,
        "frames_with_pose":     int(arr_q.shape[0]),
        "fps_source":           float(fps),
        "elapsed_sec":          round(elapsed, 2),
        "ms_per_sampled_frame": round(1000.0 * elapsed / max(1, frame_idx // skip), 1),
        "summary":              summary,
        "per_frame_quality": {
            "mean":   float(arr_q.mean())   if arr_q.size else None,
            "median": float(np.median(arr_q)) if arr_q.size else None,
            "p10":    float(np.percentile(arr_q, 10)) if arr_q.size else None,
            "p90":    float(np.percentile(arr_q, 90)) if arr_q.size else None,
        },
        "per_frame_joint_err_mean": {
            JOINT_GROUP_NAMES[i]: float(arr_j[:, i].mean()) if arr_j.size else 0.0
            for i in range(10)
        },
        "mismatch_warning_fired": per_frame_mismatch_fired,
    }


def main():
    print(f"[load] FormAnalyzer({MODEL_DIR})")
    analyzer = FormAnalyzer(model_dir=MODEL_DIR)

    reports: dict[str, dict] = {}
    for label, path in VIDEOS.items():
        print(f"\n[run ] {label:<4s}  {path}")
        reports[label] = _run_video(analyzer, path, label)

    # Pair each "good" with its "bad" counterpart by camera view.
    pairs = [("good_side", "bad_side"), ("good_angled", "bad_angled")]
    verdicts = {}

    print("\n" + "=" * 78)
    print("PHASE-F USER SQUAT VALIDATION  (locked exercise: squat; "
          f"relevant joints: {RELEVANT_NAMES})")
    print("=" * 78)

    for label, r in reports.items():
        s = r["summary"]
        print(f"\n[{label.upper():<11s}]  {r['video']}")
        print(f"  reps detected        : {s.get('total_reps')}")
        print(f"  per-rep scores       : {s.get('per_rep_scores')}")
        print(f"  avg_quality (summary): {s.get('average_quality')}")
        print(f"  per-frame quality    : mean={r['per_frame_quality']['mean']}  "
              f"median={r['per_frame_quality']['median']}")
        print(f"  shown exercise       : {s.get('exercise')}  "
              f"[raw classifier guess: {s.get('predicted_exercise')}  "
              f"mismatch flag: {s.get('exercise_mismatch')}]")
        print(f"  mismatch_warning fired during stream: {r['mismatch_warning_fired']}")
        print(f"  common_errors (masked to squat-relevant joints): {s.get('common_errors')}")
        # Relevant-joint per-frame means
        rel_means = {JOINT_GROUP_NAMES[i]: round(r["per_frame_joint_err_mean"][JOINT_GROUP_NAMES[i]], 3)
                     for i in RELEVANT_IDX}
        print(f"  per-frame relevant joint errors: {rel_means}")
        print(f"  frames w/ pose       : {r['frames_with_pose']}/{r['frames_sampled']}")
        print(f"  wall time            : {r['elapsed_sec']}s  "
              f"({r['ms_per_sampled_frame']} ms/sampled-frame)")

    for good_key, bad_key in pairs:
        good = reports[good_key]
        bad  = reports[bad_key]

        # Quality delta
        good_rep_scores = good["summary"].get("per_rep_scores") or []
        bad_rep_scores  = bad ["summary"].get("per_rep_scores") or []
        use_rep_level = len(good_rep_scores) >= 3 and len(bad_rep_scores) >= 3

        if use_rep_level:
            mean_good = float(np.mean(good_rep_scores))
            mean_bad  = float(np.mean(bad_rep_scores))
            granularity = "per-rep"
        else:
            mean_good = good["per_frame_quality"]["mean"] or 0.0
            mean_bad  = bad ["per_frame_quality"]["mean"] or 0.0
            granularity = "per-frame (insufficient reps detected)"

        delta = mean_good - mean_bad
        quality_pass = delta >= 0.15

        # Relevant joints only — knees / hips / back for squat
        diff_errs = {}
        for i in RELEVANT_IDX:
            name = JOINT_GROUP_NAMES[i]
            diff_errs[name] = round(
                bad["per_frame_joint_err_mean"][name]
                - good["per_frame_joint_err_mean"][name],
                3,
            )
        top_diffs = sorted(diff_errs.items(), key=lambda x: -x[1])
        bad_lit_joints = [k for k, v in diff_errs.items() if v > 0.05]
        joint_pass = len(bad_lit_joints) > 0

        verdict = {
            "pair":              (good_key, bad_key),
            "granularity":       granularity,
            "mean_quality_good": round(mean_good, 4),
            "mean_quality_bad":  round(mean_bad,  4),
            "delta":             round(delta, 4),
            "delta_threshold":   0.15,
            "quality_pass":      bool(quality_pass),
            "joint_deltas":      top_diffs,
            "bad_lit_joints":    bad_lit_joints,
            "joint_pass":        bool(joint_pass),
            "overall_pass":      bool(quality_pass and joint_pass),
        }
        verdicts[f"{good_key}_vs_{bad_key}"] = verdict

        print("\n" + "-" * 78)
        print(f"VERDICT  {good_key}  vs  {bad_key}   ({granularity})")
        print(f"  mean(good) = {verdict['mean_quality_good']}")
        print(f"  mean(bad)  = {verdict['mean_quality_bad']}")
        print(f"  delta      = {verdict['delta']}   (need >= 0.15)")
        print(f"  quality_pass = {verdict['quality_pass']}")
        print(f"  joint deltas (bad - good) for squat-relevant joints:")
        for name, d in top_diffs:
            marker = "*" if d > 0.05 else " "
            print(f"      {marker} {name:<15s}  {d:+.3f}")
        print(f"  bad-lit joints (>0.05): {bad_lit_joints}")
        print(f"  joint_pass (any squat-relevant joint lit) = {verdict['joint_pass']}")
        print(f"  OVERALL PASS = {verdict['overall_pass']}")
    print("=" * 78)

    # Persist
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"verdicts": verdicts, "reports": reports}
    # strip non-serialisable items (none expected here)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[save] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
