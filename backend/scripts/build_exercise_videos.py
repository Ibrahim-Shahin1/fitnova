"""
One-time script: select 4-5 short consecutive video clips per exercise from
the BTC dataset, copy them into backend/static/exercise_videos/, and generate
backend/data/exercise_videos.json with a name→slug lookup map.

Usage:
    python -m backend.scripts.build_exercise_videos \
        --source "C:\\Users\\tsh_x\\Desktop\\FitNova Drafts2\\exercises_library\\raw_data\\raw_data\\data-btc"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess


def _get_ffmpeg() -> str:
    """Return path to ffmpeg — prefers system ffmpeg, falls back to imageio-ffmpeg bundle."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _concat_no_audio(srcs: list[str], dst: str) -> bool:
    """
    Concatenate multiple MP4s into one file with no audio track.
    Uses ffmpeg concat demuxer — no re-encode, video stream copied as-is.
    Returns True on success.
    """
    ffmpeg = _get_ffmpeg()
    # Write a temp concat list
    list_path = dst + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for s in srcs:
            # ffmpeg concat list requires forward slashes and escaped single quotes
            safe = s.replace("\\", "/").replace("'", "\\'")
            f.write(f"file '{safe}'\n")
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-an", "-c:v", "copy", dst],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
_STATIC_DIR = os.path.join(_BASE_DIR, "backend", "static", "exercise_videos")
_OUTPUT     = os.path.join(_BASE_DIR, "backend", "data", "exercise_videos.json")
_CATALOG    = os.path.join(_BASE_DIR, "backend", "data", "program_catalog.pkl")

# Max clip file size to consider as a "short" clip (bytes)
_MAX_CLIP_SIZE = 2 * 1024 * 1024  # 2 MB

# Desired number of clips per exercise
_TARGET_CLIPS = 5

# ── Manual name aliases ────────────────────────────────────────────────────────
# Maps dataset slug → list of lowercase exercise name variants the LLM might generate

MANUAL_ALIASES: dict[str, list[str]] = {
    # Only genuine name variants of the SAME exercise (same movement, same equipment type)
    "bench_press": [
        "bench press", "barbell bench press", "flat bench press",
        "flat barbell bench press", "bb bench press",
        "smith machine bench press", "smith machine press",
    ],
    "incline_bench_press": [
        "incline bench press", "incline barbell bench press",
        "incline press", "incline barbell press",
        "incline smith machine press", "incline smith machine bench press",
    ],
    "decline_bench_press": [
        "decline bench press", "decline barbell bench press", "decline press",
        "decline barbell press",
    ],
    "squat": [
        "squat", "barbell squat", "back squat", "bb squat",
        "barbell back squat", "high bar squat", "low bar squat",
        "pause squat", "safety bar squat", "smith machine squat",
    ],
    "deadlift": [
        "deadlift", "barbell deadlift", "conventional deadlift",
        "conventional barbell deadlift",
    ],
    "romanian_deadlift": [
        "romanian deadlift", "rdl", "barbell rdl", "romanian dl",
        "barbell romanian deadlift", "stiff leg deadlift",
        "stiff-leg deadlift", "barbell stiff leg deadlift",
    ],
    "lat_pulldown": [
        "lat pulldown", "cable lat pulldown", "wide grip lat pulldown",
        "wide grip pulldown", "lat pull down", "lat pull-down",
        "close grip lat pulldown", "neutral grip lat pulldown",
        "underhand lat pulldown", "reverse grip lat pulldown",
        "cable pulldown",
    ],
    "pull_up": [
        "pull up", "pull-up", "pullup", "pull ups", "pull-ups",
        "weighted pull up", "weighted pull-up", "assisted pull up",
        "wide grip pull up", "neutral grip pull up",
    ],
    "push_up": [
        "push-up", "pushup", "push up", "push-ups", "pushups", "push ups",
    ],
    "shoulder_press": [
        "shoulder press", "overhead press", "ohp", "military press",
        "barbell shoulder press", "barbell overhead press",
        "standing overhead press", "standing barbell press",
        "seated barbell press", "seated overhead press",
        "smith machine shoulder press", "smith machine overhead press",
    ],
    "lateral_raise": [
        "lateral raise", "dumbbell lateral raise", "side raise",
        "side lateral raise", "cable lateral raise", "db lateral raise",
        "lateral raises", "machine lateral raise", "cable side raise",
        "seated lateral raise",
    ],
    "barbell_biceps_curl": [
        "barbell biceps curl", "barbell curl", "biceps curl",
        "standing barbell curl", "ez bar curl", "ez-bar curl",
        "ez bar bicep curl", "barbell curl close grip", "barbell curl wide grip",
    ],
    "hammer_curl": [
        "hammer curl", "dumbbell hammer curl", "hammer curls",
        "cross body curl", "cross-body hammer curl",
    ],
    "tricep_pushdown": [
        "tricep pushdown", "triceps pushdown", "cable pushdown",
        "cable tricep pushdown", "rope pushdown", "tricep cable pushdown",
        "rope tricep pushdown", "bar pushdown", "v-bar pushdown",
    ],
    "tricep_dips": [
        "tricep dips", "dips", "bench dips", "triceps dips",
        "parallel bar dips", "weighted dips",
    ],
    "leg_extension": [
        "leg extension", "machine leg extension", "leg extensions",
        "seated leg extension",
    ],
    "leg_raises": [
        "leg raises", "leg raise", "hanging leg raise", "lying leg raise",
        "lying leg raises", "hanging knee raise", "hanging knee raises",
        "captain's chair leg raise", "hanging leg raises", "knee raise",
    ],
    "hip_thrust": [
        "hip thrust", "barbell hip thrust", "glute bridge",
        "barbell glute bridge", "weighted hip thrust",
    ],
    "chest_fly_machine": [
        "chest fly machine", "pec deck", "machine chest fly",
        "pec deck fly", "machine fly",
    ],
    "plank": [
        "plank", "front plank", "forearm plank", "plank hold",
        "side plank",
    ],
    "russian_twist": [
        "russian twist", "russian twists", "weighted russian twist",
        "medicine ball russian twist",
    ],
    "t_bar_row": [
        "t bar row", "t-bar row", "landmine row", "t-bar rows",
        "chest supported t bar row",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(folder_name: str) -> str:
    """Convert folder name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")


def _numeric_index(filename: str) -> int:
    m = re.search(r"_(\d+)\.mp4$", filename, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def select_clips(folder_path: str, exercise_name: str) -> list[str]:
    """Return the first _TARGET_CLIPS .mp4 files sorted by numeric index."""
    all_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")],
        key=_numeric_index,
    )
    return all_files[:_TARGET_CLIPS]


def build_name_to_slug(slugs: list[str]) -> dict[str, str]:
    """
    Build lowercase-name → slug mapping from manual aliases.
    Then fuzzy-match catalog exercise names against the 22 dataset exercises.
    """
    mapping: dict[str, str] = {}

    # Manual aliases
    for slug, aliases in MANUAL_ALIASES.items():
        if slug in slugs:
            for alias in aliases:
                mapping[alias.lower()] = slug

    # Fuzzy-match catalog names
    try:
        from rapidfuzz import fuzz, process  # type: ignore

        with open(_CATALOG, "rb") as f:
            catalog = pickle.load(f)

        catalog_names: set[str] = set()
        for prog in catalog.values():
            for ex in prog.get("week1_exercises", []):
                n = str(ex.get("exercise_name", "")).strip()
                if n:
                    catalog_names.add(n)

        # Build target list: slug → representative name (folder name)
        slug_display = {s: s.replace("_", " ") for s in slugs}

        for cat_name in catalog_names:
            cat_lower = cat_name.lower().strip()
            if cat_lower in mapping:
                continue  # already covered by manual aliases

            # Find best matching slug
            best_slug = None
            best_score = 0.0
            for slug, display in slug_display.items():
                score = fuzz.token_sort_ratio(cat_lower, display)
                if score > best_score:
                    best_score = score
                    best_slug = slug

            if best_score >= 82 and best_slug:
                # Word overlap guard
                def words(s: str) -> set:
                    stop = {"the","a","an","with","and","or","to","from","on",
                            "in","at","single","double","one","two","three"}
                    return set(re.sub(r"[^a-z0-9 ]", " ", s).split()) - stop

                if words(cat_lower) & words(slug_display[best_slug]):
                    mapping[cat_lower] = best_slug

        print(f"  Name-to-slug map: {len(mapping)} entries")

    except ImportError:
        print("  rapidfuzz not available — skipping catalog fuzzy match")

    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True,
        help="Path to the data-btc directory containing exercise folders",
    )
    args = parser.parse_args()

    source = args.source
    if not os.path.isdir(source):
        print(f"ERROR: source directory not found: {source}")
        return

    # Discover exercise folders
    folders = sorted([
        d for d in os.listdir(source)
        if os.path.isdir(os.path.join(source, d))
    ])
    print(f"Found {len(folders)} exercise folders")

    os.makedirs(_STATIC_DIR, exist_ok=True)

    exercise_data: dict[str, dict] = {}
    slugs: list[str] = []

    for folder in folders:
        folder_path = os.path.join(source, folder)
        slug = _slug(folder)
        slugs.append(slug)

        clips = select_clips(folder_path, folder)
        if not clips:
            print(f"  SKIP {folder!r} — no clips found")
            continue

        dest_dir = os.path.join(_STATIC_DIR, slug)
        os.makedirs(dest_dir, exist_ok=True)

        # Remove old files
        for old in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, old))

        # Concatenate all clips into a single demo.mp4 (no audio)
        src_paths = [os.path.join(folder_path, c) for c in clips]
        demo_path = os.path.join(dest_dir, "demo.mp4")
        ok = _concat_no_audio(src_paths, demo_path)

        if not ok or not os.path.exists(demo_path):
            print(f"  SKIP {folder!r} — ffmpeg concat failed")
            continue

        sizes_kb = [os.path.getsize(p) // 1024 for p in src_paths]
        demo_kb  = os.path.getsize(demo_path) // 1024
        print(f"  {folder!r} -> {slug!r}: {len(clips)} clips concatenated "
              f"({', '.join(str(s)+'KB' for s in sizes_kb)}) => demo.mp4 {demo_kb}KB")

        exercise_data[slug] = {
            "display_name": folder,
            "demo": "demo.mp4",
        }

    # Build name-to-slug map
    print("\nBuilding name-to-slug mapping...")
    name_to_slug = build_name_to_slug(slugs)

    # Combine and write output
    output: dict = {"_name_to_slug": name_to_slug}
    output.update(exercise_data)

    with open(_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {_OUTPUT}")
    print(f"Static dir: {_STATIC_DIR}")
    print(f"Exercises with videos: {len(exercise_data)}")


if __name__ == "__main__":
    main()
