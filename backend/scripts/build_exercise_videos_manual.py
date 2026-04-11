"""
Manual video selection for exercises.
User specifies exact video paths in order; script concatenates them with audio stripped.

Usage:
    python -m backend.scripts.build_exercise_videos_manual
"""

from __future__ import annotations

import json
import os
import subprocess


def _get_ffmpeg() -> str:
    """Return path to ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _concat_no_audio(srcs: list[str], dst: str) -> bool:
    """Concatenate MP4s into one file with no audio track."""
    ffmpeg = _get_ffmpeg()
    list_path = dst + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for s in srcs:
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


# Manual selections: exercise_name -> (slug, [video paths])
MANUAL_VIDEOS = {
    "Barbell Biceps Curl": ("barbell_biceps_curl", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\barbell biceps curl\barbell biceps curl_0.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\barbell biceps curl\barbell biceps curl_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\barbell biceps curl\barbell biceps curl_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\barbell biceps curl\barbell biceps curl_3.mp4",
    ]),
    "Barbell Bench Press": ("bench_press", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\bench press\bench press_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\bench press\bench press_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\bench press\bench press_3.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\bench press\bench press_4.mp4",
    ]),
    "Chest Fly Machine": ("chest_fly_machine", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\chest fly machine\chest fly machine_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\chest fly machine\chest fly machine_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\chest fly machine\chest fly machine_3.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\chest fly machine\chest fly machine_4.mp4",
    ]),
    "Deadlift": ("deadlift", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\deadlift\deadlift_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\deadlift\deadlift_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\deadlift\deadlift_3.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\deadlift\deadlift_4.mp4",
    ]),
    "Hammer Curl": ("hammer_curl", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\hammer curl\hammer curl_3.mp4",
    ]),
    "Hip Thrust": ("hip_thrust", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\hip thrust\hip thrust_5.mp4",
    ]),
    "Incline Bench Press": ("incline_bench_press", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\incline bench press\incline bench press_29.mp4",
    ]),
    "Lat Pulldown": ("lat_pulldown", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lat pulldown\lat pulldown_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lat pulldown\lat pulldown_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lat pulldown\lat pulldown_3.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lat pulldown\lat pulldown_4.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lat pulldown\lat pulldown_5.mp4",
    ]),
    "Lateral Raise": ("lateral_raise", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lateral raise\lateral raise_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lateral raise\lateral raise_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\lateral raise\lateral raise_3.mp4",
    ]),
    "Leg Extension": ("leg_extension", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\leg extension\leg extension_1.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\leg extension\leg extension_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\leg extension\leg extension_3.mp4",
    ]),
    "Leg Raises": ("leg_raises", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\leg raises\leg raises_12.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\leg raises\leg raises_13.mp4",
    ]),
    "Plank": ("plank", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\plank\plank_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\plank\plank_3.mp4",
    ]),
    "Pull up": ("pull_up", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\pull Up\pull Up_21.mp4",
    ]),
    "Push up": ("push_up", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\push-up\push-up_15.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\push-up\push-up_16.mp4",
    ]),
    "Romanian Deadlift": ("romanian_deadlift", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\romanian deadlift\romanian deadlift_11.mp4",
    ]),
    "Russian Twist": ("russian_twist", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\russian twist\russian twist_1.mp4",
    ]),
    "Shoulder Press": ("shoulder_press", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\shoulder press\shoulder press_16.mp4",
    ]),
    "Squat": ("squat", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\squat\squat_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\squat\squat_3.mp4",
    ]),
    "T Bar Row": ("t_bar_row", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\t bar row\t bar row_2.mp4",
    ]),
    "Dips": ("tricep_dips", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\tricep dips\tricep dips_12.mp4",
    ]),
    "Tricep Cable Pushdown": ("tricep_pushdown", [
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\tricep Pushdown\tricep Pushdown_2.mp4",
        r"C:\Users\tsh_x\Desktop\FitNova Drafts2\exercises_library\raw_data\raw_data\data-btc\tricep Pushdown\tricep Pushdown_3.mp4",
    ]),
}


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    static_dir = os.path.join(base_dir, "backend", "static", "exercise_videos")
    output_json = os.path.join(base_dir, "backend", "data", "exercise_videos.json")

    os.makedirs(static_dir, exist_ok=True)

    exercise_data = {}
    name_to_slug = {}

    # Extra aliases per slug — all the name variants the LLM might generate
    ALIASES: dict[str, list[str]] = {
        "barbell_biceps_curl": [
            "barbell biceps curl", "barbell curl", "bicep curl", "biceps curl",
            "standing barbell curl", "ez bar curl", "ez-bar curl", "ez bar bicep curl",
            "barbell bicep curl", "21 curl (barbell)", "21s (ez bar)", "21s",
            "barbell curl (close grip)", "barbell curl (wide grip)",
            "close grip barbell curl", "wide grip barbell curl",
        ],
        "bench_press": [
            "barbell bench press", "bench press", "flat bench press",
            "flat barbell bench press", "bb bench press",
            "smith machine bench press", "smith machine press",
            "chest press", "barbell chest press",
            "3 count pause bench", "pause bench press", "feet up bench press",
            "feet up bench", "floor press (barbell)", "floor press",
            "close grip bench press", "close-grip bench press",
            "close grip barbell bench press", "narrow grip bench press",
        ],
        "chest_fly_machine": [
            "chest fly machine", "pec deck", "machine chest fly",
            "pec deck fly", "machine fly", "cable fly machine",
            "chest fly", "fly", "pec fly",
        ],
        "deadlift": [
            "deadlift", "barbell deadlift", "conventional deadlift",
            "conventional barbell deadlift", "barbell dead lift",
            "halting deadlift", "floating deadlift", "hack deadlift",
            "deficit deadlift", "pause deadlift",
        ],
        "hammer_curl": [
            "hammer curl", "dumbbell hammer curl", "hammer curls",
            "neutral grip curl", "cross body curl", "cross-body hammer curl",
            "db hammer curl", "hammer curl (dumbbell)", "hammer curl dumbbell",
            "hammer curl dumbell",
        ],
        "hip_thrust": [
            "hip thrust", "barbell hip thrust", "glute bridge",
            "barbell glute bridge", "weighted hip thrust", "bb hip thrust",
            "glute bridge (barbell)", "glute bridge (bodyweight)", "glute bridge (dumbbell)",
            "glute bridge (machine)", "glute bridge (band)", "glute bridges",
            "glute bridge hold", "glute bridge smith", "glute bridge - 2 up 1 down",
            "smith machine hip thrust", "machine hip thrust",
            "glute drive (machine)", "glute drive/hip thrust machine",
            "single leg hip thrust", "dumbbell hip thrust",
        ],
        "incline_bench_press": [
            "incline bench press", "incline barbell bench press",
            "incline barbell press", "incline press",
            "incline smith machine press", "incline chest press",
            "incline barbell chest press", "incline barbell bench",
            "high incline press", "45 degree incline db press",
            "45° incline barbell press",
        ],
        "lat_pulldown": [
            "lat pulldown", "cable lat pulldown", "wide grip lat pulldown",
            "lat pull down", "lat pull-down", "close grip lat pulldown",
            "neutral grip lat pulldown", "underhand lat pulldown",
            "reverse grip lat pulldown", "cable pulldown", "pulldown",
            "wide grip pulldown", "cable lat pull down",
            "leverage pulldown", "supinated lat pulldown",
            "pronated lat pulldown", "v-bar lat pulldown",
        ],
        "lateral_raise": [
            "lateral raise", "dumbbell lateral raise", "side raise",
            "side lateral raise", "cable lateral raise", "db lateral raise",
            "lateral raises", "machine lateral raise", "seated lateral raise",
            "dumbbell side raise", "heavy lateral raise",
            "high cable lateral raise",
        ],
        "leg_extension": [
            "leg extension", "machine leg extension", "leg extensions",
            "seated leg extension", "quad extension", "machine quad extension",
            "leg seated leg extension",
        ],
        "leg_raises": [
            "leg raises", "leg raise", "hanging leg raise", "lying leg raise",
            "lying leg raises", "hanging knee raise", "hanging knee raises",
            "captain's chair leg raise", "hanging leg raises", "knee raise",
            "captains chair leg raise", "hanging leg raise (weighted)",
            "hanging oblique knee raise", "toes to bar", "hanging toes to bar",
            "foot to bar",
        ],
        "plank": [
            "plank", "front plank", "forearm plank", "plank hold", "side plank",
            "plank exercise", "active plank", "elevated planks",
        ],
        "pull_up": [
            "pull up", "pull-up", "pullup", "pull ups", "pull-ups", "pullups",
            "weighted pull up", "weighted pull-up", "assisted pull up",
            "wide grip pull up", "neutral grip pull up", "overhand pull up",
            "explosive pull up", "archer pull up",
        ],
        "push_up": [
            "push-up", "pushup", "push up", "push-ups", "pushups", "push ups",
            "standard push up", "chest push up", "hand release push up",
            "hand release push", "elevated push ups", "explosive push-up",
            "explosive push-ups",
        ],
        "romanian_deadlift": [
            "romanian deadlift", "rdl", "barbell rdl", "romanian dl",
            "barbell romanian deadlift", "stiff leg deadlift",
            "stiff-leg deadlift", "barbell stiff leg deadlift",
            "straight leg deadlift", "barbell romanian dead lift",
            "b stance rdl", "b-stance romanian deadlift (dumbbell)",
            "elevated romanian deadlifts",
        ],
        "russian_twist": [
            "russian twist", "russian twists", "weighted russian twist",
            "medicine ball russian twist", "seated russian twist",
        ],
        "shoulder_press": [
            "shoulder press", "overhead press", "ohp", "military press",
            "barbell shoulder press", "barbell overhead press",
            "standing overhead press", "standing barbell press",
            "seated barbell press", "seated overhead press",
            "smith machine shoulder press", "smith machine overhead press",
            "barbell military press", "push press", "barbell push press",
            "arnold press", "arnold press (standing/seated)",
            "dumbbell shoulder press", "dumbbell overhead press",
            "seated dumbbell press", "seated dumbbell shoulder press",
            "db shoulder press", "db overhead press",
            "seated iso shoulder press", "standing landmine shoulder press",
            "half kneeling landmine press", "landmine press",
            "seated smith machine shoulder press",
        ],
        "squat": [
            "squat", "barbell squat", "back squat", "bb squat",
            "barbell back squat", "high bar squat", "low bar squat",
            "pause squat", "safety bar squat", "smith machine squat",
            "free weight squat", "high bar squat (barbell)",
            "high bar squat (barbell, paused)", "paused squat",
            "1-1/4 squat", "anderson squat",
        ],
        "t_bar_row": [
            "t bar row", "t-bar row", "landmine row", "t-bar rows",
            "chest supported t bar row", "t bar rows", "tbar row",
            "landmine t bar row",
            "bent over row", "bent-over row", "barbell bent over row",
            "barbell row", "bb row", "pendlay row", "barbell pendlay row",
            "yates row", "supinated barbell bent over row",
            "dumbbell row", "single arm dumbbell row", "one arm dumbbell row",
            "single arm supported dumbbell row", "db row",
            "cable row", "seated cable row", "seated row", "low cable row",
            "machine row", "close grip cable row", "iso row",
            "iso row (upper back focused)", "meadows row", "helms row",
            "1 arm machine row",
        ],
        "tricep_dips": [
            "dips", "tricep dips", "bench dips", "triceps dips",
            "parallel bar dips", "weighted dips", "bodyweight dips",
            "chest dips", "tricep bench dips", "weighted bench dip",
        ],
        "tricep_pushdown": [
            "tricep cable pushdown", "tricep pushdown", "triceps pushdown",
            "cable pushdown", "cable tricep pushdown", "rope pushdown",
            "tricep rope pushdown", "rope tricep pushdown", "bar pushdown",
            "v-bar pushdown", "cable triceps pushdown", "tricep pull down",
            "tricep pulldown", "triceps cable pushdown",
            "tricep extension", "triceps extension", "cable tricep extension",
            "skull crusher", "skullcrusher", "ez bar skull crusher",
            "ez-bar skull crusher", "lying tricep extension",
            "overhead tricep extension", "overhead triceps extension",
            "seated ez bar overhead tricep extension",
            "ez bar overhead tricep extension",
            "tricep kickback", "dumbbell tricep kickback",
            "close grip bench press tricep", "french press", "french press (barbell)",
            "cable overhead tricep extension", "cable overhead extension",
        ],
    }

    for display_name, (slug, video_paths) in MANUAL_VIDEOS.items():
        # Verify all files exist
        missing = [p for p in video_paths if not os.path.exists(p)]
        if missing:
            print(f"  SKIP {display_name!r} — missing files: {missing}")
            continue

        # Create output directory
        dest_dir = os.path.join(static_dir, slug)
        os.makedirs(dest_dir, exist_ok=True)

        # Remove old demo.mp4
        demo_path = os.path.join(dest_dir, "demo.mp4")
        try:
            os.remove(demo_path)
        except FileNotFoundError:
            pass

        # Concatenate clips with no audio
        ok = _concat_no_audio(video_paths, demo_path)
        if not ok or not os.path.exists(demo_path):
            print(f"  SKIP {display_name!r} — ffmpeg concat failed")
            continue

        demo_kb = os.path.getsize(demo_path) // 1024
        print(f"  {display_name!r} ({len(video_paths)} clips) => demo.mp4 {demo_kb}KB")

        exercise_data[slug] = {
            "display_name": display_name,
            "demo": "demo.mp4",
        }

        # Add display name + all aliases
        for alias in ALIASES.get(slug, [display_name.lower()]):
            name_to_slug[alias.lower()] = slug
        # Fallback: always include the display name itself
        name_to_slug[display_name.lower()] = slug

    # Write JSON
    output = {"_name_to_slug": name_to_slug}
    output.update(exercise_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {output_json}")
    print(f"Exercises: {len(exercise_data)}")


if __name__ == "__main__":
    main()
