"""Unit tests for backend.training.preprocessing.qevd_label_builder.

Distribution (master plan section II.4.4 + adversarial cases):
  Corrective ......... 8
  Affirmative ........ 5
  Acknowledgment ..... 4
  Informative ........ 3
  Bilateral group .... 4
  Side-specific ...... 3
  Edge cases ......... 3
  Word-boundary adv .. 4
  -------------------------
  Total fixtures ..... 34

Run from repo root:
    pytest backend/training/preprocessing/test_qevd_label_builder.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.training.preprocessing.qevd_label_builder import (
    BILATERAL_KEYWORDS,
    GROUP_KEYWORDS,
    JOINT_GROUP_NAMES,
    JOINT_GROUP_TO_IDX,
    N_JOINT_GROUPS,
    Q_DEFAULT_NO_FEEDBACK,
    build_clip_label,
    classify_feedback_type,
    derive_joint_groups,
    derive_quality_scalar,
)

# Convenience: index lookup by group name
_IDX = JOINT_GROUP_TO_IDX
LK, RK = _IDX["L_knee"], _IDX["R_knee"]
LE, RE = _IDX["L_elbow"], _IDX["R_elbow"]
LS, RS = _IDX["L_shoulder"], _IDX["R_shoulder"]
LH, RH = _IDX["L_hip"], _IDX["R_hip"]
TR = _IDX["Trunk"]
NK = _IDX["Neck"]


def _groups_set(arr: np.ndarray) -> set[int]:
    """Convert a (10,) bool array to the set of fired indices."""
    assert arr.shape == (N_JOINT_GROUPS,)
    assert arr.dtype == bool
    return {int(i) for i, v in enumerate(arr.tolist()) if v}


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Corrective × 8
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "keep your back straight",
    "don't let your knees cave in",
    "lower your hips",
    "more depth on the squat",
    "less weight on your toes",
    "bend your elbows more",
    "engage your core",
    "stop arching your back",
])
def test_corrective_strings_classify_as_corrective(text):
    assert classify_feedback_type(text) == "corrective"


# ─────────────────────────────────────────────────────────────────────────────
# Section B: Affirmative × 5
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "great job",
    "perfect form",
    "nice work on the depth",
    "excellent",
    "well done",
])
def test_affirmative_strings_classify_as_affirmative(text):
    assert classify_feedback_type(text) == "affirmative"


# ─────────────────────────────────────────────────────────────────────────────
# Section C: Acknowledgment × 4
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "okay",
    "got it",
    "yes",
    "alright",
])
def test_acknowledgment_strings_classify_as_acknowledgment(text):
    assert classify_feedback_type(text) == "acknowledgment"


# ─────────────────────────────────────────────────────────────────────────────
# Section D: Informative × 3
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "this is rep 5 of your set",
    "you have 30 seconds left",
    "we'll do squats next",
])
def test_informative_strings_classify_as_informative(text):
    assert classify_feedback_type(text) == "informative"


# ─────────────────────────────────────────────────────────────────────────────
# Section E: Bilateral × 4 (group derivation -- not type classification)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("watch your knees",         {LK, RK}),
    ("your elbows are flaring",  {LE, RE}),
    ("open your shoulders",      {LS, RS}),
    ("shift your hips back",     {LH, RH, TR}),  # 'back' fires Trunk too
])
def test_bilateral_groups_fire_both_sides(text, expected):
    assert _groups_set(derive_joint_groups(text)) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Section F: Side-specific × 3
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("your left elbow is dropping",       {LE}),
    ("right knee is collapsing inward",   {RK}),
    ("lift your right shoulder up",       {RS}),
])
def test_side_specific_groups_fire_only_one_side(text, expected):
    assert _groups_set(derive_joint_groups(text)) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Section G: Edge cases × 3
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_string_classifies_as_informative_with_no_groups():
    assert classify_feedback_type("") == "informative"
    arr = derive_joint_groups("")
    assert arr.shape == (N_JOINT_GROUPS,)
    assert arr.dtype == bool
    assert not arr.any()


def test_all_caps_corrective_still_classifies_and_fires_groups():
    text = "DON'T LOCK YOUR KNEES"
    assert classify_feedback_type(text) == "corrective"
    assert _groups_set(derive_joint_groups(text)) == {LK, RK}


def test_punctuation_density_does_not_break_lexicon():
    # 'knees' fires bilateral; 3 distinct tokens (after stripping punctuation)
    # and no corrective/affirmative keyword -> acknowledgment.
    text = "Knees, knees, KNEES!!!"
    # Joint groups: bilateral 'knees' fires both
    assert _groups_set(derive_joint_groups(text)) == {LK, RK}
    # Type: no corrective keyword present -> acknowledgment (<=3 tokens)
    assert classify_feedback_type(text) == "acknowledgment"


# ─────────────────────────────────────────────────────────────────────────────
# Section H: Word-boundary adversarial tests × 4
# ─────────────────────────────────────────────────────────────────────────────


def test_moreover_does_not_fire_more():
    # "more" appears as a prefix of "moreover" but should NOT match
    # because 'more' is not a standalone word here.
    text = "moreover, complete the set"
    assert classify_feedback_type(text) == "informative"


def test_kneeling_does_not_fire_knees_bilateral():
    # 'kneeling' contains 'knee' but is not a standalone word; bilateral
    # 'knee' / 'knees' must NOT fire.
    text = "the kneeling position is fine"
    assert _groups_set(derive_joint_groups(text)) == set()


def test_smart_quote_apostrophe_normalises_and_matches():
    # Curly apostrophe (U+2019) should normalise to ASCII so 'don't' fires.
    text = "don’t lock your knees"
    assert classify_feedback_type(text) == "corrective"
    assert _groups_set(derive_joint_groups(text)) == {LK, RK}


def test_left_knee_does_not_double_fire_via_bilateral():
    # 'left knee' should consume the match before bilateral 'knee' runs;
    # only L_knee fires.
    text = "your left knee is collapsing"
    assert _groups_set(derive_joint_groups(text)) == {LK}


# ─────────────────────────────────────────────────────────────────────────────
# Section I: derive_quality_scalar formula
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_silence_default():
    assert derive_quality_scalar([]) == pytest.approx(Q_DEFAULT_NO_FEEDBACK)


def test_quality_all_corrective_drops_to_0p4():
    # 100% corrective -> q = 1.0 - 0.6 = 0.4
    fbs = ["keep your back straight", "don't lock your knees"]
    assert derive_quality_scalar(fbs) == pytest.approx(0.4, abs=1e-6)


def test_quality_all_affirmative_caps_at_1p0():
    # 100% affirmative -> q = 1.0 + 0.2 = 1.2 -> clipped to 1.0
    fbs = ["great job", "perfect form", "excellent"]
    assert derive_quality_scalar(fbs) == pytest.approx(1.0, abs=1e-6)


def test_quality_mixed_corrective_affirmative():
    # 50/50 -> q = 1.0 - 0.6*0.5 + 0.2*0.5 = 0.8
    fbs = ["keep your back straight", "great job"]
    assert derive_quality_scalar(fbs) == pytest.approx(0.8, abs=1e-6)


def test_quality_informative_only_stays_at_1p0():
    # Informative neither rewards nor penalises; q stays at 1.0
    fbs = ["this is rep 5", "you have 30 seconds left"]
    assert derive_quality_scalar(fbs) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Section J: Lexicon hit-rate sanity (would catch a dead keyword)
# ─────────────────────────────────────────────────────────────────────────────


def test_every_group_keyword_fires_at_least_once_in_fixtures():
    """Defensive: catches a typo'd keyword that would never match anything."""
    # Build a corpus from all fixture strings used elsewhere in this file.
    corpus = " ".join([
        "keep your back straight", "don't let your knees cave in",
        "lower your hips", "more depth on the squat",
        "less weight on your toes", "bend your elbows more",
        "engage your core", "stop arching your back",
        "great job", "perfect form", "nice work on the depth",
        "excellent", "well done", "okay", "got it", "yes", "alright",
        "this is rep 5 of your set", "you have 30 seconds left",
        "we'll do squats next", "watch your knees",
        "your elbows are flaring", "open your shoulders",
        "shift your hips back", "your left elbow is dropping",
        "right knee is collapsing inward", "lift your right shoulder up",
        "DON'T LOCK YOUR KNEES", "Knees, knees, KNEES!!!",
        "the neck position is good", "tuck your chin",
        "engage your spine, lift your chest", "watch your hip alignment",
    ])
    fired = derive_joint_groups(corpus)
    # Every joint group must fire at least once across this corpus.
    missing = [
        JOINT_GROUP_NAMES[i] for i in range(N_JOINT_GROUPS) if not fired[i]
    ]
    assert not missing, f"groups never fired across corpus: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Section K: build_clip_label end-to-end
# ─────────────────────────────────────────────────────────────────────────────


def _write_fit300k_json(tmp_path: Path) -> Path:
    payload = {
        "clip_id": "0001",
        "human_id": "h_023",
        "exercise_class": "squat",
        "feedbacks": [
            "keep your back straight",
            "don't let your knees cave in",
            "great depth",
        ],
    }
    p = tmp_path / "0001.json"
    p.write_text(json.dumps(payload))
    return p


def _write_fitcoach_json(tmp_path: Path) -> Path:
    payload = {
        "clip_id": "long_007",
        "human_id": "h_test_5",
        "exercise_class": "squat",
        "feedbacks": [
            {"text": "keep your back straight", "type": "corrective",
             "t_start": 5.0, "t_end": 7.5},
            {"text": "good", "type": "affirmative",
             "t_start": 12.1, "t_end": 13.0},
            {"text": "rep 1", "type": "rep-counting",
             "t_start": 14.0, "t_end": 14.5},
            {"text": "rep 2", "type": "rep-counting",
             "t_start": 18.0, "t_end": 18.5},
        ],
    }
    p = tmp_path / "long_007.json"
    p.write_text(json.dumps(payload))
    return p


def test_build_clip_label_fit300k_roundtrip(tmp_path):
    p = _write_fit300k_json(tmp_path)
    label = build_clip_label(p, fitcoach=False)
    assert label["clip_id"] == "0001"
    assert label["human_id"] == "h_023"
    assert label["exercise_class"] == "squat"
    assert label["n_feedbacks"] == 3
    # Two corrective + one with no corrective/affirmative match
    # ("great depth" -> 'great' is affirmative)
    # 2/3 corrective, 1/3 affirmative -> q = 1 - 0.6*2/3 + 0.2*1/3 = 0.6 + 0.067 = 0.6667
    assert label["quality"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    # Joint groups: 'back' (Trunk) + 'knees' (L+R)
    fired = set(i for i, v in enumerate(label["joint_groups_per_clip"]) if v)
    assert TR in fired and LK in fired and RK in fired
    assert label["joint_groups_per_feedback"] is None  # FIT-300K has no per-feedback timeline
    assert label["rep_count_weak"] is None             # FIT-300K has no rep-count GT


def test_build_clip_label_fitcoach_with_rep_count(tmp_path):
    p = _write_fitcoach_json(tmp_path)
    label = build_clip_label(p, fitcoach=True)
    assert label["clip_id"] == "long_007"
    assert label["rep_count_weak"] == 2          # two 'rep-counting' feedbacks
    # Per-feedback timeline preserved
    assert label["joint_groups_per_feedback"] is not None
    assert len(label["joint_groups_per_feedback"]) == 4
    # First feedback fires Trunk via 'back'
    fb0 = label["joint_groups_per_feedback"][0]
    assert fb0["t_start"] == 5.0
    assert fb0["groups"][TR] is True


# ─────────────────────────────────────────────────────────────────────────────
# Section L: Expanded corrective lexicon (paper-evidenced — Fig 2 examples)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "go deeper on the squat",
    "watch your form",
    "raise your knees higher",
    "drop your hips back",
    "go slower on the way down",
    "go faster on the punches",
    "control the descent",
])
def test_paper_evidenced_corrective_phrases_classify_correctly(text):
    """Real feedback patterns from QEVD paper Fig 2 + section 3.1."""
    assert classify_feedback_type(text) == "corrective"


def test_great_job_but_watch_your_form_is_corrective_not_affirmative():
    # Mixed message: corrective wins by master plan precedence rule.
    text = "great job but watch your form"
    assert classify_feedback_type(text) == "corrective"


# ─────────────────────────────────────────────────────────────────────────────
# Section M: derive_joint_groups_from_class (paper Fig 2 pushup variations)
# ─────────────────────────────────────────────────────────────────────────────


from backend.training.preprocessing.qevd_label_builder import (
    derive_joint_groups_from_class,
)


@pytest.mark.parametrize("class_label,expected", [
    ("Hips Too Low",       {LH, RH}),
    ("Hips Too High",      {LH, RH}),
    ("Elbows Flared",      {LE, RE}),
    ("Head Down",          {NK}),
    # 'On Knees' -> bilateral 'knees' fires both
    ("On Knees",           {LK, RK}),
    # CamelCase / snake-case variants the helper must split correctly
    ("HipsTooLow",         {LH, RH}),
    ("pushup_HipsTooLow",  {LH, RH}),
    ("squat_KneesCaveIn",  {LK, RK}),
])
def test_derive_joint_groups_from_class(class_label, expected):
    got = _groups_set(derive_joint_groups_from_class(class_label))
    assert got == expected, (class_label, expected, got)


def test_derive_joint_groups_from_class_handles_empty():
    arr = derive_joint_groups_from_class("")
    assert arr.shape == (10,)
    assert not arr.any()


# ─────────────────────────────────────────────────────────────────────────────
# Section N: QEVDLabels manifest loader (schema-tolerant)
# ─────────────────────────────────────────────────────────────────────────────


from backend.training.preprocessing.qevd_label_builder import QEVDLabels


def test_qevd_labels_loads_dict_keyed_schema(tmp_path: Path):
    """Schema A: top-level dict keyed by clip_id."""
    feedbacks_data = {
        "00000000": ["great job", "watch your knees"],
        "00000001": ["go deeper on the squat"],
    }
    fine_data = {
        "00000000": {"exercise": "squat", "variation": "Hips Too Low",
                     "human_id": "h_001", "split": "train"},
        "00000001": {"exercise": "pushup", "variation": "Elbows Flared",
                     "human_id": "h_002", "split": "train"},
    }
    fbs_path = tmp_path / "feedbacks_short_clips.json"
    fbs_path.write_text(json.dumps(feedbacks_data))
    fine_path = tmp_path / "fine_grained_labels.json"
    fine_path.write_text(json.dumps(fine_data))

    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )

    assert len(labels) == 2
    rec = labels.lookup("00000000")
    assert rec["feedbacks"] == ["great job", "watch your knees"]
    assert rec["exercise"] == "squat"
    assert rec["variation"] == "Hips Too Low"
    assert rec["human_id"] == "h_001"
    assert rec["split"] == "train"
    assert rec["n_feedbacks"] == 2


def test_qevd_labels_loads_list_of_records_schema(tmp_path: Path):
    """Schema B: top-level list of records."""
    feedbacks_data = [
        {"clip_id": "00000000", "feedbacks": ["great job"]},
        {"clip_id": "00000001", "feedbacks": ["watch your form"]},
    ]
    fine_data = [
        {"clip_id": "00000000", "exercise": "squat",
         "variation": "Hips Too Low", "human_id": "h_001", "split": "train"},
        {"clip_id": "00000001", "exercise": "pushup",
         "variation": "Elbows Flared", "human_id": "h_002", "split": "train"},
    ]
    fbs_path = tmp_path / "feedbacks_short_clips.json"
    fbs_path.write_text(json.dumps(feedbacks_data))
    fine_path = tmp_path / "fine_grained_labels.json"
    fine_path.write_text(json.dumps(fine_data))

    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )

    assert len(labels) == 2
    rec = labels.lookup("00000001")
    assert rec["feedbacks"] == ["watch your form"]
    assert rec["exercise"] == "pushup"


def test_qevd_labels_lookup_pads_numeric_ids(tmp_path: Path):
    """Numeric clip IDs should match regardless of zero-padding (str vs int)."""
    feedbacks_data = {"00012345": ["great job"]}
    fbs_path = tmp_path / "fbs.json"
    fbs_path.write_text(json.dumps(feedbacks_data))
    labels = QEVDLabels.from_files(feedbacks_short_clips_path=fbs_path)

    # Lookup with same form
    assert labels.lookup("00012345")["feedbacks"] == ["great job"]
    # Lookup with int -> should auto-pad to 8-digit
    assert labels.lookup(12345)["feedbacks"] == ["great job"]


def test_qevd_labels_lookup_returns_empty_for_missing(tmp_path: Path):
    """Missing clip -> empty record, not exception."""
    labels = QEVDLabels()
    rec = labels.lookup("99999999")
    assert rec["feedbacks"] == []
    assert rec["exercise"] is None
    assert rec["n_feedbacks"] == 0


def test_qevd_labels_stats_counts_humans_and_feedbacks(tmp_path: Path):
    feedbacks_data = {"00000000": ["a", "b", "c"], "00000001": ["x"]}
    fine_data = {
        "00000000": {"human_id": "h_001"},
        "00000001": {"human_id": "h_002"},
    }
    fbs_path = tmp_path / "fbs.json"
    fbs_path.write_text(json.dumps(feedbacks_data))
    fine_path = tmp_path / "fine.json"
    fine_path.write_text(json.dumps(fine_data))
    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )
    stats = labels.stats()
    assert stats["n_clips_with_feedback"] == 2
    assert stats["n_clips_with_fine"] == 2
    assert stats["n_total_feedbacks"] == 4
    assert stats["n_unique_humans"] == 2


def test_qevd_labels_rejects_unknown_top_level_shape(tmp_path: Path):
    """Top-level int (or any non-dict/list) must raise — fail-fast."""
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(42))
    labels = QEVDLabels()
    with pytest.raises(ValueError, match="Unrecognised"):
        labels.load_feedbacks(bad_path)


# ─────────────────────────────────────────────────────────────────────────────
# Section O: Real Qualcomm schema (video_path keyed list-of-records)
# ─────────────────────────────────────────────────────────────────────────────


def test_qevd_labels_loads_real_qualcomm_schema(tmp_path: Path):
    """Real schema: video_path='./00000009.mp4', labels=[<exercise> - <var>]."""
    feedbacks_data = [
        {"video_path": "./00000009.mp4",
         "feedbacks": ["watch your form", "great effort"],
         "split": "train"},
        {"video_path": "./00000017.mp4",
         "feedbacks": ["go deeper"], "split": "train"},
    ]
    fine_data = [
        {"video_path": "./00000009.mp4",
         "labels": ["elbow plank - stopping early"],
         "labels_descriptive": ["elbow plank - User stopped the exercise early"],
         "split": "train"},
        {"video_path": "./00000017.mp4",
         "labels": ["squat - rounded back"],
         "labels_descriptive": ["squat - User has a rounded back"],
         "split": "train"},
    ]
    fbs_path = tmp_path / "feedbacks_short_clips.json"
    fbs_path.write_text(json.dumps(feedbacks_data))
    fine_path = tmp_path / "fine_grained_labels.json"
    fine_path.write_text(json.dumps(fine_data))

    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )

    rec = labels.lookup("00000009")
    assert rec["clip_id"] == "00000009"
    assert rec["feedbacks"] == ["watch your form", "great effort"]
    assert rec["exercise"] == "elbow plank"
    assert rec["variation"] == "stopping early"
    assert rec["labels"] == ["elbow plank - stopping early"]
    assert rec["split"] == "train"

    # Same clip lookup via './' prefix or with .mp4
    assert labels.lookup("./00000009.mp4")["feedbacks"] == rec["feedbacks"]


# ─────────────────────────────────────────────────────────────────────────────
# Section P: quality_from_label / quality_from_labels (tier-based)
# ─────────────────────────────────────────────────────────────────────────────


from backend.training.preprocessing.qevd_label_builder import (
    compute_clip_quality, quality_from_feedbacks, quality_from_label,
    quality_from_labels,
)


@pytest.mark.parametrize("label,bound", [
    # HIGH tier
    ("elbow plank - no obvious issue",   "ge_0.85"),
    ("squat - rom=5",                     "ge_0.85"),
    ("pushup - perfect",                  "ge_0.85"),
    ("high knees - as fast as possible", "ge_0.80"),
    # GOOD tier
    ("squat - average",                   "in_0.70_0.85"),
    ("pushup - rom=3",                    "in_0.65_0.80"),
    ("plank - head straight",             "in_0.75_0.85"),
    # MILD tier
    ("squat - rom=2",                     "in_0.45_0.65"),
    ("pushup - looking down",             "in_0.45_0.65"),
    # MAJOR tier
    ("elbow plank - stopping early",      "le_0.50"),
    ("squat - rounded back",              "le_0.50"),
    ("plank - lazy",                      "le_0.50"),
    ("pushup - too shallow",              "le_0.50"),
])
def test_quality_from_label_returns_expected_tier(label, bound):
    q = quality_from_label(label)
    assert q is not None, f"no tier matched for {label!r}"
    if bound == "ge_0.85":
        assert q >= 0.85, q
    elif bound == "ge_0.80":
        assert q >= 0.80, q
    elif bound == "in_0.70_0.85":
        assert 0.70 <= q <= 0.85, q
    elif bound == "in_0.65_0.80":
        assert 0.65 <= q <= 0.80, q
    elif bound == "in_0.75_0.85":
        assert 0.75 <= q <= 0.85, q
    elif bound == "in_0.45_0.65":
        assert 0.45 <= q <= 0.65, q
    elif bound == "le_0.50":
        assert q <= 0.50, q
    else:
        raise AssertionError(f"unknown bound {bound!r}")


def test_quality_from_label_returns_none_when_no_pattern_matches():
    # 'unknown new variation' has no matching tier
    assert quality_from_label("squat - unknown novel descriptor xyz") is None


def test_quality_from_label_handles_no_dash_separator():
    # Label without " - " is treated as the variation directly
    assert quality_from_label("no obvious issue") is not None
    assert quality_from_label("rom=5") is not None


def test_quality_from_label_handles_empty():
    assert quality_from_label("") is None
    assert quality_from_label(None) is None  # type: ignore[arg-type]


def test_quality_from_labels_averages_multi_label_clips():
    # Mix of tiers: "no obvious issue" (~0.92) + "rom=2" (0.55) -> ~0.735
    labels = ["squat - no obvious issue", "squat - rom=2"]
    q = quality_from_labels(labels)
    assert q is not None
    assert 0.65 < q < 0.85, q


def test_quality_from_labels_skips_unmatched_labels():
    # One matched + one unmatched -> use only the matched one
    labels = ["squat - no obvious issue", "squat - completely novel descriptor"]
    q = quality_from_labels(labels)
    assert q is not None
    assert q >= 0.85, q


def test_quality_from_labels_returns_none_when_all_unmatched():
    labels = ["squat - foo", "squat - bar"]
    assert quality_from_labels(labels) is None


# ─────────────────────────────────────────────────────────────────────────────
# Section Q: compute_clip_quality combiner
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_clip_quality_uses_feedback_when_only_feedback():
    # 'great job' alone -> 100% affirmative -> q = 1.0
    q = compute_clip_quality(["great job"], labels=[])
    assert q == pytest.approx(1.0, abs=1e-6)


def test_compute_clip_quality_uses_label_when_only_label():
    q = compute_clip_quality(feedbacks=[], labels=["squat - no obvious issue"])
    assert q is not None
    assert q >= 0.85


def test_compute_clip_quality_silence_default_when_neither():
    q = compute_clip_quality(feedbacks=[], labels=[])
    assert q == pytest.approx(0.7, abs=1e-6)
    # Or labels match nothing
    q2 = compute_clip_quality(
        feedbacks=[], labels=["squat - novel pattern not in tiers"],
    )
    assert q2 == pytest.approx(0.7, abs=1e-6)


def test_compute_clip_quality_averages_feedback_and_label():
    # Feedback all corrective -> q_fb = 0.4
    # Label 'no obvious issue' -> q_lbl ~ 0.92
    # Combined: (0.4 + 0.92) / 2 = 0.66
    q = compute_clip_quality(
        feedbacks=["keep your back straight", "don't lock your knees"],
        labels=["squat - no obvious issue"],
    )
    assert 0.60 <= q <= 0.72, q


def test_compute_clip_quality_label_disambiguates_silent_clip():
    # The bug we're fixing: a silent clip ('moving arms' = error in label)
    # should have q < 0.7, not the silence default.
    q_old = compute_clip_quality(feedbacks=[], labels=[])
    q_new = compute_clip_quality(
        feedbacks=[],
        labels=["dead bugs (legs only) - lazy"],   # 'lazy' -> 0.40 tier
    )
    assert q_old == pytest.approx(0.7, abs=1e-6)
    assert q_new <= 0.50, q_new


# ─────────────────────────────────────────────────────────────────────────────
# Section R: Worker IDs (= participants, verified 2026-05-06)
# ─────────────────────────────────────────────────────────────────────────────


def test_qevd_labels_loads_worker_ids_from_real_schema(tmp_path: Path):
    """Real Qualcomm worker_ids schema:
        [{"video_path": "./00000005.mp4", "worker_id": 1750}, ...]
    """
    worker_data = [
        {"video_path": "./00000005.mp4", "worker_id": 1750},
        {"video_path": "./00000006.mp4", "worker_id": 297},
        {"video_path": "./00000007.mp4", "worker_id": 297},
    ]
    fine_data = [
        {"video_path": "./00000005.mp4",
         "labels": ["squat - average"], "split": "train"},
        {"video_path": "./00000006.mp4",
         "labels": ["pushup - perfect"], "split": "train"},
        {"video_path": "./00000007.mp4",
         "labels": ["pushup - rom=4"], "split": "train"},
    ]
    fine_path = tmp_path / "fine.json"; fine_path.write_text(json.dumps(fine_data))
    workers_path = tmp_path / "workers.json"; workers_path.write_text(json.dumps(worker_data))

    labels = QEVDLabels.from_files(
        fine_grained_labels_path=fine_path,
        worker_ids_path=workers_path,
    )
    assert labels.clip_to_human == {
        "00000005": "1750",
        "00000006": "297",
        "00000007": "297",
    }
    rec = labels.lookup("00000005")
    assert rec["human_id"] == "1750"
    rec = labels.lookup("00000007")
    assert rec["human_id"] == "297"

    stats = labels.stats()
    assert stats["n_clips_with_worker_id"] == 3
    assert stats["n_unique_humans"] == 2  # {1750, 297}


def test_qevd_labels_load_worker_ids_handles_dict_schema(tmp_path: Path):
    """Defensive: also accept top-level dict."""
    worker_data = {"00000005.mp4": 1750, "00000006": 297}
    workers_path = tmp_path / "w.json"; workers_path.write_text(json.dumps(worker_data))
    labels = QEVDLabels()
    labels.load_worker_ids(workers_path)
    assert labels.clip_to_human == {"00000005": "1750", "00000006": "297"}


def test_qevd_labels_load_worker_ids_rejects_bad_shape(tmp_path: Path):
    bad_path = tmp_path / "bad.json"; bad_path.write_text(json.dumps(42))
    labels = QEVDLabels()
    with pytest.raises(ValueError, match="Unrecognised worker_ids"):
        labels.load_worker_ids(bad_path)


def test_qevd_labels_derive_label_combines_signals(tmp_path: Path):
    """End-to-end: QEVDLabels.derive_label produces enriched record."""
    feedbacks_data = [
        {"video_path": "./00000009.mp4",
         "feedbacks": ["watch your form on the elbow"],
         "split": "train"},
    ]
    fine_data = [
        {"video_path": "./00000009.mp4",
         "labels": ["elbow plank - stopping early"],
         "split": "train"},
    ]
    fbs_path = tmp_path / "fbs.json"; fbs_path.write_text(json.dumps(feedbacks_data))
    fine_path = tmp_path / "fine.json"; fine_path.write_text(json.dumps(fine_data))
    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )
    enriched = labels.derive_label("00000009")
    # quality combines: feedback corrective (q=0.4) + label 'stopping early' (q=0.4) -> ~0.4
    assert 0.30 <= enriched["quality"] <= 0.50, enriched["quality"]
    # joint groups: feedback fires Trunk (via 'form' indirectly? no — via 'elbow'),
    # label fires nothing specific (no body part in 'stopping early')
    fired = {i for i, v in enumerate(enriched["joint_groups"]) if v}
    assert (LE in fired) or (RE in fired), \
        f"expected an elbow group to fire on 'watch your form on the elbow', got {fired}"
    assert enriched["quality_source"] == "feedback+label"
