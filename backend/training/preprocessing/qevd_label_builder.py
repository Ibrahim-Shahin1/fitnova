"""Derive QEVD v6 supervision labels from natural-language coaching feedback.

QEVD ships natural-language feedback strings -- never typed labels for
joint-error or quality. This module turns each clip's feedback list into:

  * quality scalar in [0, 1]
  * 10-vector joint-group hits (matches form_session.JOINT_GROUP_NAMES order)
  * feedback type per string (corrective / affirmative / acknowledgment /
    informative)
  * weak rep-count signal (FIT-COACH only -- ground-truth from
    'rep-counting' tagged feedbacks)

The lexicon-based approach is the master-plan-locked strategy
(plans/i-am-now-on-zazzy-brooks.md section II.4.4).  It is the highest-
risk component of v6 -- if D3 spot-check on 100 real clips fails, the
fallback is GPT-4o-mini zero-shot classification of the ~8,500 unique
feedback strings (~$2 batch cost).

Key design choices:
  * Word-boundary regex (not naive substring) so 'moreover' does not fire
    'more', 'kneeling' does not fire 'knees', etc.
  * Side-specific keywords ('left knee') consume the match BEFORE bilateral
    keywords ('knees') run, so 'left knee' fires only L_knee, not both.
  * Smart-quote / curly-apostrophe normalisation so 'don't' and 'don't'
    both match.
  * No external dependencies beyond numpy + stdlib regex.

This module is pure-function and has no MediaPipe / TF dependency, which
makes the 30-case unit-test suite cheap to run (sub-second).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Joint-group taxonomy (matches form_session.JOINT_GROUP_NAMES order)
# ─────────────────────────────────────────────────────────────────────────────

JOINT_GROUP_NAMES: list[str] = [
    "L_elbow", "R_elbow", "L_shoulder", "R_shoulder",
    "L_knee",  "R_knee",  "L_hip",      "R_hip",
    "Trunk",   "Neck",
]
N_JOINT_GROUPS = len(JOINT_GROUP_NAMES)
JOINT_GROUP_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(JOINT_GROUP_NAMES)
}

# Side-specific keywords map to a single joint group.
GROUP_KEYWORDS: dict[str, list[str]] = {
    "L_elbow":    ["left elbow", "l elbow"],
    "R_elbow":    ["right elbow", "r elbow"],
    "L_shoulder": ["left shoulder"],
    "R_shoulder": ["right shoulder"],
    "L_knee":     ["left knee"],
    "R_knee":     ["right knee"],
    "L_hip":      ["left hip"],
    "R_hip":      ["right hip"],
    "Trunk":      ["back", "spine", "core", "torso", "chest", "posture"],
    "Neck":       ["neck", "head", "chin"],
}

# Bilateral keywords fire BOTH L+R groups when no side-specific match
# consumed the word first. Side-specific is checked + consumed first.
BILATERAL_KEYWORDS: dict[str, list[str]] = {
    "knees":     ["L_knee", "R_knee"],
    "knee":      ["L_knee", "R_knee"],
    "elbows":    ["L_elbow", "R_elbow"],
    "elbow":     ["L_elbow", "R_elbow"],
    "shoulders": ["L_shoulder", "R_shoulder"],
    "shoulder":  ["L_shoulder", "R_shoulder"],
    "hips":      ["L_hip", "R_hip"],
    "hip":       ["L_hip", "R_hip"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Feedback-type lexicons (master plan section II.4.4)
# ─────────────────────────────────────────────────────────────────────────────

CORRECTIVE_KEYWORDS: tuple[str, ...] = (
    # Master plan section II.4.4 base set
    "keep", "don't", "do not", "avoid", "more", "less", "lower", "higher",
    "straighten", "bend", "engage", "tuck", "arch", "stop",
    # Paper-evidenced expansions (NeurIPS 2024 paper Fig 2 + section 3.1
    # examples: "Go deeper on the squat", "watch your form",
    # "Raise your knees", "drop your hips", "slower", "faster"). Conservative
    # additions; D3 spot-check on real feedbacks_short_clips.json will
    # validate / expand further.
    "deeper", "shallower", "watch", "raise", "drop", "slower", "faster",
    "wider", "narrower", "control", "controlled", "squeeze", "extend",
    "hold",
)

AFFIRMATIVE_KEYWORDS: tuple[str, ...] = (
    "good", "great", "perfect", "nice", "well done", "excellent", "amazing",
)

# Ack = short utterance not matching the above
ACK_MAX_TOKENS = 3

FeedbackType = Literal[
    "corrective", "affirmative", "acknowledgment", "informative",
]
ALL_TYPES: tuple[FeedbackType, ...] = (
    "corrective", "affirmative", "acknowledgment", "informative",
)

# FIT-COACH JSONs ship with an additional 'rep-counting' tag. The lexicon
# classifier never produces this tag (it's a coach-action label, not a
# semantic-content label) so we only accept it on the JSON-passthrough
# path. See master plan section II.2.
FIT_COACH_PASSTHROUGH_TYPES: tuple[str, ...] = (
    *ALL_TYPES, "rep-counting",
)

# Quality formula coefficients (master plan section II.3)
Q_W_CORRECTIVE = 0.6
Q_W_AFFIRMATIVE = 0.2
Q_W_ACK = 0.1
Q_DEFAULT_NO_FEEDBACK = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Lowercase + replace smart quotes / curly apostrophes with ASCII.

    Different QEVD feedback sources may use Unicode quotes (U+2018, U+2019,
    U+201C, U+201D); normalising them prevents false negatives on the
    word-boundary regex.
    """
    if not text:
        return ""
    # Replace common smart quotes BEFORE NFKC (which would canonicalise some
    # but leave "U+2019 right single quotation mark" untouched in old envs).
    repl = {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark (most common)
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
        "–": "-",  # en dash
        "—": "-",  # em dash
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def _build_keyword_regex(keywords: Iterable[str]) -> re.Pattern[str]:
    """Compile a single regex with word boundaries around each keyword."""
    parts = sorted(keywords, key=len, reverse=True)  # match longer first
    if not parts:
        return re.compile(r"$^")  # never matches
    body = "|".join(re.escape(k) for k in parts)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)")


_CORRECTIVE_RE = _build_keyword_regex(CORRECTIVE_KEYWORDS)
_AFFIRMATIVE_RE = _build_keyword_regex(AFFIRMATIVE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Feedback-type classifier
# ─────────────────────────────────────────────────────────────────────────────


def classify_feedback_type(text: str) -> FeedbackType:
    """Return the feedback type for one feedback string.

    Order of precedence (master plan section II.4.4):
        corrective > affirmative > acknowledgment > informative
    """
    norm = _normalise(text)
    if not norm:
        return "informative"

    if _CORRECTIVE_RE.search(norm):
        return "corrective"
    if _AFFIRMATIVE_RE.search(norm):
        return "affirmative"

    # Acknowledgment: short utterance, no corrective/affirmative match.
    n_tokens = len(re.findall(r"\w+", norm))
    if 0 < n_tokens <= ACK_MAX_TOKENS:
        return "acknowledgment"

    return "informative"


# ─────────────────────────────────────────────────────────────────────────────
# Quality scalar
# ─────────────────────────────────────────────────────────────────────────────


def derive_quality_scalar(
    feedbacks: Sequence[dict | str],
    *,
    silence_default: float = Q_DEFAULT_NO_FEEDBACK,
) -> float:
    """Compute the per-clip quality scalar.

    Formula (master plan section II.3):
        q = clip(
            1.0 - 0.6 * frac_corrective
                + 0.2 * frac_affirmative
                + 0.1 * frac_acknowledgment,
            0, 1,
        )

    `feedbacks` may be a list of dicts (each with `text` and optional
    `type`) or raw strings.  Empty list returns `silence_default` (0.7).
    """
    if not feedbacks:
        return float(silence_default)

    counts = {t: 0 for t in ALL_TYPES}
    n_total = 0
    for fb in feedbacks:
        if isinstance(fb, str):
            t = classify_feedback_type(fb)
        else:
            t = fb.get("type") or classify_feedback_type(fb.get("text", ""))
            if t not in FIT_COACH_PASSTHROUGH_TYPES:
                t = classify_feedback_type(fb.get("text", ""))
        # 'rep-counting' is a counter marker, not a quality signal -- skip
        # so it doesn't dilute the corrective/affirmative fractions.
        if t == "rep-counting":
            continue
        counts[t] += 1
        n_total += 1

    if n_total == 0:
        return float(silence_default)

    frac_corr = counts["corrective"] / n_total
    frac_aff  = counts["affirmative"] / n_total
    frac_ack  = counts["acknowledgment"] / n_total

    q = 1.0 - Q_W_CORRECTIVE * frac_corr \
            + Q_W_AFFIRMATIVE * frac_aff \
            + Q_W_ACK * frac_ack

    return float(max(0.0, min(1.0, q)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-joint-group derivation
# ─────────────────────────────────────────────────────────────────────────────


def derive_joint_groups(text: str) -> np.ndarray:
    """Return shape (10,) bool array of joint-group hits."""
    out = np.zeros(N_JOINT_GROUPS, dtype=bool)
    norm = _normalise(text)
    if not norm:
        return out

    # ── Pass 1: side-specific keywords. Mask matches out of the text so
    #            bilateral keywords don't double-count ("left knee" should
    #            fire only L_knee, not both via the bilateral "knee").
    masked = norm
    for group, keywords in GROUP_KEYWORDS.items():
        idx = JOINT_GROUP_TO_IDX[group]
        for kw in sorted(keywords, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<!\w){re.escape(kw)}(?!\w)"
            )
            if pattern.search(masked):
                out[idx] = True
                masked = pattern.sub(" ", masked)

    # ── Pass 2: bilateral keywords on the masked-out text
    for kw, groups in BILATERAL_KEYWORDS.items():
        pattern = re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)")
        if pattern.search(masked):
            for g in groups:
                out[JOINT_GROUP_TO_IDX[g]] = True

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip label builder
# ─────────────────────────────────────────────────────────────────────────────


def build_clip_label(
    json_path: Path,
    *,
    fitcoach: bool = False,
) -> dict:
    """Load a QEVD per-clip JSON and derive the supervision labels.

    Parameters
    ----------
    json_path : Path
    fitcoach : bool
        Set True for FIT-COACH JSONs (timestamped feedbacks with explicit
        type tags). Set False for FIT-300K (clip-end feedbacks, no types).

    Returns
    -------
    dict with the keys documented in plans/phase-1-complete-critical-snappy-flurry.md
    section D0.6.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clip_id = data.get("clip_id") or json_path.stem
    human_id = data.get("human_id", "unknown")
    exercise_class = data.get("exercise_class", "unknown")

    raw_feedbacks = data.get("feedbacks") or data.get("feedback") or []

    # Normalise feedback entries to dicts with {text, type, t_start?, t_end?}
    feedbacks: list[dict] = []
    for fb in raw_feedbacks:
        if isinstance(fb, str):
            feedbacks.append({
                "text": fb,
                "type": classify_feedback_type(fb),
            })
        elif isinstance(fb, dict):
            text = fb.get("text", "") or fb.get("feedback", "")
            ftype = fb.get("type")
            if ftype not in FIT_COACH_PASSTHROUGH_TYPES:
                ftype = classify_feedback_type(text)
            feedbacks.append({
                "text":    text,
                "type":    ftype,
                "t_start": fb.get("t_start"),
                "t_end":   fb.get("t_end"),
            })

    # Quality scalar (clip-level for both schemas)
    q = derive_quality_scalar(feedbacks)

    # Per-clip 10-vector: union of per-feedback hits
    union_groups = np.zeros(N_JOINT_GROUPS, dtype=bool)
    per_feedback_groups = []
    for fb in feedbacks:
        groups = derive_joint_groups(fb["text"])
        union_groups |= groups
        if fitcoach:
            per_feedback_groups.append({
                "t_start": fb.get("t_start"),
                "t_end":   fb.get("t_end"),
                "groups":  groups.tolist(),
                "type":    fb["type"],
            })

    # Rep count: FIT-COACH only — count of explicit 'rep-counting' feedbacks.
    rep_count_weak: Optional[int] = None
    if fitcoach:
        rep_count_weak = sum(1 for fb in feedbacks if fb["type"] == "rep-counting")

    return {
        "clip_id":               clip_id,
        "human_id":              human_id,
        "exercise_class":        exercise_class,
        "n_feedbacks":           len(feedbacks),
        "feedbacks":             feedbacks,   # canonicalised (text+type)
        "quality":               q,
        "joint_groups_per_clip": union_groups.tolist(),
        "joint_groups_per_feedback": per_feedback_groups if fitcoach else None,
        "rep_count_weak":        rep_count_weak,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest-based loader for the actual on-disk QEVD label files
# ─────────────────────────────────────────────────────────────────────────────
#
# QEVD ships clip MP4s in 4 multi-volume archives (76,000 clips per part).
# Per-clip JSONs do NOT exist. The labels live in two manifest files inside
# the separate FIT-COACH download:
#
#   feedbacks_short_clips.json   - keyed by clip_id -> list of NL feedbacks
#   fine_grained_labels.json     - keyed by clip_id -> {exercise, variation,
#                                  human_id, split, ...}
#
# The exact schema (dict-keyed-by-id vs list-of-records) is uncertain until
# the user finishes downloading FIT-COACH.zip, so we accept BOTH shapes and
# fail loudly if neither matches.

class QEVDLabels:
    """In-memory index over QEVD per-clip labels.

    Loaded once at start-of-pipeline; subsequent calls go through fast dict
    lookup. Both manifest paths are optional so the loader can be used
    incrementally as files arrive on disk.

    Attributes
    ----------
    clip_to_feedbacks : dict[str, list[str]]
        clip_id -> list of NL feedback strings (~2 per clip in FIT-300K)
    clip_to_fine : dict[str, dict]
        clip_id -> {"exercise": str, "variation": str, "human_id": str,
                    "split": "train"|"test", ...}
    """

    def __init__(self):
        self.clip_to_feedbacks: dict[str, list[str]] = {}
        self.clip_to_fine: dict[str, dict] = {}
        # Qualcomm puts split=train|test directly on each record (FIT-300K
        # train/test is participant-disjoint per paper §3.1). We trust this
        # value verbatim.
        self.clip_to_split: dict[str, str] = {}
        # Worker IDs are PARTICIPANTS (verified empirically 2026-05-06: all
        # 2,920 unique workers fall in exactly one split, matching paper's
        # 1,800+ train + 100 test). Loaded from the optional
        # fine_grained_labels_with_worker_ids.json manifest.
        self.clip_to_human: dict[str, str] = {}

    # ── Loaders ──────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        *,
        feedbacks_short_clips_path: Optional[Path] = None,
        fine_grained_labels_path: Optional[Path] = None,
        worker_ids_path: Optional[Path] = None,
    ) -> "QEVDLabels":
        """Load any subset of the three QEVD manifests.

        ``worker_ids_path`` points to ``fine_grained_labels_with_worker_ids.json``
        — the additional download that ships participant IDs. Without it,
        ``clip_to_human`` stays empty and subject-disjoint val splits
        cannot be carved out within ``split=train``.
        """
        obj = cls()
        if feedbacks_short_clips_path is not None:
            obj.load_feedbacks(feedbacks_short_clips_path)
        if fine_grained_labels_path is not None:
            obj.load_fine_grained(fine_grained_labels_path)
        if worker_ids_path is not None:
            obj.load_worker_ids(worker_ids_path)
        return obj

    def load_worker_ids(self, path: Path) -> None:
        """Load fine_grained_labels_with_worker_ids.json into clip_to_human.

        Schema (verified 2026-05-06):
            [{"video_path": "./00000005.mp4", "worker_id": 1750}, ...]

        Worker IDs are PARTICIPANTS (people in the videos), not annotators.
        Verified by the deterministic test that 2,920/2,920 unique workers
        fall in exactly one split, matching the paper's participant-disjoint
        split design.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged: dict[str, str] = {}
        if isinstance(data, list):
            for rec in data:
                cid = _record_clip_id(rec)
                if cid is None:
                    continue
                wid = rec.get("worker_id") or rec.get("participant_id")
                if wid is None:
                    continue
                merged[cid] = str(wid)
        elif isinstance(data, dict):
            for cid, wid in data.items():
                merged[_extract_clip_id(cid)] = str(wid)
        else:
            raise ValueError(
                f"Unrecognised worker_ids JSON shape: top-level is "
                f"{type(data).__name__}, expected dict or list"
            )
        self.clip_to_human.update(merged)

    def load_feedbacks(self, path: Path) -> None:
        """Load feedbacks_short_clips.json into clip_to_feedbacks."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged: dict[str, list[str]] = {}
        if isinstance(data, dict):
            # Schema A: {clip_id: [feedbacks...]}
            for cid, fbs in data.items():
                merged[_extract_clip_id(cid)] = _coerce_feedback_list(fbs)
        elif isinstance(data, list):
            # Schema B: [{video_path|clip_id, feedbacks: [...]}, ...]
            # Real Qualcomm format uses 'video_path' = './00000009.mp4'
            for rec in data:
                cid = _record_clip_id(rec)
                if cid is None:
                    continue
                fbs = (rec.get("feedbacks") or rec.get("feedback") or
                       rec.get("texts") or [])
                merged[cid] = _coerce_feedback_list(fbs)
                # Preserve split alongside feedbacks (Qualcomm puts it on the
                # feedbacks record too — useful when fine_grained isn't loaded)
                if rec.get("split") is not None:
                    self.clip_to_split[cid] = str(rec["split"])
        else:
            raise ValueError(
                f"Unrecognised feedbacks_short_clips.json shape: "
                f"top-level is {type(data).__name__}, expected dict or list"
            )
        self.clip_to_feedbacks.update(merged)

    def load_fine_grained(self, path: Path) -> None:
        """Load fine_grained_labels.json into clip_to_fine."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged: dict[str, dict] = {}
        if isinstance(data, dict):
            for cid, rec in data.items():
                merged[_extract_clip_id(cid)] = _coerce_fine_record(rec)
        elif isinstance(data, list):
            for rec in data:
                cid = _record_clip_id(rec)
                if cid is None:
                    continue
                merged[cid] = _coerce_fine_record(rec)
                if rec.get("split") is not None:
                    self.clip_to_split[cid] = str(rec["split"])
        else:
            raise ValueError(
                f"Unrecognised fine_grained_labels.json shape: "
                f"top-level is {type(data).__name__}, expected dict or list"
            )
        self.clip_to_fine.update(merged)

    # ── Lookup ───────────────────────────────────────────────────────────

    def lookup(self, clip_id) -> dict:
        """Return a unified label dict for one clip.

        Always returns a dict with consistent keys; missing fields are None
        / empty list. Safe to call before either manifest is loaded.
        """
        cid = _extract_clip_id(clip_id)
        feedbacks = self.clip_to_feedbacks.get(cid, [])
        fine = self.clip_to_fine.get(cid, {})
        split = (fine.get("split") if fine else None) or self.clip_to_split.get(cid)
        # Prefer the dedicated worker-id manifest over any legacy human_id
        # field that may have appeared inside fine_grained_labels.json
        # (Qualcomm's main manifest does NOT ship them, but be defensive).
        human_id = self.clip_to_human.get(cid) or fine.get("human_id")
        return {
            "clip_id":       cid,
            "feedbacks":     feedbacks,
            "exercise":      fine.get("exercise"),
            "variation":     fine.get("variation"),
            "fine_class":    fine.get("fine_class"),
            "labels":        fine.get("labels", []),
            "human_id":      human_id,
            "split":         split,
            "n_feedbacks":   len(feedbacks),
        }

    def derive_label(self, clip_id) -> dict:
        """Return the FULL training-ready label dict for one clip.

        Combines:
          - quality scalar (from feedbacks + label tiers + silence default)
          - 10-vec joint groups (union of feedback-derived + class-derived)
          - exercise / variation / split passthroughs

        This is the function the dataset builder will call per clip at
        training-data assembly time.
        """
        rec = self.lookup(clip_id)
        feedbacks = rec["feedbacks"]
        labels_list = rec["labels"]

        quality = compute_clip_quality(feedbacks, labels_list)

        groups = np.zeros(N_JOINT_GROUPS, dtype=bool)
        for fb in feedbacks:
            groups |= derive_joint_groups(fb)
        for lbl in labels_list:
            groups |= derive_joint_groups_from_class(lbl)

        return {
            **rec,
            "quality":              quality,
            "joint_groups":         groups.tolist(),
            "quality_source":       _quality_source(feedbacks, labels_list),
        }

    # ── Aggregate stats (D3 sanity helpers) ──────────────────────────────

    def stats(self) -> dict:
        """Cheap summary used by D3 label-sanity gate."""
        n_clips_with_feedback = len(self.clip_to_feedbacks)
        n_clips_with_fine = len(self.clip_to_fine)
        n_total_feedbacks = sum(
            len(v) for v in self.clip_to_feedbacks.values()
        )
        # Pull human IDs from the dedicated worker-ids manifest first,
        # then any legacy human_id field on the fine record.
        unique_humans = set(self.clip_to_human.values())
        unique_humans |= {
            r.get("human_id") for r in self.clip_to_fine.values()
            if r.get("human_id") is not None
        }
        return {
            "n_clips_with_feedback":   n_clips_with_feedback,
            "n_clips_with_fine":       n_clips_with_fine,
            "n_total_feedbacks":       n_total_feedbacks,
            "n_clips_with_worker_id":  len(self.clip_to_human),
            "n_unique_humans":         len(unique_humans),
        }

    def __len__(self) -> int:
        return len(set(self.clip_to_feedbacks) | set(self.clip_to_fine))


def _quality_source(feedbacks, labels_list) -> str:
    """Diagnostic: where did this clip's quality come from?"""
    has_fb = bool(feedbacks)
    has_lbl_match = (
        labels_list is not None and quality_from_labels(labels_list) is not None
    )
    if has_fb and has_lbl_match: return "feedback+label"
    if has_fb:                   return "feedback_only"
    if has_lbl_match:            return "label_only"
    return "silence_default"


def _extract_clip_id(raw_id) -> str:
    """Normalise a raw clip identifier to canonical form.

    Accepts ``./00000009.mp4``, ``00000009.mp4``, ``00000009``, ``9`` (int)
    and returns ``00000009`` (8-digit zero-padded). Non-numeric IDs are
    returned as-is (covers FIT-COACH long-range files like ``0001`` or
    user-defined IDs).
    """
    s = str(raw_id).strip()
    if s.startswith("./"):
        s = s[2:]
    if s.endswith(".mp4"):
        s = s[:-4]
    # Strip any leading directory components
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if "\\" in s:
        s = s.rsplit("\\", 1)[-1]
    if s.isdigit():
        # Match Qualcomm's 8-digit short-clip convention
        return s.zfill(8)
    return s


def _record_clip_id(rec: dict) -> str | None:
    """Pull the clip ID out of a record by trying common key names."""
    if not isinstance(rec, dict):
        return None
    raw = (rec.get("video_path") or rec.get("clip_id") or rec.get("id")
           or rec.get("video_id") or rec.get("long_range_video_file"))
    return _extract_clip_id(raw) if raw is not None else None


def _coerce_feedback_list(fbs) -> list[str]:
    """Accept (a) list[str], (b) list[dict{text}], (c) string -> wrap as list."""
    if fbs is None:
        return []
    if isinstance(fbs, str):
        return [fbs]
    out: list[str] = []
    for fb in fbs:
        if isinstance(fb, str):
            out.append(fb)
        elif isinstance(fb, dict):
            t = fb.get("text") or fb.get("feedback") or fb.get("string")
            if t:
                out.append(str(t))
    return out


def _coerce_fine_record(rec) -> dict:
    """Pull the canonical fields out of a possibly-messy record dict.

    Real Qualcomm fine_grained_labels.json schema:
        {
          "video_path": "./00000000.mp4",
          "labels": ["elbow plank - stopping early"],
          "labels_descriptive": ["elbow plank - User stopped..."],
          "split": "train"
        }
    Each label is "<exercise> - <variation/error>". We split on " - " to
    isolate the exercise name from the variation.
    """
    if not isinstance(rec, dict):
        return {"raw": rec}
    labels = rec.get("labels") or []
    descriptive = rec.get("labels_descriptive") or []
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(descriptive, str):
        descriptive = [descriptive]

    # Best-effort split: '<exercise> - <variation>'. Take the first label
    # as canonical when there are several.
    exercise = None
    variation = None
    fine_class = None
    if labels:
        first = labels[0]
        fine_class = first
        if " - " in first:
            ex, var = first.split(" - ", 1)
            exercise = ex.strip()
            variation = var.strip()
        else:
            exercise = first.strip()

    return {
        "exercise":           rec.get("exercise") or rec.get("exercise_class") or exercise,
        "variation":          rec.get("variation") or rec.get("variation_name") or variation,
        "fine_class":         rec.get("fine_class") or rec.get("class") or fine_class,
        "labels":             labels,           # all raw labels, preserved
        "labels_descriptive": descriptive,      # all raw descriptive, preserved
        "human_id":           rec.get("human_id") or rec.get("subject") or rec.get("participant_id"),
        "split":              rec.get("split"),
        "extra":              {k: v for k, v in rec.items()
                                if k not in {"exercise", "exercise_class",
                                             "variation", "variation_name",
                                             "fine_class", "class", "label",
                                             "labels", "labels_descriptive",
                                             "human_id", "subject",
                                             "participant_id", "split",
                                             "video_path", "clip_id", "id",
                                             "video_id"}},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality derived from fine-grained variation labels
# ─────────────────────────────────────────────────────────────────────────────
#
# Empirical sweep of all 298,089 fine_grained_labels.json records
# (2026-05-06) revealed that Qualcomm encodes quality directly in the
# variation string itself, not just in NL feedback. Top patterns out of
# 661 unique variations include 'average' (23k), 'no obvious issue' (14k),
# 'as fast as possible' (9k), 'stopping early' (5k), 'rom=1..5', 'lazy',
# 'too fast', 'rounded back', etc.
#
# Without this label-based signal, ~85.5% of clips have no NL feedback at
# all and would all collapse to the silence-default of 0.7 -- a unimodal
# histogram that fails master plan §II.4.4 D3 hard-stop.

# Phrase -> quality scalar in [0, 1]. Ordered conceptually high->low; lookup
# is longest-match-first via regex word-boundary search to avoid spurious
# substring hits ("head" inside "head straight").
VARIATION_QUALITY_TIERS: dict[str, float] = {
    # ── HIGH (correct execution) ─────────────────────────────────────────
    "no obvious issue":     0.92,
    "no obvious error":     0.92,
    "no issue":             0.92,
    "no error":             0.92,
    "perfect":              0.95,
    "ideal":                0.95,
    "rom=5":                0.92,
    "rom=4":                0.85,
    "as fast as possible":  0.85,
    "with maximum effort":  0.85,
    "maximum effort":       0.85,

    # ── GOOD (mostly correct, minor descriptor) ──────────────────────────
    "average":              0.78,
    "rom=3":                0.72,
    "head straight":        0.80,
    "shoulder-width":       0.80,
    "shoulder width":       0.80,
    "arms shoulder width":  0.80,
    "neutral":              0.80,
    "arms extended":        0.80,
    "arms straight":        0.80,
    "back straight":        0.85,
    "feet shoulder width":  0.80,
    "feet hip width":       0.80,

    # ── MILD ERROR (form deviation, not severe) ──────────────────────────
    "rom=2":                0.55,
    "slow":                 0.60,
    "wide":                 0.55,
    "narrow":               0.55,
    "head up":              0.55,
    "head down":            0.55,
    "looking down":         0.55,
    "looking up":           0.55,
    "low range of motion":  0.50,
    "knee on ground":       0.55,   # often a prescribed modification

    # ── MAJOR ERROR (severe form / pacing problems) ──────────────────────
    "rom=1":                0.40,
    "stopping early":       0.40,
    "stops early":          0.40,
    "starting late":        0.40,
    "lazy":                 0.40,
    "shallow":              0.45,
    "too shallow":          0.40,
    "too fast":             0.45,
    "too slow":             0.45,
    "too low":              0.45,
    "too high":             0.45,
    "too wide":             0.45,
    "too narrow":           0.45,
    "rounded back":         0.40,
    "leaning forward":      0.45,
    "leaning back":         0.45,
    "arched back":          0.45,
    "bowed":                0.40,
    "wobbling":             0.45,
    "flaring":              0.45,
    "flared":               0.45,
    "caving":               0.40,
    "cave in":              0.40,
    "collapsing":           0.35,
    "broken":               0.30,
    "incorrect":            0.30,
    "wrong":                0.30,
    "punching down":        0.45,
}

# Pre-compile a single regex over all phrases (longest-first); used for
# fast scan over the variation string. Returns the matched phrase so we
# can index into VARIATION_QUALITY_TIERS.
_VARIATION_PATTERNS_SORTED = sorted(
    VARIATION_QUALITY_TIERS.keys(), key=len, reverse=True,
)
_VARIATION_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(p) for p in _VARIATION_PATTERNS_SORTED)
    + r")(?!\w)"
)


def quality_from_label(label: str) -> Optional[float]:
    """Return a quality scalar derived from one fine-grained label.

    `label` is e.g. ``'elbow plank - stopping early'``. We strip the
    exercise prefix (everything before " - ") and search the remainder
    against ``VARIATION_QUALITY_TIERS``. First (longest) match wins.

    Returns
    -------
    float in [0, 1] when a known variation pattern matched, else None.
    None means "no quality signal here" -- caller decides how to combine
    with feedback-derived quality or fall back to the silence default.
    """
    if not label:
        return None
    norm = _normalise(label)
    # Strip "exercise - variation" prefix; tolerate labels with no " - ".
    if " - " in norm:
        norm = norm.split(" - ", 1)[1].strip()
    if not norm:
        return None
    m = _VARIATION_RE.search(norm)
    if m is None:
        return None
    return float(VARIATION_QUALITY_TIERS[m.group(1)])


def quality_from_labels(labels: list[str]) -> Optional[float]:
    """Average quality over all fine-grained labels on a clip.

    Multi-label clips (several variations on the same clip) get averaged.
    Returns None if zero labels matched any tier.
    """
    if not labels:
        return None
    qs = [q for q in (quality_from_label(lbl) for lbl in labels) if q is not None]
    if not qs:
        return None
    return float(sum(qs) / len(qs))


def quality_from_feedbacks(feedbacks: Sequence[dict | str]) -> Optional[float]:
    """Quality from NL feedback strings, or None if feedbacks are empty.

    Same formula as ``derive_quality_scalar`` but does NOT apply the 0.7
    silence default -- callers can combine None-result with label-based
    quality before falling back.
    """
    if not feedbacks:
        return None
    return derive_quality_scalar(feedbacks)


def compute_clip_quality(
    feedbacks: Sequence[dict | str],
    labels: list[str],
    *,
    silence_default: float = Q_DEFAULT_NO_FEEDBACK,
) -> float:
    """Combine NL-feedback signal + label signal into one quality scalar.

    Both signals contribute equally when both present. When only one is
    present, that one is used. When neither, fall back to the silence
    default (0.7). Always returns a float in [0, 1].
    """
    q_fb = quality_from_feedbacks(feedbacks)
    q_lbl = quality_from_labels(labels)
    if q_fb is not None and q_lbl is not None:
        q = 0.5 * q_fb + 0.5 * q_lbl
    elif q_fb is not None:
        q = q_fb
    elif q_lbl is not None:
        q = q_lbl
    else:
        q = silence_default
    return float(max(0.0, min(1.0, q)))


# ─────────────────────────────────────────────────────────────────────────────
# Fine-grained class -> joint-group helper
# ─────────────────────────────────────────────────────────────────────────────


def derive_joint_groups_from_class(class_label: str) -> np.ndarray:
    """Treat a fine-grained class label like 'pushup_HipsTooLow' as text and
    fire the matching joint-groups via the same lexicon as feedback strings.

    Examples (from QEVD paper Fig 2 pushup variations):
      'Hips Too Low'   -> L_hip + R_hip
      'Hips Too High'  -> L_hip + R_hip
      'Elbows Flared'  -> L_elbow + R_elbow
      'Head Down'      -> Neck
      'Side to Side'   -> Trunk (no direct match -> may need calibration)
      'On Knees'       -> L_knee + R_knee  (modification, not strict error)
    """
    if not class_label:
        return np.zeros(N_JOINT_GROUPS, dtype=bool)
    # Replace _ with space so 'HipsTooLow' / 'pushup_HipsTooLow' both work
    text = class_label.replace("_", " ")
    # Insert spaces before capitals so 'HipsTooLow' -> 'Hips Too Low'
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
    return derive_joint_groups(text)


__all__ = [
    "AFFIRMATIVE_KEYWORDS",
    "BILATERAL_KEYWORDS",
    "CORRECTIVE_KEYWORDS",
    "FIT_COACH_PASSTHROUGH_TYPES",
    "FeedbackType",
    "GROUP_KEYWORDS",
    "JOINT_GROUP_NAMES",
    "JOINT_GROUP_TO_IDX",
    "N_JOINT_GROUPS",
    "QEVDLabels",
    "Q_DEFAULT_NO_FEEDBACK",
    "VARIATION_QUALITY_TIERS",
    "build_clip_label",
    "classify_feedback_type",
    "compute_clip_quality",
    "derive_joint_groups",
    "derive_joint_groups_from_class",
    "derive_quality_scalar",
    "quality_from_feedbacks",
    "quality_from_label",
    "quality_from_labels",
]
