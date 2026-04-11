"""
One-time script to build exercise_media.json from ExerciseDB API.

Fetches all exercises from ExerciseDB (RapidAPI or GitHub), fuzzy-matches
them against the FitNova program catalog exercise names, and writes a new
exercise_media.json with real animated GIF URLs.

Usage:
    # With RapidAPI key:
    python -m backend.scripts.build_exercise_media --api-key YOUR_KEY

    # From GitHub raw JSON (no key needed, may be outdated):
    python -m backend.scripts.build_exercise_media --from-github
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import requests
from rapidfuzz import fuzz, process

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
CATALOG_PATH = os.path.join(_BASE_DIR, "backend", "data", "program_catalog.pkl")
OUTPUT_PATH  = os.path.join(_BASE_DIR, "backend", "data", "exercise_media.json")

# ── ExerciseDB sources ────────────────────────────────────────────────────────

RAPIDAPI_URL = "https://exercisedb.p.rapidapi.com/exercises"
GITHUB_URL   = (
    "https://raw.githubusercontent.com/bootstrapping-lab/"
    "exercisedb-api/main/src/data/exercises.json"
)
GIF_BASE     = "https://static.exercisedb.dev/media/"

# Minimum fuzzy score (0–100) to accept a match
MATCH_THRESHOLD = 82

# Words that carry no exercise-specific meaning — ignored in overlap check
_STOPWORDS = {
    'the','a','an','with','and','or','to','from','on','in','at',
    'single','double','one','two','three','arm','leg','hand','side',
}


def _has_word_overlap(name_a: str, name_b: str) -> bool:
    """Return True if both names share at least one non-stopword."""
    def words(s: str) -> set:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        return set(s.split()) - _STOPWORDS
    return bool(words(name_a) & words(name_b))


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_variants(name: str) -> list[str]:
    """
    Return multiple normalised forms of an exercise name for fuzzy matching.

    We try two forms so we match both ExerciseDB name styles:
    - "Squat (Barbell)"     → ["squat", "barbell squat"]
    - "Bench Press (Barbell)" → ["bench press", "barbell bench press"]
    - "Lat Pulldown"        → ["lat pulldown"]
    """
    def clean(s: str) -> str:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        return " ".join(s.split())

    paren_match = re.search(r"\(([^)]+)\)", name)
    equipment   = paren_match.group(1).strip() if paren_match else ""
    base        = re.sub(r"\([^)]*\)", "", name).strip()

    variants = [clean(base)]
    if equipment:
        variants.append(clean(f"{equipment} {base}"))
    return variants


def normalize(name: str) -> str:
    """Primary normalised form (strips parens). Used for ExerciseDB side."""
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return " ".join(s.split())


def fetch_from_rapidapi(api_key: str) -> list[dict]:
    print("Fetching exercises from RapidAPI (paginated, 10 per page)...")
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "exercisedb.p.rapidapi.com",
    }
    all_exercises: list[dict] = []
    limit = 10
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset}
        resp = requests.get(RAPIDAPI_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_exercises.extend(batch)
        print(f"  Fetched {len(all_exercises)} so far (offset={offset})...")
        if len(batch) < limit:
            break
        offset += limit

    print(f"  Total: {len(all_exercises)} exercises from RapidAPI")
    return all_exercises


def fetch_from_github() -> list[dict]:
    print("Fetching exercises from GitHub...")
    resp = requests.get(GITHUB_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Got {len(data)} exercises from GitHub")
    return data


def load_catalog_names() -> set[str]:
    with open(CATALOG_PATH, "rb") as f:
        catalog = pickle.load(f)

    names: set[str] = set()
    for prog in catalog.values():
        for ex in prog.get("week1_exercises", []):
            n = str(ex.get("exercise_name", "")).strip()
            if n:
                names.add(n)
    print(f"Catalog has {len(names)} unique exercise names")
    return names


def build_mapping(edb_exercises: list[dict], catalog_names: set[str]) -> dict:
    """Fuzzy-match catalog names → ExerciseDB GIF URLs."""

    # Build ExerciseDB lookup: normalised_name → entry metadata
    edb_lookup: dict[str, dict] = {}
    for ex in edb_exercises:
        norm = normalize(ex.get("name", ""))
        if not norm:
            continue
        # gifUrl may be a full URL already; otherwise build from id
        gif = ex.get("gifUrl") or f"{GIF_BASE}{ex.get('id', '')}.gif"
        edb_lookup[norm] = {
            "gif_url":      gif,
            "original_name": ex.get("name", ""),
        }

    edb_norm_list = list(edb_lookup.keys())
    results: dict[str, dict] = {}
    unmatched: list[str] = []

    for our_name in sorted(catalog_names):
        best_match = None
        best_score = 0.0

        # Try all normalised variants, keep the highest-scoring match
        for variant in normalize_variants(our_name):
            m = process.extractOne(
                variant,
                edb_norm_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=MATCH_THRESHOLD,
            )
            if m and m[1] > best_score:
                best_match = m
                best_score = m[1]

        if best_match:
            matched_norm, score, _ = best_match
            # Reject if no significant word overlaps (e.g. "Curl" → "Squat")
            if not _has_word_overlap(our_name, edb_lookup[matched_norm]["original_name"]):
                unmatched.append(our_name)
                continue
            entry = edb_lookup[matched_norm]
            results[our_name] = {
                "gif_url":    entry["gif_url"],
                "type":       "gif",
                "_matched_to": entry["original_name"],
                "_score":      score,
            }
        else:
            unmatched.append(our_name)

    print(f"\nMatched   : {len(results):,} / {len(catalog_names):,} exercises")
    print(f"Unmatched : {len(unmatched):,}")
    if unmatched[:15]:
        print("  First 15 unmatched:", unmatched[:15])

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exercise_media.json from ExerciseDB"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--api-key", metavar="KEY", help="RapidAPI key for ExerciseDB")
    group.add_argument(
        "--from-github",
        action="store_true",
        help="Download exercise data from GitHub (no key required)",
    )
    args = parser.parse_args()

    # 1. Load catalog
    catalog_names = load_catalog_names()

    # 2. Fetch ExerciseDB data
    if args.from_github:
        edb = fetch_from_github()
    else:
        edb = fetch_from_rapidapi(args.api_key)

    # 3. Build mapping
    mapping = build_mapping(edb, catalog_names)

    # 4. Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {OUTPUT_PATH}")
    print(f"Coverage: {len(mapping):,} exercises now have GIF demos")


if __name__ == "__main__":
    main()
