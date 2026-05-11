"""Tests for the defect-name to UI-region rephrase lookup."""
from __future__ import annotations

import pytest

from backend.config.defect_to_region import (
    REGION_ARMS,
    REGION_ELBOWS,
    REGION_HEAD,
    REGION_HIPS,
    REGION_KNEES,
    REGION_LEGS,
    REGION_NECK,
    REGION_SHOULDERS,
    REGION_TRUNK,
    REGION_WRISTS,
    regions_for_defect,
    regions_for_defects,
)


# ---- Cases drawn from real QEVD squat / pushup variations -----------------


@pytest.mark.parametrize("variation,expected", [
    ("knees over toes",        {REGION_KNEES}),
    ("back not straight",      {REGION_TRUNK}),
    ("elbows flared",          {REGION_ELBOWS}),
    ("head down",              {REGION_HEAD}),
    ("hips too low",           {REGION_HIPS}),
    ("hips too high",          {REGION_HIPS}),
    ("shoulders rolled",       {REGION_SHOULDERS}),
    ("legs not straight",      {REGION_LEGS}),
    ("arms too wide",          {REGION_ARMS}),
    ("wrists collapsed",       {REGION_WRISTS}),
    ("neck strained",          {REGION_NECK}),
    ("chin tucked",            {REGION_NECK}),
    ("spine flexed",           {REGION_TRUNK}),
    ("core not engaged",       {REGION_TRUNK}),
])
def test_named_body_part_maps_to_region(variation, expected):
    assert regions_for_defect(variation) == expected


@pytest.mark.parametrize("variation", [
    "shallow",
    "narrow",
    "wide",
    "no obvious issue",
    "stopping early",
    "starting late",
    "hold",
    "plie",
    "90 degrees",
    "fast",
    "slow",
    "speed=0.40 rps",
    "squat_depth=3",
])
def test_no_body_part_returns_empty(variation):
    assert regions_for_defect(variation) == set()


# ---- Word boundary safety -------------------------------------------------


def test_shoulder_width_fires_shoulders():
    # "shoulder-width" literally contains the word "shoulder" (hyphen acts
    # as a word boundary). The function honestly returns shoulders. In the
    # production path, upstream filtering should drop "shoulder-width" from
    # the defect set before calling this function — it's a NEUTRAL stance
    # label, not a defect. The function does not special-case it.
    assert regions_for_defect("shoulder-width") == {REGION_SHOULDERS}


def test_kneeling_does_not_fire_knees():
    # "kneeling" contains "knee" but \bknee\b does not match inside it.
    assert regions_for_defect("kneeling on the floor") == set()


def test_headphones_does_not_fire_head():
    # "headphones" contains "head" but \bhead\b does not match inside it.
    assert regions_for_defect("headphones on") == set()


def test_backwards_does_not_fire_back():
    # "backwards" starts with "back" — \bback\b should NOT match.
    assert regions_for_defect("moving backwards") == set()


# ---- Multi-region defects -------------------------------------------------


def test_multiple_body_parts_in_one_variation():
    # "head down and back rounded" should fire both
    assert regions_for_defect("head down and back rounded") == {REGION_HEAD, REGION_TRUNK}


def test_legs_and_hips_on_floor():
    # Real QEVD label: "elbow plank - legs and hips on the floor"
    assert regions_for_defect("legs and hips on the floor") == {REGION_LEGS, REGION_HIPS}


# ---- Plurals + case insensitivity -----------------------------------------


@pytest.mark.parametrize("singular,plural", [
    ("knee bent",     "knees bent"),
    ("elbow flared",  "elbows flared"),
    ("shoulder up",   "shoulders up"),
    ("hip high",      "hips high"),
    ("arm wide",      "arms wide"),
    ("leg out",       "legs out"),
    ("wrist down",    "wrists down"),
])
def test_singular_and_plural_both_match(singular, plural):
    assert regions_for_defect(singular) == regions_for_defect(plural)


def test_case_insensitive():
    assert regions_for_defect("KNEES OVER TOES") == {REGION_KNEES}
    assert regions_for_defect("Back Not Straight") == {REGION_TRUNK}


# ---- Empty / null inputs --------------------------------------------------


def test_empty_string_returns_empty():
    assert regions_for_defect("") == set()


def test_whitespace_only_returns_empty():
    assert regions_for_defect("   ") == set()


# ---- Batch helper ---------------------------------------------------------


def test_regions_for_defects_unions():
    variations = ["knees over toes", "back not straight", "shallow"]
    assert regions_for_defects(variations) == {REGION_KNEES, REGION_TRUNK}


def test_regions_for_defects_empty_input():
    assert regions_for_defects([]) == set()


def test_regions_for_defects_all_empty_variations():
    assert regions_for_defects(["shallow", "narrow", "wide"]) == set()
