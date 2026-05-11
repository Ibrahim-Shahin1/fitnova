"""Pure-rephrase mapping from QEVD variation names to UI body regions.

This module is intentionally small and does NOT add any biomechanical
knowledge. It scans the variation TEXT for body-part keywords that
literally appear in it, and emits the corresponding region names for the
Flutter UI to highlight on the skeleton overlay.

Examples:
    "knees over toes"   -> {"knees"}        # text mentions knees
    "back not straight" -> {"trunk"}        # text mentions back
    "narrow"            -> set()            # no body part named
    "shallow"           -> set()            # no body part named
    "stopping early"    -> set()            # timing, no body part

For variations that do not name a body part, the UI should still display
the defect name as a chip — it just does not highlight a specific region
on the skeleton. Per plan §6: "the defect name names the joint" is the
only mapping rule. No biomech inference, no curated severity, no
quality scalar.

Used at runtime in `form_session.py` (backend) to translate the model's
predicted defect classes into highlight signals for the Flutter overlay.
"""
from __future__ import annotations

import re
from typing import Iterable

# Region keys mirror the UI region scheme. Keep alphabetical for diff
# readability. The Flutter side has a parallel constant set; if you add
# a region here, add it there too.
REGION_KNEES = "knees"
REGION_ELBOWS = "elbows"
REGION_SHOULDERS = "shoulders"
REGION_HIPS = "hips"
REGION_TRUNK = "trunk"
REGION_NECK = "neck"
REGION_HEAD = "head"
REGION_WRISTS = "wrists"
REGION_ARMS = "arms"
REGION_LEGS = "legs"

# Map word-boundary regex tokens to the region they name. Both singular
# and plural forms; both left/right qualifiers (the UI does not currently
# distinguish sides for highlighting). Ordered most-specific to least so
# "shoulders" wins before "shoulder" if the same string is matched twice.
_KEYWORD_TO_REGION: dict[str, str] = {
    r"shoulders?":   REGION_SHOULDERS,
    r"elbows?":      REGION_ELBOWS,
    r"wrists?":      REGION_WRISTS,
    r"knees?":       REGION_KNEES,
    r"hips?":        REGION_HIPS,
    r"back":         REGION_TRUNK,
    r"spine":        REGION_TRUNK,
    r"core":         REGION_TRUNK,
    r"torso":        REGION_TRUNK,
    r"posture":      REGION_TRUNK,
    r"neck":         REGION_NECK,
    r"chin":         REGION_NECK,
    r"head":         REGION_HEAD,
    r"arms?":        REGION_ARMS,
    r"legs?":        REGION_LEGS,
}

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{kw}\b", re.IGNORECASE), region)
    for kw, region in _KEYWORD_TO_REGION.items()
]


def regions_for_defect(variation: str) -> set[str]:
    """Return the set of UI regions named by the variation text.

    Pure text scan with word boundaries. No semantic inference. A
    variation like "knees over toes" returns {"knees"}; "back not
    straight" returns {"trunk"}; "shallow" returns the empty set.
    """
    if not variation:
        return set()
    out: set[str] = set()
    for pattern, region in _COMPILED:
        if pattern.search(variation):
            out.add(region)
    return out


def regions_for_defects(variations: Iterable[str]) -> set[str]:
    """Union of regions across multiple predicted variations."""
    out: set[str] = set()
    for v in variations:
        out |= regions_for_defect(v)
    return out
