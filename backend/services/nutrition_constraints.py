"""
FitNova - Nutrition Constraint Engine
=====================================
Single source of truth for dietary constraints. Expands user-facing exclusion
terms and diet labels into the concrete ingredient keywords that must NOT appear
in a recipe, and verifies meals/plans against them.

This is what makes "no fish" actually exclude tuna, salmon, cod, etc. — a literal
substring match never would. Used at BOTH selection time (so violating recipes are
never picked) and as a post-build verification pass (so the final plan is provably
constraint-clean). Pure Python, no LLM.
"""

from __future__ import annotations

import re

# Category -> concrete ingredient words that belong to it. Excluding the category
# excludes every member (e.g. "no fish" must also drop tuna, salmon, cod...).
EXCLUSION_GROUPS: dict[str, list[str]] = {
    "fish": ["fish", "tuna", "salmon", "cod", "tilapia", "halibut", "trout",
             "sardine", "anchovy", "anchovies", "mackerel", "haddock", "snapper",
             "catfish", "pollock", "mahi", "swordfish", "herring", "perch",
             "flounder", "sole", "bass", "grouper", "caviar", "roe"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
                  "scallop", "squid", "calamari", "octopus", "crayfish", "crawfish"],
    "pork": ["pork", "bacon", "ham", "sausage", "prosciutto", "pancetta",
             "chorizo", "salami", "pepperoni", "lard"],
    "beef": ["beef", "steak", "veal"],
    "poultry": ["chicken", "turkey", "duck", "poultry"],
    "meat": ["beef", "steak", "veal", "pork", "bacon", "ham", "sausage",
             "chicken", "turkey", "duck", "lamb", "mutton", "venison", "bison",
             "prosciutto", "pancetta", "chorizo", "salami", "pepperoni", "poultry"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee",
              "buttermilk", "custard", "mozzarella", "parmesan", "cheddar",
              "ricotta", "feta"],
    "egg": ["egg", "mayonnaise", "mayo"],
    "gluten": ["wheat", "flour", "bread", "pasta", "barley", "rye", "couscous",
               "cracker", "breadcrumb", "noodle", "macaroni", "spaghetti"],
    "peanut": ["peanut"],
    "tree nut": ["almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut",
                 "macadamia", "pine nut", "brazil nut"],
    "nut": ["peanut", "almond", "walnut", "cashew", "pecan", "pistachio",
            "hazelnut", "macadamia", "pine nut", "brazil nut"],
    "soy": ["soy", "tofu", "edamame", "tempeh", "miso", "soya"],
    "shrimp": ["shrimp", "prawn"],
}

# Common things a user might type -> canonical group key.
_ALIASES = {
    "fishes": "fish", "nuts": "nut", "tree nuts": "tree nut", "peanuts": "peanut",
    "shellfishes": "shellfish", "dairies": "dairy", "eggs": "egg",
    "milk": "dairy", "cheese": "dairy",
}

# Diet label -> categories that diet forbids (belt-and-suspenders on top of the
# Food.com diet tag, since tags are not 100% reliable).
DIET_IMPLIED_EXCLUSIONS: dict[str, list[str]] = {
    "vegetarian": ["fish", "shellfish", "meat"],
    "vegan": ["fish", "shellfish", "meat", "dairy", "egg"],
    "pescatarian": ["meat"],
    "dairy-free": ["dairy"],
    "gluten-free": ["gluten"],
}


def expand_exclusions(terms: list[str] | None, diet: list[str] | None = None) -> list[str]:
    """User exclusion terms + diet -> flat, lowercased, de-duplicated list of
    ingredient words to forbid (categories expanded to all members)."""
    out: set[str] = set()
    for t in (terms or []):
        s = str(t).strip().lower()
        if not s:
            continue
        out.add(s)  # keep the literal term too
        if s == "seafood":
            out.update(EXCLUSION_GROUPS["fish"])
            out.update(EXCLUSION_GROUPS["shellfish"])
            continue
        key = _ALIASES.get(s, s)
        if key in EXCLUSION_GROUPS:
            out.update(EXCLUSION_GROUPS[key])
    for d in (diet or []):
        for cat in DIET_IMPLIED_EXCLUSIONS.get(str(d).strip().lower(), []):
            out.update(EXCLUSION_GROUPS.get(cat, []))
    return sorted(out)


def build_exclusion_regex(expanded_terms: list[str]) -> re.Pattern | None:
    """Word-boundary regex over all forbidden words (plural-tolerant). Word
    boundaries avoid false positives like 'egg' matching 'eggplant' or 'ham'
    matching 'graham'. Returns None if nothing to exclude."""
    terms = [t for t in (expanded_terms or []) if t]
    if not terms:
        return None
    alt = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(r"\b(?:" + alt + r")s?\b", re.IGNORECASE)


def meal_violations(meal: dict, expanded_terms: list[str]) -> list[str]:
    """Return which forbidden words appear in a meal's ingredients/name."""
    hay = " ".join(str(i).lower() for i in meal.get("ingredients", []))
    hay += " " + str(meal.get("name", "")).lower()
    hits = []
    for t in expanded_terms:
        if t and re.search(r"\b" + re.escape(t) + r"s?\b", hay):
            hits.append(t)
    return sorted(set(hits))
