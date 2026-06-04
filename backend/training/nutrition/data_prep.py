"""
N2 data prep — convert the Kaggle Food.com CSVs into the pickle format the
vendored recipe_gen pipeline expects (the upstream loader keys on these names):

    recipes.pkl                  <- PP_recipes.csv     (list cols parsed)
    user_rep.pkl                 <- PP_users.csv       (list cols parsed, indexed by 'u')
    interactions_train.pkl       <- interactions_train.csv
    interactions_valid_new.pkl   <- interactions_validation.csv   (renamed)
    interactions_test_new.pkl    <- interactions_test.csv         (renamed)

Pure pandas, no LLM/model. Runs in ~1-2 min on Colab. Called from notebook Cell B.
"""

from __future__ import annotations

import ast
import os
from typing import Iterable

import pandas as pd

_LIST_COLS_RECIPES = ("name_tokens", "ingredient_tokens", "steps_tokens",
                      "techniques", "ingredient_ids")
_LIST_COLS_USERS = ("techniques", "items", "ratings")
_SPLIT_RENAME = {
    "interactions_train.csv": "interactions_train.pkl",
    "interactions_validation.csv": "interactions_valid_new.pkl",
    "interactions_test.csv": "interactions_test_new.pkl",
}


def _parse_lists(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].apply(ast.literal_eval)
    return df


def prepare(src_dir: str, out_dir: str) -> dict:
    """Convert all five files. Returns row counts per output for verification."""
    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}

    recipes = pd.read_csv(os.path.join(src_dir, "PP_recipes.csv"))
    _parse_lists(recipes, _LIST_COLS_RECIPES).to_pickle(os.path.join(out_dir, "recipes.pkl"))
    counts["recipes.pkl"] = len(recipes)

    users = pd.read_csv(os.path.join(src_dir, "PP_users.csv"))
    _parse_lists(users, _LIST_COLS_USERS).set_index("u").to_pickle(
        os.path.join(out_dir, "user_rep.pkl"))
    counts["user_rep.pkl"] = len(users)

    for src, dst in _SPLIT_RENAME.items():
        df = pd.read_csv(os.path.join(src_dir, src))
        df.to_pickle(os.path.join(out_dir, dst))
        counts[dst] = len(df)
    return counts


if __name__ == "__main__":  # CLI: python -m ... data_prep <src> <out>
    import json
    import sys

    src, out = sys.argv[1], sys.argv[2]
    print(json.dumps(prepare(src, out), indent=2))
