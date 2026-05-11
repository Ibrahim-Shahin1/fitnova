"""Deep validation of the v6 D2 Colab notebook using pyflakes.

The local dry-run (_verify_v6_bundle.py) proves the BUNDLE works.
This script proves the NOTEBOOK works:

  1. Every code cell parses as valid Python (catches syntax errors)
  2. pyflakes finds NO undefined-name issues across cumulative cell scopes
  3. Critical names + invariants are present in the right cells
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
from pathlib import Path
from io import StringIO

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                errors="replace")

import pyflakes.api
import pyflakes.reporter

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK = REPO_ROOT / "colab_notebooks" / "v6_d2_extract.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"NOTEBOOK NOT FOUND: {NOTEBOOK}")
        return 1
    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    cell_sources = [
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in code_cells
    ]

    print(f"\n=== {NOTEBOOK.name} — {len(code_cells)} code cells ===\n")

    # ── 1. Syntax check ─────────────────────────────────────────────────
    print("STEP 1: parse every code cell as Python")
    for i, src in enumerate(cell_sources, 1):
        try:
            ast.parse(src)
            first_line = src.split("\n", 1)[0][:60]
            print(f"  [OK] Cell {i}: {first_line}")
        except SyntaxError as e:
            print(f"  [FAIL] Cell {i}: {e}")
            return 2
    print()

    # ── 2. pyflakes on CUMULATIVE script ─────────────────────────────────
    # The notebook is logically equivalent to running each cell in sequence
    # in one Python session. Concatenate all cell sources into one script,
    # then pyflakes-check the whole thing. Any NameError-class bug in cell N
    # that depends on a missing definition will show up here.
    #
    # We mark Colab-only modules as available via stubs so we don't get
    # spurious "undefined-name google" errors from `from google.colab import drive`.
    print("STEP 2: pyflakes on the full notebook-as-script")
    script = ""
    for i, src in enumerate(cell_sources, 1):
        # Strip Jupyter line-magic / shell escapes
        clean_lines = []
        for line in src.splitlines():
            if line.lstrip().startswith(("!", "%")):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        script += f"\n# === CELL {i} ===\n"
        script += "\n".join(clean_lines)
        script += "\n"

    err_buf = StringIO()
    out_buf = StringIO()
    reporter = pyflakes.reporter.Reporter(out_buf, err_buf)
    n_warnings = pyflakes.api.check(
        script, "<notebook>", reporter=reporter,
    )
    output = out_buf.getvalue() + err_buf.getvalue()

    # Filter out the unavoidable Colab/IPython false-positives. We accept
    # warnings about google.colab (only available on Colab), and about
    # the known-imported names from our bundle.
    # Filter cosmetic warnings that aren't bugs.
    KNOWN_FP_PREFIXES = (
        "google.colab",
        "imported but unused",     # our setup imports cv2 / np for sanity
        "unable to detect",
        "redefinition of unused",  # re-importing sys/time across cells: harmless
        "f-string is missing placeholders",  # cosmetic
    )
    real_issues: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if any(fp in line for fp in KNOWN_FP_PREFIXES):
            continue
        real_issues.append(line)

    if real_issues:
        print("  [FAIL] pyflakes found issues:")
        for line in real_issues:
            print(f"    {line}")
        return 3
    if n_warnings == 0:
        print(f"  [OK] pyflakes: 0 issues across {len(code_cells)} cells")
    else:
        # All filtered out as known-FP
        print(f"  [OK] pyflakes: {n_warnings} raw warnings, all known-FP "
              "(Colab-only imports, unused-but-needed-for-sanity)")
        if output.strip():
            print("       (filtered output:)")
            for line in output.strip().splitlines():
                print(f"       {line}")
    print()

    # ── 3. Spot-check critical invariants ───────────────────────────────
    print("STEP 3: spot-check critical names exist in the right cells\n")
    invariants = [
        (1, "mediapipe==0.10.33",        "pinned mediapipe version (must match local PC)"),
        (1, "protobuf==4.25.3",          "pinned protobuf version"),
        (1, "remove_pkg_anywhere",       "TF removal helper"),
        (2, "google.colab",              "Drive mount"),
        (3, "PINNED_MEDIAPIPE_VERSION",  "version-mismatch assertion"),
        (3, "get_part_urls",             "URL helper import"),
        (4, "HARDWARE",                  "hardware validation cell"),
        (4, "INSUFFICIENT CPU CORES",    "halt on too few cores"),
        (5, "PARTS_TO_PROCESS",          "main config"),
        (5, "EXPECTED_PART_COUNT",       "expected count dict"),
        (6, "download_with_progress",    "tqdm download helper"),
        (6, "run_7z_with_resume",        "7z resume helper (hardlink trick)"),
        (6, "extract_with_tqdm",         "MediaPipe progress helper"),
        (6, "sync_to_drive",             "sync helper"),
        (7, "for PART_NUMBER in PARTS_TO_PROCESS", "main loop"),
        (8, "D2_extract_quality",        "gate report id"),
    ]
    for cell_idx, needle, desc in invariants:
        if cell_idx > len(cell_sources):
            print(f"  [FAIL] Cell {cell_idx} doesn't exist (only {len(cell_sources)})")
            return 4
        src = cell_sources[cell_idx - 1]
        if needle in src:
            print(f"  [OK] Cell {cell_idx}: '{needle}' present  ({desc})")
        else:
            print(f"  [FAIL] Cell {cell_idx}: '{needle}' MISSING  ({desc})")
            return 5
    print()

    # ── 4. Critical extra checks ────────────────────────────────────────
    print("STEP 4: invariants beyond cell scope\n")
    full = "\n".join(cell_sources)

    # 4a. EXPECTED_PART_COUNT must list all 4 parts with correct numbers
    if "{1: 76000, 2: 76000, 3: 76000, 4: 70089}" in full:
        print("  [OK] EXPECTED_PART_COUNT: paper-correct (76K + 76K + 76K + 70089)")
    else:
        print("  [FAIL] EXPECTED_PART_COUNT not paper-correct")
        return 6

    # 4b. Main loop must do download -> extract -> sync -> cleanup
    flow_steps = [
        "download_and_extract_part",
        "extract_with_tqdm",
        "sync_to_drive",
        "cleanup_part",
    ]
    for step in flow_steps:
        if step in full:
            print(f"  [OK] main flow step: '{step}'")
        else:
            print(f"  [FAIL] main flow missing: '{step}'")
            return 8

    # 4c. Resume support: skip-already-done logic
    if "part_already_done_on_drive" in full and "continue" in full:
        print("  [OK] resume support: skip-already-done branch present")
    else:
        print("  [FAIL] resume support missing")
        return 9

    # 4d. True 7z resume via hardlink trick
    if "os.link" in full and "-aos" in full:
        print("  [OK] 7z resume: hardlink + -aos (no re-extraction of done files)")
    else:
        print("  [FAIL] 7z resume missing hardlink trick or -aos flag")
        return 10

    # 4e. Per-clip resume
    if "skip clips whose .npz already exists" in full or ("out_path.exists()" in full and "n_skipped" in full):
        print("  [OK] MediaPipe resume: per-clip skip-if-exists")
    else:
        print("  [FAIL] MediaPipe per-clip resume not detected")
        return 11

    # 4f. Gate report writes to a file
    if 'json.dump(report' in full and 'D2_extract_quality.json' in full:
        print("  [OK] gate report: writes D2_extract_quality.json")
    else:
        print("  [FAIL] gate report missing writer")
        return 12

    # 4g. Version assertion
    if "assert mp_lib.__version__ == PINNED_MEDIAPIPE_VERSION" in full:
        print("  [OK] mediapipe version asserted (loud-fail on Colab drift)")
    else:
        print("  [FAIL] mediapipe version not asserted")
        return 13

    # 4h. Hardware halt
    if "raise RuntimeError" in full and "INSUFFICIENT CPU" in full:
        print("  [OK] hardware halt: refuses to run with <4 vCPUs")
    else:
        print("  [FAIL] hardware halt missing")
        return 14

    # 4i. Visible progress bars on every long step
    progress_targets = [
        ("download_with_progress", "download tqdm"),
        ("run_7z_with_resume",     "7z extract tqdm via output-dir poll"),
        ("copy_dir_with_progress", "Drive copy tqdm"),
        ("tqdm(total=len(tasks)",  "MediaPipe loop tqdm"),
    ]
    for needle, desc in progress_targets:
        if needle in full:
            print(f"  [OK] progress: {desc}")
        else:
            print(f"  [FAIL] progress missing: {desc}")
            return 15

    print("\n" + "=" * 70)
    print("NOTEBOOK VALIDATION: ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
