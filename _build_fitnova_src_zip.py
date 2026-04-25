"""
Build a fresh `fitnova_src.zip` containing the backend source code that Colab
will extract to /content/ before running training.

Usage:
    python _build_fitnova_src_zip.py

Output:
    fitnova_src.zip  (in the current working directory)

After building, upload the zip to the ROOT of your Google Drive, replacing any
previous `MyDrive/fitnova_src.zip` (the Colab notebook will happily overwrite).

If you want to re-upload after editing backend code: delete
`MyDrive/fitnova_src.zip` on Drive first, otherwise Cell 4 will restore the
old zip from Drive instead of asking for the new upload.
"""
import os
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
OUT_ZIP = ROOT / "fitnova_src.zip"

# Directory names skipped at ANY depth (caches, pytest etc.)
SKIP_ANY = {
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "logs",
}

# Subtrees skipped ONLY at the top level of backend/.  NOTE: do NOT use
# basename match here — `backend/models/` (trained weights output) must be
# skipped but `backend/training/models/` (the mt_tcn model CODE) must NOT.
SKIP_TOPLEVEL = {
    "data",       # backend/data/ (dataset tensors, built on Colab)
    "models",     # backend/models/ (trained weight blobs, output of training)
}

# Only include code files (no weight blobs, no training logs, no datasets)
ALLOWED_EXT = {".py", ".json", ".npz", ".md", ".txt"}


def should_include(path: Path) -> bool:
    """Return True if `path` is a source-code file we want in the zip."""
    # Must be inside backend/
    try:
        rel = path.relative_to(BACKEND)
    except ValueError:
        return False

    parts = rel.parts

    # Reject top-level-only skips (backend/data, backend/models)
    if parts and parts[0] in SKIP_TOPLEVEL:
        return False

    # Reject any-depth skips (caches, logs)
    if any(p in SKIP_ANY for p in parts):
        return False

    # Only include whitelisted extensions
    if path.suffix.lower() not in ALLOWED_EXT:
        return False

    return True


def main():
    if not BACKEND.is_dir():
        raise SystemExit(f"ERROR: {BACKEND} not found. Run this script from the project root.")

    # Collect all matching files
    files: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(BACKEND):
        # In-place filter so os.walk doesn't descend into any-depth skips
        dirnames[:] = [d for d in dirnames if d not in SKIP_ANY]
        # Also skip top-level subtrees (backend/data, backend/models)
        if Path(dirpath) == BACKEND:
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOPLEVEL]
        for fname in filenames:
            abs_p = Path(dirpath) / fname
            if should_include(abs_p):
                arc = "backend/" + str(abs_p.relative_to(BACKEND)).replace("\\", "/")
                files.append((abs_p, arc))

    if not files:
        raise SystemExit("ERROR: no source files matched — nothing to zip.")

    # Write zip
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_p, arc in sorted(files, key=lambda x: x[1]):
            zf.write(abs_p, arcname=arc)

    size_kb = OUT_ZIP.stat().st_size / 1024
    print(f"Wrote {OUT_ZIP}  ({size_kb:.1f} KB, {len(files)} files)")
    print()
    print("Contents:")
    for _, arc in sorted(files, key=lambda x: x[1]):
        print(f"  {arc}")
    print()
    print("Next step:")
    print("  1. Go to drive.google.com")
    print("  2. Delete MyDrive/fitnova_src.zip if it exists (old version)")
    print("  3. Upload the new fitnova_src.zip to the root of MyDrive")


if __name__ == "__main__":
    main()
