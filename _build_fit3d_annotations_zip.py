"""
Build a compact `fit3d_annotations.zip` containing only the files the Fit3D
training pipeline needs:
    - {subject}/joints3d_25/*.json   (MoCap ground-truth joint coordinates)
    - {subject}/rep_ann.json         (per-rep boundary frame indices)

Usage:
    python _build_fit3d_annotations_zip.py

Output:
    fit3d_annotations.zip  (in the current working directory)

Then upload this file to the root of your Google Drive as:
    MyDrive/fit3d_annotations.zip

Expected size: ~20-50 MB (tiny — joints3d_25 JSONs are ASCII numbers).
"""
import os
import sys
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
OUT_ZIP = Path("fit3d_annotations.zip")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not DATASET_ROOT.exists():
        print(f"ERROR: Fit3D dataset root not found: {DATASET_ROOT}")
        print("Edit DATASET_ROOT in this script if your path differs.")
        sys.exit(1)

    print(f"Scanning {DATASET_ROOT}...")
    to_zip = []  # list of (abs_path, archive_path) tuples

    for subj in SUBJECTS:
        subj_dir = DATASET_ROOT / subj
        if not subj_dir.is_dir():
            print(f"  [MISSING] {subj}/  (skipping)")
            continue

        # rep_ann.json
        rep_ann = subj_dir / "rep_ann.json"
        if rep_ann.is_file():
            to_zip.append((rep_ann, f"{subj}/rep_ann.json"))
        else:
            print(f"  [MISSING] {subj}/rep_ann.json")

        # joints3d_25/*.json
        joints_dir = subj_dir / "joints3d_25"
        if not joints_dir.is_dir():
            print(f"  [MISSING] {subj}/joints3d_25/")
            continue

        n_ex = 0
        for fp in sorted(joints_dir.glob("*.json")):
            to_zip.append((fp, f"{subj}/joints3d_25/{fp.name}"))
            n_ex += 1
        print(f"  [OK] {subj}: rep_ann.json + {n_ex} exercises")

    if not to_zip:
        print("ERROR: nothing to zip — all subjects missing.")
        sys.exit(1)

    # Write zip
    print(f"\nWriting {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arc_path in to_zip:
            zf.write(abs_path, arcname=arc_path)

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"Wrote {OUT_ZIP}  ({size_mb:.1f} MB, {len(to_zip)} files)")
    print()
    print("Next: upload fit3d_annotations.zip to the ROOT of your Google Drive")
    print("      (same level as MyDrive/videos.zip and MyDrive/fitnova_src.zip).")


if __name__ == "__main__":
    main()
