"""
Rebuild videos.zip with ONLY:
  - {subject}/rep_ann.json           (8 tiny files — rep boundaries for training)
  - {subject}/videos/{camera}/*.mp4  (216 videos)

No joints3d_25, smplx, gpp, camera_parameters — those are not needed on Colab.
Result is ~2.5 GB instead of 11.49 GB.

Run from repo root:  python rebuild_videos_zip.py
"""
import zipfile, os

video_root = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"
out_zip    = r"C:\Users\tsh_x\Desktop\FitNova Application\videos.zip"
CAMERA     = "60457274"

if not os.path.isdir(video_root):
    raise SystemExit("ERROR: video root not found: " + video_root)

print("Scanning files...")
all_files = []

for subject in sorted(os.listdir(video_root)):
    subj_dir = os.path.join(video_root, subject)
    if not os.path.isdir(subj_dir):
        continue

    # 1. rep_ann.json  (required for rep-boundary segmentation on Colab)
    rep_ann = os.path.join(subj_dir, "rep_ann.json")
    if os.path.isfile(rep_ann):
        arc = f"{subject}/rep_ann.json"
        all_files.append((rep_ann, arc))
    else:
        print(f"  WARNING: missing {rep_ann}")

    # 2. MP4 videos for the target camera only
    cam_dir = os.path.join(subj_dir, "videos", CAMERA)
    if not os.path.isdir(cam_dir):
        print(f"  WARNING: no camera dir {cam_dir}")
        continue
    for fname in sorted(os.listdir(cam_dir)):
        if not fname.endswith(".mp4"):
            continue
        abs_path = os.path.join(cam_dir, fname)
        arc_name = f"{subject}/videos/{CAMERA}/{fname}"
        all_files.append((abs_path, arc_name))

n_mp4  = sum(1 for _, a in all_files if a.endswith(".mp4"))
n_json = sum(1 for _, a in all_files if a.endswith(".json"))
print(f"Found {n_mp4} .mp4 files, {n_json} rep_ann.json files")
print(f"Writing {out_zip} ...")

with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
    for i, (abs_path, arc_name) in enumerate(all_files):
        zf.write(abs_path, arc_name)
        if i % 20 == 0:
            print(f"  {i}/{len(all_files)} ...", end='\r')

size_gb = os.path.getsize(out_zip) / 1e9
print(f"\nDone. videos.zip: {size_gb:.2f} GB  ({len(all_files)} entries, forward slashes only)")
print("Upload this file to the ROOT of your Google Drive.")
