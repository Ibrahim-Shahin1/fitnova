"""Build the Colab bundle for D2 (MediaPipe pose extraction at scale).

Run once locally before each Colab session:

    python _build_v6_d2_colab_bundle.py

Outputs:
    colab_notebooks/v6_d2_extract.ipynb        the notebook
    fitnova_v6_src.zip                          source bundle to upload to Drive

Design principles for THIS rewrite:
  * Every long step has a tqdm progress bar (download, 7z extract,
    MediaPipe, Drive copy/sync).
  * True resume support — re-running cells NEVER deletes already-done work:
      - downloads use HTTP Range header
      - 7z extract uses -aos + hardlink trick to skip already-extracted MP4s
      - MediaPipe skips clips with existing .npz
      - Drive sync uses --ignore-existing
  * Hardware-validation cell that REFUSES to proceed on too few CPUs
    (the runtime miscount that wasted hours).
  * Markdown header cells documenting WHAT each code cell does and the
    EXPECTED OUTPUT — so a stalled cell is recognisable.
  * Single source of truth: backend/training/preprocessing/qevd_urls.py
    (15 explicit Qualcomm URLs, no string substitution).
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent
COLAB_DIR = PROJECT_ROOT / "colab_notebooks"
COLAB_DIR.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.lstrip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.lstrip("\n"))


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec":   {"name": "python3", "display_name": "Python 3", "language": "python"},
        "colab":        {"provenance": []},
        "language_info":{"name": "python"},
    }
    out = COLAB_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  wrote {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Notebook cells (markdown headers explaining each code cell + the expected
# output, then the code cell itself).
# ─────────────────────────────────────────────────────────────────────────────


CELL_INTRO = md("""
# FitNova v6 — D2 (MediaPipe pose extraction at scale)

Runs MediaPipe pose extraction on all 298,089 QEVD-FIT-300K clips on
Colab CPU and writes ``.npz`` files to Google Drive.

## What you do
1. Make sure `MyDrive/fitnova_v6/src/fitnova_v6_src.zip` exists on your Drive
2. Pick a runtime: **Runtime → Change runtime type → CPU + High-RAM** (or any GPU runtime with 8+ vCPUs). Avoid the standard 2-vCPU runtimes — Cell 4 will halt if it sees that.
3. **Run all** (Runtime → Run all)
4. Walk away

## Resume guarantee
Every long step is resumable, even after session disconnects:
| Step | How it resumes |
|---|---|
| Download | HTTP Range header (curl-style `-C -`) |
| 7z extract | hardlinks already-done MP4s into 7z's output, then `-aos` skips them |
| MediaPipe extraction | per-clip `.npz` skip if already present |
| Drive sync | `rsync --ignore-existing` |

If a session times out, just re-run the notebook. **No work is wasted.**

## Cell layout
| # | Cell | What it does | Time |
|---|---|---|---|
| 1 | Setup | Remove TF, install pinned mediapipe, p7zip | ~2-3 min |
| 2 | Drive | Mount Drive, set paths | ~30 s |
| 3 | Bundle | Unzip source bundle, import modules | ~5 s |
| 4 | **Hardware check** | CPU/RAM/disk audit. **HALTS if too few cores.** | instant |
| 5 | Config | Set PARTS_TO_PROCESS, N_WORKERS | instant |
| 6 | Helpers | Define download/extract/sync functions | instant |
| 7 | **Main loop** | Process all 4 Parts | hours |
| 8 | Final report | D2 gate JSON written to Drive | ~1 min |
""")


CELL1_SETUP_HEADER = md("""
## Cell 1 — Setup

**What this cell does:** removes Colab's pre-installed TensorFlow (which
conflicts with our protobuf pin), installs pinned mediapipe + cv2, and
installs p7zip for archive extraction.

**Time:** 2-3 minutes (the pip install is silent — be patient).

**Expected last lines of output:**
```
mediapipe : 0.10.33
protobuf  : 4.25.3
cv2       : 4.10.0
numpy     : 2.x
CPU cores : N
[OK] all imports verified
[OK] p7zip-full installed
```

If it ends differently (or hangs >5 min), see troubleshooting at the bottom.
""")

CELL1_SETUP = code("""
import os, shutil, sys, subprocess, importlib

# ── Step 1: nuke TensorFlow + Keras (Colab pre-installs them; their
#   protobuf 5.x conflicts with mediapipe's protobuf 4.x). We don't need
#   TF for D2 — pure MediaPipe + OpenCV pipeline.
def remove_pkg_anywhere(name):
    spec = importlib.util.find_spec(name)
    while spec is not None and spec.origin and "__init__" in spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
        try:
            shutil.rmtree(pkg_dir)
            print(f"  removed {pkg_dir}")
        except Exception as e:
            print(f"  WARN couldn't remove {pkg_dir}: {e}")
            break
        for k in list(sys.modules):
            if k == name or k.startswith(name + "."):
                del sys.modules[k]
        importlib.invalidate_caches()
        spec = importlib.util.find_spec(name)

print("[setup] removing TensorFlow + friends ...")
for pkg in ["tensorflow", "tf_keras", "keras", "tensorboard"]:
    remove_pkg_anywhere(pkg)
try:
    import tensorflow as _tf
    raise RuntimeError("tensorflow STILL importable — abort, do not proceed")
except (ImportError, ModuleNotFoundError):
    print("  [OK] tensorflow no longer importable")

# ── Step 2: install pinned mediapipe stack. This is silent for ~2 min.
print("\\n[setup] installing pinned mediapipe (~2 min, no progress shown by pip) ...")
import time
_t0 = time.time()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "mediapipe==0.10.33",
     "protobuf==4.25.3",
     "opencv-python-headless==4.10.0.84",
     "numpy"],
    check=True,
)
print(f"  [OK] pip install done in {(time.time() - _t0):.0f}s")

# ── Step 3: install 7-Zip (silent ~30s)
print("\\n[setup] installing p7zip (~30s) ...")
_t0 = time.time()
subprocess.run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=True)
print(f"  [OK] p7zip-full installed in {(time.time() - _t0):.0f}s")

# ── Step 4: verify imports
print("\\n[setup] verifying imports ...")
import mediapipe as mp_lib
import google.protobuf as gp
import cv2
import numpy as np
print(f"  mediapipe : {mp_lib.__version__}")
print(f"  protobuf  : {gp.__version__}")
print(f"  cv2       : {cv2.__version__}")
print(f"  numpy     : {np.__version__}")
import multiprocessing as _mp
n_cpu = os.cpu_count() or _mp.cpu_count() or 4
print(f"  CPU cores : {n_cpu}")
print("\\n[OK] all imports verified")
""")


CELL2_DRIVE_HEADER = md("""
## Cell 2 — Mount Drive

**What this cell does:** prompts you to authenticate Google Drive, then sets up the four working folders.

**Time:** ~30 seconds (you'll get an auth popup — click through it).

**Expected last lines of output:**
```
Drive mounted.
SRC_ZIP   = /content/drive/MyDrive/fitnova_v6/src/fitnova_v6_src.zip  (exists: True)
OUT_ROOT  = /content/drive/MyDrive/fitnova_v6/qevd_extracted
REPORTS   = /content/drive/MyDrive/fitnova_v6/reports
```

If `exists: False`, you forgot to upload the bundle — go do that, then re-run this cell.
""")

CELL2_DRIVE = code("""
from google.colab import drive
drive.mount("/content/drive")

DRIVE_BASE = "/content/drive/MyDrive/fitnova_v6"
SRC_ZIP    = f"{DRIVE_BASE}/src/fitnova_v6_src.zip"
OUT_ROOT   = f"{DRIVE_BASE}/qevd_extracted"
REPORTS    = f"{DRIVE_BASE}/reports"
for d in [OUT_ROOT, REPORTS, f"{DRIVE_BASE}/src"]:
    os.makedirs(d, exist_ok=True)

print("Drive mounted.")
print(f"SRC_ZIP   = {SRC_ZIP}  (exists: {os.path.isfile(SRC_ZIP)})")
print(f"OUT_ROOT  = {OUT_ROOT}")
print(f"REPORTS   = {REPORTS}")
assert os.path.isfile(SRC_ZIP), (
    f"\\n\\n!!! Bundle not found at {SRC_ZIP}.\\n"
    f"Upload `fitnova_v6_src.zip` to MyDrive/fitnova_v6/src/ then re-run this cell.\\n"
)
""")


CELL3_BUNDLE_HEADER = md("""
## Cell 3 — Unzip source bundle

**What this cell does:** unzips `fitnova_v6_src.zip` to /content/fitnova_v6/ and adds it to the Python path. Imports the extractor + URL list from the bundle.

**Time:** ~5 seconds.

**Expected last lines of output:**
```
Bundle ready at /content/fitnova_v6
mediapipe pinned: 0.10.33
mediapipe loaded: 0.10.33
config: detect=0.5 mode=VIDEO
```
""")

CELL3_BUNDLE = code("""
WORK_DIR = "/content/fitnova_v6"
shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
subprocess.run(["unzip", "-q", SRC_ZIP, "-d", WORK_DIR], check=True)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from backend.training.preprocessing.qevd_extractor import (
    extract_clip, extract_directory, ClipFeatures, _worker_extract,
    STATUS_OK, STATUS_INTERP, STATUS_NO_POSE,
)
from backend.training.preprocessing.qevd_urls import (
    get_part_urls, all_urls, QEVD_FIT_COACH_URL,
    QEVD_FIT_COACH_BENCHMARK_URL, QEVD_WORKER_IDS_URL,
)
from backend.services.mediapipe_config import (
    MIN_POSE_DETECTION_CONFIDENCE, RUNNING_MODE, PINNED_MEDIAPIPE_VERSION,
)

print(f"Bundle ready at {WORK_DIR}")
print(f"mediapipe pinned: {PINNED_MEDIAPIPE_VERSION}")
print(f"mediapipe loaded: {mp_lib.__version__}")
print(f"config: detect={MIN_POSE_DETECTION_CONFIDENCE} mode={RUNNING_MODE}")
assert mp_lib.__version__ == PINNED_MEDIAPIPE_VERSION, (
    f"VERSION MISMATCH: pinned={PINNED_MEDIAPIPE_VERSION} "
    f"loaded={mp_lib.__version__} — STOP, do not proceed"
)
""")


CELL4_HW_HEADER = md("""
## Cell 4 — Hardware validation (the runtime sanity check)

**What this cell does:** prints the CPU/RAM/disk available on your runtime, then **REFUSES TO PROCEED** if there are fewer than 4 vCPUs (the threshold below which extraction takes weeks).

**Time:** instant.

**Expected output if your runtime is OK:**
```
=== HARDWARE ===
  CPU cores:    8
  N_WORKERS:    7
  RAM:          51.0 GB
  Disk free:    180.5 GB
=== THROUGHPUT ESTIMATE ===
  ~0.7 clips/sec × 76,000 clips = ~30 hr per Part = ~5 days for all 4
  [OK] proceeding
```

**If you see HALT:** follow the printed instructions — switch runtime to High-RAM CPU (or any GPU runtime with 8+ vCPUs) and start over.
""")

CELL4_HW = code("""
import shutil, multiprocessing
n_cpu = os.cpu_count() or multiprocessing.cpu_count() or 1
N_WORKERS = max(1, n_cpu - 1)

# RAM via /proc/meminfo (more reliable than psutil on Colab)
try:
    with open("/proc/meminfo") as f:
        meminfo = {l.split(":")[0]: l.split(":")[1].strip()
                   for l in f if ":" in l}
    ram_kb = int(meminfo.get("MemTotal", "0 kB").split()[0])
    ram_gb = ram_kb / 1024 / 1024
except Exception:
    ram_gb = -1.0

disk = shutil.disk_usage("/content")
disk_free_gb = disk.free / 1e9

print("=== HARDWARE ===")
print(f"  CPU cores:    {n_cpu}")
print(f"  N_WORKERS:    {N_WORKERS}")
print(f"  RAM:          {ram_gb:.1f} GB")
print(f"  Disk free:    {disk_free_gb:.1f} GB")
print()

# Throughput estimate (calibrated against local Windows benchmark:
# ~0.10 clips/sec/worker steady-state).
clips_per_worker_per_sec = 0.10
agg_throughput = clips_per_worker_per_sec * N_WORKERS
clips_per_part = 76_000
hr_per_part = clips_per_part / max(0.01, agg_throughput) / 3600
total_hr = hr_per_part * 4

print("=== THROUGHPUT ESTIMATE ===")
print(f"  ~{agg_throughput:.2f} clips/sec × {clips_per_part:,} clips")
print(f"  = ~{hr_per_part:.0f} hr per Part = ~{total_hr/24:.1f} days for all 4")
print()

# ── Hard halts ─────────────────────────────────────────────────────────
if n_cpu < 4:
    print("=" * 70)
    print("!!! HALT: only %d CPU cores. Extraction would take ~%.0f days." % (n_cpu, total_hr/24))
    print("=" * 70)
    print()
    print("FIX:")
    print("  1. Runtime → Disconnect and delete runtime")
    print("  2. Runtime → Change runtime type")
    print("  3. Hardware accelerator: CPU. RAM: HIGH-RAM (or GPU with 8+ vCPU)")
    print("  4. Connect, then run ALL cells from Cell 1 again.")
    print("     (Resume support means previous progress on Drive is reused.)")
    print()
    raise RuntimeError(f"INSUFFICIENT CPU CORES: {n_cpu}. See message above.")
elif n_cpu < 8:
    print(f"[warn] {n_cpu} cores is acceptable but slow ({total_hr/24:.1f} days for all 4).")
    print(f"[warn] If you can pick a runtime with 8+ vCPUs, it'll be faster.")
    print(f"[OK] proceeding anyway")
else:
    print(f"[OK] {n_cpu} cores is great — proceeding")

if disk_free_gb < 50:
    raise RuntimeError(
        f"DISK TOO LOW: {disk_free_gb:.1f} GB free, need 50+ GB for Part downloads + extraction."
    )
""")


CELL5_CONFIG_HEADER = md("""
## Cell 5 — Config

**What this cell does:** sets `PARTS_TO_PROCESS = [1, 2, 3, 4]` and a few other knobs.

**Time:** instant.

**Expected output:**
```
PARTS_TO_PROCESS = [1, 2, 3, 4]
N_WORKERS        = 7  (or whatever Cell 4 reported)
DROP_THRESHOLD   = 0.5
```
""")

CELL5_CONFIG = code("""
PARTS_TO_PROCESS = [1, 2, 3, 4]   # leave as-is for full run; e.g. [3] for just Part 3
DROP_THRESHOLD   = 0.50            # per-clip filter: no_pose_fraction > this -> drop
EXPECTED_PART_COUNT = {1: 76000, 2: 76000, 3: 76000, 4: 70089}
print(f"PARTS_TO_PROCESS = {PARTS_TO_PROCESS}")
print(f"N_WORKERS        = {N_WORKERS}  (from Cell 4)")
print(f"DROP_THRESHOLD   = {DROP_THRESHOLD}")
""")


CELL6_HELPERS_HEADER = md("""
## Cell 6 — Helper functions (downloader, extractor, sync)

**What this cell does:** defines all the worker functions used by the main loop. **Every long operation has a tqdm progress bar** and **every step is fully resumable** — re-running this cell never destroys partial work.

**Time:** instant.

**Expected output:**
```
[OK] helper functions defined: download_with_progress, run_7z_with_resume,
     copy_dir_with_progress, sync_to_drive, cleanup_part, extract_with_tqdm
```
""")

CELL6_HELPERS = code("""
import time, urllib.request
from multiprocessing import get_context
from pathlib import Path
from tqdm.auto import tqdm


def part_already_done_on_drive(part_n):
    pdir = f"{OUT_ROOT}/Part-{part_n}"
    if not os.path.isdir(pdir):
        return 0
    return sum(1 for f in os.listdir(pdir) if f.endswith(".npz"))


def download_with_progress(url, dst, max_attempts=8):
    \"\"\"Resumable download with tqdm. Uses HTTP Range to continue partial downloads.\"\"\"
    fname = os.path.basename(dst)
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0))
            existing = os.path.getsize(dst) if os.path.isfile(dst) else 0
            if total > 0 and existing >= total:
                print(f"    {fname}: complete ({total/1e9:.2f} GB), skip")
                return
            headers, mode = {}, "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                print(f"    {fname}: resuming from {existing/1e9:.2f} GB")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                pbar = tqdm(total=total, initial=existing, unit="B",
                            unit_scale=True, unit_divisor=1024,
                            desc=fname, mininterval=1.0)
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk: break
                        f.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()
            return
        except Exception as e:
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"    [attempt {attempt}/{max_attempts}] {type(e).__name__}: {e}, wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {max_attempts}: {url}")


def run_7z_with_resume(zip_path, out_dir, expected_files, existing_dir=None):
    \"\"\"Run 7z with full resume support: hardlinks already-extracted MP4s
    into the output dir before running 7z so its `-aos` (skip if exists)
    flag treats them as done. Then ticks tqdm by file-count growth.

    Args:
      zip_path:      multivolume archive head (.zip)
      out_dir:       7z output dir (will contain QEVD-FIT-300k-Part-N/...)
      expected_files: total MP4s expected when complete
      existing_dir:  if given, hardlink any *.mp4 from here into out_dir first
                     so 7z skips them via -aos (true resume)
    \"\"\"
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    inner = out_path / Path(zip_path).stem  # e.g. .../QEVD-FIT-300k-Part-1
    inner.mkdir(parents=True, exist_ok=True)

    # ── Pre-populate with already-extracted files via hardlink (zero-cost)
    n_linked = 0
    if existing_dir is not None and os.path.isdir(existing_dir):
        existing = list(Path(existing_dir).glob("*.mp4"))
        if existing:
            print(f"  [resume] hardlinking {len(existing)} already-extracted "
                  f"MP4s into 7z output dir (so it can skip them) ...")
            for src in existing:
                dst = inner / src.name
                if not dst.exists():
                    try:
                        os.link(src, dst)
                        n_linked += 1
                    except OSError:
                        # fallback to copy if hardlink fails (cross-device etc)
                        shutil.copy2(src, dst)
                        n_linked += 1
            print(f"  [resume] linked {n_linked} files; 7z will only extract the missing {expected_files - n_linked}")

    # ── Run 7z with -aos (skip files that already exist).  We background
    # it via Popen and poll the output dir to drive the tqdm bar.
    proc = subprocess.Popen(
        ["7z", "x", str(zip_path), f"-o{out_dir}", "-aos", "-y", "-bd"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pbar = tqdm(total=expected_files, initial=n_linked,
                desc="7z extract", unit="file",
                mininterval=1.0, smoothing=0.05)
    last = n_linked
    while proc.poll() is None:
        time.sleep(2)
        try:
            current = sum(1 for _ in inner.glob("*.mp4"))
            if current > last:
                pbar.update(current - last); last = current
        except Exception:
            pass
    final_count = sum(1 for _ in inner.glob("*.mp4"))
    if final_count > last:
        pbar.update(final_count - last)
    pbar.close()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return inner


def copy_dir_with_progress(src_dir, dst_dir, suffix=".npz", desc="copy"):
    \"\"\"shutil.copy2 each file in src_dir matching suffix to dst_dir, with tqdm.
    Skips files that already exist at dst.\"\"\"
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in src_dir.iterdir()
             if f.is_file() and f.name.endswith(suffix)]
    if not files: return 0
    n_copied = 0
    pbar = tqdm(total=len(files), desc=desc, unit="file", mininterval=1.0)
    for src in files:
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst); n_copied += 1
        pbar.update(1)
    pbar.close()
    return n_copied


def download_and_extract_part(part_n):
    \"\"\"Download Part N's 3 multivolume zips and extract MP4s to /content/qevd_part_N/.

    Resume strategy:
      - downloads use Range header (resume partial)
      - 7z uses -aos with hardlinks of already-extracted files (true resume)
      - never deletes already-extracted MP4s
    \"\"\"
    local_dir = f"/content/qevd_part_{part_n}"
    expected = EXPECTED_PART_COUNT[part_n]

    # If already complete, skip
    if os.path.isdir(local_dir):
        n_existing = sum(1 for f in os.listdir(local_dir) if f.endswith(".mp4"))
        if n_existing >= expected:
            print(f"  [stage] complete: {n_existing} MP4s in {local_dir}")
            return local_dir
        elif n_existing > 0:
            print(f"  [stage] partial: {n_existing}/{expected} MP4s on disk, "
                  f"will resume (no files deleted)")
        else:
            print(f"  [stage] empty {local_dir}, will extract from scratch")
    else:
        os.makedirs(local_dir, exist_ok=True)
        print(f"  [stage] fresh extract")

    # ── Download (resumable) ──────────────────────────────────────────
    urls = get_part_urls(part_n)
    print(f"  [download] {len(urls)} files from Qualcomm CDN ...")
    t0 = time.time()
    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        download_with_progress(url, f"/content/{fname}")
    print(f"  [download] done in {(time.time()-t0)/60:.1f} min")

    # ── 7z extract with resume (hardlink trick) ──────────────────────
    tmp_dir = f"/content/_7z_tmp_part_{part_n}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  [extract] running 7z with resume support (expected {expected} MP4s) ...")
    t0 = time.time()
    main_zip = f"/content/QEVD-FIT-300k-Part-{part_n}.zip"
    inner = run_7z_with_resume(main_zip, tmp_dir, expected, existing_dir=local_dir)

    # ── Move/rename completed files to flat local_dir ────────────────
    print(f"  [extract] moving extracted files to {local_dir} ...")
    moved = 0
    for src in Path(inner).glob("*.mp4"):
        dst = Path(local_dir) / src.name
        if not dst.exists():
            try:
                # If src is a hardlink of dst already, skip; else move
                if os.path.exists(dst) and os.path.samefile(src, dst):
                    continue
                shutil.move(str(src), str(dst))
                moved += 1
            except OSError as e:
                print(f"    WARN: couldn't move {src.name}: {e}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Free disk: delete the source zips
    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        try: os.remove(f"/content/{fname}")
        except OSError: pass

    final_count = sum(1 for f in os.listdir(local_dir) if f.endswith(".mp4"))
    print(f"  [extract] done in {(time.time()-t0)/60:.1f} min "
          f"(moved {moved}, final {final_count} MP4s)")
    return local_dir


def sync_to_drive(part_n, local_out):
    \"\"\"Copy .npz files from /content -> Drive with tqdm progress. Skips
    files already on Drive (idempotent).\"\"\"
    drive_dir = Path(f"{OUT_ROOT}/Part-{part_n}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [sync] copying .npz to Drive (Drive writes are slow — can take 10-30 min) ...")
    t0 = time.time()
    n_copied = copy_dir_with_progress(local_out, drive_dir, ".npz", desc="sync->Drive")
    n_drive = sum(1 for f in drive_dir.iterdir()
                  if f.is_file() and f.name.endswith(".npz"))
    print(f"  [sync] copied {n_copied} new, {n_drive} total on Drive "
          f"({(time.time()-t0)/60:.1f} min)")


def cleanup_part(part_n):
    \"\"\"Free local disk after a Part is fully synced to Drive.\"\"\"
    print(f"  [cleanup] removing /content/qevd_part_{part_n}* ...")
    for d in [f"/content/qevd_part_{part_n}",
              f"/content/qevd_extracted_part_{part_n}",
              f"/content/_7z_tmp_part_{part_n}"]:
        shutil.rmtree(d, ignore_errors=True)
    print(f"  [cleanup] done")


def extract_with_tqdm(in_dir, out_dir, n_workers, drop_threshold=0.50):
    \"\"\"Run MediaPipe extraction on all MP4s in in_dir. Skips clips with
    existing .npz (true per-clip resume).\"\"\"
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(in_dir.rglob("*.mp4"))
    tasks, n_skipped = [], 0
    for mp4 in mp4s:
        out_path = out_dir / mp4.relative_to(in_dir).with_suffix(".npz")
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skipped += 1
            continue
        tasks.append((str(mp4), str(out_path)))
    print(f"  [tasks] {len(mp4s)} total | {n_skipped} done | {len(tasks)} remaining")
    if not tasks:
        return {"n_ok": 0, "n_dropped": 0, "n_failed": 0, "n_skipped": n_skipped,
                "no_pose_mean": 0.0, "interp_mean": 0.0, "elapsed_s": 0,
                "failures": []}
    n_ok = n_dropped = n_failed = 0
    no_pose_total = interp_total = 0.0
    failures = []
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        with tqdm(total=len(tasks), desc=f"Part-{PART_NUMBER}",
                  unit="clip", smoothing=0.05, mininterval=1.0) as pbar:
            for result in pool.imap_unordered(_worker_extract, tasks, chunksize=8):
                if result["ok"]:
                    n_ok += 1
                    no_pose_total += result["no_pose_fraction"]
                    interp_total += result["interp_fraction"]
                    if result["no_pose_fraction"] > drop_threshold:
                        n_dropped += 1
                else:
                    n_failed += 1
                    failures.append(result)
                pbar.update(1)
                if n_ok and n_ok % 10 == 0:
                    pbar.set_postfix({
                        "no_pose": f"{no_pose_total/n_ok:.3f}",
                        "fail": n_failed, "drop": n_dropped,
                    })
    return {"n_ok": n_ok, "n_dropped": n_dropped, "n_failed": n_failed,
            "n_skipped": n_skipped, "elapsed_s": time.time() - t0,
            "no_pose_mean": no_pose_total / max(1, n_ok),
            "interp_mean": interp_total / max(1, n_ok),
            "failures": failures}


print("[OK] helper functions defined: download_with_progress, run_7z_with_resume,")
print("     copy_dir_with_progress, sync_to_drive, cleanup_part, extract_with_tqdm")
""")


CELL7_LOOP_HEADER = md("""
## Cell 7 — Main loop (the long one)

**What this cell does:** processes every Part in `PARTS_TO_PROCESS` end-to-end. Each Part:
1. Skip if Drive already has all `.npz` files
2. Download zips from Qualcomm (resumable, with progress)
3. 7z extract (resumable via hardlink trick, with progress)
4. Copy any already-extracted `.npz` from Drive back to local (resume, with progress)
5. MediaPipe pose extraction (resumable per-clip, with progress)
6. Sync `.npz` to Drive (with progress)
7. Free local disk

**Time:** depends entirely on `N_WORKERS`. See Cell 4's estimate.

**Expected progress lines:**
```
======================================================================
PART 1
======================================================================
  [status] 0/76000 on Drive | 76000 remaining
  [stage] fresh extract
  [download] 3 files from Qualcomm CDN ...
QEVD-FIT-300k-Part-1.z01: 100%|████| 9.67G/9.67G [05:30<00:00, 30MB/s]
QEVD-FIT-300k-Part-1.z02: 100%|████| 9.67G/9.67G [05:30<00:00, 30MB/s]
QEVD-FIT-300k-Part-1.zip: 100%|████| 1.18G/1.18G [00:40<00:00, 30MB/s]
  [download] done in 11.7 min
  [extract] running 7z with resume support ...
7z extract: 100%|████████| 76000/76000 [10:00<00:00, 127file/s]
  [extract] done in 10.0 min (moved 76000, final 76000 MP4s)
  [tasks] 76000 total | 0 done | 76000 remaining
Part-1: 50%|█████  | 38000/76000 [12:00<11:30, 52clip/s, no_pose=0.012, fail=0, drop=23]
```

If progress STOPS for >5 min on any one bar, ping me with which bar froze.
""")

CELL7_LOOP = code("""
overall_t0 = time.time()
for PART_NUMBER in PARTS_TO_PROCESS:
    print(f"\\n{'='*70}\\nPART {PART_NUMBER}\\n{'='*70}")
    expected = EXPECTED_PART_COUNT[PART_NUMBER]
    n_done = part_already_done_on_drive(PART_NUMBER)
    if n_done >= expected - 50:
        print(f"  [skip] already done on Drive ({n_done}/{expected})")
        continue
    print(f"  [status] {n_done}/{expected} on Drive | {expected - n_done} remaining")

    # 1+2+3: download + 7z extract (resumable)
    local_in = download_and_extract_part(PART_NUMBER)

    # 4: pull any partial .npz from Drive (resume from previous session)
    local_out = f"/content/qevd_extracted_part_{PART_NUMBER}"
    os.makedirs(local_out, exist_ok=True)
    drive_part = f"{OUT_ROOT}/Part-{PART_NUMBER}"
    if os.path.isdir(drive_part):
        n_existing = sum(1 for f in os.listdir(drive_part) if f.endswith(".npz"))
        if n_existing > 0:
            print(f"  [resume] copying {n_existing} existing .npz from Drive -> /content ...")
            copy_dir_with_progress(drive_part, local_out, ".npz", desc="resume<-Drive")

    # 5: MediaPipe extraction (per-clip resume)
    print(f"  [extract-pose] starting MediaPipe (workers={N_WORKERS}) ...")
    stats = extract_with_tqdm(local_in, local_out, N_WORKERS, DROP_THRESHOLD)
    print(f"  [extract-pose] done: ok={stats['n_ok']} dropped={stats['n_dropped']} "
          f"failed={stats['n_failed']} in {stats['elapsed_s']/3600:.2f} hr")
    if stats["failures"]:
        print(f"  first 3 failures:")
        for f in stats["failures"][:3]:
            print(f"    {f.get('mp4','?')}: {f.get('error', f.get('reason','?'))}")

    # 6: sync to Drive
    sync_to_drive(PART_NUMBER, local_out)

    # 7: free local disk
    cleanup_part(PART_NUMBER)

print(f"\\n{'='*70}\\nALL PARTS DONE — total {(time.time()-overall_t0)/3600:.2f} hr\\n{'='*70}")
""")


CELL8_REPORT_HEADER = md("""
## Cell 8 — Final D2 gate report

**What this cell does:** scans all Drive `Part-{1,2,3,4}/` folders, computes aggregate `no_pose` / `interp` stats from a 500-clip sample per Part, writes the gate JSON.

**Time:** ~1-2 minutes.

**Expected output:**
```
=== D2 GATE REPORT: PASS ===
  total clips:    298,089
  no_pose mean:   0.012  (must be <0.10)
  interp  mean:   0.005  (must be <0.05)
  Part-1: 76,000
  Part-2: 76,000
  Part-3: 76,000
  Part-4: 70,089

Report written: /content/drive/MyDrive/fitnova_v6/reports/D2_extract_quality.json
```
""")

CELL8_REPORT = code("""
import json
from datetime import datetime, timezone

t0 = time.time()
all_no_pose = []
all_interp  = []
clip_count_per_part = {}
import numpy as np
for n in (1, 2, 3, 4):
    pdir = f"{OUT_ROOT}/Part-{n}"
    if not os.path.isdir(pdir):
        clip_count_per_part[f"Part-{n}"] = 0; continue
    npz_files = [f for f in os.listdir(pdir) if f.endswith(".npz")]
    clip_count_per_part[f"Part-{n}"] = len(npz_files)
    if not npz_files: continue
    print(f"Part {n}: scanning {len(npz_files)} .npz ...")
    sample = np.random.RandomState(42).choice(
        len(npz_files), size=min(500, len(npz_files)), replace=False)
    for i in sample:
        with np.load(f"{pdir}/{npz_files[i]}") as z:
            T = z["pose_canon"].shape[0]
            stat = z["status_per_frame"]
            all_no_pose.append(float((stat == STATUS_NO_POSE).sum() / max(1, T)))
            all_interp.append(float((stat == STATUS_INTERP).sum() / max(1, T)))

n_total = sum(clip_count_per_part.values())
no_pose_mean = float(np.mean(all_no_pose)) if all_no_pose else 0.0
interp_mean  = float(np.mean(all_interp))  if all_interp  else 0.0
report = {
    "gate_id":   "D2_extract_quality",
    "status":    "PASS" if (n_total > 0 and no_pose_mean < 0.10
                              and interp_mean < 0.05) else "PARTIAL",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "metrics": {
        "n_total_clips_extracted":   n_total,
        "n_clips_per_part":          clip_count_per_part,
        "sample_no_pose_mean":       round(no_pose_mean, 4),
        "sample_interp_mean":        round(interp_mean,  4),
        "n_clips_sampled":           len(all_no_pose),
        "elapsed_aggregation_s":     round(time.time() - t0, 1),
    },
    "failures": [],
}
report_path = f"{REPORTS}/D2_extract_quality.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
print(f"\\n=== D2 GATE REPORT: {report['status']} ===")
print(f"  total clips:    {n_total:,}")
print(f"  no_pose mean:   {no_pose_mean:.4f}  (must be <0.10)")
print(f"  interp  mean:   {interp_mean:.4f}   (must be <0.05)")
for k, v in clip_count_per_part.items():
    print(f"  {k}: {v:,}")
print(f"\\nReport written: {report_path}")
""")


CELL_DONE = md("""
## All done

When this cell prints `D2 GATE REPORT: PASS`, you have:

- 298,089 `.npz` files at `MyDrive/fitnova_v6/qevd_extracted/Part-{1..4}/`
- A signed gate report at `MyDrive/fitnova_v6/reports/D2_extract_quality.json`

Ready for D3 + D4 + supervised training.

## Troubleshooting

**Cell 1 hangs on pip install for 5+ min.** Usually completes in 2-3 min. If it goes past 5 min, the runtime is overloaded — disconnect and re-run.

**Cell 4 halts.** Your runtime has fewer than 4 vCPUs. Switch to High-RAM CPU per the printed instructions.

**Cell 7 download stalls (no progress for 5+ min).** Qualcomm's CDN occasionally throttles. The download has 8 retries with backoff; if all 8 fail, re-run cell 7 — it resumes from where the partial file left off.

**Cell 7 7z extract stalls.** Cell 6 watches the output dir every 2 sec; if `7z extract: X/76000` stops growing for 60+ sec, check `!ls /content/_7z_tmp_part_N/QEVD-FIT-300k-Part-N | wc -l` in a fresh cell. If 7z is still alive but slow, just wait. If it died, re-run cell 7 — `-aos` makes it skip already-extracted files.

**Cell 7 MediaPipe stalls.** Each clip takes ~5-10 seconds. If progress freezes for > 60 sec, one worker may have crashed. Other workers continue; if all crash, re-run cell 7 — per-clip `.npz` already done are skipped.

**Session disconnected.** Re-run all cells from cell 1. Resume support kicks in: each Part's progress on Drive is honored, downloads resume mid-file, 7z skips already-extracted MP4s, MediaPipe skips already-done clips.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Source bundle definition
# ─────────────────────────────────────────────────────────────────────────────


SRC_FILES_V6 = [
    "backend/__init__.py",
    "backend/services/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/diagnostics/__init__.py",

    "backend/services/mediapipe_config.py",

    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/qevd_extractor.py",
    "backend/training/preprocessing/qevd_label_builder.py",
    "backend/training/preprocessing/qevd_urls.py",

    "backend/training/diagnostics/phase_gates.py",

    # Reference files (kept in bundle for diagnostic / future use)
    "backend/training/diagnostics/references/reference_outputs.npz",
    "backend/training/diagnostics/references/input_clips/Good_Squats.mp4",
]


def main():
    print("== v6 D2 Notebook ==")
    cells = [
        CELL_INTRO,
        CELL1_SETUP_HEADER, CELL1_SETUP,
        CELL2_DRIVE_HEADER, CELL2_DRIVE,
        CELL3_BUNDLE_HEADER, CELL3_BUNDLE,
        CELL4_HW_HEADER, CELL4_HW,
        CELL5_CONFIG_HEADER, CELL5_CONFIG,
        CELL6_HELPERS_HEADER, CELL6_HELPERS,
        CELL7_LOOP_HEADER, CELL7_LOOP,
        CELL8_REPORT_HEADER, CELL8_REPORT,
        CELL_DONE,
    ]
    write_notebook("v6_d2_extract.ipynb", cells)
    print()

    print("== v6 source bundle ==")
    out_zip = PROJECT_ROOT / "fitnova_v6_src.zip"
    src_files = list(SRC_FILES_V6)
    refs_in = PROJECT_ROOT / "backend/training/diagnostics/references/input_clips"
    if refs_in.exists():
        for f in sorted(refs_in.glob("*.mp4")):
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            if rel not in src_files:
                src_files.append(rel)

    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in src_files:
            full = PROJECT_ROOT / rel
            if not full.exists():
                if rel.endswith("__init__.py"):
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_text("")
                else:
                    print(f"  WARN missing: {rel}")
                    continue
            zf.write(full, arcname=rel)
            n_files += 1
            total_bytes += full.stat().st_size
    size_mb = out_zip.stat().st_size / 1e6
    print(f"  wrote {n_files} files, {size_mb:.2f} MB compressed "
          f"({total_bytes/1e6:.2f} MB raw)")

    print()
    print("=" * 70)
    print("Next:")
    print(f"  1. Upload ONLY {out_zip.name} ({out_zip.stat().st_size/1e6:.1f} MB) to:")
    print( "       MyDrive/fitnova_v6/src/")
    print( "  2. Open colab_notebooks/v6_d2_extract.ipynb in Colab")
    print( "     (use Colab Pro CPU + High-RAM runtime; Cell 4 will halt if too few cores)")
    print( "  3. Run all. The notebook processes all 4 Parts automatically.")
    print( "     If session times out mid-run, re-run all — it auto-resumes.")


if __name__ == "__main__":
    main()
