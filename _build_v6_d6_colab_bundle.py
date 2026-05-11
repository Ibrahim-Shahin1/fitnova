"""Build the Colab bundle for D6 (supervised v6 training).

Reuses the same fitnova_v6_src.zip as D2 (single source-of-truth bundle).
Generates a NEW notebook colab_notebooks/v6_d6_train.ipynb.

D6 vs D2 (key differences):
  * Uses GPU runtime (T4 / L4 / A100). Tensorflow is REQUIRED (kept, not removed).
  * Reads .npz files from Drive at MyDrive/fitnova_v6/qevd_extracted/Part-{1..4}/
    (output of D2 notebook).
  * COPIES .npz to /content/qevd_local FIRST so training reads from local SSD.
    Drive FUSE per-file open is ~50-100ms; local is ~5ms. Difference: ~5-10x
    faster training. The one-time copy cost (~15-30 min) is amortised over
    a 5+ hour run.
  * Uses Drive API for the existence-count step (NOT os.listdir, which
    times out on 76K-file Drive folders).
  * Streams subprocess output line-by-line so you can see per-epoch progress
    in real time.
  * Auto-retries the training subprocess up to 3 times. Each retry resumes
    from the latest epoch checkpoint, so no work is lost.
  * Force-flushes Drive at the end to ensure model weights actually land
    on Google's servers (not just FUSE cache).
  * Reads label JSONs from Drive at MyDrive/fitnova_v6/labels/.
  * Calls train_form_model_v6.py via the CLI; train script handles resume,
    checkpointing, gradient clipping, AdamW + cosine decay.

Run once locally before launching Colab:

    python _build_v6_d6_colab_bundle.py

Outputs:
    colab_notebooks/v6_d6_train.ipynb       the notebook
    fitnova_v6_src.zip                       (rebuilt — same bundle as D2)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent
COLAB_DIR = PROJECT_ROOT / "colab_notebooks"
COLAB_DIR.mkdir(exist_ok=True)


def md(text):  return nbf.v4.new_markdown_cell(text.lstrip("\n"))
def code(text): return nbf.v4.new_code_cell(text.lstrip("\n"))


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
# Notebook cells (8 cells: intro + 7 code, all hardened)
# ─────────────────────────────────────────────────────────────────────────────


CELL_INTRO = md("""
# FitNova v6 — D6 (supervised training on QEVD)

Trains the v6 ST-GCN model on QEVD-derived labels using `.npz` files
already extracted to Drive by the D2 notebook.

## Prerequisites
- D2 has finished: `MyDrive/fitnova_v6/qevd_extracted/Part-{1..4}/` exists with `.npz` files
- `MyDrive/fitnova_v6/labels/` contains `feedbacks_short_clips.json`, `fine_grained_labels.json`, and `fine_grained_labels_with_worker_ids.json`
- `MyDrive/fitnova_v6/src/fitnova_v6_src.zip` is the latest bundle

## Runtime — pick ONE
| GPU | Time | Notes |
|---|---|---|
| **A100** | 1.5-2.5 h | fastest, most expensive |
| **L4** | 3-5 h | good balance |
| **T4** | 5-9 h | works fine; bottleneck is I/O not compute |

T4 free tier disconnects after 90 min idle — keep the tab focused, or use a paid runtime. **Auto-resume** kicks in if the session dies, so worst-case you re-run the notebook and it picks up at the last completed epoch.

## What you do
1. Runtime → Change runtime type → GPU (T4/L4/A100) + High RAM
2. Run all cells
3. Walk away

## Safety + progress
| Risk | Mitigation in this notebook |
|---|---|
| Drive FUSE listdir hangs on 76K-file dirs | Cell 3 uses Drive API instead |
| Drive FUSE slow .npz reads in training | Cell 5 copies all .npz to /content first |
| Subprocess stdout buffered — no progress | Cell 6 streams output line-by-line |
| Crash mid-training | Cell 6 auto-retries 3x; trainer resumes from last checkpoint |
| Weights stuck in FUSE cache after training | Cell 7 force-flushes Drive |
| Session timeout | Re-run notebook; trainer skips already-done epochs |
""")


CELL1_HW = code("""
# ── Cell 1: hardware sanity + GPU detection ─────────────────────────────────
import os, shutil, subprocess

print(\"=== HARDWARE ===\")
n_cpu = os.cpu_count() or 1
print(f\"  CPU cores: {n_cpu}\")

try:
    with open(\"/proc/meminfo\") as f:
        for line in f:
            if line.startswith(\"MemTotal:\"):
                ram_gb = int(line.split()[1]) / 1024 / 1024
                print(f\"  RAM:       {ram_gb:.1f} GB\")
                break
except Exception:
    print(\"  RAM:       unknown\")

disk = shutil.disk_usage(\"/content\")
disk_free_gb = disk.free / 1e9
print(f\"  /content free: {disk_free_gb:.1f} GB\")

gpu_name = None
try:
    out = subprocess.check_output(
        [\"nvidia-smi\", \"--query-gpu=name,memory.total\", \"--format=csv,noheader\"],
        text=True,
    ).strip()
    print(f\"  GPU:       {out}\")
    gpu_name = out.split(\",\")[0].strip()
except Exception:
    print(\"  GPU:       NONE\")
    raise RuntimeError(\"No GPU detected. Switch runtime to GPU and re-run.\")

ETA_HOURS = {
    \"Tesla T4\":            \"5-9\",
    \"NVIDIA L4\":           \"3-5\",
    \"NVIDIA A100\":         \"1.5-2.5\",
    \"NVIDIA H100\":         \"1-1.5\",
}
eta = next((v for k, v in ETA_HOURS.items() if k in (gpu_name or \"\")), \"3-8\")
print(f\"\\n  ETA for D6 training on this GPU: ~{eta} hours\")
print(\"  (bottleneck is data I/O — Cell 5 mitigates by caching to /content)\")

if \"T4\" in (gpu_name or \"\"):
    print(\"\\n  [warn] T4 free-tier disconnects after 90 min idle.\")
    print(\"         Keep this tab focused, OR re-run after disconnect (auto-resumes).\")

if disk_free_gb < 20:
    raise RuntimeError(
        f\"Insufficient /content disk: {disk_free_gb:.1f} GB. Need 20+ GB for .npz cache.\"
    )
""")


CELL2_DRIVE = code("""
# ── Cell 2: mount Drive (with FUSE-zombie cleanup) ──────────────────────────
import os, subprocess
from google.colab import drive

# Defensive cleanup. If a previous session crashed mid-flush_and_unmount,
# /content/drive can be in a zombie state where mount() fails with
# 'Mountpoint must not already contain files'. Fixing it preemptively:
subprocess.run([\"fusermount\", \"-uz\", \"/content/drive\"], capture_output=True)
subprocess.run([\"rm\", \"-rf\", \"/content/drive\"], capture_output=True)

drive.mount(\"/content/drive\")

DRIVE_BASE = \"/content/drive/MyDrive/fitnova_v6\"
SRC_ZIP    = f\"{DRIVE_BASE}/src/fitnova_v6_src.zip\"
LABELS_DIR = f\"{DRIVE_BASE}/labels\"
NPZ_ROOT   = f\"{DRIVE_BASE}/qevd_extracted\"
MODELS_DIR = f\"{DRIVE_BASE}/models/form_model_v6\"

assert os.path.isfile(SRC_ZIP), (
    f\"\\nMissing bundle: {SRC_ZIP}\\n\"
    f\"Run `python _build_v6_d6_colab_bundle.py` locally then upload the\\n\"
    f\"resulting fitnova_v6_src.zip to MyDrive/fitnova_v6/src/.\"
)

assert os.path.isdir(LABELS_DIR), f\"missing labels dir: {LABELS_DIR}\"
expected_labels = [
    \"feedbacks_short_clips.json\",
    \"fine_grained_labels.json\",
    \"fine_grained_labels_with_worker_ids.json\",
]
missing = [f for f in expected_labels if not os.path.isfile(f\"{LABELS_DIR}/{f}\")]
if missing:
    raise RuntimeError(
        f\"Missing label files: {missing}\\n\"
        f\"Upload from QEVD-FIT-COACH download to {LABELS_DIR}/.\"
    )
os.makedirs(MODELS_DIR, exist_ok=True)
print(f\"  [OK] bundle + 3 label JSONs present\")
print(f\"  MODELS_DIR = {MODELS_DIR}\")
""")


CELL3_VERIFY_VIA_API = code("""
# ── Cell 3: verify D2 output via Drive API (NOT FUSE listdir) ───────────────
# Drive FUSE listdir on 76K-file folders takes 5-15 min and frequently times
# out. Drive API counts the same in ~5-30 sec.
import sys, logging
subprocess.run(
    [sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"google-api-python-client\"],
    check=True,
)

# Silence the per-request-timeout warning that the API client spams on every
# call (it's a known, harmless quirk of the httplib2 transport).
logging.getLogger(\"google_auth_httplib2\").setLevel(logging.ERROR)

from google.colab import auth
from googleapiclient.discovery import build
auth.authenticate_user()
svc = build(\"drive\", \"v3\")

def find_folder(path_parts):
    parent = \"root\"
    for name in path_parts:
        r = svc.files().list(
            q=f\"'{parent}' in parents and name='{name}' and trashed=false\",
            fields=\"files(id)\", pageSize=10).execute()
        if not r[\"files\"]: return None
        parent = r[\"files\"][0][\"id\"]
    return parent

def count_npz(folder_id):
    n, page_token = 0, None
    while True:
        r = svc.files().list(
            q=(f\"'{folder_id}' in parents and trashed=false \"
               f\"and name contains '.npz'\"),
            fields=\"nextPageToken, files(id)\",
            pageSize=1000, pageToken=page_token,
        ).execute()
        n += len(r[\"files\"])
        page_token = r.get(\"nextPageToken\")
        if not page_token: break
    return n

print(\"Counting .npz on Drive (via API) ...\")
import time as _t
t0 = _t.time()
total_drive = 0
PARTS = (1, 2, 3, 4)
PER_PART = {}
for n in PARTS:
    fid = find_folder([\"fitnova_v6\", \"qevd_extracted\", f\"Part-{n}\"])
    if fid is None:
        PER_PART[n] = 0
        print(f\"  Part-{n}: 0 (folder missing)\")
        continue
    c = count_npz(fid)
    PER_PART[n] = c
    total_drive += c
    print(f\"  Part-{n}: {c:,}   ({_t.time()-t0:.0f}s elapsed)\")

print(f\"\\nTotal on Drive: {total_drive:,}\")
assert total_drive >= 250_000, (
    f\"Too few clips on Drive ({total_drive:,}). \"
    f\"D2 may be incomplete — check the D2 gate report.\"
)
print(f\"[OK] D2 output verified ({total_drive:,} >= 250,000 threshold)\")
""")


CELL4_BUNDLE = code("""
# ── Cell 4: unzip source bundle ─────────────────────────────────────────────
import shutil, sys
WORK_DIR = \"/content/fitnova_v6\"
shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
subprocess.run([\"unzip\", \"-q\", SRC_ZIP, \"-d\", WORK_DIR], check=True)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from backend.training.models.st_gcn_v6 import build_v6_model
n_params = build_v6_model().count_params()
print(f\"  v6 model factory imported. trainable params -> {n_params:,}\")
assert 1_000_000 < n_params < 1_200_000, f\"unexpected param count: {n_params}\"
print(\"  [OK] bundle unzipped + import verified\")
""")


CELL5_COPY_NPZ = code("""
# ── Cell 5: cache .npz from Drive → /content via Drive API ──────────────────
# WHY: Drive FUSE listdir times out on 76K-file dirs (rsync, cp, ls all break).
# We use the Drive API for both LISTING and DOWNLOADING — bypasses FUSE entirely.
#
# Speedup rationale: training reads 32 files/batch × 8800 batches/epoch ×
# 30 epochs = 8.4M opens. From Drive FUSE: ~120 hours of I/O. From local SSD:
# ~12 hours. Caching saves ~5-10x on the training run.
import io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaIoBaseDownload
from tqdm.auto import tqdm

LOCAL_NPZ_ROOT = \"/content/qevd_local\"
os.makedirs(LOCAL_NPZ_ROOT, exist_ok=True)

def list_npz_in_folder(folder_id):
    out, page_token = [], None
    while True:
        r = svc.files().list(
            q=(f\"'{folder_id}' in parents and trashed=false \"
               f\"and name contains '.npz'\"),
            fields=\"nextPageToken, files(id, name)\",
            pageSize=1000, pageToken=page_token,
        ).execute()
        out.extend((f[\"name\"], f[\"id\"]) for f in r[\"files\"])
        page_token = r.get(\"nextPageToken\")
        if not page_token: break
    return out

def download_one(fid_, dst_path):
    \"\"\"Download one .npz via Drive API. Skips if already cached.\"\"\"
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return None
    from googleapiclient.discovery import build as _build
    local_svc = _build(\"drive\", \"v3\")
    for attempt in range(3):
        try:
            req = local_svc.files().get_media(fileId=fid_)
            with io.FileIO(dst_path, \"wb\") as fh:
                downloader = MediaIoBaseDownload(fh, req, chunksize=1024*1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return None
        except Exception as e:
            if attempt == 2:
                if os.path.exists(dst_path):
                    try: os.remove(dst_path)
                    except OSError: pass
                return f\"{os.path.basename(dst_path)}: {type(e).__name__}: {e}\"
            time.sleep(2 ** attempt)

print(\"Caching .npz from Drive -> /content via Drive API (no FUSE) ...\")
print(\"(Resumable: re-running skips files already cached)\\n\")

t_start = time.time()
per_part_local = {}
for n in (1, 2, 3, 4):
    fid = find_folder([\"fitnova_v6\", \"qevd_extracted\", f\"Part-{n}\"])
    if fid is None:
        print(f\"  Part-{n}: skip (Drive folder missing)\")
        per_part_local[n] = 0
        continue
    print(f\"  Part-{n}: listing via API ...\")
    files = list_npz_in_folder(fid)
    dst_dir = f\"{LOCAL_NPZ_ROOT}/Part-{n}\"
    os.makedirs(dst_dir, exist_ok=True)
    todo = [(name, fid_i) for name, fid_i in files
            if not (os.path.exists(f\"{dst_dir}/{name}\")
                    and os.path.getsize(f\"{dst_dir}/{name}\") > 0)]
    print(f\"  Part-{n}: {len(files):,} on Drive, {len(todo):,} to download \"
          f\"({len(files)-len(todo):,} cached)\")
    if not todo:
        per_part_local[n] = len(files)
        continue

    errors = []
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = {exe.submit(download_one, fid_i, f\"{dst_dir}/{name}\"): name
                   for name, fid_i in todo}
        with tqdm(total=len(todo), desc=f\"Part-{n}\", unit=\"file\",
                  smoothing=0.05, mininterval=1.0) as pbar:
            for fut in as_completed(futures):
                err = fut.result()
                if err: errors.append(err)
                if errors and len(errors) % 100 == 0:
                    pbar.set_postfix({\"errors\": len(errors)})
                pbar.update(1)
    n_local = sum(1 for f in os.listdir(dst_dir) if f.endswith(\".npz\"))
    per_part_local[n] = n_local
    print(f\"  Part-{n}: {n_local:,} on /content ({len(errors)} download errors)\")
    if errors:
        print(f\"    first error: {errors[0]}\")

total_local = sum(per_part_local.values())
disk = shutil.disk_usage(\"/content\")
print(f\"\\n[OK] {total_local:,} .npz on /content. \"
      f\"Free disk: {disk.free/1e9:.1f} GB. \"
      f\"Total time: {(time.time()-t_start)/60:.1f} min\")
assert total_local >= 250_000, f\"only {total_local} on /content — incomplete\"
""")


CELL6_TRAIN = code("""
# ── Cell 6: launch training (the long one) — auto-retry + live output ───────
# Hyperparameters from master plan §II.7
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 3e-4

cmd = [
    sys.executable, \"-m\", \"backend.training.train_form_model_v6\",
    \"--npz-dirs\",
    f\"{LOCAL_NPZ_ROOT}/Part-1\", f\"{LOCAL_NPZ_ROOT}/Part-2\",
    f\"{LOCAL_NPZ_ROOT}/Part-3\", f\"{LOCAL_NPZ_ROOT}/Part-4\",
    \"--labels-dir\", LABELS_DIR,
    \"--out-dir\",    MODELS_DIR,
    \"--epochs\",     str(EPOCHS),
    \"--batch-size\", str(BATCH_SIZE),
    \"--lr\",         str(LR),
]
print(\"Train cmd:\", \" \".join(cmd[:8]), \"...\")
print()

# Auto-retry up to 3 times. Each retry hits the trainer's resume path
# (scans MODELS_DIR/checkpoints/ for the latest epoch_NN.weights.h5).
import time as _t
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f\"\\n{'='*60}\\nattempt {attempt}/{MAX_ATTEMPTS}\\n{'='*60}\")
    proc = subprocess.Popen(
        cmd, cwd=WORK_DIR,
        env={**os.environ, \"PYTHONPATH\": WORK_DIR, \"PYTHONUNBUFFERED\": \"1\"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    proc.wait()
    if proc.returncode == 0:
        print(f\"\\n[OK] training finished cleanly on attempt {attempt}.\")
        break
    print(f\"\\n[WARN] training exited with code {proc.returncode} on attempt {attempt}.\")
    if attempt < MAX_ATTEMPTS:
        print(f\"      retrying in 30s (will resume from latest checkpoint) ...\")
        _t.sleep(30)
    else:
        raise RuntimeError(f\"training failed after {MAX_ATTEMPTS} attempts.\")
""")


CELL7_VERIFY = code("""
# ── Cell 7: verify outputs + API-based server-side check (NO flush hang) ────
# We DELIBERATELY do not call drive.flush_and_unmount(). That call has
# silently hung for 30-60+ minutes in past sessions. Instead we use the
# already-authed Drive API to check what's actually on Google's servers,
# and force-upload via API anything missing.
import json
from googleapiclient.http import MediaFileUpload

weights_path = f\"{MODELS_DIR}/v6_supervised.weights.h5\"
history_path = f\"{MODELS_DIR}/v6_history.json\"
config_path  = f\"{MODELS_DIR}/model_config.json\"
ex_map_path  = f\"{MODELS_DIR}/qevd_exercise_map.json\"

print(\"[1] checking outputs locally ...\")
for p in (weights_path, history_path, config_path, ex_map_path):
    assert os.path.isfile(p), f\"missing: {p}\"
    print(f\"  [OK] {os.path.basename(p)}: {os.path.getsize(p)/1e6:.2f} MB\")

print(\"\\n[2] last-3 epoch summary:\")
hist = json.loads(open(history_path).read())
for k in (\"total\", \"val_total\"):
    if k in hist:
        vals = [round(x, 4) for x in hist[k][-3:]]
        print(f\"  {k:<12s} -> {vals}\")

print(\"\\n[3] verifying server-side via Drive API ...\")
model_fid = find_folder([\"fitnova_v6\", \"models\", \"form_model_v6\"])
if model_fid is None:
    raise RuntimeError(\"models dir not found on Drive\")

server_files = {}
page_token = None
while True:
    r = svc.files().list(
        q=f\"'{model_fid}' in parents and trashed=false\",
        fields=\"nextPageToken, files(id, name, size)\",
        pageSize=200, pageToken=page_token,
    ).execute()
    for f in r[\"files\"]:
        server_files[f[\"name\"]] = int(f.get(\"size\", 0))
    page_token = r.get(\"nextPageToken\")
    if not page_token: break

expected = [
    (\"v6_supervised.weights.h5\", weights_path),
    (\"v6_history.json\",          history_path),
    (\"model_config.json\",        config_path),
    (\"qevd_exercise_map.json\",   ex_map_path),
]
for name, local_path in expected:
    if name in server_files and server_files[name] > 0:
        print(f\"  [OK] {name} on Drive: {server_files[name]/1e6:.2f} MB\")
    else:
        print(f\"  [upload] {name} not on Drive yet — pushing via API ...\")
        media = MediaFileUpload(local_path, mimetype=\"application/octet-stream\",
                                 resumable=False)
        if name in server_files:
            r = svc.files().list(
                q=f\"'{model_fid}' in parents and name='{name}' and trashed=false\",
                fields=\"files(id)\", pageSize=1).execute()
            if r[\"files\"]:
                svc.files().update(fileId=r[\"files\"][0][\"id\"],
                                    media_body=media, fields=\"id\").execute()
        else:
            svc.files().create(
                body={\"name\": name, \"parents\": [model_fid]},
                media_body=media, fields=\"id\",
            ).execute()
        print(f\"  [OK] {name} uploaded.\")

print(f\"\\n[DONE] outputs durably on Drive at: {MODELS_DIR}\")
""")


CELL_DONE = md("""
## Done

Outputs at `MyDrive/fitnova_v6/models/form_model_v6/`:
- `v6_supervised.weights.h5` ← the trained model
- `v6_history.json` ← per-epoch losses (5 heads + total)
- `model_config.json` ← target_frames / n_exercises / n_joint_groups
- `qevd_exercise_map.json` ← 24 named QEVD exercises + 'other'
- `checkpoints/epoch_NN.weights.h5` ← rotating last-3 (resume support)

You can disconnect the runtime now to stop burning units.

## Next (run LOCALLY on your PC)
The reality-check D9 conflicts with TF on protobuf, so it runs locally:

1. Download the 4 files above from Drive to `backend/models/form_model_v6/` on your PC
2. From repo root:
```
python -m backend.training.evaluation.reality_check_v6 \\
    --model-dir backend/models/form_model_v6 \\
    --report    backend/data/qevd_phase_reports/D9_reality_check.json
```
3. Then D7 in-domain eval (needs the .npz from Drive too — download Part-* dirs):
```
python -m backend.training.evaluation.qevd_in_domain_eval \\
    --model-dir       backend/models/form_model_v6 \\
    --extracted-dirs  backend/data/qevd_extracted/fit300k/test \\
    --fine-grained    backend/data/.../fine_grained_labels.json \\
    --feedbacks       backend/data/.../feedbacks_short_clips.json \\
    --report          backend/data/qevd_phase_reports/D7_in_domain.json
```

D9 quality gap ≥ +0.15 = **ship it**.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Source bundle (same as D2 — re-emitted here so D6 alone can rebuild it)
# ─────────────────────────────────────────────────────────────────────────────


SRC_FILES_V6 = [
    "backend/__init__.py",
    "backend/services/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/diagnostics/__init__.py",
    "backend/training/models/__init__.py",
    "backend/training/evaluation/__init__.py",

    "backend/services/mediapipe_config.py",

    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/qevd_extractor.py",
    "backend/training/preprocessing/qevd_label_builder.py",
    "backend/training/preprocessing/qevd_dataset.py",
    "backend/training/preprocessing/qevd_urls.py",

    "backend/training/diagnostics/phase_gates.py",

    "backend/training/models/st_gcn.py",
    "backend/training/models/st_gcn_v6.py",

    "backend/training/train_form_model_v6.py",
    "backend/training/evaluation/reality_check_v6.py",
    "backend/training/evaluation/qevd_fitcoach_eval.py",
    "backend/training/evaluation/qevd_in_domain_eval.py",

    "backend/training/diagnostics/references/reference_outputs.npz",
    "backend/training/diagnostics/references/input_clips/Good_Squats.mp4",
]


def main():
    print("== v6 D6 Notebook ==")
    cells = [
        CELL_INTRO,
        CELL1_HW,
        CELL2_DRIVE,
        CELL3_VERIFY_VIA_API,
        CELL4_BUNDLE,
        CELL5_COPY_NPZ,
        CELL6_TRAIN,
        CELL7_VERIFY,
        CELL_DONE,
    ]
    write_notebook("v6_d6_train.ipynb", cells)
    print()

    print("== v6 source bundle (re-emitted) ==")
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
    print(f"  1. Upload {out_zip.name} to MyDrive/fitnova_v6/src/ (replaces D2 bundle)")
    print( "  2. Upload feedbacks_short_clips.json + fine_grained_labels.json + ")
    print( "     fine_grained_labels_with_worker_ids.json to MyDrive/fitnova_v6/labels/")
    print( "  3. Open colab_notebooks/v6_d6_train.ipynb in Colab")
    print( "     T4/L4/A100 GPU + High-RAM. Run all cells.")
    print( "  4. Wait. Auto-resumes on disconnect, auto-retries on crash.")
    print( "  5. Download trained weights to backend/models/form_model_v6/ on your PC.")
    print( "  6. Run reality_check_v6 LOCALLY for the D9 verdict.")


if __name__ == "__main__":
    main()
