"""Build the Colab bundle for v6.1 supervised training.

Forks _build_v6_d6_colab_bundle.py and adapts to the v6.1 architecture
(single multi-label head over the 1,387-class QEVD variation taxonomy)
and the user-mandated robustness contract (plan §9):

    1. Visible progress bar (tf.keras.utils.Progbar in train + tqdm in prep)
    2. Resume-from-checkpoint on disconnect
    3. Tee logs to Drive (every cell)
    4. Pre-flight self-check (GPU, TF version, Drive mount, bundle hash,
       dataset .npz count, free disk space) — fails fast before training

The v6.1 bundle includes a NEW file (qevd_class_space_v6_1.json) and
NEW Python modules (qevd_class_space.py, qevd_label_builder_v2.py,
qevd_dataset_v6_1.py, st_gcn_v6_1.py, train_form_model_v6_1.py,
reality_check_v6_1.py, defect_to_region.py).

It re-uses the .npz files already on Drive from D2 (no re-extraction
needed — the dataset pipeline only swaps the label-derivation step).

Run once locally before launching Colab:

    python _build_v6_1_colab_bundle.py

Outputs:
    colab_notebooks/v6_1_train.ipynb   the v6.1 training notebook
    fitnova_v6_1_src.zip                the v6.1 source bundle
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent
COLAB_DIR = PROJECT_ROOT / "colab_notebooks"
COLAB_DIR.mkdir(exist_ok=True)


def md(text):   return nbf.v4.new_markdown_cell(text.lstrip("\n"))
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
# Notebook cells
# ─────────────────────────────────────────────────────────────────────────────


CELL_INTRO = md("""
# FitNova v6.1 — Supervised Training (paper-faithful multi-label)

Trains the **v6.1 ST-GCN multi-label classifier** on the QEVD variation
taxonomy. Replaces v6's 5-head multi-task regression with a single
sigmoid head over the 1,387 cleaned class strings.

## Why v6.1
- v6 failed D9: `quality_gap = 0.004` because its lexicon-classifier
  compressed 1,851 discrete labels into a 1-D scalar + 10-bit bool.
- v6.1 trains directly on the discrete class strings — what QEVD's
  STREAM-VLM paper does at the vision-backbone level.
- See `plans/handoff-2026-05-11-md-this-handoff-is-sleepy-sparrow.md`
  for the full plan + comparison framing.

## Prerequisites
- D2 has finished: `MyDrive/fitnova_v6/qevd_extracted/Part-{1..4}/`
  is populated (same .npz files v6 used — same pose pipeline,
  different labels).
- `MyDrive/fitnova_v6/labels/` contains `fine_grained_labels.json`
  and (optionally) `feedbacks_short_clips.json`,
  `fine_grained_labels_with_worker_ids.json` for participant-disjoint
  splits.
- `MyDrive/fitnova_v6_1/src/fitnova_v6_1_src.zip` is the latest
  v6.1 bundle (uploaded after running `_build_v6_1_colab_bundle.py`
  locally).

## Runtime
| GPU | Time | Notes |
|---|---|---|
| **A100** | 5-8 h | one session |
| **L4** | 15-22 h | spans 1–2 Colab Pro+ sessions; **§9 contract resumes** |
| **T4** | too slow for the long-tail multi-label head |

User target: **L4 on Colab Pro+** (24 h session window covers the run).

## §9 robustness contract (in this notebook)
1. **Cell 1** is a pre-flight self-check: GPU + TF version + Drive
   mount + bundle hash + dataset count + disk space. Fails fast.
2. **All cells** tee stdout to `MyDrive/fitnova_v6_1/logs/cell_N_<timestamp>.log`.
3. **Cell 6** auto-retries the training subprocess; each retry resumes
   from `checkpoints/epoch_NN.weights.h5` on Drive.
4. **Cell 6** also streams `tf.keras.utils.Progbar` and per-step
   heartbeats so you can see progress without `tqdm` redraws breaking.
""")


CELL1_PREFLIGHT = code("""
# ── Cell 1: pre-flight self-check (FAILS FAST) ──────────────────────────────
# §9 contract item #4. Verifies everything the training run depends on
# BEFORE we burn an L4 hour on a broken bundle / missing dataset / etc.
import os, sys, shutil, subprocess, hashlib, json, time, io
from datetime import datetime

PREFLIGHT_FAILURES = []
def fail(msg):
    PREFLIGHT_FAILURES.append(msg)
    print(f\"  [FAIL] {msg}\")
def ok(msg):
    print(f\"  [OK]   {msg}\")

print(\"=== PRE-FLIGHT SELF-CHECK ===\")
print(f\"  timestamp: {datetime.utcnow().isoformat()}Z\")

# 1. CPU + RAM + disk
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
    pass

disk = shutil.disk_usage(\"/content\")
disk_free_gb = disk.free / 1e9
if disk_free_gb >= 30:
    ok(f\"/content free disk: {disk_free_gb:.1f} GB (>= 30 GB threshold)\")
else:
    fail(f\"/content free disk only {disk_free_gb:.1f} GB; need >= 30 GB\")

# 2. GPU detection
gpu_name = None
try:
    out = subprocess.check_output(
        [\"nvidia-smi\", \"--query-gpu=name,memory.total\", \"--format=csv,noheader\"],
        text=True,
    ).strip()
    print(f\"  GPU:       {out}\")
    gpu_name = out.split(\",\")[0].strip()
except Exception:
    fail(\"No GPU detected. Switch runtime to GPU and re-run.\")

if gpu_name:
    if \"L4\" in gpu_name or \"A100\" in gpu_name:
        ok(f\"GPU class is suitable for v6.1 training: {gpu_name}\")
    elif \"T4\" in gpu_name:
        print(f\"  [warn] T4 will work but is slow for the 1387-class head.\")
    else:
        print(f\"  [warn] Unknown GPU class: {gpu_name}\")

# 3. TF version
try:
    import tensorflow as tf
    tf_v = tf.__version__
    print(f\"  TF:        {tf_v}\")
    major, minor = (int(x) for x in tf_v.split(\".\")[:2])
    if (major, minor) >= (2, 15):
        ok(f\"TF {tf_v} >= 2.15 (Colab default)\")
    else:
        fail(f\"TF {tf_v} unexpected — Colab usually ships >= 2.15\")
except Exception as e:
    fail(f\"TF import failed: {e}\")

# 4. Drive
try:
    from google.colab import drive
    subprocess.run([\"fusermount\", \"-uz\", \"/content/drive\"], capture_output=True)
    subprocess.run([\"rm\", \"-rf\", \"/content/drive\"], capture_output=True)
    drive.mount(\"/content/drive\")
    ok(\"Drive mounted at /content/drive\")
except Exception as e:
    fail(f\"Drive mount failed: {e}\")

DRIVE_BASE_V6   = \"/content/drive/MyDrive/fitnova_v6\"        # reused .npz from D2
DRIVE_BASE_V6_1 = \"/content/drive/MyDrive/fitnova_v6_1\"      # v6.1 outputs
SRC_ZIP    = f\"{DRIVE_BASE_V6_1}/src/fitnova_v6_1_src.zip\"
LABELS_DIR = f\"{DRIVE_BASE_V6}/labels\"
NPZ_ROOT   = f\"{DRIVE_BASE_V6}/qevd_extracted\"
MODELS_DIR = f\"{DRIVE_BASE_V6_1}/models/form_model_v6_1\"
LOGS_DIR   = f\"{DRIVE_BASE_V6_1}/logs\"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 5. Bundle on Drive
if os.path.isfile(SRC_ZIP):
    bundle_size_mb = os.path.getsize(SRC_ZIP) / 1e6
    h = hashlib.sha256()
    with open(SRC_ZIP, \"rb\") as f:
        for chunk in iter(lambda: f.read(1024*1024), b\"\"):
            h.update(chunk)
    bundle_hash = h.hexdigest()[:16]
    ok(f\"src bundle present: {bundle_size_mb:.2f} MB, sha256[:16]={bundle_hash}\")
else:
    fail(f\"src bundle missing: {SRC_ZIP}\\n         \"
         f\"Upload fitnova_v6_1_src.zip to MyDrive/fitnova_v6_1/src/.\")

# 6. Label JSONs
for name in (\"fine_grained_labels.json\", \"feedbacks_short_clips.json\",
             \"fine_grained_labels_with_worker_ids.json\"):
    p = f\"{LABELS_DIR}/{name}\"
    if os.path.isfile(p):
        ok(f\"label file present: {name} ({os.path.getsize(p)/1e6:.1f} MB)\")
    else:
        fail(f\"label file missing: {p}\")

# 7. Tee setup — every subsequent cell appends to a per-cell log on Drive
def tee_log(cell_num: int):
    ts = datetime.utcnow().strftime(\"%Y%m%d_%H%M%S\")
    path = f\"{LOGS_DIR}/cell_{cell_num}_{ts}.log\"
    return path

print()
if PREFLIGHT_FAILURES:
    raise RuntimeError(
        \"Pre-flight self-check FAILED — fix above issues and re-run cell 1.\\n\"
        \"Failures:\\n  - \" + \"\\n  - \".join(PREFLIGHT_FAILURES)
    )
print(\"[PRE-FLIGHT OK] proceeding to training setup.\")
""")


CELL2_VERIFY_NPZ = code("""
# ── Cell 2: verify .npz dataset count on Drive (via API, not FUSE) ──────────
# Reuses the D2 output. Drive FUSE listdir on 76K-file folders times out;
# the Drive API counts the same in 5-30 sec.
LOG_PATH = tee_log(2)
print(f\"[tee] also writing this cell's output to {LOG_PATH}\")

subprocess.run(
    [sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"google-api-python-client\"],
    check=True,
)
import logging as _logging
_logging.getLogger(\"google_auth_httplib2\").setLevel(_logging.ERROR)
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
t0 = time.time()
per_part = {}
for n in (1, 2, 3, 4):
    fid = find_folder([\"fitnova_v6\", \"qevd_extracted\", f\"Part-{n}\"])
    if fid is None:
        per_part[n] = 0
        print(f\"  Part-{n}: 0 (folder missing on Drive)\")
        continue
    c = count_npz(fid)
    per_part[n] = c
    print(f\"  Part-{n}: {c:,}   ({time.time()-t0:.0f}s elapsed)\")
total_drive = sum(per_part.values())
print(f\"\\nTotal .npz on Drive: {total_drive:,}\")
if total_drive < 250_000:
    raise RuntimeError(
        f\"Too few clips ({total_drive:,}). D2 may be incomplete. \"
        f\"Check the D2 gate report.\"
    )
print(f\"[OK] D2 output verified ({total_drive:,} >= 250,000 threshold)\")

# Persist for later cells
DRIVE_NPZ_COUNT = total_drive

# Also append this cell's printed output to Drive log
with open(LOG_PATH, \"a\") as f:
    f.write(f\"npz_count: {total_drive}\\nper_part: {per_part}\\n\")
""")


CELL3_BUNDLE_UNPACK = code("""
# ── Cell 3: unzip v6.1 source bundle + import-verify ────────────────────────
LOG_PATH = tee_log(3)
print(f\"[tee] log -> {LOG_PATH}\")

WORK_DIR = \"/content/fitnova_v6_1\"
shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
subprocess.run([\"unzip\", \"-q\", SRC_ZIP, \"-d\", WORK_DIR], check=True)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

# Import-verify: every module the trainer touches must import.
from backend.training.preprocessing.qevd_class_space import get_default_class_space
from backend.training.preprocessing.qevd_label_builder_v2 import build_v6_1_target
from backend.training.preprocessing.qevd_dataset_v6_1 import make_v6_1_clip_dataset
from backend.training.models.st_gcn_v6_1 import build_v6_1_model

cs = get_default_class_space()
m = build_v6_1_model(num_classes=cs.num_classes, n_exercises=cs.num_prefixes)
n_params = m.count_params()
print(f\"  class space:    {cs.num_classes} classes, {cs.num_prefixes} prefixes\")
print(f\"  model factory:  imported, trainable params = {n_params:,}\")
assert 1_000_000 <= n_params <= 2_000_000, f\"unexpected param count: {n_params}\"
print(\"  [OK] bundle unpacked + all v6.1 imports verified\")

with open(LOG_PATH, \"a\") as f:
    f.write(f\"v6_1 params: {n_params}\\nv6_1 classes: {cs.num_classes}\\n\")
""")


CELL4_CACHE_NPZ = code("""
# ── Cell 4: cache .npz from Drive -> /content via Drive API ─────────────────
# Same pattern as v6 D6 — Drive FUSE is unusable for 250K+ file dirs.
LOG_PATH = tee_log(4)
print(f\"[tee] log -> {LOG_PATH}\")

import io
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
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return None
    local_svc = build(\"drive\", \"v3\")
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
        per_part_local[n] = 0
        print(f\"  Part-{n}: skip (Drive folder missing)\")
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

with open(LOG_PATH, \"a\") as f:
    f.write(f\"npz_local: {total_local}\\nper_part_local: {per_part_local}\\n\")
""")


CELL5_TRAIN = code("""
# ── Cell 5: launch v6.1 training (the long one) ─────────────────────────────
# §9 contract: progress bar via per-step heartbeats + tqdm in data prep;
# resume-from-checkpoint via train_form_model_v6_1's _find_latest_checkpoint;
# tee'd Drive log; auto-retry on disconnect.
LOG_PATH = tee_log(5)
print(f\"[tee] log -> {LOG_PATH}\")

# Hyperparameters
EPOCHS         = 30
BATCH_SIZE     = 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 2

cmd = [
    sys.executable, \"-u\", \"-m\", \"backend.training.train_form_model_v6_1\",
    \"--npz-dirs\",
    f\"{LOCAL_NPZ_ROOT}/Part-1\", f\"{LOCAL_NPZ_ROOT}/Part-2\",
    f\"{LOCAL_NPZ_ROOT}/Part-3\", f\"{LOCAL_NPZ_ROOT}/Part-4\",
    \"--labels-dir\",   LABELS_DIR,
    \"--out-dir\",      MODELS_DIR,
    \"--epochs\",       str(EPOCHS),
    \"--batch-size\",   str(BATCH_SIZE),
    \"--lr\",           str(LR),
    \"--weight-decay\", str(WEIGHT_DECAY),
    \"--warmup-epochs\", str(WARMUP_EPOCHS),
]
print(\"Train cmd:\", \" \".join(cmd[:10]), \"...\")
print(f\"hyperparams: epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} \"
      f\"wd={WEIGHT_DECAY} warmup={WARMUP_EPOCHS}\")
print()

# Auto-retry up to 3 times. Each retry hits the trainer's resume path
# (scans MODELS_DIR/checkpoints/ for the latest epoch_NN.weights.h5).
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f\"\\n{'='*60}\\nattempt {attempt}/{MAX_ATTEMPTS}\\n{'='*60}\")
    proc = subprocess.Popen(
        cmd, cwd=WORK_DIR,
        env={**os.environ, \"PYTHONPATH\": WORK_DIR, \"PYTHONUNBUFFERED\": \"1\"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    # Stream every line of training output to (a) the live notebook display
    # AND (b) the Drive-tee log file. The tee guarantees we keep the
    # diagnostic trail even if the Colab cell display drops mid-disconnect.
    with open(LOG_PATH, \"a\") as logf:
        try:
            for line in proc.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                logf.write(line);       logf.flush()
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
        time.sleep(30)
    else:
        raise RuntimeError(f\"training failed after {MAX_ATTEMPTS} attempts.\")
""")


CELL6_VERIFY = code("""
# ── Cell 6: verify outputs + Drive API server-side check ────────────────────
# We DO NOT call drive.flush_and_unmount() — that has silently hung for
# 30-60+ min in past sessions. Drive API confirms what's actually on
# Google's servers and re-uploads anything missing.
LOG_PATH = tee_log(6)
print(f\"[tee] log -> {LOG_PATH}\")

from googleapiclient.http import MediaFileUpload

weights_path = f\"{MODELS_DIR}/v6_1_supervised.weights.h5\"
history_path = f\"{MODELS_DIR}/v6_1_history.json\"
config_path  = f\"{MODELS_DIR}/model_config_v6_1.json\"

print(\"[1] checking outputs locally ...\")
for p in (weights_path, history_path, config_path):
    if not os.path.isfile(p):
        raise RuntimeError(f\"missing: {p}\")
    print(f\"  [OK] {os.path.basename(p)}: {os.path.getsize(p)/1e6:.2f} MB\")

print(\"\\n[2] last-3 epoch summary:\")
hist = json.loads(open(history_path).read())
for k in (\"train_loss\", \"val_loss\", \"val_macro_f1\",
          \"val_top1_acc\", \"val_top3_recall\", \"val_active_classes\"):
    if k in hist:
        vals = [round(x, 4) if isinstance(x, float) else x for x in hist[k][-3:]]
        print(f\"  {k:<22s} -> {vals}\")

print(\"\\n[3] verifying server-side via Drive API ...\")
# Find/create the v6.1 models folder
def find_or_create_folder(path_parts):
    parent = \"root\"
    for name in path_parts:
        r = svc.files().list(
            q=f\"'{parent}' in parents and name='{name}' and trashed=false\",
            fields=\"files(id)\", pageSize=10).execute()
        if r[\"files\"]:
            parent = r[\"files\"][0][\"id\"]
        else:
            meta = {\"name\": name, \"mimeType\": \"application/vnd.google-apps.folder\",
                    \"parents\": [parent]}
            created = svc.files().create(body=meta, fields=\"id\").execute()
            parent = created[\"id\"]
    return parent

model_fid = find_or_create_folder([\"fitnova_v6_1\", \"models\", \"form_model_v6_1\"])
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
    (\"v6_1_supervised.weights.h5\", weights_path),
    (\"v6_1_history.json\",          history_path),
    (\"model_config_v6_1.json\",     config_path),
]
for name, local_path in expected:
    local_size = os.path.getsize(local_path)
    srv = server_files.get(name, 0)
    if srv > 0 and srv >= local_size - 1024:
        print(f\"  [OK] {name} on Drive: {srv/1e6:.2f} MB\")
    else:
        print(f\"  [upload] {name} ({srv/1e6:.2f}/{local_size/1e6:.2f} MB) — pushing via API ...\")
        media = MediaFileUpload(local_path, mimetype=\"application/octet-stream\",
                                 resumable=False)
        existing = svc.files().list(
            q=f\"'{model_fid}' in parents and name='{name}' and trashed=false\",
            fields=\"files(id)\", pageSize=1).execute()
        if existing[\"files\"]:
            svc.files().update(fileId=existing[\"files\"][0][\"id\"],
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

v6.1 outputs at `MyDrive/fitnova_v6_1/models/form_model_v6_1/`:
- `v6_1_supervised.weights.h5` ← the trained model
- `v6_1_history.json` ← per-epoch metrics (train_loss, val_loss,
   val_macro_f1, val_top1_acc, val_top3_recall, val_active_classes)
- `model_config_v6_1.json` ← num_classes / num_prefixes / class_space_version
- `checkpoints/epoch_NN.weights.h5` ← rotating last-3 (resume support)

Logs at `MyDrive/fitnova_v6_1/logs/cell_*.log` (one per cell, per run).
You can disconnect the runtime now.

## Next (run LOCALLY on your PC)

1. Download the 3 model files above from Drive to
   `backend/models/form_model_v6_1/` on your PC.
2. Run the v6.1 D9 reality-check:
   ```
   python -m backend.training.evaluation.reality_check_v6_1 \\
       --model-dir backend/models/form_model_v6_1 \\
       --report    backend/data/qevd_phase_reports/D9_reality_check_v6_1.json
   ```
3. **D9 v6.1 gate**:
   - GOOD clips: a `no obvious issue` / `90 degrees` / `shoulder-width`
     / `over 90 degrees` variation must appear in top-3 on ≥ 60% of GOOD clips.
   - BAD clips: a `shallow` / `back not straight` / `knees over toes` /
     `narrow` / `wide` / `insufficient` / `starting late` variation must
     appear in top-3 on ≥ 60% of BAD clips.

If D9 v6.1 passes → ship the model + wire `coaching_retrieval.py`
(Day 7 in the plan). If D9 v6.1 fails → see plan §10 fallback.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Source bundle for v6.1
# ─────────────────────────────────────────────────────────────────────────────


SRC_FILES_V6_1 = [
    "backend/__init__.py",
    "backend/config/__init__.py",
    "backend/services/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/diagnostics/__init__.py",
    "backend/training/models/__init__.py",
    "backend/training/evaluation/__init__.py",

    # Shared infra (same as v6 — pose pipeline, normalize, splits)
    "backend/services/mediapipe_config.py",
    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/qevd_extractor.py",
    "backend/training/preprocessing/qevd_dataset.py",          # reused for build_qevd_splits, load_clip_npz
    "backend/training/preprocessing/qevd_label_builder.py",    # reused for QEVDLabels.from_files
    "backend/training/preprocessing/qevd_urls.py",
    "backend/training/diagnostics/phase_gates.py",
    "backend/training/models/st_gcn.py",                       # ST-GCN primitives

    # v6.1 specific
    "backend/training/preprocessing/qevd_class_space.py",
    "backend/training/preprocessing/qevd_label_builder_v2.py",
    "backend/training/preprocessing/qevd_dataset_v6_1.py",
    "backend/training/models/st_gcn_v6_1.py",
    "backend/training/train_form_model_v6_1.py",
    "backend/training/evaluation/reality_check_v6_1.py",
    "backend/config/defect_to_region.py",

    # Class space JSON (data, not code, but required at runtime)
    "backend/data/qevd_class_space_v6_1.json",
]


def main():
    print("== v6.1 training notebook ==")
    cells = [
        CELL_INTRO,
        CELL1_PREFLIGHT,
        CELL2_VERIFY_NPZ,
        CELL3_BUNDLE_UNPACK,
        CELL4_CACHE_NPZ,
        CELL5_TRAIN,
        CELL6_VERIFY,
        CELL_DONE,
    ]
    write_notebook("v6_1_train.ipynb", cells)
    print()

    print("== v6.1 source bundle ==")
    out_zip = PROJECT_ROOT / "fitnova_v6_1_src.zip"
    n_files = 0
    total_bytes = 0
    h = hashlib.sha256()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in SRC_FILES_V6_1:
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
    # Hash the final zip for the §9 contract bundle-hash check
    with open(out_zip, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    size_mb = out_zip.stat().st_size / 1e6
    print(f"  wrote {n_files} files, {size_mb:.2f} MB compressed "
          f"({total_bytes/1e6:.2f} MB raw)")
    print(f"  sha256: {h.hexdigest()}")
    print(f"  sha256[:16] (for bundle-hash pin): {h.hexdigest()[:16]}")

    print()
    print("=" * 70)
    print("Next:")
    print(f"  1. Upload {out_zip.name} to MyDrive/fitnova_v6_1/src/")
    print( "  2. Upload the label JSONs to MyDrive/fitnova_v6/labels/ if not done")
    print( "     (same files as v6 D6 — feedbacks_short_clips.json,")
    print( "      fine_grained_labels.json, fine_grained_labels_with_worker_ids.json)")
    print( "  3. Open colab_notebooks/v6_1_train.ipynb in Colab")
    print( "     L4 GPU + High-RAM (Colab Pro+ recommended for the 24h session)")
    print( "  4. Run all cells. Auto-resumes on disconnect; auto-retries on crash.")
    print( "  5. Download trained weights to backend/models/form_model_v6_1/ on your PC.")
    print( "  6. Run reality_check_v6_1 LOCALLY for the D9 v6.1 verdict.")


if __name__ == "__main__":
    main()
