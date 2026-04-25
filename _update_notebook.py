"""
Rewrites FitNova_Colab.ipynb with the FIT3D-source training flow.

Key differences from the old MediaPipe-source flow:
  - No per-video MediaPipe extraction (Cell 12 becomes: extract Fit3D annotations).
  - Dataset is built from joints3d_25 JSONs + rep_ann.json files (Fit3D MoCap ground truth).
  - Sim-to-real robustness comes from _augment_joints3d() inside dataset_builder.py:
    random Y-axis rotation (±45°) + anisotropic Gaussian noise.
  - Videos.zip is now OPTIONAL — only needed if you also want to visualize_pipeline
    on Colab (most users will do that locally).

Run: python _update_notebook.py
"""
import json

NB = 'FitNova_Colab.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']


# ─────────────────────────────────────────────────────────────────────────────
# Cell map (index → purpose).  Indexes 0..4 are unchanged: title, GPU check,
# Drive mount.  We only rewrite cells 5..20.
# ─────────────────────────────────────────────────────────────────────────────

# ── Cell[5] — Markdown: Cell 3 (videos extraction, now OPTIONAL) ─────────────
cells[5]['source'] = [
    "## Cell 3 — Extract videos (OPTIONAL, skip unless you also want to run visualize_pipeline on Colab)\n",
    "\n",
    "The Fit3D training path does NOT need videos — it reads the MoCap JSONs directly.\n",
    "You can skip this cell. Run it only if you want mp4s on Colab for later inspection.\n",
]

# ── Cell[6] — Videos extraction (unchanged, idempotent) ──────────────────────
cells[6]['source'] = [
    "import shutil, zipfile, os\n",
    "\n",
    "target = '/content/videos'\n",
    "n_mp4 = sum(1 for _, _, fs in os.walk(target) for f in fs if f.endswith('.mp4'))\n",
    "if n_mp4 >= 100:\n",
    "    print(f'Already extracted ({n_mp4} .mp4 files). Skipping.')\n",
    "elif not os.path.exists('/content/drive/MyDrive/videos.zip'):\n",
    "    print('videos.zip not on Drive — skipping (not required for Fit3D training).')\n",
    "else:\n",
    "    print('Copying videos.zip from Drive to local disk...')\n",
    "    shutil.copy('/content/drive/MyDrive/videos.zip', '/content/videos.zip')\n",
    "    print('Extracting...')\n",
    "    with zipfile.ZipFile('/content/videos.zip') as zf:\n",
    "        zf.extractall('/content/videos')\n",
    "    os.remove('/content/videos.zip')\n",
    "    count = sum(1 for _, _, fs in os.walk('/content/videos')\n",
    "                for f in fs if f.endswith('.mp4'))\n",
    "    print(f'Extracted {count} .mp4 files')\n",
]

# ── Cell[7] — Markdown: Cell 4 (source code upload, unchanged) ───────────────
cells[7]['source'] = [
    "## Cell 4 — Load source code (crash-safe)\n",
    "\n",
    "**First time:** upload `fitnova_src.zip` via the file picker.\n",
    "It is also saved to `MyDrive/fitnova_src.zip` automatically.\n",
    "\n",
    "**After reconnect:** the cell loads from Drive automatically. No re-upload needed.\n",
    "\n",
    "If you rebuild the zip on your laptop (e.g. after editing any backend file),\n",
    "**delete `MyDrive/fitnova_src.zip` first**, otherwise this cell will restore the old one.\n",
]

# ── Cell[8] — Source code upload (unchanged) ─────────────────────────────────
cells[8]['source'] = [
    "import os, shutil, zipfile\n",
    "\n",
    "DRIVE_SRC = '/content/drive/MyDrive/fitnova_src.zip'\n",
    "LOCAL_SRC  = '/content/fitnova_src.zip'\n",
    "\n",
    "if os.path.exists(DRIVE_SRC):\n",
    "    print('Loading fitnova_src.zip from Drive (no upload needed)...')\n",
    "    shutil.copy(DRIVE_SRC, LOCAL_SRC)\n",
    "    src_zip = LOCAL_SRC\n",
    "else:\n",
    "    print('fitnova_src.zip not on Drive yet -- use file picker below.')\n",
    "    from google.colab import files\n",
    "    uploaded = files.upload()\n",
    "    src_zip = list(uploaded.keys())[0]\n",
    "    shutil.copy(src_zip, DRIVE_SRC)\n",
    "    print('Saved to Drive: MyDrive/fitnova_src.zip')\n",
    "\n",
    "# Clean any previous extraction so edits in the new zip take effect\n",
    "if os.path.exists('/content/backend'):\n",
    "    shutil.rmtree('/content/backend')\n",
    "with zipfile.ZipFile(src_zip) as zf:\n",
    "    zf.extractall('/content')\n",
    "print('Source extracted')\n",
    "!ls /content/backend/training\n",
]

# ── Cell[9] — Markdown: Cell 5 (deps) ────────────────────────────────────────
cells[9]['source'] = [
    "## Cell 5 — Install deps + set path\n",
]

# ── Cell[10] — pip install ───────────────────────────────────────────────────
cells[10]['source'] = [
    "!pip install -q mediapipe scikit-learn h5py tqdm\n",
    "import sys, os\n",
    "sys.path.insert(0, '/content')\n",
    "os.chdir('/content')\n",
    "print('Ready')\n",
]

# ── Cell[11] — Markdown: Cell 6 (Fit3D annotations, REPLACES MediaPipe cell) ─
cells[11]['source'] = [
    "## Cell 6 — Extract Fit3D annotations (~10 sec, crash-safe)\n",
    "\n",
    "Fit3D training reads `joints3d_25/*.json` + `rep_ann.json` files.\n",
    "These are small (<50 MB total for all 8 subjects) and must be uploaded to\n",
    "`MyDrive/fit3d_annotations.zip` once.\n",
    "\n",
    "**To build the zip on your laptop:**\n",
    "```\n",
    "python _build_fit3d_annotations_zip.py\n",
    "```\n",
    "Then upload the resulting `fit3d_annotations.zip` to the root of your Google Drive.\n",
    "\n",
    "- **First run**: extracts from Drive to `/content/fit3d_data/`.\n",
    "- **After reconnect**: detects existing extraction and skips.\n",
]

# ── Cell[12] — Fit3D annotations extraction (REPLACES old MediaPipe cell) ────
cells[12]['source'] = [
    "import os, shutil, zipfile\n",
    "\n",
    "FIT3D_DRIVE = '/content/drive/MyDrive/fit3d_annotations.zip'\n",
    "FIT3D_LOCAL = '/content/fit3d_data'\n",
    "\n",
    "if os.path.exists(FIT3D_LOCAL) and os.path.isdir(FIT3D_LOCAL):\n",
    "    n_json = sum(1 for _, _, fs in os.walk(FIT3D_LOCAL)\n",
    "                 for f in fs if f.endswith('.json'))\n",
    "    if n_json >= 50:\n",
    "        print(f'Fit3D annotations already extracted ({n_json} JSONs). Skipping.')\n",
    "    else:\n",
    "        shutil.rmtree(FIT3D_LOCAL)\n",
    "        print('Incomplete previous extraction removed — re-extracting...')\n",
    "\n",
    "if not os.path.exists(FIT3D_LOCAL):\n",
    "    if not os.path.exists(FIT3D_DRIVE):\n",
    "        raise FileNotFoundError(\n",
    "            'fit3d_annotations.zip not found at MyDrive/fit3d_annotations.zip.\\n'\n",
    "            'Build it on your laptop:\\n'\n",
    "            '    python _build_fit3d_annotations_zip.py\\n'\n",
    "            'Then upload to Drive root.'\n",
    "        )\n",
    "    print('Copying fit3d_annotations.zip from Drive...')\n",
    "    shutil.copy(FIT3D_DRIVE, '/content/fit3d_annotations.zip')\n",
    "    print('Extracting...')\n",
    "    with zipfile.ZipFile('/content/fit3d_annotations.zip') as zf:\n",
    "        zf.extractall(FIT3D_LOCAL)\n",
    "    os.remove('/content/fit3d_annotations.zip')\n",
    "\n",
    "# Sanity check: expected subject folders\n",
    "expected = ['s03', 's04', 's05', 's07', 's08', 's09', 's10', 's11']\n",
    "missing = [s for s in expected if not os.path.isdir(os.path.join(FIT3D_LOCAL, s))]\n",
    "if missing:\n",
    "    raise RuntimeError(f'Missing subject folders under {FIT3D_LOCAL}: {missing}')\n",
    "\n",
    "n_json = sum(1 for _, _, fs in os.walk(FIT3D_LOCAL)\n",
    "             for f in fs if f.endswith('.json'))\n",
    "print(f'Fit3D annotations ready: {n_json} JSONs across 8 subjects')\n",
    "!ls /content/fit3d_data | head -20\n",
]

# ── Cell[13] — Markdown: Cell 7 (build dataset, Fit3D path) ──────────────────
cells[13]['source'] = [
    "## Cell 7 — Build training dataset from Fit3D MoCap (~3-5 min, crash-safe)\n",
    "\n",
    "Builds `form_dataset_fit3d/` with:\n",
    "- **Source**: Fit3D `joints3d_25` MoCap (clean 3D ground truth, view-invariant).\n",
    "- **Sim-to-real**: random Y-axis rotation (±45°) + anisotropic Gaussian noise\n",
    "  applied to training split only, to bridge MoCap → noisy MediaPipe at inference.\n",
    "- **Split**: train = s03,s04,s05,s07,s08 · val = s09,s10 · **test = s11** (never touched).\n",
    "\n",
    "- **First run**: builds and saves to `MyDrive/fitnova_dataset_fit3d/`.\n",
    "- **After reconnect**: restores from Drive (~1 min). No rebuild needed.\n",
    "\n",
    "To force a fresh build: delete `MyDrive/fitnova_dataset_fit3d/` from Drive, then rerun.\n",
]

# ── Cell[14] — Build dataset (--source fit3d) ────────────────────────────────
cells[14]['source'] = [
    "import os, shutil\n",
    "\n",
    "DATASET_LOCAL = '/content/backend/data/form_dataset_fit3d'\n",
    "DATASET_DRIVE = '/content/drive/MyDrive/fitnova_dataset_fit3d'\n",
    "FIT3D_ROOT    = '/content/fit3d_data'\n",
    "\n",
    "if os.path.exists(os.path.join(DATASET_DRIVE, 'train', 'X_angles.npy')):\n",
    "    # ── Fast restore path (~1-2 min) ──────────────────────────────────────\n",
    "    print('Dataset found on Drive. Restoring to local disk...')\n",
    "    if os.path.exists(DATASET_LOCAL):\n",
    "        shutil.rmtree(DATASET_LOCAL)\n",
    "    shutil.copytree(DATASET_DRIVE, DATASET_LOCAL)\n",
    "    import numpy as np\n",
    "    splits = {}\n",
    "    for sp in ('train', 'val', 'test'):\n",
    "        p = os.path.join(DATASET_LOCAL, sp, 'X_angles.npy')\n",
    "        if os.path.exists(p):\n",
    "            splits[sp] = len(np.load(p))\n",
    "    print('  Splits:', splits)\n",
    "    print('Restored from Drive. Skipping build.')\n",
    "else:\n",
    "    # ── Build path ────────────────────────────────────────────────────────\n",
    "    assert os.path.exists(FIT3D_ROOT), \\\n",
    "        f'Fit3D root missing at {FIT3D_ROOT}. Run Cell 6 first.'\n",
    "    print(f'Building dataset from Fit3D MoCap at {FIT3D_ROOT}...')\n",
    "    print('  (Sim-to-real domain randomization applied to train split only)')\n",
    "    !python -m backend.training.preprocessing.dataset_builder \\\n",
    "        {FIT3D_ROOT} \\\n",
    "        {DATASET_LOCAL} \\\n",
    "        --source fit3d\n",
    "\n",
    "    if os.path.exists(os.path.join(DATASET_LOCAL, 'train', 'X_angles.npy')):\n",
    "        import numpy as np\n",
    "        splits = {}\n",
    "        for sp in ('train', 'val', 'test'):\n",
    "            p = os.path.join(DATASET_LOCAL, sp, 'X_angles.npy')\n",
    "            if os.path.exists(p):\n",
    "                splits[sp] = len(np.load(p))\n",
    "        print('  Splits:', splits)\n",
    "        print('Saving dataset to Drive for crash recovery (~2 min)...')\n",
    "        if os.path.exists(DATASET_DRIVE):\n",
    "            shutil.rmtree(DATASET_DRIVE)\n",
    "        shutil.copytree(DATASET_LOCAL, DATASET_DRIVE)\n",
    "        print('  Saved to MyDrive/fitnova_dataset_fit3d/')\n",
    "    else:\n",
    "        raise RuntimeError('Dataset not found after build -- check output above.')\n",
]

# ── Cell[15] — Markdown: Cell 8 (SSL pretraining) ────────────────────────────
cells[15]['source'] = [
    "## Cell 8 — SSL Skeleton Autoencoder Pretraining (~90 min on L4, crash-safe)\n",
    "\n",
    "Trains a masked autoencoder on all subjects' data. No task labels used.\n",
    "SSL encoder weights are saved to Drive immediately after training.\n",
    "\n",
    "- **After reconnect**: detects saved encoder weights on Drive and skips training.\n",
    "- **To force retrain**: delete `MyDrive/fitnova_results_fit3d/mt_tcn_encoder_ssl.weights.h5`.\n",
]

# ── Cell[16] — SSL pretraining (Fit3D dataset path) ──────────────────────────
cells[16]['source'] = [
    "import os, shutil\n",
    "\n",
    "SSL_LOCAL = '/content/backend/models/form_model/mt_tcn_encoder_ssl.weights.h5'\n",
    "DRIVE_DIR = '/content/drive/MyDrive/fitnova_results_fit3d'\n",
    "SSL_DRIVE = os.path.join(DRIVE_DIR, 'mt_tcn_encoder_ssl.weights.h5')\n",
    "\n",
    "os.makedirs('/content/backend/models/form_model', exist_ok=True)\n",
    "os.makedirs(DRIVE_DIR, exist_ok=True)\n",
    "\n",
    "if os.path.exists(SSL_DRIVE):\n",
    "    print('SSL encoder weights found on Drive. Restoring and skipping training.')\n",
    "    shutil.copy(SSL_DRIVE, SSL_LOCAL)\n",
    "    print(f'  Size: {os.path.getsize(SSL_LOCAL)/1e6:.1f} MB')\n",
    "else:\n",
    "    print('Running SSL pretraining (L4: ~90 min)...')\n",
    "    !python -m backend.training.train_ssl_pretrain \\\n",
    "        --data-dir  /content/backend/data/form_dataset_fit3d \\\n",
    "        --model-dir /content/backend/models/form_model \\\n",
    "        --epochs 80 \\\n",
    "        --patience 15 \\\n",
    "        --batch-size 128\n",
    "    if os.path.exists(SSL_LOCAL):\n",
    "        shutil.copy(SSL_LOCAL, SSL_DRIVE)\n",
    "        print(f'SSL weights saved to Drive: MyDrive/fitnova_results_fit3d/')\n",
    "    else:\n",
    "        raise RuntimeError('SSL weights not found after training -- check output above.')\n",
]

# ── Cell[17] — Markdown: Cell 9 (supervised fine-tune) ───────────────────────
cells[17]['source'] = [
    "## Cell 9 — Supervised fine-tune from SSL init (~60 min on L4, crash-safe)\n",
    "\n",
    "Loads the SSL encoder from Cell 8 and fine-tunes all 5 task heads on the\n",
    "Fit3D dataset. All model files are saved to Drive immediately after training.\n",
    "\n",
    "- **After reconnect**: detects saved model on Drive and skips training.\n",
    "- **To force retrain**: delete `MyDrive/fitnova_results_fit3d/mt_tcn_best.weights.h5`.\n",
]

# ── Cell[18] — Fine-tune (--source fit3d + Fit3D data dir) ───────────────────
cells[18]['source'] = [
    "import os, shutil\n",
    "\n",
    "MODEL_DIR   = '/content/backend/models/form_model'\n",
    "DRIVE_DIR   = '/content/drive/MyDrive/fitnova_results_fit3d'\n",
    "MODEL_DRIVE = os.path.join(DRIVE_DIR, 'mt_tcn_best.weights.h5')\n",
    "ALL_FILES   = [\n",
    "    'mt_tcn_weights.weights.h5', 'mt_tcn_best.weights.h5',\n",
    "    'model_config.json', 'angle_stats.npz', 'exercise_labels.json',\n",
    "    'training_metrics.json', 'test_metrics.json', 'training_history.json',\n",
    "]\n",
    "\n",
    "os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "os.makedirs(DRIVE_DIR, exist_ok=True)\n",
    "\n",
    "if os.path.exists(MODEL_DRIVE):\n",
    "    print('Final model found on Drive. Restoring all files and skipping training.')\n",
    "    for fn in ALL_FILES:\n",
    "        src_f = os.path.join(DRIVE_DIR, fn)\n",
    "        if os.path.exists(src_f):\n",
    "            shutil.copy(src_f, os.path.join(MODEL_DIR, fn))\n",
    "            print(f'  Restored {fn}')\n",
    "else:\n",
    "    print('Running supervised fine-tune from SSL init (L4: ~60 min)...')\n",
    "    !python -m backend.training.train_form_model \\\n",
    "        --data_dir  /content/backend/data/form_dataset_fit3d \\\n",
    "        --model_dir /content/backend/models/form_model \\\n",
    "        --source    fit3d \\\n",
    "        --ssl-init  /content/backend/models/form_model/mt_tcn_encoder_ssl.weights.h5 \\\n",
    "        --batch-size 64\n",
    "    # Save all model files to Drive immediately\n",
    "    saved = []\n",
    "    for fn in ALL_FILES:\n",
    "        p = os.path.join(MODEL_DIR, fn)\n",
    "        if os.path.exists(p):\n",
    "            shutil.copy(p, os.path.join(DRIVE_DIR, fn))\n",
    "            saved.append(fn)\n",
    "    if saved:\n",
    "        print(f'Saved to Drive: {saved}')\n",
    "    else:\n",
    "        raise RuntimeError('No model files found after training -- check output above.')\n",
]

# ── Cell[19] — Markdown: Cell 10 (verify + re-save) ──────────────────────────
cells[19]['source'] = [
    "## Cell 10 — Verify + re-save results to Drive\n",
    "\n",
    "Cells 8 and 9 already save to Drive immediately after training.\n",
    "Run this cell at the end to do a final check and re-save anything missing.\n",
    "\n",
    "After this cell prints `OK` for all files, go to `drive.google.com` and\n",
    "download **all files** from `MyDrive/fitnova_results_fit3d/` to your laptop.\n",
    "\n",
    "Place them all in: `backend/models/form_model/` on your laptop.\n",
]

# ── Cell[20] — Verify (Fit3D results dir) ────────────────────────────────────
cells[20]['source'] = [
    "import os, shutil\n",
    "\n",
    "dst = '/content/drive/MyDrive/fitnova_results_fit3d'\n",
    "os.makedirs(dst, exist_ok=True)\n",
    "src_dir = '/content/backend/models/form_model'\n",
    "\n",
    "print('=== Drive results check ===\\n')\n",
    "for fn in ['mt_tcn_weights.weights.h5', 'mt_tcn_best.weights.h5',\n",
    "           'mt_tcn_encoder_ssl.weights.h5',\n",
    "           'training_history.json', 'training_metrics.json',\n",
    "           'test_metrics.json',\n",
    "           'angle_stats.npz', 'exercise_labels.json', 'model_config.json']:\n",
    "    p = os.path.join(src_dir, fn)\n",
    "    if os.path.isfile(p):\n",
    "        shutil.copy(p, dst)\n",
    "        size = os.path.getsize(p) / 1e6\n",
    "        print(f'  OK  {fn}  ({size:.1f} MB)')\n",
    "    else:\n",
    "        print(f'  MISSING  {fn}')\n",
    "\n",
    "print()\n",
    "print('Download ALL files from MyDrive/fitnova_results_fit3d/ to your laptop.')\n",
    "print('Place them all in: backend/models/form_model/')\n",
]

nb['cells'] = cells

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook updated successfully (Fit3D flow).')
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c['source'])[:75].replace('\n', ' ').encode('ascii', errors='replace').decode()
    print(f'  [{i}] {c["cell_type"]}: {src}')
