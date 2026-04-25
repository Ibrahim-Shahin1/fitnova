# FitNova Colab Runbook — Fit3D Retrain (Phase 2+3)

End-to-end: 4 hours of compute, ~3 GB Drive, $0 if you have Colab Pro. Every cell is crash-safe (reconnect → rerun from Cell 1, every completed step is skipped).

---

## What you're retraining and why

Previous training used `source=mediapipe` (front-view videos extracted frame-by-frame into 3D via MediaPipe). That gave **quality Pearson r = 0.41** — the model couldn't learn real form signals from noisy monocular 3D.

This retrain uses `source=fit3d` (clean MoCap ground-truth joints). View-invariance is added through **sim-to-real domain randomization** (random ±45° Y-axis rotation + anisotropic Gaussian noise) applied only to the training split. The model learns form from clean data, then generalizes to noisy MediaPipe at inference.

Target metrics after retrain (defence gate G3):
- Exercise accuracy on s11 (held-out): ≥ 75%
- Quality Pearson r on s11: ≥ 0.65
- Boundary F1 on s11: ≥ 0.60

---

## Step 0 — Local prep (5 minutes, on your laptop)

On your laptop, in `C:\Users\tsh_x\Desktop\FitNova Application`:

```powershell
python _build_fit3d_annotations_zip.py
python _build_fitnova_src_zip.py
```

Expected output: both zips in the project root.
- `fit3d_annotations.zip` (~144 MB, 384 files across 8 subjects)
- `fitnova_src.zip` (~180 KB, 44 files)

### Upload to Google Drive (drive.google.com, ROOT of MyDrive)

**Required:**
1. `fit3d_annotations.zip` — if `MyDrive/fit3d_annotations.zip` already exists, **delete it first**, then upload the new one.
2. `fitnova_src.zip` — if `MyDrive/fitnova_src.zip` already exists, **delete it first**, then upload the new one.

**Optional** (only if you want to run visualize_pipeline on Colab later — most users do it locally):
3. `videos.zip` — Fit3D front-view mp4s. Can already be there from a prior run.

### If you previously ran the MediaPipe flow and want a clean slate:
Also delete from Drive (these are stale):
- `MyDrive/fitnova_dataset/` (old mediapipe dataset)
- `MyDrive/fitnova_results/` (old mediapipe model)
- `MyDrive/fitnova_mediapipe/` (old per-video extractions)

Keep these (they are separate from the new run):
- `MyDrive/fitnova_dataset_fit3d/` — will be created on first Cell 7 run
- `MyDrive/fitnova_results_fit3d/` — will be created on first Cell 8/9 run

---

## Step 1 — Start Colab

1. Open `FitNova_Colab.ipynb` in Google Colab (drag-drop or open from Drive).
2. **Runtime → Change runtime type → GPU → L4 (or T4)** → Save.
3. Make sure you're on Colab Pro (for priority GPU + longer sessions).

---

## Step 2 — Run cells 1–10 top to bottom

Run them with **Shift+Enter**. Each cell has a doc header explaining the phase.

### Cell 1 — GPU check
**Expected:**
```
TF version    : 2.17.x (or similar)
GPU available : True
  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
If GPU is False: runtime type is wrong. Go back to step 1.

### Cell 2 — Drive mount
A popup asks permission. Click through. Expected: `Mounted at /content/drive`.

### Cell 3 — Videos extraction (SKIP or very fast)
- If videos.zip isn't on Drive: `videos.zip not on Drive — skipping (not required for Fit3D training).`
- If videos.zip IS on Drive: extracts ~2-3 min. You can skip this cell entirely for Fit3D training.

### Cell 4 — Load source code
- First run: uploads from Drive (`Loading fitnova_src.zip from Drive...`) + extracts.
- After reconnect: same — Drive restore. No re-upload needed.
- Expected output ends with a directory listing showing `preprocessing models evaluation` etc.

### Cell 5 — Install deps
Expected:
```
Ready
```
(pip is silent with `-q`.)

### Cell 6 — Extract Fit3D annotations
**First run:**
```
Copying fit3d_annotations.zip from Drive...
Extracting...
Fit3D annotations ready: 384 JSONs across 8 subjects
s03  s04  s05  s07  s08  s09  s10  s11
```
**After reconnect:**
```
Fit3D annotations already extracted (384 JSONs). Skipping.
```
If you see `fit3d_annotations.zip not found at MyDrive/...` → go back to step 0 and upload it.

### Cell 7 — Build training dataset (3-5 min)
**First run:**
```
Building dataset from Fit3D MoCap at /content/fit3d_data/...
  (Sim-to-real domain randomization applied to train split only)
[progress bars via tqdm]
  Splits: {'train': ~N, 'val': ~N, 'test': ~N}
Saving dataset to Drive for crash recovery (~2 min)...
  Saved to MyDrive/fitnova_dataset_fit3d/
```
**After reconnect:**
```
Dataset found on Drive. Restoring to local disk...
  Splits: {'train': N, 'val': N, 'test': N}
Restored from Drive. Skipping build.
```
Expected dataset size: `train` should be the largest (with augmentation, ~5-10× val/test). `val` and `test` are small (s09+s10, s11 alone).

### Cell 8 — SSL pretraining (~90 min on L4)
**First run:**
```
FitNova — SSL Skeleton Autoencoder Pretraining (P3.1)
GPU available: True
  Total: N windows
  Train: M  Val: K
  Ep   tr_ang   tr_jnt  val_ang  val_jnt    best   min
  ------------------------------------------------------
  [tqdm progress bar shows: "SSL  XX%|████░░░| X/80 [time_elapsed<time_remaining, Xs/ep]"]
  1   0.82xxx  0.81xxx  0.79xxx  0.78xxx  0.79xxx   1.2m *
  2   0.68xxx  0.67xxx  0.65xxx  0.64xxx  0.65xxx   2.4m *
  ...
```
The progress bar updates in-place — no scroll spam. Every 20 epochs the cell self-clears and reprints the last 5 lines.

**Expected trajectory:** `val_total` should drop from ~0.8 to ~0.2 by epoch 40-60, then plateau. Early stopping triggers around epoch 50-70.

**After reconnect:**
```
SSL encoder weights found on Drive. Restoring and skipping training.
  Size: X.X MB
```

### Cell 9 — Supervised fine-tune (~60 min on L4)
**First run:** Two phases visible in the output.

**Phase 1** (20 epochs, encoder frozen):
```
SSL Phase 1 — encoder frozen, task heads only (20 epochs, lr=1e-3)...
Epoch 1/20
XX/XX - YYs - loss: X.XX - exercise_accuracy: X.XX - val_loss: X.XX - val_exercise_accuracy: X.XX
...
  Phase 1 done (~15 min).  Best val_loss so far: X.XXX
```

**Phase 2** (remaining 60 epochs, full unfreeze):
```
SSL Phase 2 — all layers unfrozen (60 epochs, lr=1e-4)...
```

Every 10 epochs the cell self-clears and shows the last 10 epoch summaries.

**Expected trajectory:**
- `val_exercise_accuracy` should climb from ~0.1 (chance for 27 classes ≈ 0.04) to ≥ 0.65 by end of Phase 1, ≥ 0.75 by end of Phase 2.
- `val_loss` should drop monotonically.

After training, you should see:
```
Training complete in ~60 minutes.
Evaluating on test set (s11)...
  exercise_accuracy: 0.XX
  quality_mae: 0.XX
  boundary_f1: 0.XX
Saved to Drive: ['mt_tcn_weights.weights.h5', 'mt_tcn_best.weights.h5', 'model_config.json', 'angle_stats.npz', 'exercise_labels.json', 'training_metrics.json', 'test_metrics.json', 'training_history.json']
```

**After reconnect:**
```
Final model found on Drive. Restoring all files and skipping training.
  Restored mt_tcn_weights.weights.h5
  Restored mt_tcn_best.weights.h5
  ...
```

### Cell 10 — Verify + re-save
Expected:
```
=== Drive results check ===
  OK  mt_tcn_weights.weights.h5  (X.X MB)
  OK  mt_tcn_best.weights.h5  (X.X MB)
  OK  mt_tcn_encoder_ssl.weights.h5  (X.X MB)
  OK  training_history.json  (X.X MB)
  OK  training_metrics.json  (0.0 MB)
  OK  test_metrics.json  (0.0 MB)
  OK  angle_stats.npz  (0.0 MB)
  OK  exercise_labels.json  (0.0 MB)
  OK  model_config.json  (0.0 MB)

Download ALL files from MyDrive/fitnova_results_fit3d/ to your laptop.
Place them all in: backend/models/form_model/
```

If any `MISSING` — rerun the corresponding cell (8 or 9).

---

## Step 3 — Download results

1. Go to drive.google.com.
2. Open `MyDrive/fitnova_results_fit3d/`.
3. Select all 9 files (Ctrl+A) → right-click → Download. (Will download as a zip.)
4. Extract into `C:\Users\tsh_x\Desktop\FitNova Application\backend\models\form_model\` **(replace existing files)**.

Before you do this, if you want to keep the old model for comparison:
```powershell
cd "C:\Users\tsh_x\Desktop\FitNova Application\backend\models"
Rename-Item form_model form_model_mediapipe_backup
New-Item -ItemType Directory form_model
```

---

## Step 4 — Local validation

In `C:\Users\tsh_x\Desktop\FitNova Application`:

### Smoke test the loaded weights
```powershell
python -m backend.training.evaluation.weight_probe
```
**Expected:** No kernel mismatch; training-sample probe predicts the correct class with >99% confidence.

### Tensor-level accuracy (confirms weights are healthy)
```powershell
python -m backend.training.evaluation.offline_eval
```
**Expected:** ≥ 80% val accuracy on tensorized val set. If lower → something wrong with weight loading.

### Reality check on s11 (held-out test subject)
```powershell
python -m backend.training.evaluation.reality_check --subject s11
```
**Expected (this is the gate):**
- MoCap path exercise accuracy: **≥ 75%**
- MediaPipe-video path exercise accuracy: **≥ 65%** (gap is the sim-to-real residual)
- Quality Pearson r: **≥ 0.55**

If below these numbers, the sim-to-real augmentation isn't enough; report exact numbers and we'll iterate.

### Visualize on a real s11 video
```powershell
python -m backend.training.evaluation.visualize_pipeline "C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train\s11\videos\60457274\squat.mp4" backend\models\form_model\reality_check\s11_squat.mp4
```
**Expected:**
- Skeleton tracks the person
- Quality bar moves (not flat at 0 or 1)
- Rep count increments at real rep transitions (should get ~5 for 5 squats in the clip)
- Joint-error heatmap no longer saturated at 99% on all joints

---

## Troubleshooting

### Colab runtime disconnected mid-training
Just rerun Cell 1 → Cell 10. Every cell detects previous progress on Drive and skips what's done. You'll lose at most the current epoch.

### "fit3d_annotations.zip not found" in Cell 6
You forgot to upload it. Go back to Step 0.

### Cell 4 says "fitnova_src.zip not on Drive yet" but you uploaded it
Drive sync is slow. Wait 30 seconds, rerun. Or click the file-picker fallback that appears.

### You edited a backend file and re-uploaded fitnova_src.zip but Cell 4 keeps loading the old zip
You forgot to delete the old `MyDrive/fitnova_src.zip` before re-upload. Drive created `fitnova_src (1).zip` — the cell loads the original. Delete both from Drive, rebuild, re-upload as `fitnova_src.zip`.

### Cell 8 OOM (out of memory)
Reduce `--batch-size 128` to `--batch-size 64` in Cell 16 source code.

### Cell 9 says `val_exercise_accuracy` plateaus at ~0.1
Either SSL encoder didn't transfer (check Cell 8 saved properly), or data is corrupted. Rerun Cell 7 (fresh build) + Cell 8 + Cell 9.

### Training metrics look fine but reality_check gives garbage
Weight loading mismatch — run `weight_probe.py` locally; if it reports any positional/by-name kernel diff, the weights are saved with wrong Keras version. Re-save from Colab as `.keras` format instead of `.weights.h5`.

---

## One-line summary of what to expect

Colab: 4 cells run long (6 for SSL + 8 for dataset + 8 for fine-tune, ~150 min total). The other 6 finish in seconds. If you reconnect, everything resumes from where it was. After download + local validation, the committee-visible numbers are `reality_check.py --subject s11`.
