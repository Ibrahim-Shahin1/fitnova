# OHP Form Model — MD-SSL 2-Seed Ensemble Weights

This directory holds the Phase-6 Motion-Disentangled SSL fine-tuned weights for
Overhead Press **Elbows / Knees** error detection. The `.pt` files are **NOT committed
to git** (each is a full fine-tune checkpoint with optimizer state; staged manually from
Google Drive). This README is the committed documentation of the staging procedure.

---

## Drive Source Paths (Google Drive)

```
MyDrive/FitNova/checkpoints/phase06/ohp_md_finetune_seed42/best.pt
MyDrive/FitNova/checkpoints/phase06/ohp_md_finetune_seed1337/best.pt
```

## Local Target Layout (what OHPFormService expects)

```
backend/models/form_model_ohp_md/
  seed42/best.pt       ← ohp_md_finetune_seed42/best.pt from Drive
  seed1337/best.pt     ← ohp_md_finetune_seed1337/best.pt from Drive
  README.md            ← this file (tracked)
```

## Staging Instructions (one-time per machine)

1. Copy the 2 `best.pt` files from Google Drive to the paths above with PowerShell
   `Copy-Item` (NOT git-bash `cp` — git-bash reads a stale Drive-FUSE copy).
2. Round-trip verify each: `python -c "import torch; d=torch.load(P, map_location='cpu',
   weights_only=False); assert tuple(d['model_state_dict']['fc.1.weight'].shape)==(2,512);
   assert len(d['model_state_dict'])==224"`.

## Production Protocol

- **Architecture:** R(2+1)D-18, `fc = Dropout(0.2) + Linear(512, 2)` (Elbows, Knees), fp32 CPU
- **Ensemble:** 2-seed mean-of-sigmoids (`aggregate_sigmoid_mean` from `eval/ensemble.py`)
- **Thresholds (val-tuned, LOCKED — do not retune):**
  - Elbows: **0.357**
  - Knees: **0.476**
- **Preprocessing:** identical to Squat — 32-frame uniform sample, 112² center crop, Kinetics norm (no TTA)
- **Official test macro-F1:** 0.6622 (Elbows F1 0.447 / Knees F1 0.877)

## Graceful Degradation

If a seed weight is absent, the service logs a warning and skips that seed; `model_ready`
is `True` if at least one seed loaded. When all are missing, classify returns a neutral
response with `model_not_loaded: True` — no exception raised.

## References

- Training + comparison: `docs/eval/FINDINGS_OHP.md`
- Per-clip scores + ground-truth: `backend/data/benchmark/ohp/results.pkl`
- Architecture loader: `backend/training/aqa/harness/md_finetune.py` (`build_finetune_model`)
- Service entry point: `backend/services/ohp_form_service.py`
