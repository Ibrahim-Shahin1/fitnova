# Squat Form Model — MD-SSL 3-Seed Ensemble Weights

This directory holds the Phase-4 Motion-Disentangled SSL fine-tuned weights for
Squat KIE/KFE error detection.  The `.pt` files are **NOT committed to git** (D-09:
each ~358 MB full checkpoint including optimizer state; ~1.07 GB total for 3 seeds).
This README is the committed documentation of the staging procedure.

---

## Drive Source Paths (Google Drive)

```
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed42/best.pt
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed1337/best.pt
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed7/best.pt
```

## Local Target Layout (what `SquatFormService._load` expects)

```
backend/models/form_model_squat_md/
  seed42/best.pt       ← md_finetune_seed42/best.pt from Drive
  seed1337/best.pt     ← md_finetune_seed1337/best.pt from Drive
  seed7/best.pt        ← md_finetune_seed7/best.pt from Drive
  README.md            ← this file (tracked)
```

## Staging Instructions (one-time manual step per machine)

1. Download the 3 `best.pt` files from Google Drive to the correct local paths above.
2. Verify each file is ~358 MB (full Phase-4 checkpoint with model + optimizer state).
   A small HTML stub (~few KB) means the Drive download link timed out — retry.
3. Run the load check: `python -c "from backend.services.squat_form_service import
   SquatFormService; s = SquatFormService('backend/models/form_model_squat_md');
   print('model_ready:', s.model_ready, '| seeds_loaded:', len(s._models))"`
   Expected output: `model_ready: True | seeds_loaded: 3`

## Expected Checkpoint Size

| File | Expected size |
|------|--------------|
| `seed42/best.pt` | ~358 MB (full fine-tune checkpoint) |
| `seed1337/best.pt` | ~358 MB |
| `seed7/best.pt` | ~358 MB |

Note: the RESEARCH doc (05-RESEARCH.md §1) states ~126 MB — that figure is the
model weights only.  The actual `best.pt` files include optimizer state + RNG state +
`metrics_history`, bringing each file to ~358 MB.  `SquatFormService` reads only
`ckpt["model_state_dict"]` so the extra payload is ignored at serve time.

## Production Protocol (D-01)

- **Architecture:** R(2+1)D-18, `fc = Dropout(0.2) + Linear(512, 2)`, fp32 CPU
- **Ensemble:** 3-seed mean-of-sigmoids (`aggregate_sigmoid_mean` from `eval/ensemble.py`)
- **Thresholds (val-tuned, LOCKED — do not retune here):**
  - KIE (Knee Inward Error): **0.614**
  - KFE (Knee Forward Error): **0.385**
- **No TTA** (test-time augmentation) — deterministic center-crop (`spatial_val`)
- **Test macro-F1:** 0.6304 (KIE F1 ~0.47, KFE F1 ~0.79)

## Graceful Degradation

If one or more seed weights are absent, `SquatFormService` logs a warning and skips
that seed.  `model_ready` is `True` if at least one seed loaded; `False` if all are
missing.  When `model_ready is False`, `classify_clip()` returns a neutral D-05
response with `model_not_loaded: True` — no exception is raised and both API endpoints
continue to serve (with zeroed confidence scores).

## References

- MD-SSL fine-tune trainer: `backend/training/aqa/harness/md_finetune.py`
- Service entry point: `backend/services/squat_form_service.py`
- Ensemble aggregation: `backend/training/aqa/eval/ensemble.py`
