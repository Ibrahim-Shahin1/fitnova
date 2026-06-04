# Shallow-Squat Form Model — CVCSPC 3-Seed Ensemble Weights

This directory holds the Phase-7 CVCSPC (pose-contrastive SSL) fine-tuned weights for
Shallow-Squat **depth** error detection — an **image** classifier (single crop), not a
video model. The `.pt` files are **NOT committed to git** (staged manually from Google
Drive). This README is the committed documentation of the staging procedure.

---

## Drive Source Paths (Google Drive)

```
MyDrive/FitNova/checkpoints/phase07/shallow_squat_cvcspc_finetune_seed42/best.pt
MyDrive/FitNova/checkpoints/phase07/shallow_squat_cvcspc_finetune_seed1337/best.pt
MyDrive/FitNova/checkpoints/phase07/shallow_squat_cvcspc_finetune_seed7/best.pt
```

## Local Target Layout (what ShallowSquatFormService expects)

```
backend/models/form_model_shallow_cvcspc/
  seed42/best.pt       ← shallow_squat_cvcspc_finetune_seed42/best.pt from Drive
  seed1337/best.pt     ← shallow_squat_cvcspc_finetune_seed1337/best.pt from Drive
  seed7/best.pt        ← shallow_squat_cvcspc_finetune_seed7/best.pt from Drive
  README.md            ← this file (tracked)
```

## Staging Instructions (one-time per machine)

1. Copy the 3 `best.pt` files from Google Drive to the paths above with PowerShell
   `Copy-Item` (NOT git-bash `cp` — stale Drive-FUSE copy).
2. Round-trip verify each: `python -c "import torch; d=torch.load(P, map_location='cpu',
   weights_only=False); assert tuple(d['model_state_dict']['fc.weight'].shape)==(1,512);
   assert len(d['model_state_dict'])==122"`.

## Production Protocol

- **Architecture:** ResNet-18 (ImageNet-init backbone, then CVCSPC SSL + fine-tune),
  `fc = Linear(512, 1)` single-logit head, fp32 CPU
- **Input:** a SINGLE image crop — `Resize(256) → CenterCrop(224) → ToTensor → ImageNet
  norm` ([0.485,0.456,0.406] / [0.229,0.224,0.225]). No video decode, no temporal sampling,
  no rep segmentation, no kneeaware crop.
- **Ensemble:** 3-seed INLINE scalar mean-of-sigmoids — NOT `aggregate_sigmoid_mean`
  (that expects `(N, 2)` logits; this head is a single logit per image).
- **Threshold (val-tuned, LOCKED):** **0.395** (depth-error positive)
- **Official test F1:** 0.8902 (PR-AUC 0.967), beats paper CVCSPC 0.8694

## Graceful Degradation

If a seed weight is absent, the service logs a warning and skips that seed; `model_ready`
is `True` if at least one seed loaded. When all are missing, classify returns a neutral
response — no exception raised.

## References

- Training + comparison: `docs/eval/FINDINGS_SHALLOW_SQUAT.md`
- Per-crop scores + ground-truth: `backend/data/benchmark/shallow/results.pkl`
- Image pipeline + builder: `backend/training/aqa/datasets/shallow_squat.py`, `backend/training/aqa/harness/image_supervised_train.py`
- Service entry point: `backend/services/shallow_squat_form_service.py`
