---
phase: 06-overhead-press
plan: 03
status: complete
completed: 2026-05-29
requirements: [OHP-01]
---

# Phase 6 · Plan 03 (MD-SSL pretrain) — Summary

MD-SSL pretrain of the R(2+1)D-18 backbone on the unlabeled OHP clips → `backbone.pt`
(the fork point for Plan 04's fine-tune). Executed interactively on Colab (A100 for the
pretrain after the decode-cache made it GPU-bound).

## What was done

- **Coverage filter:** the raw OHP trajectories include detector-failure clips — 98 with
  zero barbell detections + a sparse tail. `OHPSSLDataset` now drops clips with <50%
  detection coverage (401/5490) → SSL trained on **5,089 clips** (still > Squat's 4,970).
- **MD-SSL pretrain (`ohp_md_pretrain_v2`):** v2 strong-aug recipe (unchanged from Phase 4),
  half-cycle contrast (argMIN), AdamW lr 1e-4 wd 1e-4. SSL loss fell monotonically
  0.857 → 0.065; `effective_rank` stayed healthy (6–13, never collapsed — the abort never
  fired); the linear-probe rose 0.55 → **0.6065 (epoch 30)** then plateaued. **`backbone.pt`
  = epoch 30** (the linear-probe-best). No representation collapse.
- **The decode-cache delivered:** epoch 0 ~414 s (builds the cache), epochs 1+ **~180 s**
  (read cache → GPU-bound). Without it, ~40 min/epoch (cv2-decode-bound).

## Deviations from PLAN (necessary)

- **cv2 decode fallback** — Colab torchvision 0.26 removed `read_video`; added a cv2 backend
  in `transforms.py` (version-robust, verified bit-identical contract).
- **`drop_last=True` on the SSL loaders** — 5,089/8 left a size-1 final batch that crashed
  the projector BatchNorm; latent bug from squat_ssl (Squat's count dodged it).
- **Decode cache** (`decode_clip_cached`) — decode-once/reuse so SSL epochs go GPU-bound
  (the user's A100 was otherwise wasted on CPU decode). `cv2.setNumThreads(0)` per worker +
  SSL `num_workers` 4→8.
- **`md_pretrain` OHP seams** — `ssl_dataset_cls` / `probe_dataset_cls` / `checkpoint_phase`
  (it hardcoded Squat + phase04; the linear-probe would have trained on Squat labels).
- **No SSL convergence early-stop** — the trainer ran the full 60-epoch ceiling (only a
  collapse-abort exists). `backbone.pt` = the probe-best (ep30) regardless, so the overrun
  wasted compute but not the deliverable. A lenient SSL early-stop is a follow-up (the noisy
  probe + a late ep30 peak mean an aggressive one would mis-stop).

## Artifacts

- Drive: `phase06/ohp_md_pretrain_v2/backbone.pt` (ep30) + per-epoch checkpoints.
- `datasets/{ohp_ssl,squat_ssl,ohp,squat}.py` (cache_dir + coverage filter + drop_last),
  `transforms.py` (cv2 decode + cache), `harness/md_pretrain.py` (seams + worker threads),
  notebook `07_ohp_md_ssl.{py,ipynb}`.

## Next

Plan 04 — 2-seed fine-tune from `backbone.pt` → ensemble → test F1 (done).
