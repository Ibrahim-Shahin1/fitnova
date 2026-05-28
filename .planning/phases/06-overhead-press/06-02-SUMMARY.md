---
phase: 06-overhead-press
plan: 02
status: complete
completed: 2026-05-29
requirements: [OHP-01]
---

# Phase 6 · Plan 02 (gated probe + supervised baseline) — Summary

Executed interactively on a fresh L4 Colab, cell-by-cell with paste-back. OHP-01-e
(baseline trains on the official splits, val macro-F1 improves) satisfied; baseline test
F1 per error recorded as the SSL-lift control. OHP-01 NOT yet complete (phase-spanning —
the MD-SSL fine-tune is Plans 03/04).

## What was done

- **Gated trajectory probe (Task 1, the highest-leverage gate):** on real unlabeled OHP
  clips. BBox format confirmed (frame = `[region_0, region_1, region_2]`; region_0 =
  barbell `[x1,y1,x2,y2,conf]`; `y_center=(y1+y2)/2`); traj↔frame mapping **1:1**;
  half-cycle sign **argMIN** (`bottom_is_argmax=False`) confirmed visually — argMIN lands
  at the overhead lockout, phase1 = press up; trajectory dir is **NESTED**
  (`bar_trajectories_raw/*.json`); empty-region-0 ~0% in samples; multi-rep 3.5% (global
  argMIN split sufficient — no find_peaks).
- **`ohp_ssl.py` finalized (Task 2):** `glob`→`rglob` for the nested layout; argMIN
  unchanged. SSL smoke: `len(ds)`~5490, triplet shapes `[3,16,112,112]`, descent/ascent
  sanity grid.
- **Supervised baseline (Tasks 3-5):** R(2+1)D-18 Kinetics-V1, joint Elbows/Knees head,
  the Phase-3 recipe (Adam lr 1e-4, batch 16, 50ep/8-patience cosine, `pos_weight`
  ~[2.89,1.92]), via `run_supervised_epoch(dataset_cls=OHPElbowsKneesDataset,
  checkpoint_phase="phase06")`. ~206 s/epoch on L4; 13 epochs trained (early-stop; best
  epoch 4, val macro-F1 0.6448 @0.5). Per-error threshold sweep on val → test eval.

## Headline (baseline control — official 339-clip OHP test split, val-tuned thresholds)

| Error | F1 | AP | threshold | confusion (TN,FP / FN,TP) |
|---|---|---|---|---|
| Elbows | 0.4167 | 0.4084 | 0.704 | 187,66 / 46,40 |
| Knees | 0.8069 | 0.9099 | 0.438 | 200,11 / 34,94 |
| **macro** | **0.6118** | | | |

**No paper Kinetics baseline exists for OHP** (paper Table 4 has no Kinetics row, unlike
Squat's Table 2) — this measured baseline is the ONLY control for the Plan-04 MD-SSL lift.
The baseline is already close to the paper's Ours-MD targets (Elbow 0.4552 / Knees 0.8452,
macro ~0.650), gaps ~0.04 each — so the SSL lift to match the paper is smaller than Squat's
was (0.543→0.626). Elbows is the hard error (modest, like Squat KIE); Knees is strong
(AP 0.91).

## Deviations from PLAN (necessary, not planned)

- **cv2 decode fallback (`transforms.py`):** Colab now ships torchvision 0.26.0+cu128, which
  REMOVED `read_video`/`read_video_timestamps` (project note said 0.27 — actually 0.26). This
  broke the whole video decode path. Added `_HAS_TV_READ_VIDEO` detection + `count_frames` +
  `_decode_clip_cv2` (mirrors `services/clip_decode`, kept in the training layer to avoid a
  training→services import); `decode_clip` routes to cv2 when `read_video` is absent;
  `ohp.py`/`squat.py` probes use `count_frames`. Verified locally with a real-mp4 round-trip;
  29 tests green, no Squat regression.
- **`run_supervised_epoch` seam gap (`supervised_train.py`):** Plan 01 added the `dataset_cls`
  seam to `_build_dataloaders` + `md_finetune` but NOT `run_supervised_epoch` (the baseline
  trainer), and it hardcoded `phase03`. Added `dataset_cls` + `checkpoint_phase`
  (backward-compatible defaults) — without this the OHP baseline couldn't select OHP data or
  write to `phase06`. The Plan-01 seam test didn't cover `run_supervised_epoch`.
- **Trajectory layout NESTED, not flat:** Plan-01 `ohp_ssl` assumed a flat `glob`; the probe
  found `bar_trajectories_raw/` nesting → `rglob`.
- **No `results.pkl` here** (Plan 04 owns it); baseline numbers recorded above for the
  Plan-04 comparison.
- **Colab friction:** torchvision 0.26 surfaced as an `ImportError` on a stale module after
  `git pull`; resolved by a runtime restart (the known `git pull` ≠ module reload trap).

## Artifacts

- `backend/training/aqa/notebooks/06_ohp_supervised_baseline.{py,ipynb}` — Cell A + Steps 0-7.
- `backend/training/aqa/datasets/ohp_ssl.py` (rglob), `transforms.py` (cv2 fallback +
  `count_frames`), `harness/supervised_train.py` (`dataset_cls`/`checkpoint_phase` seam),
  `datasets/ohp.py` + `squat.py` (`count_frames` probe).
- Drive: `phase06/ohp_supervised_v1/best.pt` + `best_thresholds={elbows 0.704, knees 0.438}`.
- `figures/ohp_traj_probe.png`, `figures/ohp_ssl_triplet_sanity.png`.
- Commits: `c246207`/`02b5ba1` (notebook + tv0.26 probe fix), `fe6cb8a` (cv2 decode),
  `e97157d` (rglob + seam + Steps 2-4), `f63f5f9` (Step 5), `a9ad92f` (Steps 6-7).

## Next

**Plan 03 — MD-SSL pretrain** on the 5,490 unlabeled OHP clips (the thesis contribution) →
`backbone.pt`. ~15-26 h on L4 — needs a FRESH L4 notebook (heavy-training rule); prompt the
user before kicking off. The finalized `OHPSSLDataset` (rglob, argMIN, 1:1) is the input.
