# Phase 1 — Fitness-AQA Dataset Report

**Phase:** 01 — Dataset Consolidation & EDA
**Completed:** 2026-05-20
**Source notebook:** `backend/training/aqa/notebooks/01_dataset_eda.py`

## Executive summary

The Fitness-AQA dataset is verified end-to-end against the published Parmar et al. (ECCV 2022) figures. All 7 errors reconcile to two decimal places; every official train/val/test split is internally disjoint and fully label-covered; Colab reads the dataset from the user's Drive shortcut without any preprocessing of the Drive layout. **The Squat-first vertical slice is on perfectly clean, paper-matching ground. Nothing in this phase derails Phase 2.**

## Dataset structure

Used in place from `My Drive/Fitness-AQA_dataset_release` (Drive shortcut to a "Shared with me" folder — works in Colab):

```
Fitness-AQA_dataset_release/
├─ BarbellRow/Labeled_Dataset/
│   ├─ barbellrow_images_raw.zip       (33,759 jpg)
│   ├─ Labels/  labels_{lumbar,torso_angle}_error.json
│   └─ Splits/  Splits_{Lumbar,TorsoAngle}_Error/{train,val,test}_ids.json
├─ OHP/
│   ├─ Labeled_Dataset/   videos.zip (2,367 mp4) + Labels + Splits
│   └─ Unlabeled_Dataset/ videos.zip (5,490) + bar_trajectories_raw.zip (5,490)
└─ Squat/
    ├─ Labeled_Dataset/
    │   ├─ videos.zip                  (1,739 mp4)
    │   ├─ Labels/  error_knees_{inward,forward}.json
    │   ├─ Splits/  train/val/test_keys + traj_nan.json (19)
    │   └─ Shallow_Squat_Error_Dataset/  images.zip (3,738) + labels + splits
    └─ Unlabeled_Dataset/   videos.zip (4,970) + bar_trajectories_raw.zip (4,970)
```

All 8 nested archives are readable from Colab via Drive FUSE. Archive open + member read ≈ 2.5s — fine for one-off audits, **untenable for per-epoch training reads**. Training data must be copied from Drive to Colab local disk once per session.

## Class balance (official benchmark set)

| Error | Modality | Total | Erroneous | % erroneous |
|---|---|---:|---:|---:|
| Squat KIE (Knees-Inward) | video | 1,623 | 232 | **14.29%** |
| Squat KFE (Knees-Forward) | video | 1,623 | 1,109 | **68.33%** |
| Squat Shallow-Depth | image | 3,611 | 1,584 | **43.87%** |
| OHP Elbows | video | 2,260 | 576 | **25.49%** |
| OHP Knees | video | 2,260 | 777 | **34.38%** |
| BarbellRow Lumbar | image | 14,778 | 2,317 | **15.68%** |
| BarbellRow Torso-Angle | image | 17,030 | 1,568 | **9.21%** |

Every % matches the Parmar et al. published figure to two decimals.

## Official splits

Across all 7 errors: **disjoint** (no ID in two splits) and **fully label-covered** (every split ID has a label).

| Error | train | val | test |
|---|---:|---:|---:|
| Squat KIE/KFE | 1,136 | 243 | 244 |
| Squat Shallow | 2,542 | 529 | 540 |
| OHP | 1,582 | 339 | 339 |
| BarbellRow Lumbar | 10,360 | 2,227 | 2,191 |
| BarbellRow Torso | 11,881 | 2,564 | 2,585 |

## Error co-occurrence

**Squat KIE × KFE** (n=1,623):

|       | KFE+ | KFE- |
|-------|-----:|-----:|
| KIE+  | 192  | 40   |
| KIE-  | 917  | 474  |

KIE almost never occurs alone — 192 of 232 KIE+ clips (83%) also have KFE. The two knee faults are biomechanically linked. **Implication: a joint multi-label head over a shared R(2+1)D-18 backbone is justified for Squat.**

**OHP Elbows × Knees** (n=2,260):

|       | Kne+ | Kne- |
|-------|-----:|-----:|
| Elb+  | 117  | 459  |
| Elb-  | 660  | 1024 |

Much more independent — only 117 co-occur, each mostly alone. Multi-label still works.

## Clip properties (video errors)

- **fps:** uniformly 30 across all samples (Squat + OHP).
- **Frame count:** Squat 49–404 (median 111, mean 120.6); OHP 49–271 (median 107, mean 123.8). The paper's "single repetition" claim holds for the median; a long tail (clips up to 13.5s) suggests multi-rep or slow-tempo cases.
- **Resolution:** width pinned at 480; height varies (270, 324, 480, 584, 592, 600) — in-the-wild portrait/square/landscape footage, width-normalized.
- **Pipeline implication:** uniformly sample 32 frames across each clip's full span (absorbs variable length); resize/crop must handle non-square aspect ratios. The paper's 320² resize → 224² center crop is the starting point; exact handling deferred to Phase 2 against the official `Code_Release`.

## Barbell trajectories (unlabeled set, for MD SSL)

**Squat:** processed flat list of y-centers per clip. Clean parabolic curves (see notebook Cell 5). Length 26–441 frames, **0% NaN**. MD half-cycle splitting can use them directly in Phase 4.

**OHP:** *raw* YOLO output — per-frame list of 3 detections, each `[[x1,y1,x2,y2,conf]]`. The barbell track = the one with the largest y-variance over the clip (= the moving thing). Track index 0 is the bar in **192/200 (96%)** of sampled clips; variance-pick handles the remaining 4%. **34% of OHP trajectories contain at least one NaN frame** — likely occlusion at the press peak. Phase 6 OHP needs: track-0 default + NaN interpolation + light smoothing. Phase 4 (Squat SSL) is unaffected.

## Visual sanity (Squat sample-frame grids)

User-verified via mid-rep and error-interval-midpoint frames (notebook Cells 6 and 7):

- **Shallow+/-** — clearly distinguishable; depth difference visible from any angle. Labels visually trustworthy.
- **KIE+** — confirmed by domain (powerlifting) read. Knees-inward is **subtle from side views** (most clips), visible in rear/quarter views. KIE often shows during the *ascent* rather than at rep bottom. **KIE is angle-and-temporal-dependent** — exactly why the paper's KIE F1 caps at ~0.53. The model must learn temporal/pose cues, not direct knee deviation in any one frame.
- **KFE+/-** — KFE+ shows knees obviously over the toes; KFE- is "better form, less obvious" — the +/- is a **severity gradient, not a binary presence**. Consistent with the 68% positive rate (most lifters get flagged).
- **Confound observation:** KFE+ clips visually correlate with worse back-arching / lumbar flexion. The dataset does *not* label a separate Squat "lumbar" error (that exists only for BarbellRow), so back-rounding is unlabeled in Squat. The KFE model may implicitly learn back-rounding cues — worth flagging during the modeling phase.

## Anomalies & findings (record for the build)

1. **Squat labeled `videos.zip` ships 1,739 clips but only 1,623 carry KIE/KFE labels.** 116 extra unlabeled videos cluster in the `339xx_x` ID range, excluded by the official splits. KIE and KFE label the same 1,623 clips.
2. **BarbellRow label files are supersets of the official splits.** Lumbar 14,978 ⊃ 14,778; Torso 17,440 ⊃ 17,030. The official split files define the authoritative benchmark.
3. **Squat `traj_nan.json` lists 19 labeled clips with NaN trajectories** — exclude from MD SSL pretraining.
4. **Dibenedetto et al. (UMAP '25) Table 1 swaps OHP Elbow/Knee percentages.** The dataset filenames are correct (cross-verified against Parmar Table 4 F1 ordering — `error_knees.json` has 34.38% positives and the higher-F1 paper number ~0.72–0.85; `error_elbows.json` has 25.49% and the lower ~0.45 F1). Use dataset filenames as ground truth. Phase 6 concern.
5. **Drive shortcut to a "Shared with me" folder DOES work in Colab** (8/8 archives readable). No restructuring needed.
6. **Drive ReadMe files are `.gdoc`** (Google Docs format, 0-byte pointers). Their content was extracted from the local `.docx` copies during planning; code must not attempt to read `.gdoc` as files.
7. **OHP raw trajectory structure** is stable: 3 detections per frame, every frame, every clip sampled. The 3 tracks correspond to ~persistent objects (barbell, lifter, floor plates).

## Implications baked into the build

- **Squat MD model:** joint multi-label head (KIE & KFE), R(2+1)D-18 (Kinetics init), 32-frame uniform sample, 224² crop after aspect-aware resize. Class-weighted BCE for KIE imbalance.
- **Data pipeline:** for each Colab session, copy the needed `.zip` from Drive → `/content/` local, extract, train from local. Drive layout stays exactly as-is.
- **Splits:** use the official `{train,val,test}_keys`/`{train,val,test}_ids` JSONs verbatim. F1 on the official *test* split is the comparison metric — identical to the paper.
- **Trajectory preprocessing:** Squat — ready as-is. OHP (Phase 6) — track-0 default with variance fallback, plus NaN interpolation and light smoothing.

## Artifacts

- Source notebook: `backend/training/aqa/notebooks/01_dataset_eda.py` (jupytext percent format; opens in Colab via jupytext or by pasting cells).
- This report.
- Phase plan: `.planning/phases/01-dataset-consolidation-eda/01-01-PLAN.md`.
- Phase summary: `.planning/phases/01-dataset-consolidation-eda/01-01-SUMMARY.md`.
- Master implementation plan: `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`.

---
*Phase 1 closed 2026-05-20.*
