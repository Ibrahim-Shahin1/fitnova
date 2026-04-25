# Fit3D Dataset & AIFit Paper — What We Used, What We Ignored

> **Sibling document to `HANDOFF.md`.** This file is the academic-honesty ledger: for every decision in the form-analysis pipeline, what was inherited from the Fit3D MoCap dataset and the AIFit paper, and what was deliberately rejected or replaced. The goal is twofold: (a) so the next chat session can attribute credit / blame correctly, and (b) so the defence committee can see exactly where we follow precedent and where we depart.

---

## 0. The two sources at a glance

| Source | Type | What it gives us |
|---|---|---|
| **Fit3D** (Fieraru et al., 2021) | MoCap dataset — 8 subjects, 27 exercises, 4 camera views, 25-joint skeletons, per-frame rep annotations | Ground-truth 3D joint trajectories for *real* exercise reps performed by *real* humans. Foundation of the synthetic training corpus. |
| **AIFit** (same authors, accompanying paper) | Method paper — proposes a multi-task 3D-pose-based model for fitness coaching | Architectural inspiration (multi-task TCN with multiple heads), the *idea* of comparing a trainee's joint trajectory to an instructor's, the joint-group abstraction for error attribution. |

We **used Fit3D** as the data backbone and **borrowed AIFit's architectural spine** (multi-task TCN, joint-group output head). We **rejected AIFit's quality-labelling strategy** (instructor-trajectory alignment) and replaced it with a synthetic-corruption scheme — for reasons documented in §3.

---

## 1. Fit3D — what we used

### 1.1 Subjects
We use **8 subjects: s03, s04, s05, s07, s08, s09, s10, s11** — the full set of public Fit3D subjects with rep annotations.

- **Train:** s03–s10 (7 subjects)
- **Held-out test:** s11 (1 subject — the one the v4 metrics in HANDOFF §3 are reported on)

Code: `backend/training/preprocessing/fit3d_loader.py::ALL_SUBJECTS`.

### 1.2 Exercises
We use **27 of Fit3D's 28 exercise classes** — every exercise that has rep annotations in `rep_ann.json`.

The full list is in `backend/training/preprocessing/fit3d_loader.py::SUPPORTED_EXERCISES`. Examples: `squat`, `deadlift`, `pushup`, `barbell_row`, `dumbbell_overhead_shoulder_press`, `burpees`, `man_maker`.

The **`walk_the_box`** class is excluded (see §2.1).

### 1.3 The joints stream
We consume **`joints3d_25.json`** — the 25-joint 3D pose trajectories produced by Fit3D's marker-based MoCap rig.

For each subject × exercise we load `subject/joints3d_25/<exercise>.json`, parse `data["joints3d_25"]` as `(N_frames, 25, 3) float32`, and segment it into per-rep tensors using `rep_ann.json` boundaries.

Code: `fit3d_loader.py::load_joints` and `fit3d_loader.py::segment_reps`.

### 1.4 Rep boundaries
We use Fit3D's hand-curated **`rep_ann.json`** — boundary frame indices `[b0, b1, ..., bn]`, where rep *i* spans frames `[b_i, b_{i+1})`.

These drive (a) per-rep tensor segmentation during preprocessing, (b) the rep-boundary supervision target for the boundary head, (c) the rep-count target.

Code: `fit3d_loader.py::load_rep_boundaries` and `dataset_builder.py` rep-segmentation logic.

### 1.5 Joint mapping (25 → 15 canonical)
Fit3D's 25-joint skeleton is collapsed to a **15-joint canonical representation** that aligns with what MediaPipe Pose can produce at inference time. This is the bridge that lets us pretrain on MoCap and infer on phone video.

The mapping (with Fit3D source indices in parentheses):

| Canonical idx | Joint | From Fit3D |
|---:|---|---|
| 0, 1 | l/r shoulder | 16, 17 |
| 2, 3 | l/r elbow | 18, 19 |
| 4, 5 | l/r wrist | 20, 21 |
| 6, 7 | l/r hip | 1, 2 |
| 8, 9 | l/r knee | 4, 5 |
| 10, 11 | l/r ankle | 7, 8 |
| 12 | pelvis | 0 |
| 13 | neck | 12 |
| 14 | spine_mid | 6 |

Code: `joint_mapping.py::CANONICAL_JOINTS`, `FIT3D_TO_CANONICAL`, `MEDIAPIPE_TO_CANONICAL`.

### 1.6 Camera angle
We use a **single camera view — `60457274`** — for the (small number of) places we touch the RGB videos (visualization only; never as model input).

Fit3D ships four cameras. Using only one keeps the pose-quality consistent and avoids spending preprocessing budget on perspectives we never serve at inference.

Code: `extract_fit3d_videos.py`.

### 1.7 Normalization
Each rep's joints stream is **pelvis-centred and torso-scaled** so that two trainees of different heights / camera distances produce numerically comparable inputs:

1. Subtract pelvis position from every joint, every frame → translation invariance.
2. Compute torso length (mean shoulder-midpoint to pelvis distance over all frames in the rep) → divide every coordinate by it → scale invariance.

We **do not** apply azimuthal alignment — we deliberately let Y-rotation vary so that the augmentation pipeline (§1.10) can teach the model to be view-invariant in the joint channel. (This required Phase A1's radial-projection fix to angular features 16–17 — see HANDOFF §2.)

Code: `normalize.py::normalize_skeleton`.

### 1.8 Resampling to 64 frames
Reps in Fit3D are variable-length; the MT-TCN expects a fixed 64-frame window. Each rep is **time-resampled** to 64 frames using simple linear interpolation in joint-coordinate space.

64 frames at the model's nominal 10 Hz is ~6.4 s — long enough for a slow squat, short enough that two reps don't fit (so each window is one rep, simplifying the boundary head).

### 1.9 22 angular features
On top of the 15 canonical joints (45 raw coordinates per frame) we compute **22 hand-engineered angular features** per frame: joint angles (knee, elbow, hip flexion), inter-segment angles (trunk lean, torso tilt), and radially-projected spine angles.

These feed the model's `angles_input` stream — the channel that drives the **quality head** specifically, because angular features are translation- *and* scale-invariant by construction (whereas raw joints are only after §1.7).

Two of the 22 features (idx 16 = `spine_vert_sag`, idx 17 = `spine_vert_front`) needed Phase A1's radial-projection rewrite to also be Y-rotation invariant. After that fix, all 22 features are fully invariant under the augmentation pipeline.

Code: `angular_features.py`.

### 1.10 Augmentation pipeline (Y-rotation + anisotropic noise)
Each clean rep is fed through `_augment_joints3d`:

1. **Y-rotation** by a random θ ∈ [-180°, 180°] — simulates camera azimuth variation a real user would create.
2. **Anisotropic Gaussian noise** with σ_xy = 2 cm, σ_z = 5 cm — simulates MediaPipe's depth uncertainty.

This is applied to the *joints* stream only. The angular-features stream is recomputed *after* augmentation, so it inherits the rotation/noise but stays physically meaningful.

Code: `augmentation.py::_augment_joints3d`, `dataset_builder.py` augmentation branch.

### 1.11 SSL pretraining target
We use the unaugmented Fit3D joints stream as the target for **masked-autoencoder self-supervised pretraining** (Cell 8 of the Colab notebook). Random 15% of frames are masked; the encoder learns to reconstruct the missing frames from temporal context.

This gives the encoder a "skeleton-completion" prior before any supervised training touches it.

Code: `train_ssl_pretrain.py`.

---

## 2. Fit3D — what we ignored (and why)

### 2.1 `walk_the_box` exercise
Fit3D's 28th class. Excluded because it has no `rep_ann.json` entries — there are no discrete reps, just continuous walking. Including it would require a separate rep-detection strategy and pollute the rep-count head with a nonsense target.

### 2.2 The other three camera angles
Fit3D ships RGB from `50591643`, `58860488`, `60457274`, `65906101`. We use only `60457274` (§1.6). Reason: we never use the RGB pixels as model input — MediaPipe is what consumes pixels at inference time, and at training time we only need the MoCap joints. The other three views would add 4× preprocessing time for zero model benefit.

### 2.3 The RGB video stream itself (as model input)
We **do not feed RGB pixels into the MT-TCN**. The model is purely pose-based: 15 canonical joints + 22 angular features. The RGB videos are used only for `visualize_pipeline.py` overlays.

This is a deliberate design decision, not a Fit3D limitation: a pose-only model is faster (~60 ms/frame on CPU vs. several hundred for any image-based model), platform-independent (works wherever MediaPipe runs), and dramatically smaller (~25 MB weights vs. hundreds of MB for any CNN-backed alternative).

### 2.4 The `smplx_neutral_J_regressor` and SMPL-X-fitted body params
Fit3D ships SMPL-X body model fits (`smplx_neutral_J_regressor.npy`, `params/`). We ignore them entirely — they would let us recover joint angles in a parametric body model, but the 22 hand-engineered angular features in `angular_features.py` are sufficient for our quality head and far cheaper to compute at inference time.

### 2.5 The instructor-trajectory comparison (the v1 strategy — discarded)
Fit3D includes subject **s03 as the instructor**: the idea being that other subjects' performance can be evaluated by aligning their trajectories to s03's via DTW and measuring deviation. This was our **v1** quality-labelling strategy.

It collapsed because:
- DTW alignment between two real human trajectories produces a *continuous* deviation distribution, but the AIFit-style threshold for "good vs bad" is a single cut-point — and on our small subject pool, **89% of training reps ended up labelled `quality = 0`** (every non-instructor rep deviated enough to fail). The model could trivially achieve high accuracy by predicting 0 always; the head learned nothing useful.
- s03 is one human — they have idiosyncrasies. Aligning to one person's specific squat tempo and depth is not the same as aligning to good form.
- We have no labels of *how* the trainee was wrong; the deviation is a scalar with no joint-level attribution.

Replacement: the synthetic-corruption strategy in §3.2.

---

## 3. AIFit — what we used

### 3.1 Multi-task TCN backbone
AIFit's core architecture is a **temporal convolutional network** that processes a pose sequence and produces multiple task-specific outputs from a shared encoder. We adopt this directly:

```
joints (T, 45)  ─┐
                  ├─→ shared TCN encoder ─→ ┬─→ exercise-class head (27-way softmax)
angles (T, 22)  ─┘                          ├─→ quality head (scalar regression)
                                            ├─→ rep-boundary head (per-frame sigmoid)
                                            ├─→ rep-count head (scalar regression)
                                            └─→ joint-error head (10-channel per-frame sigmoid)
```

Code: `backend/training/models/mt_tcn.py`.

### 3.2 The five-head structure
The five output heads are AIFit-aligned:

| Head | AIFit equivalent | Purpose |
|---|---|---|
| Exercise classifier | "exercise classification" | Identify which of the 27 exercises is being performed. |
| Quality scalar | "form-quality assessment" | Single number summarising overall form correctness. |
| Rep boundary | "rep segmentation" | Per-frame probability that this frame is a rep transition. |
| Rep count | "rep counting" | Total reps in the window (we use 0–4). |
| Joint-error head | "form-error attribution" | Which of the 10 joint groups is the rep going wrong on. |

### 3.3 The 10-joint-group abstraction
AIFit's contribution that we depend on most: errors are attributed to **joint groups, not individual joints**. We use the 10-group split:

```
0,1: l/r elbow      6,7: l/r hip
2,3: l/r shoulder   8:   trunk / spine
4,5: l/r knee       9:   neck
```

This is what makes the feedback interpretable to a human — saying "your left knee error is 0.7" is meaningful; saying "joint 8's third coordinate deviated 1.4 cm" is not.

Code: `JOINT_GROUP_NAMES` in `backend/services/form_session.py` and `JOINT_GROUP_SPECS` in `corruption.py`.

### 3.4 MoCap-as-pretraining
The general idea: use MoCap to teach the model what skeletons-doing-exercises *look like*, then deploy on a noisier real-world pose source (MediaPipe). AIFit takes this for granted because their inference is on the same MoCap distribution; we adopt it as a training prior and accept that there's a downstream domain gap to bridge — which is exactly the open problem in HANDOFF §5–§7.

---

## 4. AIFit — what we ignored (and why)

### 4.1 Instructor-trajectory quality labelling
**The single biggest deliberate departure.** AIFit assumes you have access to a "perfect" reference trajectory for each exercise (their instructor) and labels quality as `1 - normalised_DTW_distance(trainee, instructor)`.

Reasons we rejected it:
1. On Fit3D's small subject pool with one instructor (s03), the strategy collapses to "is this s03 or someone else?" — see §2.5.
2. The label is a single scalar with no joint-level meaning; it cannot supervise the joint-error head.
3. It's brittle to natural inter-subject variation in tempo, depth, and stance — penalising form differences that are fine in practice.

**Replacement: synthetic corruption.** For each clean rep we generate K corrupted variants by:
1. Picking a joint group (1 of 10).
2. Sampling a severity tier (MILD / MODERATE / SEVERE).
3. Applying a *known geometric perturbation* to that joint group across the whole rep — e.g., for "knee" we rotate the shin around the thigh axis to simulate valgus collapse.
4. Setting `quality = 1 - severity` and `joint_errors[:, group] = 1`.

This gives:
- **Balanced labels by construction.** No 89%-zeros collapse.
- **Joint-level ground truth.** The joint-error head has a real supervision signal.
- **Interpretable corruption types.** Each corruption maps to a real-world fault (knee valgus, butt-wink, forward lean, flared elbows).

Code: `corruption.py::JOINT_GROUP_SPECS` and the `corrupt_rep` family of functions.

> **Caveat for the defence committee.** The cost of this departure is exactly the failure mode HANDOFF §5 documents: synthetic corruption is *too clean* — real bad form on phone video doesn't look like a constant geometric offset; it looks like noisy, inconsistent, MediaPipe-occluded mush. The model learns to recognise the synthetic signature, not the real one. Roads A and B in HANDOFF §7 are about closing this gap.

### 4.2 AIFit's specific quality-scalar definition
AIFit defines quality as a continuous DTW-derived value. We replace it with a **discrete severity tier mapped to a scalar**:

| Severity | Quality label |
|---|---|
| (clean) | 1.0 |
| MILD | 0.7 |
| MODERATE | 0.4 |
| SEVERE | 0.1 |

This is a coarser supervision signal but it's also more stable — the severity tiers are designed to be visibly distinguishable to a coach.

### 4.3 AIFit's specific TCN hyperparameters
We don't claim hyperparameter parity. Our MT-TCN dimensions (kernel sizes, dilation schedule, number of residual blocks) were tuned for our 64-frame window and our two-stream input (joints + angles), and don't necessarily match AIFit's published numbers. Architectural *shape* is from AIFit; specific *sizes* are ours.

### 4.4 AIFit's evaluation protocol (their leave-one-subject-out)
We adopt **subject-disjoint train/test** in spirit (s11 is held out), but we don't run the full leave-one-subject-out cross-validation that AIFit reports in their tables. With 8 subjects and a Colab budget that has to also cover SSL pretraining + 2-phase fine-tuning, an LOSO sweep would be 8× the compute. The single held-out subject is the practical compromise; metrics reported in HANDOFF §3 are on s11.

### 4.5 AIFit's specific augmentation set
We use Y-rotation + anisotropic noise (§1.10). AIFit-the-paper describes their augmentation differently. The space of "augmentations on MoCap to bridge to a noisier inference distribution" is well-studied and we picked the two that we could verify were rotation- and translation-invariant after the §1.7 normalization — not necessarily the same two AIFit picked.

---

## 5. Summary table — credit ledger

| Component | From Fit3D | From AIFit | Ours |
|---|:---:|:---:|:---:|
| Joint coordinates (training) | ✓ | | |
| Rep annotations | ✓ | | |
| 25→15 joint mapping | ✓ (the source) | | ✓ (the design) |
| Pelvis-centring + torso-scaling | | | ✓ |
| 64-frame resampling | | | ✓ |
| 22 angular features | | | ✓ |
| Y-rotation + noise augmentation | | | ✓ |
| Multi-task TCN architecture | | ✓ | |
| 5-head output structure | | ✓ | |
| 10 joint-group abstraction | | ✓ | |
| MoCap-as-pretraining | | ✓ | |
| **Quality labelling (synthetic corruption)** | | | ✓ (replaces AIFit's instructor-DTW) |
| 3-tier severity → scalar | | | ✓ |
| SSL masked-autoencoder pretraining | | | ✓ |
| Single held-out subject (s11) eval | | | ✓ (not AIFit's LOSO) |
| MediaPipe at inference time | | | ✓ |

---

## 6. The honest assessment for defence

**Things we can defend.**
- The MT-TCN architecture and 5-head structure are AIFit-aligned and academically standard.
- The Fit3D usage is faithful: 8 subjects, 27 exercises, rep annotations, single-camera, public-data only.
- Subject-disjoint test split (s11 held out) avoids the most common form of leakage.
- The replacement of instructor-DTW with synthetic corruption is well-motivated by the v1 collapse and gives us joint-level supervision the original method couldn't.

**Things we cannot defend, and what the next chat is fixing.**
- The synthetic-pretrain-only training distribution does not transfer to phone video. Quality saturates, knee/hip channels flatline, classifier collapses. HANDOFF §5 has the failure-mode breakdown; HANDOFF §7 has the two roads forward.
- We did not run AIFit's full LOSO protocol (compute budget). Single held-out subject is a weaker generalisation claim.
- Our quality scalar is a 4-level discretization, not a continuous deviation measure. Coarser feedback than AIFit promises.

---

*Last updated: 2026-04-25. Sibling: `HANDOFF.md`.*
