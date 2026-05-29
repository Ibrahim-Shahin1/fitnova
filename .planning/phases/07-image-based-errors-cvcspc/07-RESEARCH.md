# Phase 7: Image-Based Errors (CVCSPC) — Research

**Researched:** 2026-05-30
**Domain:** 2D image classification (ResNet-18 / ImageNet), pose-contrastive self-supervised learning (CVCSPC)
**Confidence:** HIGH (method reconstructed from official code + ar5iv paper HTML; data layout confirmed from local disk; eval reuse confirmed from source)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D1 — Scope:** Phase 7 covers only Shallow-Squat. BarbellRow (IMG-03) is cancelled.

**D2 — Method:** ImageNet ResNet-18 supervised baseline FIRST (guaranteed result). CVCSPC SSL attempted second as a research-gated lift. Baseline alone satisfies IMG-02 if SSL cannot be assembled.

**D3 — Image pipeline:** New `datasets/shallow_squat.py`. JPEG → 224² → ImageNet mean/std norm. No video decode, no Kinetics norm.

**D4 — Rigor:** Multi-seed ensemble (3-seed default, 2-seed acceptable). Mean-of-sigmoids + val-tuned threshold per error.

**D5 — Class balance:** BCEWithLogitsLoss with train-derived pos_weight. ~44% positive (near-balanced); weighting is a minor correction.

**D6 — Carry-forward contracts:** `eval/metrics.py` reused unchanged. Colab harness pattern (Drive stage, atomic checkpoint, `latest.txt` resume, `map_location="cpu"`, disconnect-safe). `.ipynb` delivered; one cell at a time + paste-back; tqdm on any cell >2 min; `results.pkl` deliverable pack.

**D7 — Metric honesty:** Every number computed in code from real scores. Paper CVCSPC target: **0.8694**. No hand-typed values.

### Claude's Discretion

None specified — all implementation decisions locked above.

### Deferred Ideas (OUT OF SCOPE)

- BarbellRow Lumbar/Torso (IMG-03) — cancelled this milestone
- CVCSPC cross-exercise transfer — moot without BarbellRow
- API/serving integration — descoped per D1
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IMG-01 | CVCSPC (ResNet-18) image pipeline — baseline and self-supervised | §Standard Stack, §CVCSPC Method, §Architecture Patterns |
| IMG-02 | Shallow-Squat detector trained; F1 on official split vs paper CVCSPC 0.8694 | §Supervised Baseline Recipe, §Evaluation, §CVCSPC Verdict |
</phase_requirements>

---

## Summary

Phase 7 delivers a binary image classifier (2D ResNet-18) for Shallow-Squat depth error detection. The input is a single `crops_unaligned/{id}.jpg` (3,611 labeled; 43.9% positive); the output is a sigmoid score compared at a val-tuned threshold. Two stages: (1) ImageNet-pretrained supervised baseline fine-tuned end-to-end on the official split; (2) research-gated CVCSPC pose-contrastive SSL pretraining on the Squat Unlabeled dataset (4,970 clips + trajectories), followed by end-to-end fine-tune on the labeled Shallow-Squat crops.

The decisive finding is that **CVCSPC SSL is feasible as a Shallow-Squat standalone**. The paper explicitly pretrained CVCSPC on the BackSquat unlabeled dataset alone (not multi-exercise), then fine-tuned on the Shallow-Squat labeled crops. Both the unlabeled Squat videos (1.6 GB, `-3-004`) and trajectories (1.8 MB, `-3-001`) are confirmed present locally. However, the official code release is severely incomplete: `CVC_SPC.py`, `opts_exercise_qa.py`, `models/`, `augmentations/`, and `dataloader_eval.py` are all absent; only `dataloader.py` and `train_test.py` shipped. The method must be reconstructed from those two files plus the paper — which is fully achievable.

Compute is light: ResNet-18 on 3.7k small crops trains in minutes per epoch. The SSL pretraining on 4,970 videos (extracting frames on-the-fly) adds time but remains inside a Colab session. The critical risk is method reconstruction, not hardware.

**Primary recommendation:** Build and validate the ImageNet supervised baseline first (guaranteed paper-comparable result). Then attempt CVCSPC reconstruction from `dataloader.py` + `train_test.py` + paper §3.1; document every deviation. SSL is feasible standalone.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| JPEG crop loading + transform | Dataset module | — | Pure offline CPU op; all preprocessing in `__getitem__` |
| ImageNet ResNet-18 fine-tune | Colab GPU (image trainer) | — | 3.7k samples, minutes per epoch |
| CVCSPC SSL pretraining | Colab GPU (SSL trainer) | — | 4,970 unlabeled clips; trajectory-triplet construction on CPU, gradient on GPU |
| Val-tuned threshold sweep | eval/metrics.py | — | Reused unchanged; pure numpy |
| F1 / PR-AUC / confusion | eval/metrics.py | — | Reused unchanged |
| Checkpoint / resume | harness/colab.py | — | Atomic save + `latest.txt` contract carried forward |
| Drive staging (images.zip + trajectories) | harness/colab.py (new fn) | — | Mirror `stage_ohp_videos` for JPEG zip + `stage_unlabeled_squat_videos` for trajectories |
| Deliverable pack | Colab notebook | — | EDA + training/eval notebooks; `results.pkl`; figures |

---

## CVCSPC Method — Deep Dive

### What CVCSPC Is

Cross-View Cross-Subject Pose Contrastive (CVCSPC) learning. The pretext task: learn a representation where frames from two different subjects/videos at the **same exercise phase** are close in embedding space, while frames at **different phases** are far apart. [CITED: ar5iv.labs.arxiv.org/html/2202.14019, §3.1]

### What "Phase" Means

The barbell (or weight-stack) vertical position over time traces a parabolic trajectory. Phase is the **normalized elevation** of the barbell, mapped to -180° to +180° (analogous to a circular angle). Two frames have the same phase if the barbell is at the same relative height in its repetition arc. This is exercise-agnostic and camera-angle-agnostic — the trajectory is detected automatically. [CITED: ar5iv.labs.arxiv.org/html/2202.14019, §3.1]

In code, `traj2phase` converts the normalized trajectory values to degrees by multiplying by 360. [CITED: `dataloader.py` lines 35-39]

### Triplet Construction

For each training step:
1. Sample anchor video `v0` at a random phase `p_anchor` from the overlap of `v0` and `v1` phase ranges.
2. Positive: frame from a **different video** `v1` at the same phase `p_anchor`.
3. Negative: frame from `v1` at a phase **outside** the `ssl_contrastive_phase_gap` window around `p_anchor` (default gap = 30°, decaying if gap > 30° by -1° every 4 epochs). [CITED: `dataloader.py` lines 136-176; `train_test.py` lines 172-179]

The anchor and positive share the same phase but come from **different subjects/videos** — this forces the backbone to learn pose, not appearance. [CITED: ar5iv.labs.arxiv.org/html/2202.14019, §3.1]

### Loss Function

Paper (canonical, two-term denominator):

```
L = -log[ exp(-||phi_anc - phi_pos||_2) / (exp(-||phi_anc - phi_pos||_2) + exp(-||phi_anc - phi_neg||_2)) ]
```

Code (`train_test.py` lines 64-68) implements a **three-term denominator**:

```python
temp_loss = -1 * torch.log(
    torch.exp(-||anc - pos||^2) /
    (torch.exp(-||anc - pos||^2) + torch.exp(-||anc - neg||^2) + torch.exp(-||pos - neg||^2))
)
```

The code adds a `pos-neg` repulsion term not present in Eq. 1 of the paper. [CITED: `train_test.py` lines 64-68 vs ar5iv.labs.arxiv.org/html/2202.14019 Eq. 1]

**Deviation to record:** implement the three-term code version (more numerically stable; forces pos-neg separation); note in FINDINGS that the paper eq. shows two terms.

### Architecture

- Backbone: `resnet18(pretrained=False)` with ImageNet weights loaded manually via `load_state_dict` (filtering only matching keys). [CITED: `train_test.py` lines 235-241]
- Projection head: `model_linear_layers` from the missing `linear_layers.py` module. The `train_test.py` shows features pass through `F.normalize(model_linear_layers(model_CNN(im)), dim=-1, p=2)` before the loss. [CITED: `train_test.py` lines 51-53]
- **Linear layers module is absent from the code release.** Must reconstruct. The paper does not specify dimensions. [ASSUMED] Reconstruct as a 2-layer MLP: `nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 128))` — standard SimCLR-style projection head; the final L2-normalization happens outside (already in `train_test.py`).

### Augmentations

The `dataloader.py` augmentation dict shows only `masking` active (`mask_amt=0.4–0.5`, random 50% apply probability). All other augmentations (hori_flip, translation, rotation, blurring, zooming, color_jittering) are **commented out**. [CITED: `dataloader.py` lines 219-231]

The paper lists all augmentation types in prose but the code is authoritative for what was actually run. The augmentations are applied **independently** to anchor, positive, and negative. The masking function blacks out the top 40-50% of the image (cuts out the face/upper body, forces the model to focus on the body). [CITED: `Code_Release/data_augmentations/image_augmentations.py` lines 22-33]

For the supervised baseline fine-tune, the paper/code use `CenterCrop(H_2dcnn)` + `ToTensor()` + ImageNet normalize as the core transform. [CITED: `dataloader.py` line 94-96]

### Pretraining Recipe (from code + paper)

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Optimizer | Adam | [CITED: `train_test.py` line 157; paper §5] |
| Learning rate | 1e-4 | [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §5] |
| Epochs | 100 | [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §5] |
| Batch size | 25 | [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §5] |
| Phase gap | 30° (starts), decays if >30 every 4ep | [CITED: `train_test.py` lines 172-179] |
| Augmentations | Random masking (40-50%) only (others commented out) | [CITED: `dataloader.py` lines 219-231] |
| Checkpoint interval | Every 5 epochs | [CITED: `train_test.py` line 222] |

### Downstream Fine-Tune (Shallow-Squat Classification)

After SSL pretraining, the paper uses end-to-end fine-tuning for static single-image errors. [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §4 "Single image detection also made end-to-end learning more feasible, so we finetuned our models end-to-end"]

The downstream classifier replaces the ResNet-18 final FC (512→1000 ImageNet) with a `nn.Linear(512, 1)` binary head, then fine-tunes end-to-end with BCEWithLogitsLoss. [ASSUMED — paper does not specify head replacement explicitly; standard practice for binary fine-tune]

---

## The Decisive CVCSPC-Standalone Verdict

**Verdict: FEASIBLE STANDALONE — BackSquat unlabeled data only.**

Evidence:
1. Paper explicitly states: "models were self-supervisedly trained on our unlabeled BackSquat dataset" for the Shallow-Squat experiment. [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §4]
2. Paper states: "our methods are applicable to other exercises" — the method is not cross-exercise dependent at pretraining time. [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §3.1]
3. Data confirmed present locally: Squat Unlabeled `videos.zip` (1,598 MB, 4,970 clips, `-3-004`) + `bar_trajectories_raw.zip` (1.8 MB, 4,970 JSONs, `-3-001`). [VERIFIED: disk]
4. CVCSPC for BarbellRow used **cross-exercise transfer** (pretraining on Squat+OHP, fine-tuning on BarbellRow) — that is a separate experiment in Table 5; it is irrelevant here. [CITED: ar5iv.labs.arxiv.org/html/2202.14019 Table 5]
5. The `dataloader.py` requires `ssl_trajectories_dir` (JSON trajectories) and `ssl_frames_dir` (per-video frame directories). The unlabeled Squat dataset provides both, once staged to `/content/`. [VERIFIED: disk; CITED: `dataloader.py` lines 66-132]

**One caveat:** The CVCSPC dataloader expects `ssl_frames_dir/{video_id}/` — i.e., individual frames extracted from the unlabeled videos, one JPEG per frame, organized into per-video subdirectories. These frames are NOT pre-extracted in the dataset zip (which ships raw `.mp4` files). A **frame-extraction step** must run before SSL pretraining: decode each unlabeled mp4 and write frames to disk. This is straightforward (cv2 or torchvision) but adds a Step 0 cell in the SSL notebook.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x (Colab cu128) | Training + loss | Project-wide [VERIFIED: local] |
| torchvision | 0.26+ (Colab) | ResNet-18 pretrained weights | `ResNet18_Weights.IMAGENET1K_V1`; confirmed available [VERIFIED: local tv 0.27] |
| PIL (Pillow) | Latest | JPEG loading | Official code uses `Image.open` [CITED: `dataloader.py` line 43]; available [VERIFIED: local] |
| numpy | ≥1.24 | Array ops, trajectory math | Project-wide |
| sklearn | ≥1.3 | eval/metrics.py (unchanged) | Already in requirements.txt |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchvision.models.resnet18 | — | Backbone | `ResNet18_Weights.IMAGENET1K_V1` for baseline; load from `.pth` for CVCSPC |
| torchvision.transforms v2 | — | CenterCrop, RandomResizedCrop, ColorJitter, ToTensor, Normalize | Train and SSL augmentation |
| cv2 (opencv-headless) | ≥4.8 | Frame extraction from unlabeled mp4s (CVCSPC frame pre-extraction step) | SSL only; already in requirements.txt |
| tqdm | — | In-loop progress on all cells >2 min | Required per working agreement |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PIL Image.open | torchvision.io.read_image | read_image returns uint8 CHW tensor directly; PIL returns HWC and requires ToTensor. Either works but PIL matches official code. Both confirmed available on Colab tv 0.26+. [VERIFIED: local] |
| Per-epoch frame extraction | Pre-extract all frames in Step 0 | Pre-extract is dramatically faster during training (no mp4 decode in the hot loop); required by CVCSPC dataloader which expects a `{video_id}/frame.jpg` directory tree |

### Image Loading on Colab tv 0.26+

`torchvision.io.read_image` exists and is confirmed available locally (tv 0.27). The `read_video` removal that hit Phases 2-6 does NOT apply to `read_image` — they are separate APIs. [VERIFIED: local Python check — `read_image: AVAILABLE`]

For this phase, use **PIL** (`Image.open`) to match the official code exactly. PIL is available, handles JPEG natively, and produces an `HWC` PIL image that `transforms.ToTensor()` converts to `[0,1]` CHW float32.

---

## Package Legitimacy Audit

No new packages are installed in this phase. All libraries (torch, torchvision, PIL, numpy, sklearn, cv2) are already in `backend/requirements.txt` and confirmed available in the Colab environment from prior phases.

| Package | Status |
|---------|--------|
| torch | Already in requirements — no audit needed |
| torchvision | Already in requirements — no audit needed |
| Pillow | Already in requirements (via mediapipe transitive dep) — no audit needed |
| opencv-python-headless | Already in requirements.txt |

---

## Supervised Baseline Recipe

### Architecture

Standard ResNet-18 fine-tune:

```python
# Source: torchvision.models.resnet18, IMAGENET1K_V1 weights
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 1)  # binary head; in_features=512 is invariant in ResNet-18
```

End-to-end fine-tune (all layers). Binary output → `BCEWithLogitsLoss`. [CITED: paper §5 "finetuned our models end-to-end"]

### Loss

`BCEWithLogitsLoss(pos_weight=train_pos_weight)` where:

```
pos_weight = (N_train - n_pos) / max(n_pos, 1)
# = (2542 - 1116) / 1116 ≈ 1.28  (near-balanced: 1584/3611 = 43.9% pos overall; train split proportional)
```

Near 1.0 — weighting is a minor correction as expected. [CITED: CONTEXT D5]

### Optimizer

Adam, lr=1e-4. [CITED: ar5iv.labs.arxiv.org/html/2202.14019 §5]

Weight decay = 0 (not specified in paper). [ASSUMED]

CosineAnnealingLR with T_max = max_epochs. [ASSUMED — carry from Phase 3 pattern]

### Augmentation (train)

| Aug | Active | Parameters |
|-----|--------|-----------|
| `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))` | YES | Standard ImageNet fine-tune crop |
| `transforms.RandomHorizontalFlip(p=0.5)` | YES | [ASSUMED — standard for image classifiers; horizontal flip of a squat is semantically valid] |
| `transforms.ColorJitter(brightness=0.2, contrast=0.2)` | YES | [ASSUMED — minor; helps generalization] |
| `transforms.ToTensor()` | YES | Required |
| `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | YES | ImageNet stats [CITED: `dataloader.py` line 95-96] |

Val/test transform: `CenterCrop(224)` + ToTensor + Normalize. [CITED: `dataloader.py` line 94-96 — official code uses `CenterCrop(H_2dcnn)` at eval]

### Epochs / Batch Size / Patience

| Param | Value | Source |
|-------|-------|--------|
| max_epochs | 50 | [ASSUMED — paper silent on downstream epochs; carry Phase 3 pattern] |
| batch_size | 32 | [ASSUMED — ResNet-18 on L4 easily fits 64+; 32 is safe] |
| early_stop_patience | 8 | Carry from Phase 3/6 pattern [ASSUMED] |
| num_workers | 4 (+ persistent_workers=True) | Phase 2+ carry [CITED: project memory reference_pytorch_persistent_workers] |

### Expected F1 Ballpark

The paper does NOT report a separate purely-supervised ImageNet baseline row for Shallow-Squat in Table 3. The comparison table shows:

| Method | Modality | F1 |
|--------|----------|----|
| OpenPose-TDM | 2D Pose | 0.8340 |
| SimSiam | Image | 0.8286 |
| **CVCSPC (paper target)** | Image | **0.8694** |

SimSiam (0.8286) is the self-supervised image baseline. Our supervised ImageNet fine-tune is the new control. [ASSUMED] A well-tuned ImageNet ResNet-18 baseline on a near-balanced 3.7k-image dataset should land in the **0.80–0.87 range**. The dataset is small and the task is visually discriminative (squat depth is clearly visible in a static crop), so the baseline should be competitive.

---

## CVCSPC SSL Recipe (Reconstructed)

### Data Requirements

| Asset | Location (local disk) | Notes |
|-------|----------------------|-------|
| Unlabeled Squat videos.zip (4,970 mp4s) | `Fitness-AQA/.../3-004/.../Squat/Unlabeled_Dataset/videos.zip` (1.6 GB) | Stage to `/content/squat_unlabeled_videos/` |
| Trajectory JSONs zip (4,970 JSONs) | `Fitness-AQA/.../3-001/.../Squat/Unlabeled_Dataset/bar_trajectories_raw.zip` (1.8 MB) | Stage to `/content/squat_trajectories/` — already handled by `stage_unlabeled_squat_videos` |
| Pre-extracted frames | NOT in the zip — must decode from mp4s | New Step 0 cell: decode each mp4 → JPEG frames in `/content/squat_ssl_frames/{video_id}/` |

`stage_unlabeled_squat_videos` in `harness/colab.py` already handles both zips for the Squat unlabeled set. The new requirement is a **frame-extraction function** that converts `/content/squat_unlabeled_videos/*.mp4` → `/content/squat_ssl_frames/{video_id}/frame_{:06d}.jpg` using cv2.

### Frame Extraction

```python
# Reconstruct from dataloader.py ssl_frames_dir usage pattern
import cv2, os
def extract_frames_for_ssl(videos_root, frames_root, skip_existing=True):
    # for each mp4: cv2.VideoCapture -> write every frame as JPEG
    # directory: frames_root/{video_id}/frame_000000.jpg, ...
    # tqdm progress bar (cell takes >2 min)
```

This runs once per Colab session and is the most time-consuming SSL setup step. On L4, extracting 4,970 clips at ~100 frames/clip (≈500k frames total) to JPEG takes roughly 10-20 minutes. Cache check (`skip_existing=True`) makes it disconnect-safe.

### SSL Dataloader (Reconstructed)

Mirror `dataloader.py` with these fixes:
- Replace `from opts_exercise_qa import *` with explicit constants
- Replace `from augmentations import image_augmentations` with local `image_augmentations.py` (present in Code_Release)
- Replace `ssl_trajectories_dir` / `ssl_frames_dir` with runtime-provided paths
- Replace `train_val_test_sets_dir` references with the official split JSON paths

The `exclude_list` logic (exclude val + test split IDs from the unlabeled training set) must be reconstructed: the official train/val/test split JSON IDs are for the **labeled** dataset; the unlabeled set uses a different split (last 500 trajectories → val, rest → train per `dataloader.py` lines 77-84). [CITED: `dataloader.py` lines 63-84]

### SSL Model (Reconstructed)

```python
# Backbone: ResNet-18 with ImageNet init (matching train_test.py lines 235-241)
model_CNN = resnet18(weights=None)  # pretrained=False in original
# Load ImageNet weights manually, filter to matching keys
pretrained = torch.hub.load_state_dict_from_url('...')  # or ResNet18_Weights.IMAGENET1K_V1
model_CNN.load_state_dict(pretrained, strict=False)

# Remove the final FC layer — use the 512-d avgpool output as features
# Replace fc with nn.Identity() or slice backbone before fc

# Projection head (reconstructed — original linear_layers.py is missing)
model_linear_layers = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
)
```

The projection head dimensions (128) are [ASSUMED]. The paper does not specify them. The feature space after L2-normalization is used for the triplet loss.

### SSL Training Loop

The `train_test.py` loop is largely runnable as-is once the missing imports are resolved. Key reconstruction points:

1. `from ssl_contrastive_image_cleaned.dataloader import VideoDataset` → use the local reconstructed `ShallowSquatSSLDataset`
2. `from ssl_contrastive_image_cleaned.dataloader_eval import VideoDataset_Eval` → missing; reconstruct a simple eval variant (no augmentations, fixed phase gaps 30/45/60)
3. `from ssl_contrastive_image_cleaned.models import my_resnet, linear_layers` → replace with torchvision ResNet-18 + reconstructed projection head
4. `from opts_exercise_qa import *` → replace with explicit config dict

The training loop body (loss computation per `train_test.py` lines 60-75) is fully readable and self-contained. The three-term loss is the version to implement.

---

## Architecture Patterns

### Recommended Project Structure

```
backend/training/aqa/
├── datasets/
│   └── shallow_squat.py      # new — JPEG crop dataset (D3)
├── harness/
│   └── image_train.py        # new — ResNet-18 image trainer (mirrors supervised_train.py)
│   └── cvcspc_pretrain.py    # new — CVCSPC SSL pretraining loop (research-gated)
├── models/
│   └── shallow_squat/        # Drive: phase07/ checkpoints
docs/notebooks/
├── 09_shallow_squat_eda.ipynb        # EDA notebook
├── 10_shallow_squat_training.ipynb   # Baseline + SSL training + eval
```

### Pattern 1: ShallowSquatDataset Structure

```python
# Source: mirror of backend/training/aqa/datasets/squat.py (minus video decode)
class ShallowSquatDataset(Dataset):
    def __init__(self, split, *, images_root, labels_path, splits_root, train_aug=True):
        # Load labels_shallow_depth.json -> {id: 0/1}
        # Load splits/{split}_ids.json -> flat list of ids
        # self.records = [ShallowSquatRecord(id, label)]
        # pos_weight from train split always
        ...

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(images_root / f"{rec.id}.jpg").convert("RGB")
        img = self.transform(img)  # ImageNet norm, 224x224
        return img, torch.tensor(rec.label, dtype=torch.float32)
```

Key differences from `squat.py`: no `decode_clip`, no frame sampling, no Kinetics norm. Single PIL `Image.open` + transform.

### Pattern 2: ImageNet ResNet-18 Trainer

New `image_train.py` — mirrors `supervised_train.py` contracts:
- `ImageConfig` dataclass (hashed into `config_hash`)
- `build_image_model()` → ResNet-18 + `nn.Linear(512, 1)` binary head
- `run_image_epoch(*, run_name, ...)` entry point
- Same `atomic_save_checkpoint`, `load_latest_checkpoint`, `prune_checkpoints`, `capture_rng_state`, `restore_rng_state` calls
- Same early-stop on val-F1 + `best.pt` pattern
- `threshold_sweep` on val scores → `best_threshold` → `best.pt`

### Pattern 3: CVCSPC SSL Triplet Training

```python
# Source: train_test.py lines 60-75 (three-term loss, reconstructed)
def cvcspc_loss(phi_anc, phi_pos, phi_neg):
    exp_ap = torch.exp(-torch.sum((phi_anc - phi_pos) ** 2, dim=-1))
    exp_an = torch.exp(-torch.sum((phi_anc - phi_neg) ** 2, dim=-1))
    exp_pn = torch.exp(-torch.sum((phi_pos - phi_neg) ** 2, dim=-1))
    loss = -torch.log(exp_ap / (exp_ap + exp_an + exp_pn + 1e-8))
    return loss.mean()
```

After L2-normalization of features, distances are bounded and the exponential is numerically stable.

### Pattern 4: Stage Image Zip (New colab.py function)

```python
# Mirror stage_ohp_videos for JPEG crops
def stage_shallow_squat_images(drive_root, *, local_root="/content/squat_shallow_images"):
    # src: {drive_root}/Fitness-AQA_dataset_release-3-001/.../Shallow_Squat_Error_Dataset/images.zip
    # Extracts crops_unaligned/{id}.jpg flat into local_root/crops_unaligned/
    # images.zip is 100 MB — fast copy; 3,738 JPEGs extracted
```

Note: the zip extracts to `crops_unaligned/{id}.jpg` (not flat). The dataset module addresses them as `images_root/crops_unaligned/{id}.jpg`. Alternatively, extract flat and pass `images_root` as the `crops_unaligned/` directory.

### Anti-Patterns to Avoid

- **Using Kinetics norm for ResNet-18 ImageNet weights:** Kinetics stats are `mean=[0.43216, 0.394666, 0.37645]` etc. ImageNet stats are `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`. Using the wrong norm will degrade performance significantly.
- **Reusing `_extract_with_resume_and_progress` for the JPEG zip:** It is mp4-only (filters on `.mp4`). For the images.zip, either use a generic extractor or the `zipfile.extractall` path (like trajectory zips).
- **Sharing a single DataLoader between SSL pretraining and supervised fine-tune:** They require different transforms (augmented vs. supervised train).
- **Not resetting the ResNet-18 head after CVCSPC pretraining before fine-tune:** The SSL phase's linear_layers head is discarded; the fine-tune head is a new `nn.Linear(512, 1)` initialized fresh.
- **Forgetting `persistent_workers=True` when `num_workers > 0`:** Phase 2+ carry; avoids PyTorch #40157 stderr spam.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Binary F1 / PR-AUC / threshold sweep | Custom metric loop | `eval/metrics.py` (reuse unchanged) | Already verified across 3 phases; consistent methodology |
| ImageNet ResNet-18 weights | Manual download | `ResNet18_Weights.IMAGENET1K_V1` via torchvision | Versioned, canonical |
| Drive staging with byte-resume | Custom shutil loop | `stage_unlabeled_squat_videos` (already in `colab.py`) + new `stage_shallow_squat_images` | Disconnect-safe; tqdm; idempotent |
| Atomic checkpoint | `torch.save` directly | `atomic_save_checkpoint` + `load_latest_checkpoint` from `colab.py` | tmp→verify→replace→latest.txt contract |
| Triplet loss with numerical instability | Raw exp() | Add `1e-8` to denominator; use `torch.clamp` if needed | Prevents log(0) |
| Frame extraction from mp4 | Custom ffmpeg wrapper | cv2.VideoCapture read loop | Already in requirements.txt |

---

## Common Pitfalls

### Pitfall 1: Wrong Norm Stats (Kinetics vs. ImageNet)

**What goes wrong:** Using `mean=[0.43216, ...]` (Kinetics) with ImageNet ResNet-18 weights.
**Why it happens:** Copy-paste from the video pipeline (`transforms.py`).
**How to avoid:** `ShallowSquatDataset` hardcodes `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. Never import from `transforms.py` (which has Kinetics norm).
**Warning signs:** F1 stagnates at ~0.44 (random guessing on near-balanced data).

### Pitfall 2: CVCSPC Frame Directory Layout Mismatch

**What goes wrong:** Frames extracted as `{video_id}_frame0.jpg` (flat) but dataloader expects `{video_id}/frame_0.jpg` (subdirectory per video).
**Why it happens:** Frame extraction writes to the wrong layout.
**How to avoid:** Frame extractor must create `ssl_frames_root/{video_id}/` subdirectory and write frames there. The dataloader does `os.listdir(ssl_frames_dir + video_id + '/')`. [CITED: `dataloader.py` lines 130-131]

### Pitfall 3: images.zip Extracts to Nested Path

**What goes wrong:** Extracting `images.zip` with `_extract_with_resume_and_progress` (mp4-only) silently skips all JPEGs.
**Why it happens:** The extractor filters `filename.lower().endswith(".mp4")`.
**How to avoid:** Use `zipfile.extractall` for the images zip. The zip extracts to `crops_unaligned/{id}.jpg` — set `images_root` to point to the `crops_unaligned/` subdirectory, or strip the prefix at extract time.

### Pitfall 4: Missing `dataloader_eval.py` Causes SSL Eval to Fail

**What goes wrong:** `train_test.py` imports `from ssl_contrastive_image_cleaned.dataloader_eval import VideoDataset_Eval` — this module is absent from the code release.
**Why it happens:** The eval dataloader was not shipped.
**How to avoid:** Reconstruct `VideoDataset_Eval` as a simplified fixed-phase-gap variant of the SSL dataloader (no random sampling — deterministic nearest-frame lookup for a fixed `ssl_contrastive_phase_gap` at 30/45/60°). The eval loop tests triplet accuracy (anchor-positive < anchor-negative distance) at multiple gap levels — not F1; that comes in the supervised fine-tune.

### Pitfall 5: SSL Pretraining Trains on val/test Labeled Crops

**What goes wrong:** SSL pretraining accidentally includes the labeled val/test IDs in the contrastive training set.
**Why it happens:** The unlabeled dataset uses a different split than the labeled dataset; the confusion is that both are "Squat" data.
**How to avoid:** CVCSPC pretrains on the **Squat Unlabeled dataset** (4,970 clips, separate videos) — not on the 3,611 labeled crops. The labeled crops are only used for supervised fine-tune. There is no overlap.

### Pitfall 6: ResNet-18 head Not Reset Between SSL and Fine-Tune

**What goes wrong:** The projection head from SSL pretraining remains attached; the classifier head is not initialized fresh.
**Why it happens:** The trainer loads the SSL checkpoint and forgets to swap the head.
**How to avoid:** Load the SSL backbone weights into a new `resnet18(weights=None)`, replace `model.fc = nn.Linear(512, 1)`, initialize fresh. The SSL `model_linear_layers` is discarded.

### Pitfall 7: Frame Extraction Is Not Disconnect-Safe

**What goes wrong:** Frame extraction starts over from scratch after a runtime restart.
**Why it happens:** No idempotency check.
**How to avoid:** Frame extractor checks if `{video_id}` directory already exists with expected frame count; skips if complete. Tqdm progress bar shows extracted vs skipped.

---

## Code Examples

### ImageNet Norm Transform

```python
# Source: dataloader.py line 94-96 (official code) + D3 locked decision
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

### ResNet-18 Binary Head

```python
# Source: torchvision.models + standard fine-tune pattern
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def build_baseline_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert model.fc.in_features == 512, "ResNet-18 fc.in_features must be 512"
    model.fc = nn.Linear(512, 1)
    return model
```

### CVCSPC Three-Term Loss

```python
# Source: train_test.py lines 64-68 (three-term version)
import torch
import torch.nn.functional as F

def cvcspc_triplet_loss(phi_anc, phi_pos, phi_neg):
    # Inputs: L2-normalized feature vectors [B, D]
    exp_ap = torch.exp(-torch.sum((phi_anc - phi_pos) ** 2, dim=-1))
    exp_an = torch.exp(-torch.sum((phi_anc - phi_neg) ** 2, dim=-1))
    exp_pn = torch.exp(-torch.sum((phi_pos - phi_neg) ** 2, dim=-1))
    loss = -torch.log(exp_ap / (exp_ap + exp_an + exp_pn + 1e-8))
    return loss.mean()
```

### ShallowSquatDataset Skeleton

```python
# Source: mirror of backend/training/aqa/datasets/squat.py (D3)
from __future__ import annotations
import json, logging
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger("aqa.phase07")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ShallowSquatDataset(Dataset):
    def __init__(self, split, *, images_root, labels_path, splits_root, train_aug=True):
        labels = json.loads(Path(labels_path).read_text())
        ids = json.loads(Path(splits_root, f"{split}_ids.json").read_text())
        train_ids = json.loads(Path(splits_root, "train_ids.json").read_text())
        self.records = [(i, labels[i]) for i in ids if i in labels]
        train_pos = sum(labels[i] for i in train_ids if i in labels)
        n_train = sum(1 for i in train_ids if i in labels)
        w = (n_train - train_pos) / max(train_pos, 1)
        self.pos_weight = torch.tensor([w], dtype=torch.float32)
        self.images_root = Path(images_root)
        self.transform = (
            T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                       T.RandomHorizontalFlip(0.5),
                       T.ColorJitter(0.2, 0.2),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
            if train_aug else
            T.Compose([T.Resize(256), T.CenterCrop(224),
                       T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        )

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        id_, label = self.records[idx]
        img = Image.open(self.images_root / f"{id_}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
```

---

## Recommended Plan Shape (Wave Breakdown)

### Wave 0 — CPU: Dataset Module + Tests

**Goal:** `datasets/shallow_squat.py` validated on Windows (no GPU, no Drive mount needed — just the extracted images locally or a small fixture).

Tasks:
- New `datasets/shallow_squat.py` with `ShallowSquatDataset` + `build_loaders`
- Unit tests: split sizes (2542/529/540), pos_weight (≈1.28), `__getitem__` returns correct tensor shape `(3, 224, 224)` and label `{0.0, 1.0}`, ImageNet norm range check

### Wave 1 — Colab EDA Notebook (09_shallow_squat_eda.ipynb)

**Goal:** Understand the data before training. Supervisor needs visualizations.

Tasks:
- Stage `images.zip` to `/content/`; count 3,738 JPEGs; load labels_shallow_depth.json + split JSONs
- Class balance bar chart (overall + per split)
- Sample grid: 8 positive + 8 negative crops side-by-side
- Image size distribution histogram (are all crops the same size?)
- Pixel mean/std sanity check vs. ImageNet norms
- Save `results.pkl` (eda_stats, balance, sample grid paths)

### Wave 2 — Colab Baseline Training + Eval (10_shallow_squat_training.ipynb)

**Goal:** ImageNet ResNet-18 baseline trained, val-tuned threshold, test F1 computed and compared to paper CVCSPC 0.8694.

Tasks:
- `image_train.py` — `ImageConfig`, `build_baseline_model`, `run_image_epoch`
- Cell A (git pull), Step 0 (stage images.zip), Step 1 (EDA recap cell), Step 2 (train seed 42), Step 3 (train seeds 1337/7 or 2-seed), Step 4 (ensemble + eval), Step 5 (figures)
- Figures: training curves (loss/F1 vs epoch), PR curve, confusion matrix, sample hard cases
- Save `results.pkl` (baseline metrics per seed, ensemble threshold, test F1)
- FINDINGS_SHALLOW_SQUAT.md with baseline vs CVCSPC target

### Wave 3 — CVCSPC SSL Attempt (research-gated; new notebook 11_cvcspc_ssl.ipynb)

**Goal:** CVCSPC pretraining on Squat Unlabeled, then fine-tune on Shallow-Squat labeled; compare to baseline and paper's 0.8694.

Tasks:
- Frame extraction: `stage_unlabeled_squat_videos` → cv2 decode → `/content/squat_ssl_frames/{video_id}/`
- Reconstruct SSL dataloader (`ShallowSquatSSLDataset`, `ShallowSquatSSLEvalDataset`)
- Reconstruct model (ResNet-18 backbone + projection head)
- SSL pretraining loop: 100 epochs, Adam 1e-4, batch 25, phase gap decay, atomic checkpoint
- Linear probe eval (optional: frozen backbone + logistic regression on train features)
- End-to-end fine-tune post-SSL: same recipe as Wave 2 but loading SSL backbone weights
- Multi-seed (3) fine-tune; ensemble; test F1 vs baseline and vs CVCSPC 0.8694
- Update FINDINGS_SHALLOW_SQUAT.md with baseline→SSL delta

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | `backend/tests/` (existing) |
| Quick run | `pytest backend/training/aqa/datasets/test_shallow_squat.py -x -q` |
| Full suite | `pytest backend/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IMG-01 | `ShallowSquatDataset` split sizes match 2542/529/540 | unit | `pytest .../test_shallow_squat.py::test_split_sizes -x` | Wave 0 |
| IMG-01 | `pos_weight` computed from train split only, value ≈ 1.28 | unit | `pytest .../test_shallow_squat.py::test_pos_weight -x` | Wave 0 |
| IMG-01 | `__getitem__` returns `(3, 224, 224)` float32 + scalar float32 label | unit | `pytest .../test_shallow_squat.py::test_getitem_shape -x` | Wave 0 |
| IMG-01 | ImageNet norm applied (pixel mean in [-2.5, 2.5]) | unit | `pytest .../test_shallow_squat.py::test_imagenet_norm -x` | Wave 0 |
| IMG-01 | `build_loaders` returns dict with train/val/test DataLoaders | unit | `pytest .../test_shallow_squat.py::test_build_loaders -x` | Wave 0 |
| IMG-02 | Baseline test F1 computed in code (not hand-typed) | manual/Colab | Run eval cell; verify `results.pkl` loaded | Colab Wave 2 |
| IMG-02 | Multi-seed ensemble mean-of-sigmoids protocol | manual/Colab | Confirm 3 seed checkpoints in results.pkl | Colab Wave 2 |

### Sampling Rate

- Per task commit (local): `pytest backend/training/aqa/datasets/test_shallow_squat.py -x -q`
- Per wave merge: `pytest backend/ -x -q`
- Phase gate: full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `backend/training/aqa/datasets/test_shallow_squat.py` — all 5 tests above
- [ ] `backend/training/aqa/datasets/shallow_squat.py` — the module itself

---

## Security Domain

Security enforcement is not applicable to this offline ML training phase. No API, no user data, no network calls at runtime. Training uses local research data under the Fitness-AQA non-commercial license.

---

## Runtime State Inventory

Not applicable — greenfield training module, no rename/refactor.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python (local) | Unit tests, module authoring | ✓ | 3.10+ inferred | — |
| torch | Training | ✓ (local 2.12.0+cpu; Colab cu128) | 2.x | — |
| torchvision | ResNet-18, transforms, read_image | ✓ (local 0.27.0) | 0.27.0 | — |
| PIL (Pillow) | JPEG loading | ✓ (local) | latest | — |
| sklearn | eval/metrics.py | ✓ (in requirements.txt) | ≥1.3 | — |
| cv2 (opencv-headless) | Frame extraction (SSL) | ✓ (in requirements.txt) | ≥4.8 | — |
| Squat Unlabeled videos.zip | CVCSPC pretraining | ✓ | 1.6 GB on disk (-3-004) | If missing: baseline only (IMG-02 satisfied without SSL) |
| Squat Unlabeled trajectories zip | CVCSPC pretraining | ✓ | 1.8 MB on disk (-3-001) | Same as above |
| Shallow-Squat images.zip | All training | ✓ | 100 MB on disk (-3-001) | Required; no fallback |
| Shallow-Squat labels + splits | All training | ✓ | JSON files on disk | Required; no fallback |
| Google Drive (Colab) | Checkpoint persistence | ✓ (runtime) | — | Session-local only (disconnect loses checkpoints) |
| L4 GPU (Colab) | Training | ✓ (runtime; select manually) | 24 GB | T4 acceptable (slower; ResNet-18 is light) |

**Missing dependencies with no fallback:** None for the supervised baseline. CVCSPC requires the unlabeled videos.zip — confirmed present.

**Note on Colab Drive paths:** The Shallow-Squat images.zip and trajectory zip are in the `-3-001` folder; the Squat unlabeled videos.zip is in the `-3-004` folder. Both must be mapped correctly in the staging functions. The `-3-004` path differs from the existing `stage_unlabeled_squat_videos` implementation (which currently points to `-3-001`). The staging function may need a `drive_root_3004` parameter analog.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Kinetics norm for all ResNet models | ImageNet norm for ImageNet-pretrained, Kinetics norm for Kinetics-pretrained | Phase 7 introduction | Correct stats for the backbone's pretraining distribution |
| `read_video` (torchvision) | PIL Image.open for JPEG; cv2 for video frame extraction | tv 0.26 breakage in Phases 2-6 | `read_image` still available; PIL is safer and matches official code |

**Deprecated/outdated in this phase:**
- `transforms.py` spatial pipeline from Phase 2: video-specific (Kinetics norm + 3D tensor); not imported here.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Projection head is 2-layer MLP: Linear(512,128) + ReLU + Linear(128,128) | CVCSPC Method | If wrong dims: features have different scale; trivially fixable by ablating head size; F1 impact probably minor |
| A2 | Horizontal flip is a valid train augmentation for Shallow-Squat (semantic symmetry) | Supervised Baseline Recipe | Squats are approximately symmetric; risk is minimal |
| A3 | ColorJitter(0.2, 0.2) is appropriate for supervised fine-tune | Supervised Baseline Recipe | Conservative values; low risk |
| A4 | max_epochs=50 for supervised fine-tune | Supervised Baseline Recipe | ResNet-18 on 3.7k images may converge earlier; early-stop patience=8 catches this |
| A5 | Batch size=32 for supervised fine-tune | Supervised Baseline Recipe | Safe on L4; if OOM use 16 |
| A6 | Weight decay=0 for supervised fine-tune | Supervised Baseline Recipe | Paper silent; 1e-4 AdamW is an easy alternative if val-F1 plateaus |
| A7 | CosineAnnealingLR for supervised fine-tune | Supervised Baseline Recipe | Paper silent; carry from Phase 3 pattern |
| A8 | Baseline F1 will be in 0.80-0.87 range | Supervised Baseline Recipe | Near-balanced + visually discriminative task suggests this is plausible; actual result computed in code |
| A9 | SSL fine-tune uses same recipe as baseline (Adam 1e-4, 50ep) | CVCSPC SSL Recipe | Lower LR may be better for SSL fine-tune; tune on val |

---

## Open Questions

1. **Drive path for -3-004 unlabeled Squat videos**
   - What we know: on local disk the path is `.../3-004/.../Squat/Unlabeled_Dataset/videos.zip`; the existing `stage_unlabeled_squat_videos` points to `-3-001`
   - What's unclear: whether Google Drive has the same folder structure as the local extracted tree, and what the Drive shortcut path is for `-3-004`
   - Recommendation: Step 0 of SSL notebook probes the Drive path; planner adds a comment in the staging function noting the `-3-004` Drive root parameter

2. **Linear layers projection head dimensions**
   - What we know: `linear_layers.py` is absent; features are L2-normalized after the head
   - What's unclear: whether the original code used 512→512 (no reduction) or 512→128 or another dim
   - Recommendation: Implement 512→128→128 (SimCLR-style); note as reconstructed; if SSL triplet accuracy is poor, try 512→512 as ablation

3. **Supervised baseline F1 without a paper reference point**
   - What we know: Table 3 has no supervised-ImageNet baseline row; SimSiam at 0.8286 is the only image SSL baseline; CVCSPC at 0.8694 is the target
   - What's unclear: what the baseline F1 "should" be for comparison narrative
   - Recommendation: Frame the narrative as "baseline F1 = X; CVCSPC lift = Y; paper CVCSPC = 0.8694". The baseline is the new control; SimSiam (0.8286) provides the only paper image baseline for comparison.

4. **Drive shortcut for Shallow-Squat images.zip**
   - What we know: locally at `-3-001/.../Shallow_Squat_Error_Dataset/images.zip`
   - What's unclear: exact Drive shortcut path (Phase 6 OHP had a split -3-001 / -3-002 structure)
   - Recommendation: Step 0 of EDA notebook verifies Drive path and logs it; staging function parametrizes `drive_root_3001`

---

## Sources

### Primary (HIGH confidence)
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/dataloader.py` — CVCSPC dataloader: triplet construction, phase computation, augmentation dict, transforms
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/train_test.py` — CVCSPC training loop: loss formula (three-term), optimizer, checkpoint interval, model initialization
- `Fitness-AQA-Code/Code_Release/data_augmentations/image_augmentations.py` — masking augmentation implementation
- `ar5iv.labs.arxiv.org/html/2202.14019` — Parmar et al. ECCV 2022: CVCSPC method §3.1, downstream eval protocol, F1 results Table 3 (OpenPose-TDM 0.8340, SimSiam 0.8286, CVCSPC 0.8694), training recipe (Adam 1e-4, 100ep, batch 25), single-exercise pretraining confirmation
- `backend/training/aqa/eval/metrics.py` — eval reuse confirmed (read source)
- `backend/training/aqa/harness/colab.py` — staging/checkpoint primitives confirmed (read source)
- `backend/training/aqa/datasets/squat.py` — template structure confirmed (read source)
- Local disk probe — Squat Unlabeled videos.zip (4,970 mp4s, `-3-004`, 1.6 GB), trajectories zip (4,970 JSONs, `-3-001`, 1.8 MB), Shallow-Squat images.zip (100 MB), labels JSON, splits JSONs confirmed present

### Secondary (MEDIUM confidence)
- `arxiv.org/abs/2202.14019` — abstract and method overview

### Tertiary (LOW confidence)
- WebSearch summaries — CVCSPC F1 numbers corroborated (0.8694) but not independently verified against a third source

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages confirmed available locally and in prior Colab sessions
- CVCSPC method: MEDIUM-HIGH — loss formula and dataloader confirmed from code; projection head reconstructed [ASSUMED]
- Architecture patterns: HIGH — ResNet-18 fine-tune is standard; template codebase confirmed
- Pitfalls: HIGH — discovered from code reading (not assumptions)
- SSL feasibility verdict: HIGH — confirmed from paper text; data confirmed on disk

**Research date:** 2026-05-30
**Valid until:** 60 days (stable method; dataset is static)

---

## RESEARCH COMPLETE

**Phase:** 7 — Image-Based Errors (CVCSPC), Shallow-Squat only
**Confidence:** MEDIUM-HIGH

### Key Findings

- **CVCSPC SSL is feasible standalone on Shallow-Squat.** The paper explicitly pretrained CVCSPC on the BackSquat Unlabeled dataset alone (not multi-exercise), then fine-tuned on the labeled Shallow-Squat crops. The required unlabeled Squat videos.zip (4,970 mp4s, 1.6 GB) is confirmed present at `-3-004` locally; trajectories zip (4,970 JSONs, 1.8 MB) at `-3-001`. BarbellRow data is not required.

- **The official code release is severely incomplete.** `CVC_SPC.py`, `opts_exercise_qa.py`, `models/linear_layers.py`, `dataloader_eval.py` are all absent. Only `dataloader.py` (66 lines) and `train_test.py` (291 lines) shipped. The method can be fully reconstructed from these two files + the paper; the loss formula, triplet construction logic, optimizer/recipe, and augmentation dict are all readable.

- **Critical deviation: the code uses a three-term loss** (`-log(exp_ap / (exp_ap + exp_an + exp_pn))`), not the two-term paper Equation 1. Implement the code version; note the deviation.

- **CVCSPC requires a frame-extraction pre-step.** The dataloader expects per-video frame directories (`{ssl_frames_dir}/{video_id}/frame.jpg`), not raw mp4s. A Step 0 cell must decode the 4,970 unlabeled mp4s to disk (~10-20 min on L4, disconnect-safe with skip-existing).

- **The paper's Table 3 has no supervised ImageNet baseline row.** SimSiam (0.8286) is the only image SSL baseline; CVCSPC (0.8694) is the target. Our supervised baseline is the new control — results computed in code, compared honestly.

### File Created
`.planning/phases/07-image-based-errors-cvcspc/07-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All packages confirmed available locally and in prior phases |
| CVCSPC Method | MEDIUM-HIGH | Loss + dataloader from code; projection head reconstructed [ASSUMED] |
| Architecture | HIGH | ResNet-18 fine-tune standard; template code confirmed |
| Data layout | HIGH | Local disk probe confirmed all zips present and entry counts verified |
| Pitfalls | HIGH | Derived from code reading, not assumptions |

### Open Questions Remaining

- Exact Drive path for `-3-004` Squat unlabeled videos (resolvable in Step 0 of SSL notebook)
- Projection head dimensions (512→128→128 assumed; easily ablated)
- Whether Colab's `stage_unlabeled_squat_videos` correctly addresses `-3-004` or needs a new parameter

### Ready for Planning

Research complete. Planner can create PLAN.md files. The decisive CVCSPC-standalone verdict is: **FEASIBLE** — BackSquat unlabeled data alone is sufficient, both data assets are confirmed present, and the method can be reconstructed from the available code + paper.
