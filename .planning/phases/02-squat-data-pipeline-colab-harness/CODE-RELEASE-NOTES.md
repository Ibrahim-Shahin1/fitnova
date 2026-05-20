# Code_Release Gap Analysis

**Source:** `github.com/ParitoshParmar/Fitness-AQA`, cloned to `Fitness-AQA-Code/` at repo root, sha at clone time = current HEAD of `master`.
**Read by:** Phase 2 Task 2 — before writing any pipeline code.
**Purpose:** record what the official repo ships, what it doesn't, what idioms we reuse, and what Phase 2/3/4 reconstruct from the paper.

---

## (a) Full `Code_Release/` inventory

Verified by direct read of the local clone. Sizes from `Get-ChildItem`.

```
Code_Release/
├── ReadMe.md                                                        23 B
├── data_augmentations/
│   ├── ReadMe.MD                                                   118 B
│   └── image_augmentations.py                                    2,786 B
├── motion_disentanglement/
│   └── README.md                                                     2 B   <-- empty placeholder
└── pose_contrastive_learning/
    ├── README.md                                                     2 B   <-- empty placeholder
    └── self_supervised_learning/
        ├── dataloader.py                                         (251 lines)
        └── train_test.py                                         (257 lines)
```

The top-level `ReadMe.md` says exactly: `_Work in Progress..._`. That is the entirety of the upstream documentation.

The repo's whole executable surface is **three Python files** totaling under 550 lines: `image_augmentations.py`, `pose_contrastive_learning/self_supervised_learning/dataloader.py`, `pose_contrastive_learning/self_supervised_learning/train_test.py`.

---

## (b) Verdict — what's missing

The repo ships **none of the code Phase 2 promised to adapt**. Specifically:

1. **`motion_disentanglement/` is empty.** The MD-SSL pretext task — the *thesis contribution* of Parmar et al. (ECCV 2022) — has zero published code. README is 2 bytes, just a CRLF.
2. **No supervised video baseline trainer.** No `train.py`, no `model.py` defining the R(2+1)D-18 wrapper, no video dataloader, no Squat-specific KIE/KFE label loading code. The supervised numbers from paper Tables 3–5 cannot be reproduced from the release.
3. **No image-modality classifier head.** Even the CVCSPC code (image-side SSL for Squat-Shallow / BarbellRow Lumbar+Torso) only contains the SSL **pretext** trainer — no `classifier.py`, no fine-tune script, no `dataloader_eval.py` for downstream evaluation.
4. **The CVCSPC code that *does* ship is not runnable as-released.** Three independent fatal problems:
   - `dataloader.py` line 12: `from opts_exercise_qa import *` — this module is **not in the repo**. It is expected to define constants like `randomseed`, `input_resize_2dcnn`, `H_2dcnn`, `W_2dcnn`, `C_2dcnn`, `ssl_trajectories_dir`, `train_val_test_sets_dir`, `ssl_frames_dir`. The user is expected to reconstruct it from context.
   - `dataloader.py` line 13: `from augmentations import image_augmentations` — but the actual directory is `data_augmentations/`, not `augmentations/`. Import path is wrong.
   - `train_test.py` line 7: `from ssl_contrastive_image_cleaned.dataloader import VideoDataset` — but the file is at `pose_contrastive_learning/self_supervised_learning/dataloader.py`. Different package name.
5. **`image_augmentations.py` is missing the `apply_augmentations()` function** that `dataloader.py` line 51 actually calls. The file defines individual transforms (`hori_flip`, `masking`, `masking_checker_ol`, `masking_checker_nool`) but the wrapper that consumes the `augmentations` dict and applies them in order is absent.
6. **`train_test.py` references modules that don't exist in the repo at all**: `dataloader_eval`, `ssl_contrastive_image_cleaned.models.my_resnet`, `ssl_contrastive_image_cleaned.models.linear_layers`. The trainer can't even be imported standalone.
7. **Hardcoded paths inside `train_test.py`**: line 206 (`/data/paritosh_trained_wts/...`), line 237 (`/home/dockeruser/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth`). The release was extracted from a private Docker container without sanitization.

Net: **Phase 2 cannot "clone and adapt" any video-pipeline code**, and Phase 7 (the only one where CVCSPC is relevant) will still need to reconstruct `apply_augmentations`, `opts_exercise_qa`, `dataloader_eval`, and the model heads from paper context.

---

## (c) CVCSPC idioms we reuse anyway

The released CVCSPC code is incomplete, but the parts that exist surface a few concrete idioms worth borrowing as-is or as a starting point:

1. **Single-line determinism block.** `train_test.py` lines 16–18:
   - Seeds `torch`, `torch.cuda`, `random`, `numpy` all from one shared `randomseed`.
   - Sets `cudnn.deterministic=True`, `cudnn.benchmark=False`.
   - We extend it with `torch.use_deterministic_algorithms(True)` and the `CUBLAS_WORKSPACE_CONFIG` env var — see CONTEXT.md D13 and the Determinism Checklist in PLAN.md.
2. **Adam(lr=1e-4) as the default.** `train_test.py` line 157: `optim.Adam(parameters_2_optimize, lr=learning_rate)` where `learning_rate = base_learning_rate` (the constant in the missing `opts_exercise_qa`). Combined with the paper's stated `1e-4` for the supervised baseline, this is the Phase 3 starting point.
3. **Augmentation-as-config-dict pattern.** `dataloader.py` lines 219–232 build an `augmentations` list of dicts, one per sample, with `'apply': random.choice([0, 1])` toggles. The pattern is reusable; we just need to write the missing `apply_augmentations(image, dict)` dispatcher ourselves.
4. **Split exclusion pattern.** `dataloader.py` lines 66–74 compute the SSL training set by **subtracting** val/test from the full file list, then dropping `traj_nan` clips. Our `splits.py` uses the union of split files directly (more idiomatic), but the `traj_nan.json` exclusion idea is correct and we apply it in Phase 4.
5. **Checkpoint-save signature.** `train_test.py` lines 21–23 save only the model state dict via `torch.save(model.state_dict(), model_path)`. We **diverge** here — Phase 2 saves an atomic payload with `epoch`, `optimizer_state`, `rng_state`, `metrics_history`, `config_hash`. The released pattern is too thin for resumability.

---

## (d) What Phase 2 reconstructs from the paper

Because there is no video-pipeline code to mirror, Phase 2 reconstructs the following from Parmar et al. (ECCV 2022, arXiv:2202.14019) and torchvision conventions:

| Reconstructed from | Source |
|---|---|
| 32-frame uniform sampling across full clip span | Paper §5 (R(2+1)D input) |
| Aspect-aware resize → 224² (paper's Waseda 2DCNN path) / **112² (R(2+1)D path — RESEARCH.md §1)** | Paper line 495–496 + torchvision `R2Plus1D_18_Weights.KINETICS400_V1` |
| Kinetics-400 normalization (`mean=[0.43216, 0.394666, 0.37645]`, `std=[0.22803, 0.22145, 0.216989]`) | torchvision Kinetics-400 stats (R(2+1)D path uses Kinetics, not ImageNet) |
| Horizontal flip **OFF** | Paper line 576's augmentation list does not include flip; CVCSPC code (`dataloader.py` line 222) has it commented out |
| Class imbalance handled at the loss via `BCEWithLogitsLoss(pos_weight=...)` | Paper line 413 (class-weighted CE); `pos_weight` is the multi-label generalization |
| Atomic checkpoint write (tmp → reload-verify → `os.replace` → `latest.txt` last) | Drive FUSE rename is not atomic; CVCSPC's bare `torch.save` is insufficient |
| Decoder wrapped behind `decode_clip(path, indices)` | Defends against `torchvision.io.read_video` deprecation (0.22+) and OOM on long clips |
| RNG capture/restore including DataLoader worker generators | Needed for the bitwise-resume assertion (Task 14) |

Everything else in Phase 2 — the `SquatKIEKFEDataset`, `build_loaders` factory, `splits.py` index, `colab.py` harness primitives, `tiny_train.py` toy model — is fresh code.

---

## (e) What Phase 4 will face

Phase 4 is **Motion-Disentangling SSL** — the paper's main contribution. With `motion_disentanglement/` completely empty, Phase 4 must reconstruct *the whole pretext task* from:

- **Paper §4 + Figure 3**: the half-cycle splitting of a squat repetition, the MD network architecture (a R(2+1)D-18 encoder with a per-window head that disentangles "motion phase" from "style"), the loss formulation.
- **The shipped Squat bar trajectories** (`Squat/Unlabeled_Dataset/bar_trajectories_raw.zip`): clean y-center curves over the rep, ~3000 samples, 0% NaN per Phase 1. These define the "phase" axis for the half-cycle split.
- **The CVCSPC contrastive-loss skeleton** (`train_test.py` lines 60–71): a 3-way triplet with exp-based softmax denominator. **Not directly reusable** — MD-SSL is *not* CVCSPC; the pretext task is different (half-cycle reconstruction vs. anchor-positive-negative phase contrast). But the optimizer wiring and training-loop structure can be borrowed.

**Phase 4 will have to read paper §4 line-by-line with the figures and reconstruct.** No shortcut from the release. Budget the phase accordingly when its turn comes.

---

## (f) Notable observations / pitfalls in the released code (logged for future phases)

- **ImageNet vs. Kinetics normalization.** `dataloader.py` line 96 uses ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` — appropriate because CVCSPC is a 2D image model. We must **not** copy this for the R(2+1)D-18 video path; we use Kinetics-400 stats (RESEARCH.md §4).
- **Bilinear interpolator hardcoded.** `dataloader.py` line 47: the random interpolator selector is commented out, fixed to BILINEAR. `torchvision.io.read_video` already decodes RGB uint8 frames natively, so we don't need PIL resize here, but if we later use PIL-based resizing we should pick BILINEAR explicitly to match.
- **`VideoDataset` class name is misleading.** The CVCSPC class is called `VideoDataset` (`dataloader.py` line 61) but it serves **single image frames** sampled from videos — not clips. Don't get confused if you grep for `VideoDataset` later.
- **Determinism level is incomplete in the release.** CVCSPC's `dataloader.py` only sets `cudnn.deterministic=True` (not `benchmark=False`); `train_test.py` sets both but stops short of `torch.use_deterministic_algorithms(True)` and the `CUBLAS_WORKSPACE_CONFIG` env var. Our determinism is stricter (see PLAN.md `<determinism_checklist>`).
- **Implicit globals in the trainer.** `train_phase` and `test_phase` in `train_test.py` reference `model_CNN`, `model_linear_layers` as closure-captured globals from the `__main__` block. We will not do this — our `run_tiny_epoch` takes the model as a parameter (interfaces block in PLAN.md).
- **No resume code anywhere.** `save_model` writes raw weights only. There is no `load_model`, no epoch-restore, no RNG restore, no resume-aware training loop in the release. Our atomic-checkpoint + auto-resume harness has no upstream antecedent.

---

## References

- **Paper.** Parmar, Gharat, Rhodin. *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment.* ECCV 2022. arXiv:2202.14019.
- **Upstream repo.** https://github.com/ParitoshParmar/Fitness-AQA
- **Phase 1 dataset report.** `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md`
- **Phase 2 research.** `.planning/phases/02-squat-data-pipeline-colab-harness/02-RESEARCH.md` §2 (Code_Release findings), §4 (paper-faithful transforms), §6 (atomic checkpoint), §8 (class imbalance).

---

*Authored as Phase 2 Task 2. Not a module — do not import. Reference only.*
