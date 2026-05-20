# Phase 2: Squat Data Pipeline & Colab Harness — Research

**Researched:** 2026-05-20
**Domain:** PyTorch video data loading, Colab resumable training, R(2+1)D-18 supervised baseline preparation
**Confidence:** HIGH for paper/code; MEDIUM-HIGH for ecosystem; LOW only for the input-resolution gap (paper does not specify).

---

## 1. Summary

**The biggest finding changes the plan:** `Code_Release/motion_disentanglement/` is empty (1-byte README). The only complete training code is `pose_contrastive_learning/self_supervised_learning/` — the **CVCSPC image-side SSL trainer**, not the Squat KIE/KFE video baseline. The supervised baseline of paper §5 has **no released code**. Phase 2 reconstructs from paper text + CVCSPC idioms + R(2+1)D community defaults.

Operational takeaways:

- **Sampler:** uniform-32 across the full clip span for train *and* val/test; train adds ±2-frame jitter. Paper §5 line 421: *"We used 32 frames for all types of errors."*
- **Spatial:** paper's `320→224 center crop` is for the **Waseda 2DCNN path only** (line 495–496); **no in-the-wild R(2+1)D resolution is specified.** Recommend `112×112` (matches torchvision's R2Plus1D_18 Kinetics defaults), parameterized for Phase 3 ablation.
- **Normalization:** Kinetics-400 mean `[0.43216, 0.394666, 0.37645]`, std `[0.22803, 0.22145, 0.216989]` — torchvision-verified, CONTEXT.md-locked.
- **Horizontal flip OFF.** Paper line 576 enumerates strong augs as "rotation, translation, masking image regions, color channel order changing, zooming, blurring" — flip is absent. CVCSPC code has `hori_flip` defined but commented out.
- **Decoder:** `torchvision.io.read_video` for Phase 2 (zero install). Note: deprecated from torchvision 0.22, removed in 0.24. Wrap behind `decode_clip(path)` so TorchCodec swap is one-line.
- **Checkpoint:** `torch.save` (safetensors cannot store optimizer/RNG state). Payload = model + optimizer + scheduler + 4-RNG dict + metrics + `config_hash`. Drive FUSE does not guarantee atomic rename — write-to-temp + verify-by-reload + `os.replace`, then write `latest.txt` last.
- **Imbalance:** paper line 413: *"For imbalanced datasets, we used class weights (in cross-entropy loss) inversely proportional to the class size."* Multi-label generalization: `BCEWithLogitsLoss(pos_weight=Tensor([w_KIE, w_KFE]))`, not `WeightedRandomSampler`.

---

## 2. Code_Release findings

Full repo tree verified via GitHub git-trees API (`git/trees/main?recursive=1`, sha `9a20d751`):

```
Code_Release/
├── ReadMe.md                                    (22 B — "_Work in Progress..._")
├── data_augmentations/
│   ├── ReadMe.MD                                (117 B — one-line)
│   └── image_augmentations.py                   (2,715 B — image-only helpers)
├── motion_disentanglement/
│   └── README.md                                (1 B — empty)
└── pose_contrastive_learning/
    ├── README.md                                (1 B — empty)
    └── self_supervised_learning/
        ├── dataloader.py                        (11,545 B — CVCSPC triplet loader)
        └── train_test.py                        (11,549 B — CVCSPC trainer)
```

That is the **entire** `Code_Release/`. No `train.py` for the supervised baseline. No video dataloader. No R(2+1)D wiring. No `motion_disentanglement` implementation. The released `pose_contrastive_learning/` code is the **image-modality** Squat-Shallow detector (Phase 7), not video Squat KIE/KFE. [VERIFIED]

**Reusable idioms from the CVCSPC trainer:**
- Single `randomseed` applied at module top to `torch`, `torch.cuda`, `random`, `numpy`. `cudnn.deterministic=True; cudnn.benchmark=False`. This is our determinism recipe.
- `optim.Adam(parameters, lr=1e-4)`. Paper §5 confirms: *"ADAM optimizer with an initial learning rate of 1e-4 for 20 epochs with a batch size of 5"* (line 439–440) for MD; *"1e-4 ... 100 epochs ... batch size of 25"* (line 434–435) for CVCSPC.
- Only `state_dict` saved — **no optimizer, no scheduler, no RNG.** We must do better (Colab disconnects = default case).
- Dataloader imports `from opts_exercise_qa import *` — a config module **not in the repo**. Their code is not runnable as-shipped.
- `dataloader.py` augmentation dict has **only `masking` enabled**; `hori_flip`, `translation`, `rotation`, `blurring`, `zooming`, `color_jittering` all commented out. The published numbers were achieved with **less augmentation than the paper text suggests**.

**Must be reconstructed from paper text:**

| Component | Source |
|---|---|
| Squat KIE/KFE video dataloader | Paper §5 (32 frames, Adam 1e-4, class-weighted CE) + CVCSPC idioms |
| R(2+1)D-18 wiring | `torchvision.models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)` |
| Train script | Adapt CVCSPC `train_test.py` structure; BCE instead of contrastive |
| MD half-cycle splitter | Phase 4, not Phase 2 |
| Spatial transforms for video | Paper has no spec — see §4 |

**Implication:** Code_Release is a style guide and partial precedent, not a runnable reference. Every Phase 2 transform decision is a new authored decision justified by the paper text.

---

## 3. Sampler decision

**Train:** uniform 32 indices across `[0, F-1]` via `round(linspace(0, F-1, 32))`, with **±2-frame temporal jitter** per index, clamped. Magnitude is config-tunable (`train_jitter_frames`, default 2).

**Val/test:** uniform 32 indices, **no jitter**, RNG-free.

**Rationale:**
- Paper line 421: *"We used 32 frames for all types of errors."* [CITED: ECCV §5]
- No "random clip" language anywhere. Paper's Waseda discussion (line 498–500) explicitly rejects temporal downsampling that drops information.
- Phase 1 clip lengths (49–404 frames, median ~111) give uniform-32 strides 1.5–13 frames. Paper line 379: *"Our samples were automatically processed to contain a single repetition."* Long-tail clips are a data-quality tail, not the intended distribution.
- Val/test must be RNG-free or the tiny-run bitwise-identical resume test breaks.

**Anti-recommendation:** do **not** use random-clip-of-length-32. That's standard Kinetics recipe but inappropriate for already-trimmed single-rep clips — we want temporal coverage, not augmentation. [ASSUMED — paper doesn't contrast.]

---

## 4. Transform decision

**Output shape per clip:** `Tensor[float32, (3, 32, 112, 112)]` → DataLoader batches to `(B, 3, 32, 112, 112)`.

| Stage | Train | Val/Test |
|---|---|---|
| Decode | `[T, H_raw, W_raw, 3]` uint8 RGB | same |
| Frame select | 32 uniform + ±2 jitter | 32 uniform |
| Resize | short side → 128 (aspect preserved) | same |
| Crop | random 112×112 | center 112×112 |
| Horizontal flip | **OFF** | OFF |
| Rescale | `/ 255.0` | same |
| Normalize | mean `[0.43216, 0.394666, 0.37645]`, std `[0.22803, 0.22145, 0.216989]` | same |
| Permute | `(T, H, W, C) → (C, T, H, W)` | same |

**Rationale:**

- **Normalization values** verbatim from torchvision: *"first rescaled to [0.0, 1.0] and then normalized using mean=[0.43216, 0.394666, 0.37645] and std=[0.22803, 0.22145, 0.216989]"* [VERIFIED: torchvision R2Plus1D_18 docs]
- **Crop 112×112** matches torchvision's R2Plus1D_18 native input. The paper does **not** specify in-the-wild R(2+1)D resolution; the only given resolution (`320→224`) is the 2DCNN/Waseda path. 112 is paper-faithful by Kinetics inheritance; parameterize for Phase 3 ablation if KIE F1 disappoints.
- **Short-side-128 aspect-preserving resize** instead of torchvision's fixed `[128, 171]`. Fitness-AQA's aspect range (width 480, heights {270, 324, 480, 584, 592, 600}) covers landscape, square, portrait. Fixed two-tuple would distort.
- **Flip OFF:** paper line 576 lists strong augmentations without flip; CVCSPC has `# 'hori_flip': {...}` commented in the aug dict. KIE is symmetric to mirror so flipping is label-safe, but training without is paper-faithful. Phase 3 may revisit.
- **Color jitter / rotation / blur / zoom:** paper-mentioned but parameter ranges not given. **Defer to Phase 3.** Phase 2's minimal stack keeps tiny run deterministic and visualization interpretable.

**Channel-ordering reminder:** `torchvision.io.read_video` returns `(T, H, W, C)`. R(2+1)D-18 expects `(B, C, T, H, W)`. The Dataset's `__getitem__` must permute. The shape-strict toy model in the tiny run errors immediately if wrong.

---

## 5. Decoder decision

**Default:** `torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")`.

**Swap trigger:** if profiling shows decode > ~30% of batch time on Colab T4, swap to **TorchCodec** (`torchcodec.decoders.VideoDecoder`).

| Decoder | Status (May 2026) | Colab install | Verdict |
|---|---|---|---|
| `torchvision.io.read_video` | **Deprecated from 0.22, removed in 0.24** [VERIFIED: docs.pytorch.org] | Pre-installed; needs `av` | **Phase 2 default.** Colab ships 0.20.x — safe through Phase 4. |
| `torchcodec` | Official replacement, actively maintained | `pip install torchcodec --index-url=...` | **Swap-target.** Clean indexing API. |
| `decord` | **Inactive** — no PyPI release in >12 months [CITED: snyk.io/advisor] | `pip install decord` fails on modern Colab [CITED: dmlc/decord #213, #240, #273, #290, #366] | **Not viable.** |
| `pyav` directly | Stable, low-level | Pre-installed | Too verbose. |

**Known footguns** (mark inline):
- `read_video` returns `(video, audio, info)` — discard latter two.
- Set `output_format="TCHW"` explicitly; assert shape on first decoded clip.
- Long clips: gate by clip duration to avoid OOM at high res.
- Silence the packed-B-frame `UserWarning` with `warnings.filterwarnings`.

---

## 6. Checkpoint / resume decision

### Payload schema (single `.pt` file via `torch.save`)

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ... | None,
    "rng_state": {
        "python":         random.getstate(),
        "numpy":          np.random.get_state(),
        "torch_cpu":      torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    },
    "metrics_history": list[dict],
    "config_hash":  str,                          # see below
    "config_repr":  dict,                         # debug-readable
    "code_version": str,                          # git HEAD, best-effort
}
```

### Why `torch.save`, not safetensors

`safetensors` only serializes pure tensor data — **cannot** store optimizer state (nested Python scalars), RNG state (tuples), or scalar metadata. Use safetensors for distributing weights, `torch.save` for training-state. PyTorch Lightning + HF Trainer both use `torch.save` for full checkpoints. [CITED: lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html]

### RNG capture for bitwise-identical resume

Must capture all four [VERIFIED: docs.pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html]: `random.getstate/setstate`, `np.random.get_state/set_state`, `torch.get_rng_state/set_rng_state`, `torch.cuda.get_rng_state_all/set_rng_state_all`. Plus `cudnn.deterministic=True; cudnn.benchmark=False; torch.use_deterministic_algorithms(True, warn_only=True)` and env `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS.

**Not in scope for Phase 2:** DataLoader worker RNGs. With `num_workers > 0`, each worker has its own RNG; restoring main-process RNG is insufficient. Phase 2 tiny run uses `num_workers=0`; Phase 3 adds `worker_init_fn`. [ASSUMED: deterministic resume guarantee conditional on `num_workers=0` in Phase 2.]

### Atomic write — Drive FUSE caveat

Google Drive FUSE does **not** guarantee atomic rename [VERIFIED: github.com/linuxmint/cinnamon#13555]:

```python
def atomic_torch_save(payload, target_path):
    tmp = os.path.join(os.path.dirname(target_path),
                       f".tmp_{os.path.basename(target_path)}.{os.getpid()}")
    torch.save(payload, tmp)
    _ = torch.load(tmp, map_location="cpu")   # verify round-trip readable
    os.replace(tmp, target_path)              # POSIX atomic on local; "best-effort" on Drive
```

Layout: `My Drive/FitNova/checkpoints/phase02/{run_name}/epoch_{NNN}.pt` + `latest.txt` (one line, filename). **Write `epoch_NNN.pt` first, then `latest.txt`** — a `latest.txt` pointing at a non-existent file is louder than corrupt-but-readable.

### Compatibility check (`config_hash`)

```python
config_hash = hashlib.sha256(
    json.dumps(config, sort_keys=True, default=str).encode()
).hexdigest()[:16]
```

Hash inputs: backbone + weights identifier, `crop_size`, `num_frames`, `mean`, `std`, `batch_size`, `learning_rate`, `optimizer_name`, `weight_decay`, `seed`, label semantics order. Not hashed: Drive paths, log verbosity, retention.

On mismatch: raise `CheckpointConfigMismatchError` — never auto-overwrite, never auto-reinit.

### Retention

Last 3 `epoch_*.pt` + one `best.pt` (overwritten on new best val metric). Prune **after** new save succeeds. For Phase 2 tiny run, "best" = "lowest train_loss" (single epoch — val metric not yet meaningful).

---

## 7. Edge cases

**Clips < 32 frames:** none exist (Phase 1 min = 49). Defensive: `np.clip(indices, 0, F-1)` so indices saturate if a future clip is shorter.

**Long multi-rep clips (>300 frames, up to 13.5s):** paper says (line 379): *"Our samples were automatically processed to contain a single repetition."* The long tail is a data-quality tail. Three options:

| Option | Effect | Recommendation |
|---|---|---|
| Uniform 32 across full span | Coarse; may straddle reps | **Default** — paper intent |
| Random 32-frame consecutive window | Captures one rep, loses context | Defer Phase 3 |
| Crop middle 96 → uniform-32 | Biases to middle | Defer Phase 3 |

The decoded-batch visualization will surface long-clip cases. If visualized clips clearly span multiple reps, planner-checker flags for Phase 3.

**Aspect after resize:** Phase 1 heights {270, 324, 480, 584, 592, 600} at width 480. After short-side-128 resize, long side ranges 156 (portrait) to 227 (landscape). Center crop 112² **drops ~30% vertical on portrait clips, ~50% horizontal on landscape**. The paper's `320→224` recipe avoids this by aspect-distorting to square — but that would force portrait phone clips through 1:1 squashing, which is worse than dropping side regions. **Decision:** aspect-preserve. Phase-3 ablation candidates: `crop_size=224`, `crop_size=160`.

**MediaPipe is NOT in this pipeline.** CONTEXT.md confirms — raw-pixel CNN only. Phase 2 loader must not import MediaPipe. Legacy `backend/services/mediapipe_config.py` is Phase 5's deletion concern.

---

## 8. Class-imbalance pattern

**Recommended:** `BCEWithLogitsLoss(pos_weight=Tensor([w_KIE, w_KFE]))` with per-class weights from training-split positive counts:

```python
N_train = 1136
pos_KIE = count of train clips where error_knees_inward.json[id] is non-empty
pos_KFE = count of train clips where error_knees_forward.json[id] is non-empty
w_KIE = (N_train - pos_KIE) / max(pos_KIE, 1)   # ≈ 6.0 from Phase 1's ~14% rate
w_KFE = (N_train - pos_KFE) / max(pos_KFE, 1)   # ≈ 0.47 from Phase 1's ~68% rate
pos_weight = torch.tensor([w_KIE, w_KFE], dtype=torch.float32)
```

### Why `pos_weight`, not `WeightedRandomSampler`

[VERIFIED: docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html — *"`pos_weight` must be a tensor with equal size along the class dimension to the number of classes"*]

- `pos_weight` is **multi-label-native**: per-class, independently scales each class's positives. Multi-label BCE is a sum of per-class terms.
- `WeightedRandomSampler` is **single-label-native**. Per-clip weight from joint labels is awkward: 4 combinations with uneven occupancy (~9.5% double-positive, ~24% double-negative) — every heuristic ends up biased.
- **Paper-faithful:** line 413 — `pos_weight` is the multi-label generalization of "class weights inversely proportional to the class size."

### Phase 2 deliverable

Loader **computes** `pos_weight` and exposes it as `SquatKIEKFEDataset.pos_weight`. Phase 2 tiny run does **not** apply it (toy model + unweighted BCE = simplest green smoke test). Phase 3 reads `dataset.pos_weight` and passes to `BCEWithLogitsLoss`. Honors CONTEXT.md's "loss design deferred to Phase 3" while making data Phase-3-ready on day one.

---

## 9. Open questions for the user

1. **Input crop size: 112 or 224?** Paper unspecified for in-the-wild R(2+1)D. Recommend **112** (paper-faithful via torchvision Kinetics defaults; ~4× faster forward); parameterize for Phase 3 ablation. **Awaiting confirm.**
2. **Horizontal flip:** paper omits flip from enumerated lists; CVCSPC code comments it out. Recommend **OFF**. KIE is mirror-symmetric so flipping is label-safe but a divergence. **Awaiting confirm.**
3. **Temporal jitter magnitude:** paper says "strong augmentations" without naming jitter. Recommending **±2 frames**, configurable. **Plausible-but-arbitrary — flagging.**
4. **`num_workers=0` for tiny run determinism?** Sidesteps worker RNG complexity; Phase 3 adds `worker_init_fn`. **Awaiting confirm acceptable for tiny run.**
5. **Retention: hard 3 + best?** Caps Drive at ~4 files per run. **Confirming the simple policy is fine.**
6. **Decoder swap timing.** Recommend benchmarking inside the tiny run and deciding from measured numbers, not estimates.

---

## 10. References

### Primary (HIGH confidence)

- Parmar, Gharat, Rhodin. *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment.* ECCV 2022. [paper PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980104.pdf). §5 (paper lines 406–442 of the extracted text).
- Tran et al. *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* CVPR 2018. arXiv:1711.11248.
- Fitness-AQA Code_Release tree at sha `9a20d751`: [github.com/ParitoshParmar/Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA)
  - [`image_augmentations.py`](https://raw.githubusercontent.com/ParitoshParmar/Fitness-AQA/master/Code_Release/data_augmentations/image_augmentations.py)
  - [`pose_contrastive_learning/.../dataloader.py`](https://raw.githubusercontent.com/ParitoshParmar/Fitness-AQA/master/Code_Release/pose_contrastive_learning/self_supervised_learning/dataloader.py)
  - [`pose_contrastive_learning/.../train_test.py`](https://raw.githubusercontent.com/ParitoshParmar/Fitness-AQA/master/Code_Release/pose_contrastive_learning/self_supervised_learning/train_test.py)
- [torchvision R2Plus1D_18 docs](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r2plus1d_18.html)
- [torchvision read_video docs](https://docs.pytorch.org/vision/main/generated/torchvision.io.read_video.html)
- [torch.nn.BCEWithLogitsLoss docs](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html)
- [torch.cuda.get_rng_state docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html)
- [pytorch/torchcodec README](https://github.com/pytorch/torchcodec)

### Secondary (MEDIUM confidence)

- [Snyk: decord](https://snyk.io/advisor/python/decord) and [Libraries.io decord versions](https://libraries.io/pypi/decord/versions) — inactive maintenance.
- dmlc/decord install failures: [#213](https://github.com/dmlc/decord/issues/213), [#240](https://github.com/dmlc/decord/issues/240), [#273](https://github.com/dmlc/decord/issues/273), [#290](https://github.com/dmlc/decord/issues/290), [#366](https://github.com/dmlc/decord/issues/366).
- [linuxmint/cinnamon#13555](https://github.com/linuxmint/cinnamon/issues/13555) — Drive FUSE non-atomic rename.

### Project-internal (HIGH confidence)

- `.planning/phases/02-squat-data-pipeline-colab-harness/02-CONTEXT.md` — locked decisions.
- `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md` — verified clip/aspect/balance facts.
- `.planning/REQUIREMENTS.md` SQUAT-01, SQUAT-02; `.planning/ROADMAP.md` Phase 2 success criteria.
- `backend/training/aqa/notebooks/01_dataset_eda.py` — jupytext-percent template.

---

## Metadata

**Confidence breakdown:**
- Code_Release findings — **HIGH** (full tree read from raw.githubusercontent.com).
- Paper text (frames, normalization, optimizer, class weights) — **HIGH** (direct quotes).
- Spatial input size (112 vs 224) — **MEDIUM** (paper unspecified; design choice via torchvision default).
- Horizontal flip default OFF — **HIGH** (paper-supported absence + CVCSPC commented-out).
- Decoder — **HIGH** for deprecation timeline; **MEDIUM** for swap-when-needed (no Colab measurement yet).
- Class-imbalance pattern — **HIGH** (paper-cited; `pos_weight` is the standard idiom).
- Drive atomic-write mitigation — **MEDIUM-HIGH** (FUSE behavior documented; verify-by-reload is defensive standard).

**Research date:** 2026-05-20
**Valid until:** 2026-06-20 (torchvision deprecation timeline and TorchCodec API are the closest moving targets; re-verify if Phase 2 pushes past mid-June)
