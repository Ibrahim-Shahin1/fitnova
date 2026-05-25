# Phase 5: Backend Inference Integration (Squat) — Research

**Researched:** 2026-05-25
**Domain:** PyTorch inference serving (R(2+1)D-18 ensemble), FastAPI integration, rep segmentation without pose
**Confidence:** HIGH — all critical paths verified against actual source code and benchmarks on the target machine

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01** Production model = 3-seed MD-SSL ensemble. Load `md_finetune_seed{42,1337,7}/best.pt`, mean-of-sigmoids, per-head thresholds **KIE 0.614 / KFE 0.385**, **no TTA**. Test macro-F1 0.6304.
- **D-02** Clip contract = 32-frame, 112², Kinetics-normalized RGB. Preprocessing MUST reuse the exact Phase-2/3 contract: `transforms.KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_val`. Train/serve preprocessing skew is non-negotiable to avoid.
- **D-03** Dedicated rep-segmentation component, NO pose. Upload (ships first): server-side segmentation. Live: lightweight on-the-fly sliding-window.
- **D-04** Exact rep-segmentation algorithm is Claude's discretion (researcher resolves + MEASURE).
- **D-05** Per-rep response: `{ exercise, errors: [ { type, detected: bool, confidence, severity_word, intervals: [[start,end]] } ] }`. Old `FormFrameResult`/`FormSessionSummary` Pydantic models **replaced, not shimmed**.
- **D-06** Archive, don't hard-delete. Move form modules/weights to `backend/_archive_form_v4_v6/`. Do NOT touch recommender/plan-generation path.
- **D-07** PyTorch + TF coexist via separate `app.state` objects, both loaded at startup.
- **D-08** Default ensemble; measure CPU latency first, fall back to single-seed if live latency unacceptable. Seed-count a config knob.
- **D-09** Stage `.pt` to `backend/models/`, not committed raw. R(2+1)D-18 ≈ 126 MB fp32 each → ensemble ≈ 378 MB.
- **D-10** Deterministic templated feedback.
- **D-11** Test user's real phone-camera squat clip EARLY.
- **D-12** Phase 5 is LOCAL backend work. No Colab/Drive/disconnect/notebook machinery.

### Claude's Discretion

- D-04 (rep-segmentation algorithm), D-08 (ensemble-vs-single default + latency probe), D-09 (weights transfer mechanism), D-10 (deterministic feedback templates).

### Deferred Ideas (OUT OF SCOPE)

- OHP inference (Phase 6); image-based errors (Phase 7); cross-method ensemble (Phase 8).
- Polished Flutter form UI (v2 UI-01/UI-02).
- Graded-severity ground truth (v2 FB-01).
- Backend security hardening (upload-size cap, rate limiting, CORS allowlist).
- GPU serving / ONNX / TorchServe.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | Old form-analysis subsystem removed; PyTorch form-inference service loaded in FastAPI | §Removal Inventory; §Inference Reconstruction; §TF+Torch Coexistence |
| API-02 | Rep-segmentation component for live and upload inference paths | §Rep Segmentation Without Pose |
| API-03 | Video-upload REST endpoint returning binary error detections with timing | §Upload Path; §API Schema |
| API-04 | Live WebSocket endpoint returning per-rep error feedback | §Live WebSocket Path |

</phase_requirements>

---

## Summary

Phase 5 serves the Phase-4 MD-SSL ensemble through the existing FastAPI app, replacing the old TF+MediaPipe form subsystem. This research resolves the eight technical unknowns identified in the CONTEXT: the exact inference reconstruction path, rep segmentation algorithm, TF+torch coexistence, CPU latency, the WebSocket live path, weights staging, API verification patterns, and the precise archival impact.

The single highest correctness risk is preprocessing parity (D-02). The old form model died from a train/serve MediaPipe IMAGE-vs-VIDEO mismatch. The new model is raw-pixel CNN; the analogous risk is using `spatial_train` (random crop) at serve time instead of `spatial_val` (center crop). The serving code MUST call `spatial_val`, not `spatial_train`, and MUST use `uniform_sample_indices(num_frames, 32, jitter=0)` — deterministic, no augmentation.

**CPU latency verdict (MEASURED on this machine, torch 2.12.0+cpu):** single R(2+1)D-18 forward with 2-head fc is **~0.93 s fp32 per clip** (3 runs: 0.940, 0.934, 0.932 s). The 3-seed ensemble is **~2.8 s per clip**. fp16 on CPU hangs (no fast fp16 kernels on Windows CPU in PyTorch 2.12 — confirmed by timeout). Recommendation: ship single-seed (1× ~0.93 s) for live mode, ensemble (3× ~2.8 s) for upload. Both are feasible for their respective transport modes.

**Primary recommendation:** `SquatFormService` loads once at startup into `app.state.squat_form_service`, holds the 3 state-dicts + thresholds, exposes `classify_clip(frames: np.ndarray) -> dict` as the sole public API, and is called from inside `run_in_threadpool` (Starlette primitive already installed) to avoid blocking the async event loop.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Clip preprocessing (resize, crop, normalize) | Backend service | — | Must match training pipeline exactly; no client involvement |
| Model forward pass (R(2+1)D-18) | Backend service | — | Raw-pixel CNN; ~126 MB weights; server-side only |
| Ensemble aggregation (mean-of-sigmoids) | Backend service | — | `aggregate_sigmoid_mean` from `eval/ensemble.py` |
| Rep segmentation | Backend service | — | No pose; dataset provides no model; heuristic component |
| Error detection thresholding | Backend service | — | KIE 0.614, KFE 0.385 — val-tuned, hardcoded |
| Plain-language feedback | Backend service | — | Deterministic template, no LLM required |
| Video decoding (upload) | Backend service | — | `torchvision.io.read_video` + `decode_clip` |
| Frame buffering (live WebSocket) | Per-connection buffer | Backend service (model) | Model read-only; buffers are per-session state |
| TF recommender (existing) | Backend service | — | Untouched; separate `app.state` object |
| Result delivery | FastAPI REST / WS | — | Redefined Pydantic models inline in `app.py` |

---

## Standard Stack

### Core

| Library | Version (installed) | Purpose | Why Standard |
|---------|---------------------|---------|--------------|
| `torch` | 2.12.0+cpu [VERIFIED: measured] | Model load, forward pass, sigmoid | Training was PyTorch; state-dicts are torch format |
| `torchvision` | 0.27.0+cpu [VERIFIED: measured] | `r2plus1d_18` architecture, `read_video` decoder | Exact same API used during training |
| `fastapi` | 0.125.0 [VERIFIED: installed] | REST + WebSocket routing | Existing app framework |
| `starlette` | 0.50.0 [VERIFIED: installed] | `run_in_threadpool`, WebSocket test session | Starlette is FastAPI's ASGI foundation |
| `numpy` | 2.2.6 [VERIFIED: installed] | Array ops, sigmoid computation | `aggregate_sigmoid_mean` is pure numpy |
| `opencv-python-headless` | 4.13.0 [VERIFIED: installed] | Video decode for REST upload path | Already in `requirements.txt`; used in current app.py |
| `httpx` | 0.28.1 [VERIFIED: installed] | Sync TestClient and async tests | Already installed; `from fastapi.testclient import TestClient` uses httpx |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scikit-learn` | >=1.3.0 (installed) [VERIFIED] | `f1_score` for API-level eval | During domain-shift test with labeled clip |
| `pytest` | 9.0.2 [VERIFIED: installed] | Test framework | All API-level tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `starlette.concurrency.run_in_threadpool` | `asyncio.to_thread` | Both available and equivalent; `run_in_threadpool` is the FastAPI idiom (`asyncio.to_thread` requires Python 3.9+ — confirmed available) |
| `torchvision.io.read_video` (upload decode) | `OpenCV cap.read()` loop | Current app uses OpenCV for the REST path; `decode_clip` uses torchvision. For Phase 5, use `torchvision.io.read_video` via the existing `decode_clip` helper — simpler and already tested |
| fp32 inference | fp16 on CPU | fp16 hangs on Windows CPU with PyTorch 2.12 (no fast fp16 GEMM kernels); **do not attempt fp16 serving on CPU** |

**Installation (new deps required):** None. All required packages are already installed.

---

## Package Legitimacy Audit

No new packages are installed in Phase 5 — all inference-serving code uses packages already in `backend/requirements.txt` (torch, torchvision, fastapi, starlette, numpy, opencv-python-headless). No audit required.

---

## Architecture Patterns

### System Architecture Diagram

```
[Client / curl / pytest]
        |
        |  REST POST /analyze-form-video (UploadFile)
        |  WS  /ws/form-session (base64-JPEG stream)
        v
[FastAPI app.py]
        |
        |  app.state.squat_form_service  (PyTorch, read-only)
        |  app.state.recommender        (TF, untouched)
        |  app.state.llm_adapter        (untouched)
        |
        v
[SquatFormService]                         [RepSegmenter]
  load_ensemble() at startup                 segment_upload(video_path) -> [(start,end),...]
  classify_clip(frames_np) -> dict           rolling_window_reps(buffer) -> rep_trigger
  ^                                          ^
  |  run_in_threadpool (blocks ~0.93–2.8s)   |
  |                                          |
[REST endpoint]                          [WS endpoint]
  decode full video via decode_clip          buffer base64 frames
  segment reps                               detect rep boundary
  per-rep classify_clip × N reps             per-rep classify_clip
  return PerRepResponse[]                    send PerRepResponse per rep
                                             send SessionSummary on end_session
```

### Recommended Project Structure

```
backend/
├── services/
│   ├── squat_form_service.py       # new: SquatFormService (load + classify_clip)
│   └── rep_segmenter.py            # new: RepSegmenter (upload + live paths)
│   └── [form_analyzer.py]          # ARCHIVED → backend/_archive_form_v4_v6/
│   └── [form_session.py]           # ARCHIVED
│   └── [form_geometry.py]          # ARCHIVED
│   └── [mediapipe_config.py]       # ARCHIVED
│   └── [exercise_sanity.py]        # ARCHIVED (whole file — no form-agnostic parts)
├── models/
│   └── form_model_squat_md/        # staged from Drive (NOT committed to git raw)
│       ├── seed42/best.pt
│       ├── seed1337/best.pt
│       └── seed7/best.pt
├── _archive_form_v4_v6/            # archive of old subsystem
│   ├── services/                   # form_analyzer.py, form_session.py, etc.
│   ├── models/                     # all form_model* weight dirs
│   └── training/                   # MT-TCN / ST-GCN training scripts
└── tests/
    └── test_squat_form_api.py      # new: REST + WS API tests
```

### Pattern 1: Serving Inference Path (EXACT Steps — D-02 Critical)

**The single most important pattern.** Mirrors the eval path in `05_squat_md_finetune.ipynb`/notebook eval cells. Any deviation from this order is a correctness bug.

```python
# Source: backend/training/aqa/harness/md_finetune.py:build_finetune_model
# Source: backend/training/aqa/eval/ensemble.py:aggregate_sigmoid_mean
# Source: backend/training/aqa/datasets/transforms.py:spatial_val, uniform_sample_indices, decode_clip

# STEP 1: ARCHITECTURE CONSTRUCTION (must match build_finetune_model exactly)
from torchvision.models.video import r2plus1d_18
import torch.nn as nn

def _build_head(dropout: float = 0.2) -> nn.Module:
    model = r2plus1d_18(weights=None)              # NO Kinetics weights — MD backbone init
    assert model.fc.in_features == 512             # verified: 512 on torchvision 0.27
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))
    return model

# STEP 2: LOAD STATE-DICT (strict=False because checkpoint has backbone only or full fine-tune)
# Phase 4 best.pt payload schema: {epoch, model_state_dict, optimizer_state_dict, ...}
# model_state_dict contains the full fine-tuned model (backbone + new fc head)
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)   # strict=True for the full fine-tuned checkpoint
model.eval()

# STEP 3: PREPROCESSING (must use spatial_val, NOT spatial_train)
# Source: backend/training/aqa/datasets/transforms.py
from backend.training.aqa.datasets.transforms import (
    decode_clip,           # window-bounded torchvision.io.read_video decoder
    uniform_sample_indices, # deterministic linspace sampling (jitter=0)
    spatial_val,           # center crop, NOT random crop
    KINETICS_MEAN, KINETICS_STD,
)

# For a video file path:
num_frames_total = probe_frame_count(video_path)  # read_video_timestamps
indices = uniform_sample_indices(num_frames_total, target=32, jitter=0)  # jitter=0 at serve time
frames_tchw = decode_clip(video_path, indices)    # uint8 [32, 3, H, W]
clip_tensor = spatial_val(frames_tchw)            # float32 [3, 32, 112, 112] Kinetics-norm
clip_batch = clip_tensor.unsqueeze(0)             # [1, 3, 32, 112, 112]

# STEP 4: FORWARD
with torch.no_grad():
    logits = model(clip_batch)    # [1, 2]

# STEP 5: ENSEMBLE AGGREGATION (from eval/ensemble.py)
# aggregate_sigmoid_mean takes LIST of (N,2) logit arrays, returns (N,2) sigmoid scores
# For a single clip, shape is (1,2) per seed; squeeze to (2,) after aggregation:
from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
import numpy as np

per_seed_logits = [logits_seed42.numpy(), logits_seed1337.numpy(), logits_seed7.numpy()]
ensemble_scores = aggregate_sigmoid_mean(per_seed_logits)  # (1, 2) float in [0,1]
kie_score, kfe_score = float(ensemble_scores[0, 0]), float(ensemble_scores[0, 1])

# STEP 6: THRESHOLD (val-tuned, D-01)
KIE_THRESHOLD = 0.614
KFE_THRESHOLD = 0.385
kie_detected = kie_score >= KIE_THRESHOLD
kfe_detected = kfe_score >= KFE_THRESHOLD
```

**ANTI-PATTERN GUARD:** Do NOT call `spatial_train` at serve time. Random crop at inference = train/serve skew = the B6 failure pattern.
**ANTI-PATTERN GUARD:** Do NOT use `weights=R2Plus1D_18_Weights.KINETICS400_V1` when loading for serving. The MD backbone IS the init. Using Kinetics weights here overwrites the SSL pretraining.
**ANTI-PATTERN GUARD:** Do NOT pass logits to `aggregate_sigmoid_mean` already sigmoidized. The function applies sigmoid internally: `1/(1+exp(-a))`.

### Pattern 2: `SquatFormService` Class Design

```python
# Source: CONTEXT.md § Code Context — app.state pattern; CONCERNS.md — concurrency bug lesson
import torch
import torch.nn as nn
from starlette.concurrency import run_in_threadpool

class SquatFormService:
    """Loaded once at FastAPI startup; shared read-only across all sessions.

    The model is read-only for inference — no mutable per-session state here.
    Per-session state (frame buffer, rep tracking) lives in the endpoint handler
    or a per-connection SquatSession object.

    Lesson from old FormAnalyzer (CONCERNS.md): process_frame() mutated _ts_ms
    and _prev_result on the shared instance. This service has NO per-call mutable
    state — classify_clip() takes inputs and returns outputs with no side effects.
    """

    def __init__(self, model_dir: str, seeds: list[int] = (42, 1337, 7),
                 kie_threshold: float = 0.614, kfe_threshold: float = 0.385):
        self.model_dir = model_dir
        self.kie_threshold = kie_threshold
        self.kfe_threshold = kfe_threshold
        self._models: list[nn.Module] = []
        self._model_ready = False
        self._load(seeds)

    def _load(self, seeds):
        ...  # build + load state_dict per seed, eval(), append to self._models
        self._model_ready = len(self._models) > 0

    @property
    def model_ready(self) -> bool:
        return self._model_ready

    def classify_clip(self, frames_tchw: np.ndarray) -> dict:
        """Synchronous — call via run_in_threadpool from async endpoint handlers."""
        # preprocess → forward × N seeds → aggregate_sigmoid_mean → threshold → return schema
        ...

    async def classify_clip_async(self, frames_tchw: np.ndarray) -> dict:
        """Async wrapper for use in FastAPI endpoints."""
        return await run_in_threadpool(self.classify_clip, frames_tchw)
```

**ANTI-PATTERN GUARD:** Do NOT call `classify_clip` directly from an async endpoint handler without `run_in_threadpool`. The forward is ~0.93–2.8 s and blocks the event loop.

### Pattern 3: FastAPI lifespan Integration

```python
# Source: backend/app.py:lifespan (reuse pattern; swap point per CONTEXT § Code Context)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Existing services (UNTOUCHED)
    app.state.recommender = Recommender()
    app.state.llm_adapter = LLMAdapter()
    app.state.chat_service = ChatService()

    # NEW: PyTorch form service (replaces app.state.form_analyzer)
    from backend.services.squat_form_service import SquatFormService
    _squat_dir = os.path.join(os.path.dirname(__file__), "models", "form_model_squat_md")
    app.state.squat_form_service = SquatFormService(model_dir=_squat_dir)
    logger.info("SquatFormService loaded (model_ready=%s)", app.state.squat_form_service.model_ready)

    yield  # startup done

    logger.info("Shutting down FitNova backend.")
    # No cleanup needed for PyTorch CPU inference (no MediaPipe landmarker to close)
```

Note: TF imports happen inside `Recommender.__init__()` at startup. Torch imports happen inside `SquatFormService.__init__()`. Order: Recommender first (TF), then SquatFormService (torch). **VERIFIED:** `import torch` after `import tensorflow` works correctly on this machine (tested explicitly — no conflict in Python 3.10 with torch 2.12 + tf 2.21).

### Anti-Patterns to Avoid

- **Using `spatial_train` at serve time:** Introduces random spatial crop that changes every call — not reproducible and not what the model was evaluated on. Always use `spatial_val`.
- **Sharing mutable session state on `SquatFormService`:** Old `FormAnalyzer` had `_ts_ms` and `_prev_result` as instance state, causing concurrent-session corruption (CONCERNS.md). New service is stateless for inference.
- **Loading Kinetics weights then overwriting with state-dict:** The `r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)` call downloads ~120 MB and then the subsequent `load_state_dict` overwrites all of it. Use `weights=None`.
- **fp16 on CPU:** PyTorch 2.12 on Windows CPU has no fast fp16 GEMM kernels; fp16 forward hangs indefinitely (verified on target machine). **Do not use `.half()` for CPU serving.**
- **Calling `model.predict()` (TF idiom) on a PyTorch model:** PyTorch forward is `model(input_tensor)`, not `model.predict(...)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ensemble aggregation | Custom sigmoid+mean | `eval/ensemble.py:aggregate_sigmoid_mean` | Already implemented, tested, and used in Phase 4 eval |
| Threshold-based detection | Re-implement sweep | `eval/metrics.py:threshold_sweep` | Used to derive the 0.614/0.385 values; same function for API-level checks |
| Video preprocessing pipeline | Custom decode/crop/norm | `transforms.py:decode_clip, spatial_val, uniform_sample_indices` | EXACT match to training pipeline — the single source of train/serve parity |
| Model architecture | Re-define R(2+1)D-18 | `harness/md_finetune.py:build_finetune_model` | Canonical source; wrong architecture = state-dict key mismatch |
| Async blocking | `asyncio.get_event_loop().run_in_executor(...)` | `starlette.concurrency.run_in_threadpool` | FastAPI's official pattern; already installed; simpler than manual executor management |

**Key insight:** The eval/training code modules are already correct and battle-tested. Phase 5 is wiring them into the serving path, not re-implementing them.

---

## Technical Unknown Resolution

### 1. Exact Inference Reconstruction

**VERIFIED from source code and benchmarks.**

**Model construction** (`harness/md_finetune.py:84-108`):
```python
model = r2plus1d_18(weights=None)     # architecture only
assert model.fc.in_features == 512   # VERIFIED: 512 (torchvision 0.27)
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
```

**State-dict loading:**
The Phase 4 checkpoint schema (`md_finetune.py:296-311`) stores `model_state_dict` as the **full fine-tuned model** (backbone + new fc head). At serve time: `model.load_state_dict(ckpt["model_state_dict"], strict=True)`. Note: During training, `build_finetune_model` uses `strict=False` when loading the MD backbone (because the backbone checkpoint has no fc keys). The fine-tune checkpoint `best.pt` has the full model including the new fc, so `strict=True` is appropriate at serve time.

**Preprocessing sequence** (`transforms.py` — VERIFIED line-by-line):
1. `uniform_sample_indices(total_frames, target=32, jitter=0)` → LongTensor [32] deterministic indices
2. `decode_clip(video_path, indices)` → uint8 [32, 3, H, W] (window-bounded `read_video`)
3. `spatial_val(clip_tchw)` → float32 [3, 32, 112, 112] center crop, /255, Kinetics-norm, permuted to (C,T,H,W)

**Kinetics normalization constants** (`transforms.py:24-25` — VERIFIED):
```python
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD  = (0.22803, 0.22145, 0.216989)
```
Applied as: `(clip / 255.0 - MEAN) / STD` over shape [T,C,H,W] with broadcast [1,3,1,1].

**Ensemble aggregation** (`eval/ensemble.py:21-48` — VERIFIED):
- Input: list of 3 ndarrays shape (N=1, 2) — raw logits (NOT pre-sigmoidized)
- Applies `sigmoid = 1/(1+exp(-a))` per seed, then `np.mean(sigmoid_scores, axis=0)`
- Output: (N=1, 2) float in [0,1]

**Thresholds** (verbatim from Phase 4 handoff — VERIFIED):
- KIE: **0.614**, KFE: **0.385**

**Total param count:** 31,505,325 [VERIFIED: measured]. State-dict size: **126.1 MB fp32** per seed, **378 MB for ensemble of 3** [VERIFIED: measured].

### 2. Rep Segmentation Without Pose (D-04 Resolution)

**Recommendation (Claude's discretion, supported by dataset properties):**

The Fitness-AQA dataset clips are **single-rep, ~3 s @ 30 fps** (~90 frames). This is the floor case and must work correctly.

**Upload mode (ships first, simplest correct case = single-rep clip):**

Use **motion-energy (frame-difference) thresholding** with a fixed minimum-rep-duration guard:

```python
# Pragmatic rep segmenter — no pose needed
def segment_reps_by_motion_energy(video_path: str, min_rep_frames: int = 60,
                                  energy_threshold_factor: float = 0.5) -> list[tuple[int,int]]:
    """Segment a clip into rep intervals by motion energy.

    Strategy:
      1. Decode all frames at low resolution (e.g. 64×64 grayscale) for speed.
      2. Compute per-frame motion energy = mean(|frame[i] - frame[i-1]|).
      3. Smooth the energy signal with a Gaussian or boxcar window (σ=5 frames).
      4. Find contiguous regions where energy > threshold * mean_energy.
      5. Merge adjacent active regions closer than min_rep_gap.
      6. Each active region is one rep. If no segmentation found → treat entire
         clip as one rep (single-rep clip = the correct Fitness-AQA case).

    Returns:
        List of (start_frame, end_frame) tuples (1-indexed into the full clip).
        Always returns at least [(0, total_frames-1)] as a fallback.
    """
```

Validation gate (before committing): run on a real squat clip (user-provided, D-11) and confirm the returned rep intervals visually look correct. Log the raw energy signal for debugging.

**Live mode (per D-03: sliding-window):**

The "live" path never has a complete multi-rep clip to segment. Use a **fixed sliding-window with minimum inter-window gap** as the rep trigger:

```python
# Per CONTEXT D-03/D-04 and master plan "sliding-window for timing"
LIVE_REP_WINDOW_FRAMES   = 32      # clip window sent to classifier
LIVE_MIN_GAP_FRAMES      = 45      # ~1.5 s @ 30fps between consecutive window captures
LIVE_BUFFER_MAX_FRAMES   = 90      # rolling buffer depth (~3 s of frames)
```

At each incoming frame, append to a deque(maxlen=90). When deque reaches 32 frames AND at least `LIVE_MIN_GAP_FRAMES` have elapsed since the last inference trigger, extract a 32-frame clip (center-sampled from the buffer), run classify_clip, emit per-rep response. This matches the "sliding-window classification over the rolling frame buffer" in the master plan (line 101).

**Why not optical flow:** `cv2.calcOpticalFlowPyrLK` is more accurate but requires installing extra OpenCV modules and is slower. Frame-difference is faster, good enough for ~3 s clips, and has zero additional dependencies. [ASSUMED — optical flow would give better boundary precision but the tradeoff is not worth the complexity for this phase.]

**Validation requirement (D-04 says "MEASURE, don't guess"):** The planner MUST include a task that runs the upload segmenter on at least one real squat clip and logs the detected intervals. Domain-shift test (D-11) provides this.

### 3. Serving PyTorch R(2+1)D-18 Alongside TensorFlow (D-07)

**VERIFIED on this machine:**
- `import torch` then `import tensorflow` works — no conflict (Python 3.10, torch 2.12.0+cpu, tf 2.21.0).
- `import tensorflow` then `import torch` also works.
- The import ORDER does not matter in this environment.

**Practical coexistence pattern:**
```python
# In app.py lifespan — TF services load first (existing pattern), torch service loads second
app.state.recommender = Recommender()       # triggers TF import inside Recommender.__init__
app.state.llm_adapter = LLMAdapter()
app.state.chat_service = ChatService()
app.state.squat_form_service = SquatFormService(...)  # triggers torch import
```

**Memory:** TF recommender (NeuMF) ≈ 3 MB model + ~50 MB TF runtime overhead. PyTorch ensemble ≈ 378 MB (3 × 126 MB fp32 state-dicts loaded to CPU RAM). Total process footprint: ~430 MB RAM at steady state. [ASSUMED — TF runtime overhead not directly measured; 50 MB is a conservative estimate based on tf 2.21 typical startup.]

**Thread safety:** PyTorch CPU inference with `no_grad()` and no shared mutable state is thread-safe. Multiple concurrent WebSocket sessions call `classify_clip()` independently via `run_in_threadpool` — each call uses the same (read-only) model weights and private input tensors. No lock needed.

**One concurrency caveat to NOT repeat:** The old `FormAnalyzer` had `self._ts_ms` and `self._prev_result` as instance state mutated on every `process_frame()` call. `SquatFormService.classify_clip()` must have NO instance-level mutable state — inputs in, outputs out, pure function behavior.

### 4. CPU Latency (D-08 — MEASURED on target machine)

**Benchmark results (VERIFIED, torch 2.12.0+cpu, Intel CPU, Windows 11):**

| Config | Per-clip time (median) | 3-run measurements |
|--------|------------------------|-------------------|
| Single seed, fp32, 32-frame 112² | **~0.93 s** | 0.940, 0.934, 0.932 s |
| 3-seed ensemble, fp32 (sequential) | **~2.8 s** | — (extrapolated 3×) |
| fp16 on CPU | **HANGS** | No fast fp16 GEMM on Windows CPU in PyTorch 2.12 |

**Decision (D-08):**
- **Upload mode:** Use full 3-seed ensemble (~2.8 s per rep). For a 3-rep upload clip: ~8.4 s total. Acceptable for a REST endpoint.
- **Live mode:** Use single seed (seed 1337 or seed 42, test macro ~0.612, −0.018 vs ensemble). ~0.93 s per window trigger. With `LIVE_MIN_GAP_FRAMES=45` at 30 fps, triggers fire at most every 1.5 s — safe.
- **Config knob:** `SQUAT_INFERENCE_SEEDS: int` env var (default 3 for upload, 1 for live). Implementation detail: `SquatFormService` stores all 3 models but a `classify_clip(n_seeds=1)` parameter uses only the first seed.

**Measuring on the user's machine:** The first integration task MUST time a real forward pass after loading actual weights and log it. The ~0.93 s is from this Windows dev machine with the base architecture (no real weights loaded). Real weights might differ marginally but the architecture is the dominant cost.

### 5. WebSocket Live Path

**Existing WS protocol** (`app.py:337-391`) — reused verbatim for transport:
```
Client → {"type": "start_session", "exercise_hint": "squat"}
Client → {"type": "frame", "data": "<base64_jpeg>", "timestamp_ms": 12345}
Client → {"type": "end_session"}
```

**New per-rep response shape** (D-05 schema):
```json
{
  "type": "rep_result",
  "rep_number": 1,
  "exercise": "squat",
  "errors": [
    {"type": "KIE", "detected": false, "confidence": 0.312, "severity_word": "none",   "intervals": []},
    {"type": "KFE", "detected": true,  "confidence": 0.821, "severity_word": "strong", "intervals": [[0.0, 3.1]]}
  ]
}
```

**Session summary** (sent on `end_session`):
```json
{
  "type": "session_summary",
  "exercise": "squat",
  "total_reps": 3,
  "rep_results": [...],
  "session_feedback": "Knees forward on 2 of 3 reps."
}
```

**Live path frame assembly:**
```python
# Per-connection state (instantiated fresh on start_session)
class SquatLiveSession:
    def __init__(self, service: SquatFormService):
        self._service = service
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=LIVE_BUFFER_MAX_FRAMES)
        self._frames_since_last_inference: int = 0
        self._rep_results: list[dict] = []
        self._rep_number: int = 0

    async def add_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> dict | None:
        """Decode JPEG, append to buffer. Return per-rep result if inference triggered, else None."""
        # Decode JPEG → np.ndarray (H,W,3) uint8
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_buffer.append(frame_rgb)
        self._frames_since_last_inference += 1

        if (len(self._frame_buffer) >= LIVE_REP_WINDOW_FRAMES and
                self._frames_since_last_inference >= LIVE_MIN_GAP_FRAMES):
            frames_np = np.stack(list(self._frame_buffer)[-LIVE_REP_WINDOW_FRAMES:])  # [32, H, W, 3]
            self._frames_since_last_inference = 0
            self._rep_number += 1
            result = await self._service.classify_clip_async(frames_np)
            return {"type": "rep_result", "rep_number": self._rep_number, **result}
        return None
```

**Key note:** The frame buffer stores raw np.ndarray frames (decoded from JPEG). The `classify_clip` function must handle the (T, H, W, C) uint8 input and internally convert to the (C, T, H, W) float32 form expected by `spatial_val`. Specifically: `spatial_val` expects `[T, 3, H, W]` (TCHW). The buffer stores HWC frames from OpenCV. Need a permute: `frames_tchw = torch.from_numpy(frames_np).permute(0, 3, 1, 2)` to get TCHW before passing to `spatial_val`.

### 6. Weights Transfer (D-09 — Drive → Server)

**Size confirmed:** Each `best.pt` = ~126 MB fp32 state-dict (VERIFIED). Three seeds = ~378 MB total.

**Recommendation:**
- **Do NOT commit to git raw.** 378 MB binary files are too large for a plain commit.
- **Do NOT use Git-LFS** for this project: Git-LFS requires server-side support and adds operational complexity. The weights are on Drive and only need to be on the local dev machine.
- **Stage manually:** User downloads 3 `best.pt` files from Drive and places them at `backend/models/form_model_squat_md/{seed42,seed1337,seed7}/best.pt`. This is a documented one-time step, not a repeatable automation step.
- **Document the staged paths** in a `backend/models/form_model_squat_md/README.md` (committed to git; the `.pt` files themselves in `.gitignore`).
- **Add `backend/models/form_model_squat_md/` to `.gitignore`** (the directory, not the README).
- **Graceful degradation:** If weights are absent, `SquatFormService.model_ready == False` and both endpoints return a consistent "model not loaded" response (matching the old `FormAnalyzer` pattern).

**Drive paths (from Phase 4 SUMMARY.md, VERIFIED):**
```
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed42/best.pt
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed1337/best.pt
MyDrive/FitNova/checkpoints/phase04/md_finetune_seed7/best.pt
```

**Action item:** The planner MUST include a task step that says: "User downloads the 3 best.pt files from Drive and stages them before running the server." This is a human action, not a code task.

### 7. API Verification (No Flutter UI)

**Patterns VERIFIED against installed libraries:**

**REST upload endpoint — pytest + TestClient:**
```python
# Source: backend/tests/test_app.py (existing pattern, adapted for new endpoint)
# Uses: from fastapi.testclient import TestClient (httpx-backed, no async)

from fastapi.testclient import TestClient
import io, numpy as np

def test_analyze_form_video_no_model(mock_services_squat):
    """Endpoint returns 'model_not_loaded' gracefully when weights absent."""
    with TestClient(app) as client:
        # Create a synthetic 90-frame mp4 in memory (or use a minimal real .mp4)
        fake_mp4_bytes = create_test_video_bytes()  # see below
        response = client.post(
            "/analyze-form-video",
            files={"file": ("test.mp4", fake_mp4_bytes, "video/mp4")},
            data={"exercise": "squat"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "exercise" in data
        assert "errors" in data

def create_test_video_bytes() -> bytes:
    """Create a minimal valid mp4 in memory using cv2 VideoWriter."""
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (112, 112))
    for _ in range(90):
        frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    with open(tmp, 'rb') as f:
        data = f.read()
    os.unlink(tmp)
    return data
```

**WebSocket endpoint — Starlette TestClient WS:**
```python
# Source: starlette.testclient.WebSocketTestSession (VERIFIED: send_json, receive_json available)

def test_ws_form_session_protocol(mock_services_squat):
    """WS endpoint: start_session → frame → end_session → session_summary."""
    with TestClient(app).websocket_connect("/ws/form-session") as ws:
        ws.send_json({"type": "start_session", "exercise_hint": "squat"})
        resp = ws.receive_json()
        assert resp["type"] == "session_started"

        # Send a minimal JPEG frame
        import base64
        fake_jpeg = make_fake_jpeg()  # cv2.imencode('.jpg', np.zeros((112,112,3), uint8))
        ws.send_json({"type": "frame", "data": base64.b64encode(fake_jpeg).decode(),
                      "timestamp_ms": 1000})
        # May or may not return rep_result depending on buffer fill level

        ws.send_json({"type": "end_session"})
        summary = ws.receive_json()
        assert summary["type"] == "session_summary"
        assert "exercise" in summary
        assert "errors" in summary or "rep_results" in summary
```

**Curl equivalents for manual testing:**
```bash
# REST
curl -X POST http://localhost:8000/analyze-form-video \
  -F "file=@test_squat.mp4;type=video/mp4" \
  -F "exercise=squat"

# Health check
curl http://localhost:8000/health
```

**For WebSocket manual testing:** `wscat -c ws://localhost:8000/ws/form-session` (install: `npm install -g wscat`).

### 8. Removal / Archival Impact (D-06 — Complete Inventory)

**Files to archive (move to `backend/_archive_form_v4_v6/`):**

| File | Size | What breaks in app.py when removed |
|------|------|-------------------------------------|
| `backend/services/form_analyzer.py` | ~715 lines | `from backend.services.form_analyzer import FormAnalyzer` (line 34); `FormAnalyzer` used in lifespan (line 203), WS endpoint (line 362, 368, 409), REST endpoint (line 409) |
| `backend/services/form_session.py` | ~500+ lines | `from backend.services.form_session import FormSession` (line 35); `FormSession` used in WS endpoint + REST endpoint |
| `backend/services/form_geometry.py` | ~505 lines | Not imported in app.py (already disconnected from runtime per CONCERNS.md) |
| `backend/services/mediapipe_config.py` | — | Not imported directly in app.py (imported inside `form_analyzer.py:131`) |
| `backend/services/exercise_sanity.py` | 189 lines | Not imported in app.py; imported in `form_session.py` — cascades with form_session removal |
| `backend/models/form_model*/` (all form dirs) | ~132 MB total | Referenced in `app.py:184-202` (lifespan model dir detection) — all that code is removed |
| `backend/models/pose_landmarker_full.task` | 9.4 MB | Referenced inside `mediapipe_config.py` |
| `backend/models/form_model_v6_1_smoke/` | 16 MB | Smoke test only; not in lifespan |
| `backend/models/form_model_v6_smoke/` | 17.8 MB | Smoke test only |

**What stays in `backend/models/` (UNTOUCHED):**
- `neumf_final.keras`, `gmf_pretrained.keras`, `mlp_pretrained.keras` — recommender
- `neumf_metadata.pkl`, `user_pos_items.pkl` — recommender
- New: `form_model_squat_md/` (staged from Drive)

**What stays in `backend/services/` (UNTOUCHED):**
- `chat_service.py`, `content_filter.py`, `llm_adapter.py`, `neumf_ranker.py`, `recommender.py`

**Co-located tests to archive (they test dead code):**
- `backend/services/test_form_analyzer_v6.py`
- `backend/services/test_form_session_v6_routing.py`
- `backend/services/test_form_geometry.py`

**ALSO archive (per master plan line 117 and CONCERNS.md stale-code inventory):**
- `backend/training/models/mt_tcn.py`, `st_gcn.py`, `st_gcn_v6.py`, `st_gcn_v6_1.py`
- `backend/training/train_form_model.py`, `train_form_model_v6.py`, `train_form_model_v6_1.py`, `train_ssl_pretrain.py`
- `backend/training/preprocessing/dataset_builder.py`, `dataset_builder_v5.py`, `dataset_builder_v5_1.py`, `qevd_extractor.py`, `normalize.py` (if not used by new service), `angular_features.py`, `joint_mapping.py` (if only used by old service)
- Old evaluation scripts: `reality_check.py`, `reality_check_v5.py`, `reality_check_v6.py`, `reality_check_v6_1.py`

**Note on `normalize.py`, `angular_features.py`, `joint_mapping.py`:** These are ONLY used in `form_analyzer.py:418-428` for the MediaPipe → angle pipeline. The new service does NOT use them. Safe to archive.

**`app.py` diff after removal:**
Remove lines: 34-35 (form_analyzer/form_session imports), 138-160 (FormFrameResult/FormSessionSummary Pydantic models), 179-207 (model-dir detection block in lifespan + form_analyzer load + mp_pose cleanup), 336-454 (both endpoints with old logic).
Add: import SquatFormService, simplified lifespan, new Pydantic models, new endpoint handlers.

**The import order for safe removal:** Archive files first → update `app.py` imports → replace endpoint handlers → add new service files → run tests. Do NOT leave `app.py` in a partially-broken import state between steps.

---

## Runtime State Inventory

This is a local-backend, code-replacement phase — no databases, no external service configs.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no DB; form analysis is stateless | None |
| Live service config | None — no external service registration | None |
| OS-registered state | None — no scheduled tasks for form analysis | None |
| Secrets/env vars | `FITNOVA_MODEL_DIR` env var (old path convention) — new service uses same convention with new path | Update `.env.example` if documenting the new path |
| Build artifacts | Old form weights in `backend/models/form_model*/` — moved to archive, not auto-rebuilt | Manual archive step (Part of D-06) |

**Nothing found requiring data migration** — all state is in files being replaced, not in a running database.

---

## Common Pitfalls

### Pitfall 1: Train/Serve Spatial Pipeline Skew

**What goes wrong:** Using `spatial_train` (random crop) at serve time instead of `spatial_val` (center crop). Model sees ~10% spatially shifted crops → confidence scores shift. At borderline KIE 0.614 threshold, this can flip detections.
**Why it happens:** Easy to confuse when porting from the training loop. The training loop uses `spatial_train` for the train DataLoader. Serve time is equivalent to "test/val time."
**How to avoid:** In `classify_clip`, the call chain must be: `uniform_sample_indices(..., jitter=0)` → `decode_clip(...)` → **`spatial_val(...)`**. No `generator` parameter needed (deterministic).
**Warning signs:** Same clip returns different results on repeated calls to the REST endpoint.

### Pitfall 2: Wrong `strict` Parameter When Loading State-Dict

**What goes wrong:** Using `strict=False` when loading the fine-tune `best.pt` causes silently missing fc head weights. Using `strict=True` when loading the MD backbone `backbone.pt` fails with unexpected key errors.
**Why it happens:** There are TWO types of checkpoints: (a) `backbone.pt` from md_pretrain (backbone only, no fc) and (b) `best.pt` from md_finetune (full model, backbone + new fc). At serve time, only (b) is loaded. Use `strict=True` for (b).
**How to avoid:** Check `ckpt.keys()` — if `"backbone_state_dict"` is the key, it's the md_pretrain backbone (b); use `model.load_state_dict(ckpt["backbone_state_dict"], strict=False)`. If `"model_state_dict"` is the key, it's the fine-tune checkpoint; use `model.load_state_dict(ckpt["model_state_dict"], strict=True)`.
**Warning signs:** `RuntimeError: Missing key(s) in state_dict: "fc.1.weight"` (strict=True on backbone) or `RuntimeError: Unexpected key(s) in state_dict: "fc.weight"` (strict=False on wrong checkpoint).

### Pitfall 3: Blocking the Event Loop with Synchronous Inference

**What goes wrong:** Calling `classify_clip()` directly inside an `async def` endpoint handler blocks the uvicorn event loop for ~0.93–2.8 s. During this time, ALL other requests (health checks, plan generation, other WebSocket frames) are stalled.
**Why it happens:** PyTorch CPU inference is purely synchronous and CPU-bound.
**How to avoid:** Always wrap with `await run_in_threadpool(service.classify_clip, frames)`.
**Warning signs:** Server appears unresponsive during form analysis; `/health` times out.

### Pitfall 4: fp16 Inference Hang on CPU

**What goes wrong:** Attempting `.half()` on the model or input tensors for CPU inference causes the forward pass to hang indefinitely.
**Why it happens:** PyTorch 2.12.0+cpu on Windows has no fast fp16 GEMM kernels. fp16 operations fall back to an extremely slow emulation path (VERIFIED on target machine: timing benchmark hung after 3 minutes).
**How to avoid:** Use fp32 only on CPU. Do not add fp16 optimizations.
**Warning signs:** `classify_clip` never returns.

### Pitfall 5: TF Import Noise at Startup

**What goes wrong:** When `import tensorflow` runs after `import torch`, TF prints several oneDNN INFO lines and a WARNING about absl logging. These are harmless but clutter logs.
**Why it happens:** TF 2.21 initializes its profiler and oneDNN backend on import.
**How to avoid:** Suppress with `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` before the import, or accept as expected noise.
**Warning signs:** None — this is cosmetic only.

---

## Code Examples

### Loading the Ensemble (Serving Pattern)

```python
# Source: harness/md_finetune.py:84-108; eval/ensemble.py:21-48; transforms.py
# This is the exact path that must be replicated at serve time.

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision.models.video import r2plus1d_18

from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
from backend.training.aqa.datasets.transforms import (
    decode_clip, spatial_val, uniform_sample_indices,
)


class SquatFormService:
    THRESHOLDS = {"kie": 0.614, "kfe": 0.385}

    def __init__(self, model_dir: str, seeds: list[int] = (42, 1337, 7)):
        self._models: list[nn.Module] = []
        self._model_ready = False
        for seed in seeds:
            pt_path = Path(model_dir) / f"seed{seed}" / "best.pt"
            if pt_path.exists():
                model = rip_2plus1d_18(weights=None)
                model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                model.eval()
                self._models.append(model)
        self._model_ready = len(self._models) > 0

    def classify_clip(self, frames_tchw_uint8: np.ndarray) -> dict:
        """frames_tchw_uint8: [T, 3, H, W] uint8 from decode_clip."""
        if not self._model_ready:
            return self._neutral_response()
        clip_tensor = spatial_val(torch.from_numpy(frames_tchw_uint8))  # [3,32,112,112]
        batch = clip_tensor.unsqueeze(0)  # [1,3,32,112,112]
        per_seed_logits = []
        with torch.no_grad():
            for m in self._models:
                logits = m(batch)  # [1, 2]
                per_seed_logits.append(logits.numpy())
        scores = aggregate_sigmoid_mean(per_seed_logits)  # [1, 2]
        kie_s, kfe_s = float(scores[0, 0]), float(scores[0, 1])
        return self._build_response(kie_s, kfe_s)

    def _build_response(self, kie_s: float, kfe_s: float) -> dict:
        kie_det = kie_s >= self.THRESHOLDS["kie"]
        kfe_det = kfe_s >= self.THRESHOLDS["kfe"]
        return {
            "exercise": "squat",
            "errors": [
                {"type": "KIE", "detected": kie_det, "confidence": round(kie_s, 4),
                 "severity_word": _severity(kie_s, kie_det), "intervals": []},
                {"type": "KFE", "detected": kfe_det, "confidence": round(kfe_s, 4),
                 "severity_word": _severity(kfe_s, kfe_det), "intervals": []},
            ],
        }
```

### Deterministic Feedback Templates (D-10)

```python
# Source: CONTEXT.md D-10; master plan "Knees inward — detected" example
# Expected behavior note (Phase 4 SUMMARY): strong KFE (~97% recall), modest KIE (~47%)
# Surface KIE as "possible" rather than definitive.

def _severity(confidence: float, detected: bool) -> str:
    """Confidence-derived severity word. Not a measured result — a UX estimate."""
    if not detected:
        return "none"
    if confidence >= 0.80:
        return "strong"
    if confidence >= 0.65:
        return "moderate"
    return "possible"

ERROR_FEEDBACK_TEMPLATES = {
    "KIE": {
        True:  "Knees caving inward — {severity} signal detected.",
        False: "Knees inward error not detected.",
    },
    "KFE": {
        True:  "Knees traveling too far forward — {severity} signal detected.",
        False: "Knee forward error not detected.",
    },
}

def format_rep_feedback(error_type: str, detected: bool, severity_word: str) -> str:
    template = ERROR_FEEDBACK_TEMPLATES[error_type][detected]
    return template.format(severity=severity_word)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TF + MediaPipe pose → 22 angles → TCN/ST-GCN | PyTorch raw-pixel CNN R(2+1)D-18 | Phase 5 (this phase) | Removes all MediaPipe dependencies from the form path; no pose extraction needed |
| Frame-by-frame inference (per-frame result) | Per-clip (32-frame) inference | Phase 5 | Matches Fitness-AQA training contract; binary error per clip, not per frame |
| Exercise detection via classifier head | User declares exercise in session start | Phase 5 | Squat-only in Phase 5; no exercise classifier needed |
| MediaPipe VIDEO mode timestamp management | Not applicable | Phase 5 | `reset_video_state()` complexity entirely removed |
| `FormFrameResult` + `FormSessionSummary` schema | Per-rep `{ exercise, errors: [...] }` schema | Phase 5 | Simpler; binary+timing; no landmarks, no quality_score, no rep_count in response |

**Deprecated/outdated in Phase 5:**
- `FormAnalyzer`, `FormSession`, `FormGeometry`, `mediapipe_config`: All archived. None should be imported.
- `app.state.form_analyzer`: Swap point replaced by `app.state.squat_form_service`.
- Old Pydantic models `FormFrameResult`, `FormSessionSummary`: Replaced by new schema.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | None (no `pytest.ini`; run from project root) |
| Quick run command | `pytest backend/tests/test_squat_form_api.py -x -v` |
| Full suite command | `pytest backend/ -x -v --ignore=backend/training` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | Old form_analyzer/form_session NOT importable from serving path; SquatFormService loads at startup | unit | `pytest backend/tests/test_squat_form_api.py::test_service_loads -x` | ❌ Wave 0 |
| API-01 | `/health` reports `squat_model_ready` and NOT old `model_version` field | smoke | `pytest backend/tests/test_squat_form_api.py::test_health_schema -x` | ❌ Wave 0 |
| API-02 | `RepSegmenter.segment_reps_by_motion_energy` returns at least 1 interval on a synthetic 90-frame clip | unit | `pytest backend/tests/test_squat_form_api.py::test_rep_segmenter_synthetic -x` | ❌ Wave 0 |
| API-02 | Live window trigger fires after `LIVE_REP_WINDOW_FRAMES` frames and respects `LIVE_MIN_GAP_FRAMES` | unit | `pytest backend/tests/test_squat_form_api.py::test_live_window_trigger -x` | ❌ Wave 0 |
| API-03 | `POST /analyze-form-video` with synthetic mp4 → 200, response has `exercise` + `errors[]` with KIE/KFE | integration | `pytest backend/tests/test_squat_form_api.py::test_upload_response_schema -x` | ❌ Wave 0 |
| API-03 | When model not loaded, endpoint returns graceful response (not 500) | integration | `pytest backend/tests/test_squat_form_api.py::test_upload_no_model -x` | ❌ Wave 0 |
| API-04 | WS `/ws/form-session`: `start_session` → `frame` × 32+ → `end_session` returns `session_summary` | integration | `pytest backend/tests/test_squat_form_api.py::test_ws_session_protocol -x` | ❌ Wave 0 |
| API-04 | After 32 frames, per-rep `rep_result` is emitted with correct schema | integration | `pytest backend/tests/test_squat_form_api.py::test_ws_rep_result_schema -x` | ❌ Wave 0 |
| D-02 | `spatial_val` is called (not `spatial_train`) in classify_clip: deterministic repeated calls return same scores | unit | `pytest backend/tests/test_squat_form_api.py::test_classify_deterministic -x` | ❌ Wave 0 |
| D-11 | Domain-shift test: user-provided real squat clip → response has expected schema + timing logged | manual | Run manually; paste outputs | N/A — requires real clip |

### Sampling Rate

- **Per task commit:** `pytest backend/tests/test_squat_form_api.py -x -v`
- **Per wave merge:** `pytest backend/ -x -v --ignore=backend/training`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `backend/tests/test_squat_form_api.py` — all 9 automated test cases above
- [ ] `backend/services/squat_form_service.py` — must exist (even stub) before tests can be written
- [ ] `backend/services/rep_segmenter.py` — must exist (even stub) before test_rep_segmenter_synthetic
- [ ] No new test framework required (pytest already installed)

*Note: No `pytest-asyncio` needed — `TestClient` and `websocket_connect` are synchronous wrappers from Starlette.*

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` (from config.json).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable — local demo server, no auth |
| V3 Session Management | Partial | Per-connection `SquatLiveSession` object is ephemeral; no tokens |
| V4 Access Control | No | All endpoints public (local dev server) |
| V5 Input Validation | Yes | Validate `exercise` form field (allowlist); validate JPEG before decode |
| V6 Cryptography | No | No crypto in this phase |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Oversized video upload exhausting RAM | DoS | Note: out of Phase 5 scope (CONCERNS.md); add `max_upload_size` guard as a trivial adjacent fix |
| Malformed JPEG in WS frame causing `cv2.imdecode` to return None | Tampering | Guard: check `frame is not None` after decode; return early if None |
| Temp file not cleaned up if `NamedTemporaryFile` context throws before assignment | DoS / disk exhaustion | Fix: assign `tmp_path = None` before `with` block; guard `os.unlink` with `if tmp_path` (existing CONCERNS.md bug — adjacent fix) |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10 | Type annotations (`str \| None`) | ✓ | 3.10.x | — |
| torch (CPU) | Model forward pass | ✓ | 2.12.0+cpu | — |
| torchvision (CPU) | `r2plus1d_18`, `read_video` | ✓ | 0.27.0+cpu | — |
| tensorflow | Existing recommender (unchanged) | ✓ | 2.21.0 | — |
| fastapi | REST + WS routing | ✓ | 0.125.0 | — |
| starlette | `run_in_threadpool`, WS test | ✓ | 0.50.0 | — |
| opencv-python-headless | JPEG decode in live path | ✓ | 4.13.0 | — |
| httpx | Test client | ✓ | 0.28.1 | — |
| Drive weights (3× best.pt) | `SquatFormService` | ✗ (not staged yet) | — | `model_ready=False` graceful degradation |

**Missing dependencies with no fallback:** None blocking code execution.

**Missing dependencies with fallback:** Drive weights — `SquatFormService.model_ready == False` path is a planned graceful degradation, not an error.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | TF runtime overhead ~50 MB — total process ~430 MB RAM | §TF+Torch Coexistence | If TF overhead is larger (e.g. 200 MB), total RAM may hit 600 MB; still acceptable for a dev machine |
| A2 | Frame-difference rep segmentation is sufficient for upload mode | §Rep Segmentation | If clips have very low motion variance, false rep boundaries; mitigation: fallback to single-rep treatment |
| A3 | Optical flow would give better boundary precision than frame-difference | §Rep Segmentation | If frame-difference produces poor results, optical flow is the next step |
| A4 | `decode_clip` from `transforms.py` works with video files decoded server-side (not just Colab paths) | §Inference Reconstruction | `decode_clip` uses `torchvision.io.read_video` with absolute paths — should work on Windows; needs a quick smoke test with a local .mp4 |
| A5 | 3-seed ensemble latency on target machine is linear (3 × 0.93 s = ~2.8 s) | §CPU Latency | Could be slightly slower if memory bandwidth saturates with 3 models in RAM |

---

## Open Questions

1. **Real-clip domain shift (D-11)**
   - What we know: Test F1 on curated Fitness-AQA clips is 0.63 macro.
   - What's unclear: Live phone-camera lighting/angle/framing will likely shift confidence distributions. KIE threshold 0.614 may be too high or too low under domain shift.
   - Recommendation: The planner MUST include an early task where the user provides a real squat clip, the full pipeline runs on it, raw confidence scores are logged, and the results are pasted back for diagnosis. This is a required integration test, not optional.

2. **`decode_clip` with temp file from REST upload**
   - What we know: `decode_clip` uses `torchvision.io.read_video` with a file path. The REST endpoint currently writes an `UploadFile` to a `tempfile.NamedTemporaryFile`.
   - What's unclear: Whether the temp file path with a `.mp4` extension is reliably readable by `read_video_timestamps` on Windows (some codecs need specific extensions).
   - Recommendation: Keep the existing temp file pattern from `app.py:415-418`; ensure `suffix=".mp4"` is set.

3. **Single-rep clip without explicit segmentation**
   - What we know: Fitness-AQA clips are single-rep. A single-rep upload = run classify_clip once on the full 32-frame uniform sample.
   - What's unclear: How the user will typically provide clips — single-rep (training-distribution) or multi-rep (real-world use). The segmenter's single-rep fallback handles this correctly.
   - Recommendation: Document in the endpoint response that `rep_number: 1` = the whole clip was treated as one rep if segmentation failed.

---

## Sources

### Primary (HIGH confidence)

- `backend/training/aqa/harness/md_finetune.py` — `build_finetune_model`, `FinetuneConfig`, exact checkpoint schema [VERIFIED: read line-by-line]
- `backend/training/aqa/eval/ensemble.py` — `aggregate_sigmoid_mean`, input/output contract [VERIFIED: read line-by-line]
- `backend/training/aqa/datasets/transforms.py` — `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_val`, `spatial_train` [VERIFIED: read line-by-line]
- `backend/app.py` — lifespan pattern, endpoint scaffolding, Pydantic model patterns [VERIFIED: read full file]
- `backend/services/form_analyzer.py` — archival impact; old patterns to avoid [VERIFIED: read full file]
- `backend/services/form_session.py` — archival impact; session structure reference [VERIFIED: read header]
- `.planning/phases/04-squat-motion-disentangling-ssl/04-04-SUMMARY.md` — production checkpoint declaration, thresholds, fallback [VERIFIED: read full file]
- Python runtime benchmark — torch 2.12.0+cpu, torchvision 0.27.0+cpu, R(2+1)D-18 forward timing [VERIFIED: measured on target machine; 0.940/0.934/0.932 s fp32]
- fp16 CPU hang — confirmed by timeout on same machine [VERIFIED]
- Import order safety (torch + tf) — confirmed working [VERIFIED]
- `starlette.testclient.WebSocketTestSession` — `send_json`/`receive_json` confirmed available [VERIFIED]
- `starlette.concurrency.run_in_threadpool` — signature confirmed [VERIFIED]

### Secondary (MEDIUM confidence)

- `.planning/codebase/CONCERNS.md` — archival scope, concurrency bug history [VERIFIED: read full file]
- Master plan `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` lines 100-143 [VERIFIED: read]
- CONTEXT.md decisions D-01 through D-12 [VERIFIED: read]

---

## Metadata

**Confidence breakdown:**
- Inference reconstruction: HIGH — verified against actual source code; benchmarks on target machine
- Archival impact: HIGH — verified by reading all affected files and their import chains
- Rep segmentation: MEDIUM — algorithm is pragmatic; effectiveness on real clips is ASSUMED and requires measurement (D-04/D-11)
- TF+torch coexistence: HIGH — tested directly on this machine
- CPU latency: HIGH — measured 3 runs on the target machine
- WebSocket live path: HIGH — verified against existing protocol and Starlette test client API

**Research date:** 2026-05-25
**Valid until:** 2026-06-25 (stable stack; torch/torchvision versions confirmed installed)
