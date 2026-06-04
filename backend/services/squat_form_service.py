"""SquatFormService — PyTorch R(2+1)D-18 ensemble inference service for Squat KIE/KFE detection.

Loaded once at FastAPI startup into app.state.squat_form_service.  Shared read-only
across all sessions — NO per-call mutable state.

The old FormAnalyzer mutated _ts_ms and _prev_result on the shared instance,
corrupting concurrent sessions.  This service has NO per-call mutable state —
classify_clip() takes inputs and returns outputs with no side effects.

Preprocessing parity is the #1 correctness risk.  The service reuses the EXACT
offline contract:
  - kneeaware_spatial_val (portrait lower-body pre-crop → spatial_val center crop;
    landscape input passes through to spatial_val unchanged — offline-eval parity)
  - uniform_sample_indices with jitter=0
  - decode_clip
  - KINETICS_MEAN / KINETICS_STD
  - aggregate_sigmoid_mean on RAW logits (sigmoid applied internally)

Anti-patterns explicitly guarded against:
  - Using Kinetics pretrained weights when building the serve model (overwrites SSL)
  - Using the random-crop augmentation pipeline at inference time (train/serve skew)
  - Pre-sigmoidizing logits before passing to aggregate_sigmoid_mean (sigmoid applied internally)
  - strict=False on the fine-tune checkpoint (silently drops fc head weights)
  - fp16 inference on CPU (hangs on Windows CPU in PyTorch 2.12)
  - calling classify_clip() directly in async handlers (use classify_clip_async)
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from starlette.concurrency import run_in_threadpool
from torchvision.models.video import r2plus1d_18

from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
from backend.services.clip_decode import kneeaware_spatial_val

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:  # pragma: no cover - onnxruntime optional; PyTorch is the fallback
    ort = None  # type: ignore
    _ORT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module constants (validation-tuned on the 3-seed ensemble)
# ─────────────────────────────────────────────────────────────────────────────

KIE_THRESHOLD: float = 0.614   # Knee-Inward Error threshold (val-tuned)
KFE_THRESHOLD: float = 0.385   # Knee-Forward Error threshold (val-tuned)
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 7)

# ONNX Runtime intra-op thread count (CPU). 0 = let onnxruntime choose (default).
# Override via SQUAT_NUM_THREADS for machine-specific tuning.
_ONNX_INTRA_OP_THREADS: int = int(os.environ.get("SQUAT_NUM_THREADS", "0"))


# ─────────────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────────────


class SquatFormService:
    """Load the MD-SSL 3-seed ensemble and run EXACT inference path.

    Construction:
        svc = SquatFormService(model_dir="backend/models/form_model_squat_md")

    Usage:
        # Synchronous (call from threadpool — ~0.93 s fp32 per clip on CPU):
        result = svc.classify_clip(frames_tchw_uint8)

        # Async (use inside FastAPI endpoint handlers):
        result = await svc.classify_clip_async(frames_tchw_uint8)

    model_dir layout:
        model_dir/
          seed42/best.pt
          seed1337/best.pt
          seed7/best.pt

    Each best.pt is a fine-tune checkpoint with keys:
        model_state_dict, epoch, optimizer_state_dict, ...
    The full fine-tuned model (backbone + Dropout(0.2)+Linear(512,2) head) is stored
    under "model_state_dict" — loaded with strict=True.
    """

    def __init__(
        self,
        model_dir: str,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        kie_threshold: float = KIE_THRESHOLD,
        kfe_threshold: float = KFE_THRESHOLD,
    ) -> None:
        self.model_dir = model_dir
        self.kie_threshold = kie_threshold
        self.kfe_threshold = kfe_threshold
        self._models: list[nn.Module] = []
        self._sessions: list = []          # onnxruntime sessions, parallel to _models
        self._use_onnx: bool = False
        self._onnx_input_name: str = "clip"
        self._model_ready: bool = False

        # SQUAT_INFERENCE_SEEDS env var sets the default number of seeds to use per
        # inference call.  Clamped to [1, len(seeds)].
        n_seeds_env = int(os.environ.get("SQUAT_INFERENCE_SEEDS", len(seeds)))
        self._default_n_seeds: int = max(1, min(n_seeds_env, len(seeds)))

        self._load(seeds)

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, seeds: tuple[int, ...]) -> None:
        """Load each seed's best.pt into an eval-mode R(2+1)D-18 model.

        Missing seeds are skipped with a warning (graceful degradation — model_ready
        stays False if ALL seeds are missing).  A corrupt checkpoint that raises during
        torch.load surfaces at startup (fail-fast), not per-request.

        File-presence check per seed; missing → skip, not crash.
        Log only seed count + path; never log clip contents.
        """
        for seed in seeds:
            pt_path = Path(self.model_dir) / f"seed{seed}" / "best.pt"
            if not pt_path.exists():
                logger.warning(
                    "SquatFormService: seed%d weight not found at %s — skipping.",
                    seed,
                    pt_path,
                )
                continue

            # weights=None — the MD backbone IS the initialization; loading Kinetics
            # pretrained weights here would overwrite SSL.
            m = r2plus1d_18(weights=None)
            assert m.fc.in_features == 512, (
                f"R(2+1)D-18 fc.in_features={m.fc.in_features}, expected 512"
            )
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))

            # strict=True — best.pt contains the FULL fine-tuned model (backbone + new fc head).
            # strict=False is for backbone-only backbone.pt — a different checkpoint type.
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"], strict=True)
            m.eval()
            self._models.append(m)
            logger.info(
                "SquatFormService: loaded seed%d from %s (state_dict keys: %d).",
                seed,
                pt_path,
                len(ckpt["model_state_dict"]),
            )

            # Export (cached) + open an ONNX Runtime session for this seed. ONNX is
            # numerically identical to PyTorch (validated parity ~2e-7) but ~1.6x
            # faster on CPU. On any failure _sessions stays short of _models and
            # classify_clip falls back to the PyTorch forward.
            sess = self._load_or_export_onnx(
                m, Path(self.model_dir) / f"seed{seed}" / "best.onnx"
            )
            if sess is not None:
                self._sessions.append(sess)

        self._model_ready = len(self._models) > 0
        # Use ONNX only if EVERY loaded seed has a matching session (consistent ensemble);
        # otherwise fall back to PyTorch for all seeds.
        self._use_onnx = (
            _ORT_AVAILABLE
            and self._model_ready
            and len(self._sessions) == len(self._models)
        )
        logger.info(
            "SquatFormService: model_ready=%s, seeds_loaded=%d/%d, onnx_enabled=%s.",
            self._model_ready,
            len(self._models),
            len(seeds),
            self._use_onnx,
        )

    def _load_or_export_onnx(self, model: nn.Module, onnx_path: Path):
        """Export `model` to ONNX (cached on disk) and return a warmed InferenceSession.

        Returns None — caller falls back to PyTorch — if onnxruntime is unavailable
        or export/load fails. ONNX output is numerically identical to the PyTorch
        forward (validated parity ~2e-7); this is purely a CPU speed optimization,
        NOT a model/quality change.
        """
        if not _ORT_AVAILABLE:
            return None
        try:
            if not onnx_path.exists():
                dummy = torch.zeros(1, 3, 32, 112, 112, dtype=torch.float32)
                logger.info("SquatFormService: exporting ONNX (one-time) -> %s", onnx_path)
                torch.onnx.export(
                    model,
                    dummy,
                    str(onnx_path),
                    input_names=[self._onnx_input_name],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
            so = ort.SessionOptions()
            if _ONNX_INTRA_OP_THREADS > 0:
                so.intra_op_num_threads = _ONNX_INTRA_OP_THREADS
            sess = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            # Warmup forward — avoids first-request jitter on the live path.
            sess.run(None, {self._onnx_input_name: torch.zeros(1, 3, 32, 112, 112).numpy()})
            return sess
        except Exception as exc:  # pragma: no cover - export/runtime env dependent
            logger.warning(
                "SquatFormService: ONNX export/load failed for %s (%s) — using PyTorch.",
                onnx_path,
                exc,
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def model_ready(self) -> bool:
        """True if at least one seed model loaded successfully."""
        return self._model_ready

    @property
    def onnx_enabled(self) -> bool:
        """True if inference runs through ONNX Runtime (else the PyTorch fallback)."""
        return self._use_onnx

    def classify_clip(
        self,
        frames_tchw_uint8: np.ndarray,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Run the EXACT inference path and return the per-rep response schema.

        SYNCHRONOUS — blocks ~0.93 s (1 seed) or ~2.8 s (3 seeds) on CPU fp32.
        Call via classify_clip_async from async endpoint handlers to avoid blocking
        the uvicorn event loop.

        Args:
            frames_tchw_uint8: uint8 ndarray [T, 3, H, W] from decode_clip.
                               The live path must permute (T, H, W, C) → (T, C, H, W)
                               before calling.
            n_seeds:           number of seeds to use (default: self._default_n_seeds).
                               Lower = faster; fewer = slightly less accurate.

        Returns:
            dict with keys: exercise, errors (list of KIE + KFE dicts).
            When model_ready is False, returns _neutral_response() (no exception).
        """
        if not self._model_ready:
            return self._neutral_response()

        # Use the deterministic center-crop pipeline, not the random-crop augmentation
        # pipeline used during training. Random crop at serve time = non-deterministic
        # results = train/serve skew.
        # kneeaware_spatial_val applies a portrait lower-body pre-crop so the knees land
        # inside the 112² crop on phone video; landscape input passes through to
        # spatial_val UNCHANGED (preserves offline-eval parity on dataset clips).
        # INPUT: [T, 3, H, W] uint8 tensor → OUTPUT: [3, T, 112, 112] float32 Kinetics-norm.
        clip = kneeaware_spatial_val(torch.from_numpy(frames_tchw_uint8))
        batch = clip.unsqueeze(0)  # [1, 3, T, 112, 112]

        n = n_seeds if n_seeds is not None else self._default_n_seeds

        # Pass RAW logits to aggregate_sigmoid_mean.
        # Sigmoid is applied internally — do NOT pre-sigmoidize.
        # fp32 ONLY — never use fp16 on CPU (hangs on Windows CPU in PyTorch 2.12).
        # ONNX path and PyTorch path are numerically identical (parity ~2e-7); ONNX is
        # ~1.6x faster on CPU. Both emit RAW logits into aggregate_sigmoid_mean.
        per_seed_logits: list[np.ndarray] = []
        if self._use_onnx:
            inp = batch.numpy()
            for sess in self._sessions[:n]:
                per_seed_logits.append(sess.run(None, {self._onnx_input_name: inp})[0])  # [1, 2]
        else:
            with torch.no_grad():
                for m in self._models[:n]:
                    per_seed_logits.append(m(batch).numpy())  # [1, 2]

        scores = aggregate_sigmoid_mean(per_seed_logits)  # [1, 2] in [0, 1]
        kie_s = float(scores[0, 0])
        kfe_s = float(scores[0, 1])
        return self._build_response(kie_s, kfe_s)

    async def classify_clip_async(
        self,
        frames_tchw_uint8: np.ndarray,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Async wrapper for classify_clip — required for use in FastAPI endpoint handlers.

        Offloads the synchronous CPU-bound forward to a thread via Starlette's
        run_in_threadpool, which is the FastAPI-idiomatic pattern for blocking I/O or compute.
        """
        fn = functools.partial(self.classify_clip, frames_tchw_uint8, n_seeds=n_seeds)
        return await run_in_threadpool(fn)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(self, kie_s: float, kfe_s: float) -> dict:
        """Assemble the per-rep response dict from raw sigmoid scores.

        intervals=[] here — per-clip timing is attached by the endpoint handler,
        not the service (the service is stateless).
        """
        kie_det = kie_s >= self.kie_threshold
        kfe_det = kfe_s >= self.kfe_threshold
        return {
            "exercise": "squat",
            "errors": [
                {
                    "type": "KIE",
                    "detected": kie_det,
                    "confidence": round(kie_s, 4),
                    "threshold": self.kie_threshold,
                    "intervals": [],
                },
                {
                    "type": "KFE",
                    "detected": kfe_det,
                    "confidence": round(kfe_s, 4),
                    "threshold": self.kfe_threshold,
                    "intervals": [],
                },
            ],
        }

    def _neutral_response(self) -> dict:
        """Response schema returned when model_ready is False — no exception raised.

        Neutral degradation rather than 500 error.
        """
        return {
            "exercise": "squat",
            "errors": [
                {
                    "type": "KIE",
                    "detected": False,
                    "confidence": 0.0,
                    "threshold": self.kie_threshold,
                    "intervals": [],
                },
                {
                    "type": "KFE",
                    "detected": False,
                    "confidence": 0.0,
                    "threshold": self.kfe_threshold,
                    "intervals": [],
                },
            ],
            "model_not_loaded": True,
        }
