"""SquatFormService — PyTorch R(2+1)D-18 ensemble inference service for Squat KIE/KFE detection.

Loaded once at FastAPI startup into app.state.squat_form_service.  Shared read-only
across all sessions — NO per-call mutable state.

Lesson from old FormAnalyzer (CONCERNS.md): process_frame() mutated _ts_ms and
_prev_result on the shared instance, corrupting concurrent sessions.  This service
has NO per-call mutable state — classify_clip() takes inputs and returns outputs with
no side effects.

Preprocessing parity (D-02) is the #1 correctness risk.  The service reuses the EXACT
Phase-2/3 contract:
  - spatial_val  (center crop — the deterministic val/test pipeline, not the train pipeline)
  - uniform_sample_indices with jitter=0
  - decode_clip
  - KINETICS_MEAN / KINETICS_STD
  - aggregate_sigmoid_mean on RAW logits (sigmoid applied internally)

Anti-patterns explicitly guarded against (see RESEARCH §Common Pitfalls):
  - Using Kinetics pretrained weights when building the serve model (Pitfall 2 — overwrites SSL)
  - Using the random-crop augmentation pipeline at inference time (Pitfall 1 — train/serve skew)
  - Pre-sigmoidizing logits before passing to aggregate_sigmoid_mean (sigmoid applied internally)
  - strict=False on the fine-tune checkpoint (Pitfall 2 — silently drops fc head weights)
  - fp16 inference on CPU (Pitfall 4 — hangs on Windows CPU in PyTorch 2.12)
  - calling classify_clip() directly in async handlers (Pitfall 3 — use classify_clip_async)

See: .planning/phases/05-backend-inference-integration-squat/05-RESEARCH.md § Pitfalls 1-4.
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
from backend.training.aqa.datasets.transforms import spatial_val

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module constants (D-01 — val-tuned on the 3-seed ensemble)
# ─────────────────────────────────────────────────────────────────────────────

KIE_THRESHOLD: float = 0.614   # Knee-Inward Error threshold (val-tuned)
KFE_THRESHOLD: float = 0.385   # Knee-Forward Error threshold (val-tuned)
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper (private — tested directly in test_severity_word)
# ─────────────────────────────────────────────────────────────────────────────


def _severity(confidence: float, detected: bool) -> str:
    """Return a confidence-derived severity word (D-10 UX estimate, not a measured result).

    Args:
        confidence: sigmoid score in [0, 1].
        detected:   True if score >= the per-head threshold.

    Returns:
        "none" if not detected; "strong" / "moderate" / "possible" by confidence band.

    Note: KIE has modest recall (~47%) — surface as "possible" to avoid over-claiming.
    """
    if not detected:
        return "none"
    if confidence >= 0.80:
        return "strong"
    if confidence >= 0.65:
        return "moderate"
    return "possible"


# ─────────────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────────────


class SquatFormService:
    """Load the Phase-4 MD-SSL 3-seed ensemble and run EXACT inference path.

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

    Each best.pt is a Phase-4 fine-tune checkpoint with keys:
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
        self._model_ready: bool = False

        # D-08 config knob: SQUAT_INFERENCE_SEEDS env var sets the default number of
        # seeds to use per inference call.  Clamped to [1, len(seeds)].
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

        T-05-02 mitigation: file-presence check per seed; missing → skip, not crash.
        T-05-03 mitigation: log only seed count + path; never log clip contents.
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

            # ANTI-PATTERN GUARD (Pitfall 2): weights=None — the MD backbone IS the
            # initialization; loading Kinetics pretrained weights here would overwrite SSL.
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

        self._model_ready = len(self._models) > 0
        logger.info(
            "SquatFormService: model_ready=%s, seeds_loaded=%d/%d.",
            self._model_ready,
            len(self._models),
            len(seeds),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def model_ready(self) -> bool:
        """True if at least one seed model loaded successfully."""
        return self._model_ready

    def classify_clip(
        self,
        frames_tchw_uint8: np.ndarray,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Run the EXACT Phase-4 inference path and return the D-05 response schema.

        SYNCHRONOUS — blocks ~0.93 s (1 seed) or ~2.8 s (3 seeds) on CPU fp32.
        Call via classify_clip_async from async endpoint handlers to avoid blocking
        the uvicorn event loop (Pitfall 3).

        Args:
            frames_tchw_uint8: uint8 ndarray [T, 3, H, W] from decode_clip.
                               The live path must permute (T, H, W, C) → (T, C, H, W)
                               before calling (see RESEARCH §5 live-path note).
            n_seeds:           number of seeds to use (default: self._default_n_seeds).
                               Lower = faster; fewer = slightly less accurate.

        Returns:
            D-05 dict with keys: exercise, errors (list of KIE + KFE dicts).
            When model_ready is False, returns _neutral_response() (no exception).
        """
        if not self._model_ready:
            return self._neutral_response()

        # ANTI-PATTERN GUARD (Pitfall 1): use the deterministic center-crop pipeline
        # (spatial_val), not the random-crop augmentation pipeline used during training.
        # Random crop at serve time = non-deterministic results = B6-class train/serve skew.
        # INPUT: [T, 3, H, W] uint8 tensor → OUTPUT: [3, T, 112, 112] float32 Kinetics-norm.
        clip = spatial_val(torch.from_numpy(frames_tchw_uint8))
        batch = clip.unsqueeze(0)  # [1, 3, T, 112, 112]

        n = n_seeds if n_seeds is not None else self._default_n_seeds
        selected = self._models[:n]

        # ANTI-PATTERN GUARD: pass RAW logits to aggregate_sigmoid_mean.
        # Sigmoid is applied internally — do NOT pre-sigmoidize.
        # fp32 ONLY — never use fp16 on CPU (Pitfall 4: hangs on Windows CPU in PyTorch 2.12).
        per_seed_logits: list[np.ndarray] = []
        with torch.no_grad():
            for m in selected:
                logits = m(batch)  # [1, 2]
                per_seed_logits.append(logits.numpy())

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
        """Assemble the D-05 per-rep response dict from raw sigmoid scores.

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
                    "severity_word": _severity(kie_s, kie_det),
                    "intervals": [],
                },
                {
                    "type": "KFE",
                    "detected": kfe_det,
                    "confidence": round(kfe_s, 4),
                    "severity_word": _severity(kfe_s, kfe_det),
                    "intervals": [],
                },
            ],
        }

    def _neutral_response(self) -> dict:
        """D-05 schema returned when model_ready is False — no exception raised.

        T-05-02 mitigation: neutral degradation rather than 500 error.
        """
        return {
            "exercise": "squat",
            "errors": [
                {
                    "type": "KIE",
                    "detected": False,
                    "confidence": 0.0,
                    "severity_word": "none",
                    "intervals": [],
                },
                {
                    "type": "KFE",
                    "detected": False,
                    "confidence": 0.0,
                    "severity_word": "none",
                    "intervals": [],
                },
            ],
            "model_not_loaded": True,
        }
