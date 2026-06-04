"""OHPFormService — PyTorch R(2+1)D-18 ensemble inference service for OHP Elbows/Knees detection.

Loaded once at FastAPI startup into app.state.ohp_form_service.  Shared read-only
across all sessions — NO per-call mutable state.

Preprocessing parity: the service reuses the EXACT training contract:
  - spatial_val (plain center crop — OHP training used spatial_val, not kneeaware_spatial_val;
    OHP dataset has portrait clips that trained with standard center crop, so we must match it)
  - uniform_sample_indices with jitter=0
  - decode_clip
  - KINETICS_MEAN / KINETICS_STD
  - aggregate_sigmoid_mean on RAW logits (sigmoid applied internally)
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
# Module constants (val-tuned on the 2-seed ensemble)
# ─────────────────────────────────────────────────────────────────────────────

ELBOWS_THRESHOLD: float = 0.357
KNEES_THRESHOLD: float = 0.476
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337)


# ─────────────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────────────


class OHPFormService:
    """Load the MD-SSL 2-seed ensemble and run EXACT inference path.

    Construction:
        svc = OHPFormService(model_dir="backend/models/form_model_ohp_md")

    Usage:
        # Synchronous (call from threadpool):
        result = svc.classify_clip(frames_tchw_uint8)

        # Async (use inside FastAPI endpoint handlers):
        result = await svc.classify_clip_async(frames_tchw_uint8)

    model_dir layout:
        model_dir/
          seed42/best.pt
          seed1337/best.pt

    Each best.pt is a fine-tune checkpoint with keys:
        model_state_dict, epoch, optimizer_state_dict, ...
    The full fine-tuned model (backbone + Dropout(0.2)+Linear(512,2) head) is stored
    under "model_state_dict" — loaded with strict=True.
    """

    def __init__(
        self,
        model_dir: str,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        elbows_threshold: float = ELBOWS_THRESHOLD,
        knees_threshold: float = KNEES_THRESHOLD,
    ) -> None:
        self.model_dir = model_dir
        self.elbows_threshold = elbows_threshold
        self.knees_threshold = knees_threshold
        self._models: list[nn.Module] = []
        self._model_ready: bool = False

        n_seeds_env = int(os.environ.get("OHP_INFERENCE_SEEDS", len(seeds)))
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
        """
        for seed in seeds:
            pt_path = Path(self.model_dir) / f"seed{seed}" / "best.pt"
            if not pt_path.exists():
                logger.warning(
                    "OHPFormService: seed%d weight not found at %s — skipping.",
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
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"], strict=True)
            m.eval()
            self._models.append(m)
            logger.info(
                "OHPFormService: loaded seed%d from %s (state_dict keys: %d).",
                seed,
                pt_path,
                len(ckpt["model_state_dict"]),
            )

        self._model_ready = len(self._models) > 0
        logger.info(
            "OHPFormService: model_ready=%s, seeds_loaded=%d/%d.",
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
        """Run the EXACT inference path and return the raw-output response schema.

        SYNCHRONOUS — blocks on CPU fp32.
        Call via classify_clip_async from async endpoint handlers to avoid blocking
        the uvicorn event loop.

        Args:
            frames_tchw_uint8: uint8 ndarray [T, 3, H, W] from decode_clip.
            n_seeds:           number of seeds to use (default: self._default_n_seeds).

        Returns:
            Dict with keys: exercise, errors (list of ELBOWS + KNEES dicts).
            When model_ready is False, returns _neutral_response() (no exception).
        """
        if not self._model_ready:
            return self._neutral_response()

        # spatial_val: plain center crop, matching the OHP training preprocessing exactly.
        # OHP training used spatial_val; kneeaware_spatial_val is squat-only.
        clip = spatial_val(torch.from_numpy(frames_tchw_uint8))
        batch = clip.unsqueeze(0)  # [1, 3, T, 112, 112]

        n = n_seeds if n_seeds is not None else self._default_n_seeds

        # Pass RAW logits to aggregate_sigmoid_mean — sigmoid applied internally.
        per_seed_logits: list[np.ndarray] = []
        with torch.no_grad():
            for m in self._models[:n]:
                per_seed_logits.append(m(batch).numpy())  # [1, 2]

        scores = aggregate_sigmoid_mean(per_seed_logits)  # [1, 2] in [0, 1]
        elbows_s = float(scores[0, 0])
        knees_s = float(scores[0, 1])
        return self._build_response(elbows_s, knees_s)

    async def classify_clip_async(
        self,
        frames_tchw_uint8: np.ndarray,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Async wrapper for classify_clip — required for use in FastAPI endpoint handlers.

        Offloads the synchronous CPU-bound forward to a thread via Starlette's
        run_in_threadpool.
        """
        fn = functools.partial(self.classify_clip, frames_tchw_uint8, n_seeds=n_seeds)
        return await run_in_threadpool(fn)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(self, elbows_s: float, knees_s: float) -> dict:
        """Assemble the raw-output response dict from sigmoid scores.

        intervals=[] here — per-clip timing is attached by the endpoint handler,
        not the service (the service is stateless).
        """
        elbows_det = elbows_s >= self.elbows_threshold
        knees_det = knees_s >= self.knees_threshold
        return {
            "exercise": "ohp",
            "errors": [
                {
                    "type": "ELBOWS",
                    "detected": elbows_det,
                    "confidence": round(elbows_s, 4),
                    "threshold": self.elbows_threshold,
                    "intervals": [],
                },
                {
                    "type": "KNEES",
                    "detected": knees_det,
                    "confidence": round(knees_s, 4),
                    "threshold": self.knees_threshold,
                    "intervals": [],
                },
            ],
        }

    def _neutral_response(self) -> dict:
        """Response returned when model_ready is False — no exception raised."""
        return {
            "exercise": "ohp",
            "errors": [
                {
                    "type": "ELBOWS",
                    "detected": False,
                    "confidence": 0.0,
                    "threshold": self.elbows_threshold,
                    "intervals": [],
                },
                {
                    "type": "KNEES",
                    "detected": False,
                    "confidence": 0.0,
                    "threshold": self.knees_threshold,
                    "intervals": [],
                },
            ],
            "model_not_loaded": True,
        }
