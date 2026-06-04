"""ShallowSquatFormService — PyTorch ResNet-18 3-seed image classifier for Shallow-Squat depth-error detection.

Loaded once at FastAPI startup into app.state.shallow_form_service.  Shared read-only
across all sessions — NO per-call mutable state.

Structurally different from the video services: single-frame image classifier.
Pipeline: PIL open -> Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet norm ->
3-seed inline scalar mean-of-sigmoids -> single DEPTH error at threshold 0.395.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from starlette.concurrency import run_in_threadpool
from torchvision.models import resnet18

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module constants
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_THRESHOLD: float = 0.395
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 7)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Val/test transform matching shallow_squat.py:86-92 exactly — no random crops at serve time.
_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────────────


class ShallowSquatFormService:
    """Load the CVCSPC 3-seed ResNet-18 ensemble and run single-frame inference.

    Construction:
        svc = ShallowSquatFormService(model_dir="backend/models/form_model_shallow_cvcspc")

    Usage:
        # Synchronous (call from threadpool):
        result = svc.classify_image(image_path)

        # Async (use inside FastAPI endpoint handlers):
        result = await svc.classify_image_async(image_path)

    model_dir layout:
        model_dir/
          seed42/best.pt
          seed1337/best.pt
          seed7/best.pt

    Each best.pt is a fine-tune checkpoint with keys:
        model_state_dict, epoch, optimizer_state_dict, ...
    The full fine-tuned ResNet-18 (backbone + Linear(512,1) head) is stored
    under "model_state_dict" — 122 keys, loaded with strict=True.
    """

    def __init__(
        self,
        model_dir: str,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        threshold: float = DEPTH_THRESHOLD,
    ) -> None:
        self.model_dir = model_dir
        self.threshold = threshold
        self._models: list[nn.Module] = []
        self._model_ready: bool = False

        # SHALLOW_INFERENCE_SEEDS env knob: number of seeds used per classify_image call.
        # Clamped to [1, len(seeds)] so it is always safe to slice self._models[:n].
        n_seeds_env = int(os.environ.get("SHALLOW_INFERENCE_SEEDS", len(seeds)))
        self._default_n_seeds: int = max(1, min(n_seeds_env, len(seeds)))

        self._load(seeds)

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, seeds: tuple[int, ...]) -> None:
        """Load each seed's best.pt into an eval-mode ResNet-18 model.

        Missing seeds are skipped with a warning (graceful degradation — model_ready
        stays False if ALL seeds are missing).  A corrupt checkpoint that raises during
        torch.load surfaces at startup (fail-fast), not per-request.

        File-presence check per seed; missing -> skip, not crash.
        """
        for seed in seeds:
            pt_path = Path(self.model_dir) / f"seed{seed}" / "best.pt"
            if not pt_path.exists():
                logger.warning(
                    "ShallowSquatFormService: seed%d weight not found at %s — skipping.",
                    seed,
                    pt_path,
                )
                continue

            # weights=None — the fine-tuned checkpoint carries the weights (not ImageNet init).
            m = resnet18(weights=None)
            assert m.fc.in_features == 512, (
                f"ResNet-18 fc.in_features={m.fc.in_features}, expected 512"
            )
            m.fc = nn.Linear(512, 1)

            # strict=True — best.pt has the full fine-tuned model (122 keys, fc.weight (1,512)).
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"], strict=True)
            m.eval()
            self._models.append(m)
            logger.info(
                "ShallowSquatFormService: loaded seed%d from %s (state_dict keys: %d).",
                seed,
                pt_path,
                len(ckpt["model_state_dict"]),
            )

        self._model_ready = len(self._models) > 0
        logger.info(
            "ShallowSquatFormService: model_ready=%s, seeds_loaded=%d/%d.",
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

    def classify_image(
        self,
        image_path: str,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Run single-frame inference and return the raw DEPTH error dict.

        SYNCHRONOUS — blocks on PIL decode + forward pass.
        Call via classify_image_async from async endpoint handlers.

        Args:
            image_path: Path to a JPEG crop (crops_unaligned/{id}.jpg).
            n_seeds:    Number of seeds to use (default: self._default_n_seeds).

        Returns:
            Dict with keys: exercise, errors (list with one DEPTH dict).
            When model_ready is False, returns _neutral_response() (no exception).
        """
        if not self._model_ready:
            return self._neutral_response()

        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224] float32

        n = n_seeds if n_seeds is not None else self._default_n_seeds
        per_seed_scalars: list[float] = []
        with torch.no_grad():
            for m in self._models[:n]:
                per_seed_scalars.append(float(m(tensor).item()))

        # Inline scalar mean-of-sigmoids — shallow logits are scalars, not (N,2) arrays.
        score = float(np.mean([1.0 / (1.0 + np.exp(-l)) for l in per_seed_scalars]))
        detected = score >= self.threshold
        return self._build_response(score, detected)

    async def classify_image_async(
        self,
        image_path: str,
        *,
        n_seeds: int | None = None,
    ) -> dict:
        """Async wrapper for classify_image — required for use in FastAPI endpoint handlers.

        Offloads the synchronous CPU-bound forward to a thread via Starlette's
        run_in_threadpool, which is the FastAPI-idiomatic pattern for blocking compute.
        """
        fn = functools.partial(self.classify_image, image_path, n_seeds=n_seeds)
        return await run_in_threadpool(fn)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(self, score: float, detected: bool) -> dict:
        """Assemble the raw DEPTH response dict from the sigmoid score.

        intervals=[] — timing is not applicable for single-image inference.
        """
        return {
            "exercise": "shallow",
            "errors": [
                {
                    "type": "DEPTH",
                    "detected": detected,
                    "confidence": round(score, 4),
                    "threshold": self.threshold,
                    "intervals": [],
                }
            ],
        }

    def _neutral_response(self) -> dict:
        """Raw DEPTH schema returned when model_ready is False — no exception raised.

        Neutral degradation rather than 500 error.
        """
        return {
            "exercise": "shallow",
            "errors": [
                {
                    "type": "DEPTH",
                    "detected": False,
                    "confidence": 0.0,
                    "threshold": self.threshold,
                    "intervals": [],
                }
            ],
            "model_not_loaded": True,
        }
