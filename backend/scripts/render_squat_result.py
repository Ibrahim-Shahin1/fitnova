"""Render the Squat KIE/KFE inference result overlaid on sample frames from a clip.

Phase-5 visualization deliverable ([[project_supervisor_visualizations]]).
Produces a matplotlib figure showing 6 sample frames and a KIE/KFE detection
legend panel with detected status, severity word, and confidence.

Usage:
    python -m backend.scripts.render_squat_result <clip.mp4> [--out figure.png]

    If --out is not given, the figure is saved to:
        .planning/phases/05-backend-inference-integration-squat/figures/squat_result_overlay.png

Requires staged weights in backend/models/form_model_squat_md/ (Plan 02, D-09).
If the model is not ready, the script prints a diagnostic and exits 1.

IMPORTANT: fp32 only -- do NOT add .half() calls.
fp16 hangs on CPU with PyTorch 2.12 on Windows (Pitfall 4).
Uses matplotlib.use("Agg") for headless rendering.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering -- must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "backend" / "models" / "form_model_squat_md"
_DEFAULT_OUT = (
    _REPO_ROOT
    / ".planning"
    / "phases"
    / "05-backend-inference-integration-squat"
    / "figures"
    / "squat_result_overlay.png"
)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap imports
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(_REPO_ROOT))

from backend.services.clip_decode import decode_clip_cv2, get_frame_count_and_fps  # noqa: E402
from backend.services.squat_form_service import SquatFormService                   # noqa: E402
from backend.training.aqa.datasets.transforms import uniform_sample_indices         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("render_squat_result")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLE_FRAMES: int = 32  # standard training window — full 32-frame clip
N_DISPLAY_FRAMES: int = 6  # how many frames to show in the figure


def _error_color(detected: bool) -> str:
    return "#d9534f" if detected else "#5cb85c"  # red=detected, green=clear


def render(clip_path: str, out_path: str) -> None:
    """Run inference on the clip and save a visualization PNG to out_path."""
    # 1. Load service
    svc = SquatFormService(model_dir=str(_MODEL_DIR))
    if not svc.model_ready:
        print(
            "ERROR: SquatFormService could not load any model weights from "
            f"{_MODEL_DIR}. Stage the Phase-4 best.pt files first "
            "(see backend/models/form_model_squat_md/README.md)."
        )
        sys.exit(1)

    # 2. Sample a representative 32-frame window using cv2 helpers
    #    (decode_clip_cv2 replaces the dead read_video_timestamps on torchvision 0.27)
    total, fps = get_frame_count_and_fps(clip_path)
    logger.info("Clip: %s | frames=%d  fps=%.1f", clip_path, total, fps)
    idx = uniform_sample_indices(total, N_SAMPLE_FRAMES, jitter=0)  # deterministic
    frames = decode_clip_cv2(clip_path, idx)  # uint8 [T, 3, H, W] RGB

    # 3. Run classify_clip (fp32 only -- no .half())
    result = svc.classify_clip(frames.numpy())
    logger.info("Inference result: %s", result)

    # ── Extract per-error info ────────────────────────────────────────────────
    errors = {e["type"]: e for e in result.get("errors", [])}
    kie = errors.get("KIE", {"detected": False, "confidence": 0.0, "severity_word": "none"})
    kfe = errors.get("KFE", {"detected": False, "confidence": 0.0, "severity_word": "none"})

    # 4. Pick 6 evenly-spaced display frames from the 32 sampled (indices into frames)
    t = frames.shape[0]
    display_indices = [int(i) for i in np.linspace(0, t - 1, N_DISPLAY_FRAMES, dtype=int)]
    display_frames = [frames[i].permute(1, 2, 0).numpy() for i in display_indices]
    # frames are uint8 RGB [H, W, 3] — matplotlib imshow expects this directly

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Render the figure
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5), dpi=120)
    fig.patch.set_facecolor("#1a1a2e")

    # 5a. Frame grid (top row)
    for col, frame in enumerate(display_frames):
        ax = fig.add_axes([col / N_DISPLAY_FRAMES, 0.38, 1 / N_DISPLAY_FRAMES - 0.005, 0.55])
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Frame {idx[display_indices[col]].item()}",
            fontsize=7,
            color="#cccccc",
            pad=2,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # 5b. Legend panel (bottom row) -- two side-by-side boxes for KIE and KFE
    for i, (label, err) in enumerate([("KIE — Knee Inward Error", kie), ("KFE — Knee Forward Error", kfe)]):
        ax = fig.add_axes([i * 0.5 + 0.02, 0.02, 0.46, 0.32])
        ax.set_facecolor("#12122a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        detected = err["detected"]
        severity = err["severity_word"]
        confidence = err["confidence"]
        color = _error_color(detected)
        status_text = "DETECTED" if detected else "CLEAR"

        ax.text(
            0.5, 0.82, label,
            ha="center", va="center", fontsize=10, color="#cccccc",
            fontweight="bold", transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.57, status_text,
            ha="center", va="center", fontsize=22, color=color,
            fontweight="bold", transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.33, severity.upper(),
            ha="center", va="center", fontsize=13, color=color,
            fontstyle="italic", transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.13, f"confidence: {confidence:.3f}",
            ha="center", va="center", fontsize=8, color="#888899",
            transform=ax.transAxes,
        )
        # Colored indicator strip on the left edge
        patch = mpatches.FancyBboxPatch(
            (0.01, 0.06), 0.04, 0.88,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)

    # Title bar
    clip_name = Path(clip_path).name
    fig.text(
        0.5, 0.975,
        f"Squat Form Analysis — {clip_name}",
        ha="center", va="top",
        fontsize=11, color="#e0e0ff", fontweight="bold",
    )
    note_text = (
        "KIE recall is modest (~47%) — 'possible' signals directional risk, not certainty.  "
        "KFE is more reliable (~70% recall).  Severity: strong ≥0.80, moderate ≥0.65, possible <0.65."
    )
    fig.text(
        0.5, 0.005, note_text,
        ha="center", va="bottom",
        fontsize=6.5, color="#666688",
    )

    # 6. Save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure saved to: {out}")

    # 7. Print result summary
    print("\n-- Inference Result --")
    print(f"  KIE detected={kie['detected']}  confidence={kie['confidence']:.4f}"
          f"  severity={kie['severity_word']}")
    print(f"  KFE detected={kfe['detected']}  confidence={kfe['confidence']:.4f}"
          f"  severity={kfe['severity_word']}")
    print("----------------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Squat KIE/KFE inference result on a clip."
    )
    parser.add_argument("clip", help="Path to an .mp4 squat clip")
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Output PNG path (default: phase figures/ dir)",
    )
    args = parser.parse_args()
    render(clip_path=args.clip, out_path=args.out)


if __name__ == "__main__":
    main()
