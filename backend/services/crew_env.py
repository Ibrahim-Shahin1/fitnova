"""
crew_env — guarantees the isolated CrewAI virtualenv (backend/.crewenv) is present
and functional before the backend starts serving.

The 4-agent plan pipeline (plan_crew.py) runs the REAL crew in this dedicated venv
as a subprocess, because CrewAI's dependency tree (chromadb, onnxruntime, litellm,
protobuf<6) conflicts with the backend's TensorFlow/mediapipe stack. That venv is
gitignored, so a clean checkout, a disk/AV cleanup, or a stray `git clean` can leave
it missing — which previously surfaced mid-use as "Crew environment not set up".

This module is the startup self-heal: verify crewai is installed in the venv and,
if it is not, rebuild the venv in place from requirements-crew.txt. It restores the
real crew — it is NOT a fallback and never substitutes a different plan generator.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from backend.services import plan_crew

logger = logging.getLogger("fitnova.crew.env")

_CREW_PY = plan_crew._CREW_PY
_VENV_DIR = os.path.abspath(os.path.join(plan_crew._BACKEND_DIR, ".crewenv"))
_REQUIREMENTS = os.path.abspath(
    os.path.join(plan_crew._BACKEND_DIR, "requirements-crew.txt"))

# An explicit interpreter override means the user manages that env themselves —
# verify it, but never create a venv at an arbitrary overridden path.
_USING_OVERRIDE = bool(os.environ.get("FITNOVA_CREW_PYTHON"))


def _crewai_version() -> str | None:
    """Installed crewai version in the crew venv, or None if the venv python is
    absent or crewai is not installed. Fast — reads package metadata, does not
    import the heavy crewai tree (keeps `--reload` startups snappy)."""
    if not os.path.exists(_CREW_PY):
        return None
    try:
        proc = subprocess.run(
            [_CREW_PY, "-c",
             "import importlib.metadata as m; print(m.version('crewai'))"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else None


def ensure_crew_env(*, install_timeout: int = 1200) -> dict:
    """Verify the crew venv is healthy; rebuild it in place if not.

    Idempotent and fast (~0.5s) when already healthy. Blocks during a rebuild
    (auto-rebuild-on-startup), so the real 4-agent crew is guaranteed available
    once this returns ok. Returns {ok, version, rebuilt, error}.
    """
    version = _crewai_version()
    if version:
        return {"ok": True, "version": version, "rebuilt": False, "error": None}

    if _USING_OVERRIDE:
        return {"ok": False, "version": None, "rebuilt": False,
                "error": f"FITNOVA_CREW_PYTHON points at '{_CREW_PY}', which has no "
                         "working crewai install (refusing to auto-create a venv at "
                         "an overridden path)."}

    logger.warning("CrewAI venv missing/incomplete at %s — rebuilding from %s "
                   "(one-time; this can take a few minutes)...",
                   _VENV_DIR, _REQUIREMENTS)
    try:
        # `python -m venv --clear` only when the interpreter is gone (a partial
        # install just needs the deps reinstalled, not a full venv rebuild).
        if not os.path.exists(_CREW_PY):
            subprocess.run([sys.executable, "-m", "venv", "--clear", _VENV_DIR],
                           timeout=300, check=True)
        # Output is inherited (not captured) so pip's progress is visible during
        # the multi-minute install instead of the server looking frozen.
        subprocess.run(
            [_CREW_PY, "-m", "pip", "install", "--disable-pip-version-check",
             "-r", _REQUIREMENTS],
            timeout=install_timeout, check=True)
    except Exception as exc:
        return {"ok": False, "version": None, "rebuilt": True, "error": str(exc)[:600]}

    version = _crewai_version()
    if version:
        logger.info("CrewAI venv rebuilt successfully — crewai %s", version)
        return {"ok": True, "version": version, "rebuilt": True, "error": None}
    return {"ok": False, "version": None, "rebuilt": True,
            "error": "crewai still not importable after rebuild"}
