"""Phase-gate assertion framework for the QEVD v6 pipeline.

Every phase D0-D9 calls assert_phase_passed(prior_gate) on entry and
write_gate_report(this_gate, ...) on exit. No phase has a soft-failure path.
This is the spine of the "zero silent bugs" mandate.

Spec: plans/i-am-now-on-zazzy-brooks.md §II.5
API:  plans/phase-1-complete-critical-snappy-flurry.md §D0.4
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

# This file lives at backend/training/diagnostics/phase_gates.py
# parents[3] resolves to the repo root regardless of caller cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "backend" / "data" / "qevd_phase_reports"

GATE_ORDER: list[str] = [
    "D0_tooling",
    "D1_extract_integrity",
    "D2_extract_quality",
    "D3_label_sanity",
    "D4_dataset_assembly",
    "D5_smoke",
    "D6_val_curves",
    "D7_in_domain",
    "D8_ood",
    "D9_reality_check",
]

GateStatus = Literal["PASS", "FAIL", "PARTIAL"]
_VALID_STATUSES: tuple[GateStatus, ...] = ("PASS", "FAIL", "PARTIAL")


class GatePendingError(RuntimeError):
    """Raised when a required prior gate has no report on disk."""


class GateFailedError(RuntimeError):
    """Raised when a required prior gate's status is FAIL."""


@dataclass
class GateContext:
    gate_id: str
    metrics: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    forced_status: GateStatus | None = None


def _resolve_dir(reports_dir: Path | None) -> Path:
    return reports_dir if reports_dir is not None else REPORTS_DIR


def _report_path(gate_id: str, reports_dir: Path | None = None) -> Path:
    return _resolve_dir(reports_dir) / f"{gate_id}.json"


def assert_phase_passed(
    gate_id: str, *, reports_dir: Path | None = None,
) -> dict:
    """Read REPORTS_DIR/<gate_id>.json. Raise on missing/non-PASS."""
    p = _report_path(gate_id, reports_dir)
    if not p.exists():
        raise GatePendingError(
            f"Gate '{gate_id}' has no report at {p}. Run the corresponding "
            "phase before continuing."
        )
    report = json.loads(p.read_text(encoding="utf-8"))
    status = report.get("status")
    if status != "PASS":
        raise GateFailedError(
            f"Gate '{gate_id}' status is {status!r} (expected PASS). "
            f"Failures: {report.get('failures', [])}. Fix the underlying "
            "issue and rewrite the report before continuing."
        )
    return report


def write_gate_report(
    gate_id: str,
    status: GateStatus,
    metrics: dict,
    failures: list,
    *,
    reports_dir: Path | None = None,
) -> Path:
    """Write the gate report JSON and print a human-readable summary."""
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"status must be one of {_VALID_STATUSES}, got {status!r}"
        )

    out_dir = _resolve_dir(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "gate_id":   gate_id,
        "status":    status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics":   metrics,
        "failures":  failures,
    }

    out_path = out_dir / f"{gate_id}.json"
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )

    banner = "=" * 60
    print(banner, file=sys.stderr)
    print(f"GATE REPORT: {gate_id} -- {status}", file=sys.stderr)
    print(banner, file=sys.stderr)
    if metrics:
        print("Metrics:", file=sys.stderr)
        for k, v in sorted(metrics.items()):
            print(f"  {k} = {v}", file=sys.stderr)
    if failures:
        print(f"Failures ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  [X] {f}", file=sys.stderr)
    else:
        print("Failures: 0", file=sys.stderr)
    print(banner, file=sys.stderr)
    return out_path


def expect(
    condition: bool,
    metric_name: str,
    actual: Any,
    bound: str,
    failures: list,
) -> None:
    """If condition is False, append a failure record. Always log to stderr."""
    glyph = "[OK]" if condition else "[FAIL]"
    print(
        f"  {glyph} {metric_name} = {actual!r} (expected {bound})",
        file=sys.stderr,
    )
    if not condition:
        failures.append(
            {"metric": metric_name, "actual": _coerce_jsonable(actual),
             "bound": bound}
        )


def _coerce_jsonable(value: Any) -> Any:
    """Best-effort coercion so ``actual`` can land in JSON."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _prior_gate(gate_id: str) -> str | None:
    """Return the gate that must PASS before `gate_id`, or None if first."""
    if gate_id not in GATE_ORDER:
        return None
    idx = GATE_ORDER.index(gate_id)
    return GATE_ORDER[idx - 1] if idx > 0 else None


@contextmanager
def gate(
    gate_id: str,
    *,
    skip_prior_check: bool = False,
    reports_dir: Path | None = None,
) -> Iterator[GateContext]:
    """Context manager: assert prior gate passed on enter; write the report
    on exit based on whether failures are empty (or forced_status is set).

    Re-raises any exception from the body AFTER writing a FAIL report,
    so the caller's traceback is preserved while the gate JSON still lands
    on disk for committee evidence.
    """
    if not skip_prior_check:
        prior = _prior_gate(gate_id)
        if prior is not None:
            assert_phase_passed(prior, reports_dir=reports_dir)

    ctx = GateContext(gate_id=gate_id)
    body_exception: BaseException | None = None
    try:
        yield ctx
    except BaseException as e:
        body_exception = e
        ctx.failures.append({
            "metric": "exception",
            "actual": f"{type(e).__name__}: {e}",
            "bound":  "no exception raised in gate body",
        })
    finally:
        if body_exception is not None:
            status: GateStatus = "FAIL"
        elif ctx.forced_status is not None:
            status = ctx.forced_status
        else:
            status = "PASS" if not ctx.failures else "FAIL"
        write_gate_report(
            ctx.gate_id, status, ctx.metrics, ctx.failures,
            reports_dir=reports_dir,
        )

    if body_exception is not None:
        raise body_exception


__all__ = [
    "GATE_ORDER",
    "GateContext",
    "GateFailedError",
    "GatePendingError",
    "REPORTS_DIR",
    "assert_phase_passed",
    "expect",
    "gate",
    "write_gate_report",
]
