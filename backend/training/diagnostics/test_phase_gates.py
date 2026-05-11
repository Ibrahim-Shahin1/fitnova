"""Unit tests for backend.training.diagnostics.phase_gates.

Run from repo root:
    pytest backend/training/diagnostics/test_phase_gates.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training.diagnostics.phase_gates import (
    GateFailedError,
    GatePendingError,
    assert_phase_passed,
    expect,
    gate,
    write_gate_report,
)


def _make_report(
    tmp_path: Path,
    gate_id: str,
    status: str,
    metrics: dict | None = None,
    failures: list | None = None,
) -> Path:
    report = {
        "gate_id":   gate_id,
        "status":    status,
        "timestamp": "2026-05-06T00:00:00+00:00",
        "metrics":   metrics or {},
        "failures":  failures or [],
    }
    path = tmp_path / f"{gate_id}.json"
    path.write_text(json.dumps(report))
    return path


# 1
def test_assert_phase_passed_raises_when_file_missing(tmp_path):
    with pytest.raises(GatePendingError, match="no report"):
        assert_phase_passed("D0_tooling", reports_dir=tmp_path)


# 2
def test_assert_phase_passed_raises_when_status_fail(tmp_path):
    _make_report(
        tmp_path, "D0_tooling", "FAIL",
        failures=[{"metric": "x", "actual": 1, "bound": ">2"}],
    )
    with pytest.raises(GateFailedError, match="FAIL"):
        assert_phase_passed("D0_tooling", reports_dir=tmp_path)


# 3
def test_assert_phase_passed_returns_dict_when_pass(tmp_path):
    _make_report(
        tmp_path, "D0_tooling", "PASS",
        metrics={"installed": True},
    )
    result = assert_phase_passed("D0_tooling", reports_dir=tmp_path)
    assert result["status"] == "PASS"
    assert result["metrics"]["installed"] is True


# 4
def test_write_gate_report_produces_documented_keys(tmp_path):
    out_path = write_gate_report(
        "D1_extract_integrity", "PASS",
        metrics={"num_clips": 303117}, failures=[],
        reports_dir=tmp_path,
    )
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert set(report.keys()) >= {
        "gate_id", "status", "timestamp", "metrics", "failures",
    }
    assert report["gate_id"] == "D1_extract_integrity"
    assert report["status"] == "PASS"
    assert report["metrics"]["num_clips"] == 303117


# 5
def test_expect_appends_failure_when_condition_false():
    failures: list = []
    expect(False, "noisy_metric", 0.42, "<0.10", failures)
    assert len(failures) == 1
    assert failures[0]["metric"] == "noisy_metric"
    assert failures[0]["actual"] == 0.42
    assert failures[0]["bound"] == "<0.10"


# 6
def test_expect_no_append_when_condition_true():
    failures: list = []
    expect(True, "good_metric", 0.05, "<0.10", failures)
    assert failures == []


# 7
def test_gate_raises_pending_when_prior_missing(tmp_path):
    # D2_extract_quality requires D1_extract_integrity; absent -> Pending.
    with pytest.raises(GatePendingError):
        with gate("D2_extract_quality", reports_dir=tmp_path) as g:
            g.metrics["x"] = 1


# 8
def test_gate_writes_fail_when_failures_non_empty(tmp_path):
    with gate(
        "D0_tooling", skip_prior_check=True, reports_dir=tmp_path,
    ) as g:
        expect(False, "missing_thing", 0, ">0", g.failures)
    out = tmp_path / "D0_tooling.json"
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["status"] == "FAIL"
    assert len(report["failures"]) == 1
    assert report["failures"][0]["metric"] == "missing_thing"


# Bonus: verify gate writes PASS when failures empty
def test_gate_writes_pass_when_failures_empty(tmp_path):
    with gate(
        "D0_tooling", skip_prior_check=True, reports_dir=tmp_path,
    ) as g:
        g.metrics["installed"] = True
        expect(True, "always_true", 1, "==1", g.failures)
    report = json.loads((tmp_path / "D0_tooling.json").read_text())
    assert report["status"] == "PASS"
    assert report["metrics"]["installed"] is True


# Bonus: forced_status overrides automatic resolution
def test_gate_honours_forced_partial(tmp_path):
    with gate(
        "D0_tooling", skip_prior_check=True, reports_dir=tmp_path,
    ) as g:
        g.forced_status = "PARTIAL"
        g.metrics["installed"] = False
    report = json.loads((tmp_path / "D0_tooling.json").read_text())
    assert report["status"] == "PARTIAL"


# Bonus: exceptions in body still write a FAIL report and re-raise
def test_gate_writes_fail_and_reraises_on_exception(tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with gate(
            "D0_tooling", skip_prior_check=True, reports_dir=tmp_path,
        ):
            raise ValueError("boom")
    report = json.loads((tmp_path / "D0_tooling.json").read_text())
    assert report["status"] == "FAIL"
    assert any("boom" in f["actual"] for f in report["failures"])
