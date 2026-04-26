from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_golden_readiness_reports_missing_real_artifacts_as_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_golden_readiness.py", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["summary"]["datasets_total"] == 10
    assert 0 <= payload["summary"]["datasets_ready"] < 10
    assert payload["summary"]["expected_total"] == 100
    assert payload["summary"]["expected_ready"] == 0
    assert payload["summary"]["strategies_total"] == 10
    assert payload["summary"]["strategies_ready"] == 0
    if payload["blockers"]["datasets"]:
        assert payload["blockers"]["datasets"][0]["reason"] == "missing real parquet"
    assert payload["blockers"]["expected"][0]["reason"] == "missing expected v1.0 result"
    assert payload["blockers"]["strategies"][0]["reason"] == "adapter still blocks execution"


def test_golden_readiness_text_output_is_actionable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_golden_readiness.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "golden-readiness:ok=False" in result.stdout
    assert "datasets=" in result.stdout
    assert "/10" in result.stdout
    assert "expected=0/100" in result.stdout
    assert "strategies=0/10" in result.stdout
    assert "blocked: missing expected v1.0 result" in result.stdout
