from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_golden_manifest_available_for_consistency_suite() -> None:
    manifest = json.loads((ROOT / "tests" / "golden" / "manifest.json").read_text())

    assert manifest["version"] == "0.1-alpha"
    assert len(manifest["datasets"]) * len(manifest["strategies"]) == 100


@pytest.mark.parametrize(
    "future_check",
    [
        "test_metrics.py",
        "test_indicators_core.py",
        "test_indicators_tdx.py",
        "test_indicators_tv.py",
        "test_backtest_rust.py",
        "test_factor.py",
        "test_report.py",
        "test_e2e_pipeline.py",
    ],
)
def test_future_consistency_checks_are_stage_gated(future_check: str) -> None:
    pytest.skip(f"{future_check} is specified in docs/specs/CONSISTENCY.md for later stages")
