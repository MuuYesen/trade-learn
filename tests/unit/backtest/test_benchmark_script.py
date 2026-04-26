from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_stage3_benchmark.py"
BASELINE = ROOT / "benchmarks" / "baseline.json"


def test_stage3_benchmark_script_emits_machine_readable_results() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--single-bars",
            "5",
            "--portfolio-bars",
            "4",
            "--portfolio-symbols",
            "2",
            "--max-single-ms",
            "1000",
            "--max-portfolio-ms",
            "1000",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["single"]["bars"] == 5
    assert payload["portfolio"]["bars"] == 4
    assert payload["portfolio"]["symbols"] == 2
    assert payload["single"]["elapsed_ms"] >= 0.0
    assert payload["portfolio"]["elapsed_ms"] >= 0.0


def test_stage3_benchmark_script_fails_when_threshold_is_exceeded() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--single-bars",
            "5",
            "--portfolio-bars",
            "4",
            "--portfolio-symbols",
            "2",
            "--max-single-ms",
            "0",
            "--max-portfolio-ms",
            "0",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["single"]["ok"] is False
    assert payload["portfolio"]["ok"] is False


def test_stage3_full_benchmark_result_is_recorded_in_baseline() -> None:
    payload = json.loads(BASELINE.read_text())

    stage3 = payload["measured"]["stage3_backtest"]

    assert stage3["single"]["bars"] == 2520
    assert stage3["single"]["symbols"] == 1
    assert stage3["single"]["elapsed_ms"] <= payload["targets"]["single_symbol_10y_daily_ms"]
    assert stage3["portfolio"]["bars"] == 2520
    assert stage3["portfolio"]["symbols"] == 500
    assert stage3["portfolio"]["elapsed_ms"] <= (
        payload["targets"]["portfolio_500_symbols_10y_daily_s"] * 1000
    )
