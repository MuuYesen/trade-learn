from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_backtrader_benchmark_runner_completes() -> None:
    result = subprocess.run(
        [sys.executable, "tests/runners/benchmark_bt.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "FAILED" not in result.stdout
    assert "❌ DIFF" not in result.stdout
    assert "QuickstartSmaCross" in result.stdout
    assert "OrderExecutionStrategy" in result.stdout
    assert "vs Prev TL" in result.stdout


def test_backtesting_compare_results_runner_completes() -> None:
    result = subprocess.run(
        [sys.executable, "compare_results.py"],
        cwd=ROOT / "examples" / "backtesting",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "RESULTS COMPARISON: BTCUSDT" in result.stdout
    assert "RESULTS COMPARISON: ETHUSDT" in result.stdout
