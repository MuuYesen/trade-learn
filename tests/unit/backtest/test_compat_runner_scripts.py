from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.runners.benchmark_bt import _benchmark_passed

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


def test_backtrader_benchmark_gate_enforces_exact_and_min_speedup() -> None:
    exact_fast = {
        "SmaCross": {
            "Tradelearn": {"final_value": 100.0, "elapsed_ms": 5.0},
            "Backtrader": {"final_value": 100.0, "elapsed_ms": 10.0},
        }
    }
    exact_slow = {
        "SmaCross": {
            "Tradelearn": {"final_value": 100.0, "elapsed_ms": 9.5},
            "Backtrader": {"final_value": 100.0, "elapsed_ms": 10.0},
        }
    }
    mismatch = {
        "SmaCross": {
            "Tradelearn": {"final_value": 100.1, "elapsed_ms": 5.0},
            "Backtrader": {"final_value": 100.0, "elapsed_ms": 10.0},
        }
    }

    assert _benchmark_passed(exact_fast, min_speedup=1.2) is True
    assert _benchmark_passed(exact_slow, min_speedup=1.2) is False
    assert _benchmark_passed(mismatch, min_speedup=1.2) is False


def test_backtesting_compare_results_runner_completes() -> None:
    result = subprocess.run(
        [sys.executable, "tests/runners/compare_backtesting.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "RESULTS COMPARISON: BTCUSDT" in result.stdout
    assert "RESULTS COMPARISON: ETHUSDT" in result.stdout
    assert "Bars/s" in result.stdout
