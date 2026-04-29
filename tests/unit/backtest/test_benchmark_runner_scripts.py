from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.runners.benchmark_bt import _benchmark_passed
from benchmarks.runners.benchmark_throughput import make_data, run_benchmark, run_engine, run_lite

ROOT = Path(__file__).resolve().parents[3]


def test_backtrader_benchmark_runner_completes() -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/runners/benchmark_bt.py"],
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
    assert "TL Bars/s" in result.stdout
    assert "BT Bars/s" in result.stdout


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


def test_throughput_benchmark_reports_bars_per_second_without_backtesting_py() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/runners/benchmark_throughput.py",
            "--bars",
            "1000",
            "--no-backtrader",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Tradelearn Engine" in result.stdout
    assert "Tradelearn Lite" in result.stdout
    assert "Bars/s" in result.stdout
    assert "Fills" in result.stdout
    assert "Closed Trades" in result.stdout
    assert "| {'Trades':>6}" not in result.stdout
    assert "419,552" in result.stdout
    assert "backtesting.py" not in result.stdout


def test_throughput_benchmark_function_returns_results() -> None:
    results = run_benchmark(n_bars=1000, repeat=1, warmup=0, include_backtrader=False)

    assert [result.name for result in results] == ["Tradelearn Engine", "Tradelearn Lite"]
    assert all(result.bars_per_sec > 0 for result in results)
    assert all(result.fills >= result.closed_trades for result in results)
    engine, lite = results
    assert engine.final_value == lite.final_value
    assert engine.fills == lite.fills
    assert engine.closed_trades == lite.closed_trades


def test_throughput_engine_and_lite_trade_on_close_are_same_semantics() -> None:
    data = make_data(1000)

    engine_value, engine_fills, engine_closed = run_engine(data, trade_on_close=True)
    lite_value, lite_fills, lite_closed = run_lite(data, trade_on_close=True)

    assert engine_value == lite_value
    assert engine_fills == lite_fills
    assert engine_closed == lite_closed


def test_throughput_benchmark_uses_same_default_semantics_for_all_runners() -> None:
    results = run_benchmark(n_bars=1000, repeat=1, warmup=0, include_backtrader=True)
    values = [result.final_value for result in results]
    fills = {result.fills for result in results}
    closed_trades = {result.closed_trades for result in results}

    assert max(values) - min(values) < 1e-4
    assert len(fills) == 1
    assert len(closed_trades) == 1
