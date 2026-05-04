from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.runners.benchmark_bt import _benchmark_passed
from benchmarks.runners.benchmark_target_weights import (
    run_benchmark as run_target_weights_benchmark,
)
from benchmarks.runners.benchmark_target_weight_parity import run_parity_benchmark
from benchmarks.runners.benchmark_throughput import make_data, run_benchmark, run_engine, run_lite
from benchmarks.runners.benchmark_research_pipeline import (
    run_benchmark as run_research_pipeline_benchmark,
)

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


def test_target_weights_benchmark_reports_profile_segments() -> None:
    timings, stats, profile = run_target_weights_benchmark(
        symbols=10,
        bars=45,
        holdings=3,
        rebalance_every=7,
        cash=100_000.0,
        seed=3,
    )

    timing_names = {timing.name for timing in timings}
    assert {"data_generate", "init_runner_feed", "run_loop", "stats_read", "total"} <= timing_names
    assert profile.rebalance_count > 0
    assert profile.signal_rank_seconds >= 0.0
    assert profile.target_weights_seconds >= profile.order_submit_seconds >= 0.0
    assert profile.order_count >= 3
    assert float(stats["final_value"]) > 0.0


def test_target_weight_parity_benchmark_uses_same_sell_first_semantics() -> None:
    results = run_parity_benchmark(
        symbols=20,
        bars=80,
        holdings=5,
        rebalance_every=11,
        cash=100_000.0,
        seed=11,
    )

    by_name = {result.name: result for result in results}
    engine = by_name["Tradelearn Engine"]
    lite = by_name["Tradelearn Lite"]
    backtrader = by_name["Backtrader"]

    assert engine.order_count == backtrader.order_count
    assert abs(engine.final_value - backtrader.final_value) < 1e-2
    assert lite.target_history == engine.target_history
    assert lite.order_count > 0
    assert all(result.bars_per_sec > 0 for result in results)


def test_research_pipeline_benchmark_reports_stage12_segments() -> None:
    result = run_research_pipeline_benchmark(
        symbols=6,
        bars=80,
        holdings=2,
        lookback=5,
        rebalance_every=10,
        split_ratio=0.6,
        seed=13,
        write_report=True,
    )

    assert result.total_bars == 480
    assert result.final_value > 0.0
    assert result.weight_dates > 0
    assert result.report_path is not None and result.report_path.name == "report.html"
    assert result.artifact_count >= 4
    assert result.bars_per_sec > 0.0
    assert [segment.name for segment in result.segments] == [
        "panel",
        "factor",
        "weights",
        "backtest",
        "report",
        "mlflow_artifacts",
        "total",
    ]
    assert all(segment.seconds >= 0.0 for segment in result.segments)
