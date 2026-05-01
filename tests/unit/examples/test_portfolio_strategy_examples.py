from __future__ import annotations

import importlib
import math

import pandas as pd
import pytest

import tradelearn.engine as bt
from benchmarks.runners import benchmark_bt


def _portfolio_data() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=90, freq="D", tz="UTC")
    specs = {
        "GLD": (80.0, 0.03, "gold"),
        "DBC": (25.0, 0.01, "commodity"),
        "SPY": (100.0, 0.08, "equity"),
        "TLT": (60.0, 0.04, "bond_lt"),
        "IEF": (55.0, 0.02, "bond_it"),
    }
    frames = {}
    for name, (base, drift, _shareclass) in specs.items():
        close = [
            base + drift * i + ((i % 9) - 4) * 0.05
            for i in range(len(index))
        ]
        frames[name] = pd.DataFrame(
            {
                "open": close,
                "high": [value + 0.2 for value in close],
                "low": [value - 0.2 for value in close],
                "close": close,
                "volume": [1000.0 + i for i in range(len(index))],
            },
            index=index,
        )
    return frames


def _run_strategy(strategy_name: str) -> tuple[float, object]:
    strategy_cls = getattr(
        importlib.import_module("examples.engine"),
        strategy_name,
    )
    cerebro = bt.Cerebro(trade_on_close=True)
    for name, frame in _portfolio_data().items():
        cerebro.adddata(bt.feeds.PandasData(dataname=frame, name=name))
    cerebro.broker.setcash(100000.0)
    cerebro.addstrategy(strategy_cls, rebalance_bars=15, lookback=10)
    [strategy] = cerebro.run()
    return float(cerebro.broker.getvalue()), strategy


def test_portfolio_strategy_examples_export_and_run() -> None:
    strategy_names = [
        "AssetClassTargetPortfolioStrategy",
        "UniformAssetClassPortfolioStrategy",
        "TrendFilteredPortfolioStrategy",
        "InverseVolatilityPortfolioStrategy",
    ]

    for strategy_name in strategy_names:
        final_value, strategy = _run_strategy(strategy_name)

        assert math.isfinite(final_value)
        assert final_value > 0
        assert strategy.target_history
        assert strategy.order_history


def test_portfolio_benchmark_examples_align_with_backtrader() -> None:
    results = benchmark_bt.run_portfolio_benchmark()

    for strategy_name, result in results.items():
        tradelearn = result["Tradelearn"]
        backtrader = result["Backtrader"]

        assert tradelearn is not None, strategy_name
        assert backtrader is not None, strategy_name
        assert tradelearn["final_value"] == pytest.approx(
            backtrader["final_value"],
            abs=benchmark_bt.EXACT_TOLERANCE,
        )
        assert tradelearn["order_count"] == backtrader["order_count"]
