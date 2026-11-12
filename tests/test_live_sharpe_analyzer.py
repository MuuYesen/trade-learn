"""Observed-value Sharpe on the shared native Runtime, with bounded moment state."""

from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
import backtrader as official

from tests.unit.engine.test_bt_returns_actual import returns_calls
from tradelearn import engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.components import LiveComponents, validate_components


def test_live_sharpe_matches_real_bt_periods_without_backtest_stats_or_full_history():
    state = {"cash": 100.0, "equity": 100.0}
    calls = []

    def account():
        calls.append(dict(state))
        return dict(state)

    broker = NautilusBroker(SimpleNamespace(account=account))
    runtime = StrategyRuntime(bt.Strategy, feeds={"A": "bar"}, broker=broker)
    e = bt.Cerebro(stdstats=False)
    options = dict(timeframe=bt.TimeFrame.Days, factor=365, annualize=True)
    e.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", **options)
    components = LiveComponents(e, runtime, account_provider=account)
    runtime.start()
    components.start()
    stamps = [
        "2026-01-30T10:00:00Z",
        "2026-01-30T10:01:00Z",
        "2026-02-02T10:00:00Z",
        "2026-02-02T10:03:00Z",
    ]
    for stamp, value in zip(stamps, [100.0, 110.0, 90.0, 120.0], strict=True):
        state["equity"] = value
        runtime.step_market("A", dict(ts=stamp, open=10, high=10, low=10, close=10, volume=1))
        components.after_market()
    runtime.stop()
    components.finish()
    components.finish()
    expected = returns_calls(
        official,
        dict(timeframe="Days", factor=365, annualize=True),
        analyzer="SharpeRatio",
        sample_from_feed=True,
    )
    assert e.analyzer_results["sharpe"] == pytest.approx(expected)
    assert runtime.strategy.analyzers.sharpe._count == 2
    assert len(calls) == 5
    assert getattr(runtime.strategy, "stats", None) is None


def test_live_sharpe_requires_authoritative_value_provider():
    e = bt.Cerebro(stdstats=False)
    e.addanalyzer(bt.analyzers.SharpeRatio)
    with pytest.raises(ValueError, match="authoritative account"):
        validate_components(e, account_provider=None)
