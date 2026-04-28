from __future__ import annotations

import pandas as pd

from tradelearn.backtest import BatchIndicatorCache, RollingBarBuffer, RollingIndicatorCache
from tradelearn.backtest.core.event_runner import EventRunner
from tradelearn.backtest.core.strategy import Strategy
from tradelearn.compat.backtrader import Cerebro
from tradelearn.core import BrokerEvent, BrokerEventPump, StreamBar


def test_event_runner_drives_single_live_bar_and_broker_events() -> None:
    events = [BrokerEvent("fill", order_id=7, payload={"price": 10.5})]
    fills: list[object] = []

    class RecordingStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[tuple[pd.Timestamp, float]] = []

        def next(self) -> None:
            self.rows.append((self.data.datetime[0], self.data.close[0]))

    strategy = RecordingStrategy()
    pump = BrokerEventPump(lambda: events)
    pump.on_fill(fills.append)
    runner = EventRunner(strategy=strategy, broker_event_pump=pump, buffer_capacity=4)

    snapshot = runner.on_bar(
        StreamBar(
            ts=pd.Timestamp("2026-01-01", tz="UTC"),
            symbol="AAPL",
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            volume=1000.0,
        )
    )

    assert snapshot.cursor == 0
    assert snapshot.dispatched_events == 1
    assert fills == [{"price": 10.5}]
    assert strategy.rows == [(pd.Timestamp("2026-01-01", tz="UTC"), 10.5)]


def test_backtest_namespace_exposes_event_runner_building_blocks() -> None:
    assert BatchIndicatorCache is not None
    assert RollingIndicatorCache is not None
    assert RollingBarBuffer is not None


def test_cerebro_accepts_backtest_paper_live_modes() -> None:
    assert Cerebro(mode="backtest").mode == "backtest"
    assert Cerebro(mode="paper").mode == "paper"
    assert Cerebro(mode="live").mode == "live"
