from __future__ import annotations

import pandas as pd

import tradelearn.backtest as backtest
from tradelearn.backtest.event_runner import (
    EventRunner,
    HistoricalDriver,
    LiveDriver,
    PaperDriver,
)
from tradelearn.core.broker_events import BrokerEvent, BrokerEventPump
from tradelearn.core.contracts import StreamBar
from tradelearn.engine import Cerebro, Strategy


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
    assert [event.kind for event in snapshot.events] == ["fill"]
    assert snapshot.bar.symbol == "AAPL"
    assert fills == [{"price": 10.5}]
    assert strategy.rows == [(pd.Timestamp("2026-01-01", tz="UTC"), 10.5)]


def test_backtest_namespace_exposes_event_runner_drivers() -> None:
    assert backtest.__all__ == []
    assert not hasattr(backtest, "HistoricalDriver")
    assert not hasattr(backtest, "PaperDriver")
    assert not hasattr(backtest, "LiveDriver")
    assert not hasattr(backtest, "BatchIndicatorCache")
    assert not hasattr(backtest, "RollingIndicatorCache")
    assert not hasattr(backtest, "RollingBarBuffer")
    assert HistoricalDriver.__name__ == "HistoricalDriver"
    assert PaperDriver.__name__ == "PaperDriver"
    assert LiveDriver.__name__ == "LiveDriver"


def test_cerebro_accepts_backtest_paper_live_modes() -> None:
    assert Cerebro(mode="backtest").mode == "backtest"
    assert Cerebro(mode="paper").mode == "paper"
    assert Cerebro(mode="live").mode == "live"


def test_historical_driver_replays_bars_through_event_runner() -> None:
    class RecordingStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.closes: list[float] = []

        def next(self) -> None:
            self.closes.append(self.data.close[0])

    strategy = RecordingStrategy()
    runner = EventRunner(strategy=strategy, buffer_capacity=4)
    bars = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    )

    snapshots = HistoricalDriver(runner, bars, symbol="AAPL").run()

    assert [snapshot.cursor for snapshot in snapshots] == [0, 1]
    assert strategy.closes == [1.5, 2.5]


def test_live_and_paper_drivers_share_event_runner_path() -> None:
    class CountingStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def next(self) -> None:
            self.count += 1

    bars = [
        StreamBar(
            ts=pd.Timestamp("2026-01-01", tz="UTC"),
            symbol="AAPL",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )
    ]

    paper_strategy = CountingStrategy()
    live_strategy = CountingStrategy()

    assert len(PaperDriver(EventRunner(paper_strategy), bars).run_once()) == 1
    assert len(LiveDriver(EventRunner(live_strategy), lambda: bars).poll_once()) == 1
    assert paper_strategy.count == 1
    assert live_strategy.count == 1


def test_cerebro_paper_and_live_modes_run_through_event_runner() -> None:
    class CountingStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.closes: list[float] = []

        def next(self) -> None:
            self.closes.append(self.data.close[0])

    bars = [
        StreamBar(
            ts=pd.Timestamp("2026-01-01", tz="UTC"),
            symbol="AAPL",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.5,
            volume=1.0,
        )
    ]

    paper = Cerebro(mode="paper", event_bars=bars)
    paper.addstrategy(CountingStrategy)
    [paper_strategy] = paper.run()

    live = Cerebro(mode="live", live_poller=lambda: bars)
    live.addstrategy(CountingStrategy)
    [live_strategy] = live.run()

    assert paper_strategy.closes == [1.5]
    assert live_strategy.closes == [1.5]
