from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradelearn.backtest.data import RollingBarBuffer
from tradelearn.backtest.lines import LineSeries
from tradelearn.core import BrokerEvent, BrokerEventPump, StreamBar


@dataclass(frozen=True)
class EventSnapshot:
    """State summary returned after one EventRunner step."""

    cursor: int
    dispatched_events: int = 0
    orders: tuple[Any, ...] = ()


class _BufferDataFeed:
    """Minimal live data facade exposing Backtrader-style OHLCV lines."""

    def __init__(self, buffer: RollingBarBuffer) -> None:
        self._buffer = buffer
        self.datetime = LineSeries(
            buffer.arrays["datetime"],
            is_datetime=True,
            buffer=buffer,
            buffer_name="datetime",
        )
        self.open = LineSeries(buffer.arrays["open"], buffer=buffer, buffer_name="open")
        self.high = LineSeries(buffer.arrays["high"], buffer=buffer, buffer_name="high")
        self.low = LineSeries(buffer.arrays["low"], buffer=buffer, buffer_name="low")
        self.close = LineSeries(buffer.arrays["close"], buffer=buffer, buffer_name="close")
        self.volume = LineSeries(buffer.arrays["volume"], buffer=buffer, buffer_name="volume")

    def __len__(self) -> int:
        return self._buffer.cursor + 1

    def _advance(self, cursor: int) -> None:
        for line in (self.datetime, self.open, self.high, self.low, self.close, self.volume):
            line._advance(cursor)


class EventRunner:
    """Single-event runner shared by backtest, paper, and live drivers."""

    def __init__(
        self,
        strategy: Any,
        *,
        broker_event_pump: BrokerEventPump | None = None,
        buffer: RollingBarBuffer | None = None,
        buffer_capacity: int = 512,
    ) -> None:
        self.strategy = strategy
        self.buffer = buffer or RollingBarBuffer(buffer_capacity)
        self.data = _BufferDataFeed(self.buffer)
        self.broker_event_pump = broker_event_pump
        self._bind_strategy_data()

    def on_bar(self, bar: StreamBar | dict[str, Any]) -> EventSnapshot:
        self.buffer.append(bar)
        cursor = self.buffer.cursor
        self.data._advance(cursor)
        dispatched = self.poll_broker_events()
        self.strategy._pre_next(cursor)
        self.strategy.next()
        return EventSnapshot(cursor=cursor, dispatched_events=dispatched)

    def on_broker_event(self, event: BrokerEvent | dict[str, Any]) -> None:
        if self.broker_event_pump is not None:
            self.broker_event_pump.dispatch(self.broker_event_pump._coerce_event(event))

    def poll_broker_events(self) -> int:
        if self.broker_event_pump is None:
            return 0
        return self.broker_event_pump.poll_once()

    def _bind_strategy_data(self) -> None:
        self.strategy.datas = [self.data]
        self.strategy.data = self.data
        if hasattr(self.strategy, "_set_bar_advancers"):
            self.strategy._set_bar_advancers(())


class HistoricalDriver:
    """Historical bar source that feeds EventRunner one bar at a time."""

    def __init__(self, runner: EventRunner, bars: pd.DataFrame, symbol: str = "data0") -> None:
        self.runner = runner
        self.bars = bars
        self.symbol = symbol

    def run(self) -> list[EventSnapshot]:
        return [self.runner.on_bar(bar) for bar in self._iter_bars()]

    def _iter_bars(self) -> Iterable[StreamBar]:
        for ts, row in self.bars.iterrows():
            timestamp = pd.Timestamp(ts)
            if timestamp.tzinfo is None:
                timestamp = pd.Timestamp(ts, tz="UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            yield StreamBar(
                ts=timestamp,
                symbol=self.symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )


class PaperDriver:
    """Paper driver facade that reuses the live EventRunner path."""

    def __init__(self, runner: EventRunner, bars: Iterable[StreamBar | dict[str, Any]]) -> None:
        self.runner = runner
        self.bars = bars

    def run_once(self) -> list[EventSnapshot]:
        return [self.runner.on_bar(bar) for bar in self.bars]


class LiveDriver:
    """Live driver facade driven by a user-supplied non-blocking poller."""

    def __init__(
        self,
        runner: EventRunner,
        poller: Callable[[], Iterable[StreamBar | dict[str, Any]]],
    ) -> None:
        self.runner = runner
        self.poller = poller

    def poll_once(self) -> list[EventSnapshot]:
        return [self.runner.on_bar(bar) for bar in self.poller()]
