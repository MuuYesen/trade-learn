from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradelearn.backtest.core.data import RollingBarBuffer
from tradelearn.compat.backtrader.base import LineSeries
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
