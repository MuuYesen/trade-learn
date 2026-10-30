from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradelearn.backtest.data import RollingBarBuffer
from tradelearn.core.broker_events import BrokerEvent, BrokerEventPump
from tradelearn.core.contracts import StreamBar


@dataclass(frozen=True)
class EventSnapshot:
    """State summary returned after one EventRunner step."""

    cursor: int
    dispatched_events: int = 0
    events: tuple[BrokerEvent, ...] = ()
    bar: StreamBar | None = None
    orders: tuple[Any, ...] = ()


class EventRunner:
    """Market-event driver delegating strategy execution to StrategyRuntime."""

    def __init__(
        self,
        strategy: Any = None,
        *,
        broker_event_pump: BrokerEventPump | None = None,
        buffer: RollingBarBuffer | None = None,
        buffer_capacity: int = 512,
        feeds: dict[str, str] | None = None,
        runtime: Any = None,
    ) -> None:
        if runtime is not None and strategy is not None and runtime.strategy is not strategy:
            raise ValueError("strategy must match runtime.strategy")
        if runtime is None and strategy is None:
            raise ValueError("strategy or runtime is required")
        if feeds is not None and not feeds:
            raise ValueError("feeds must contain at least one symbol")
        self.strategy = runtime.strategy if runtime is not None else strategy
        self.runtime = runtime
        self.broker_event_pump = broker_event_pump
        self._feeds = feeds
        self._buffer = buffer if buffer is not None else RollingBarBuffer(buffer_capacity)
        self._capacity = buffer_capacity
        self._stopped = False
        if runtime is None and feeds is not None:
            self._ensure_runtime(next(iter(feeds)))

    def _ensure_runtime(self, symbol: str) -> None:
        if self._stopped:
            raise RuntimeError("EventRunner has stopped")
        if self.runtime is None:
            from tradelearn.backtest.event_runtime import StrategyRuntime

            self.runtime = StrategyRuntime(
                strategy=self.strategy,
                feeds=self._feeds or {symbol: "bar"},
                capacity=self._capacity,
                buffers={symbol: self._buffer} if self._buffer is not None else None,
            )

    @property
    def data(self) -> Any:
        return next(iter(self.runtime.feeds.values())) if self.runtime is not None else None

    @property
    def buffer(self) -> RollingBarBuffer | None:
        return self.data._buffer if self.data is not None else self._buffer

    def on_bar(self, bar: StreamBar | dict[str, Any]) -> EventSnapshot:
        stream_bar = _coerce_stream_bar(bar)
        self._ensure_runtime(stream_bar.symbol)
        self.runtime.start()
        events = self.poll_broker_events()
        self.runtime.step_market(stream_bar.symbol, stream_bar, kind="bar")
        feed = self.runtime.feeds[stream_bar.symbol]
        return EventSnapshot(
            cursor=feed._buffer.cursor,
            dispatched_events=len(events),
            events=events,
            bar=stream_bar,
        )

    def on_broker_event(self, event: BrokerEvent | dict[str, Any]) -> None:
        event = BrokerEvent.coerce(event)
        symbol = getattr(getattr(self.strategy, "data", None), "_name", None) or "data0"
        self._ensure_runtime(symbol)
        self.runtime.start()
        if self.broker_event_pump is not None:
            self.broker_event_pump.dispatch(event)
        self.runtime.step_broker(event)

    def poll_broker_events(self) -> tuple[BrokerEvent, ...]:
        if self._stopped:
            raise RuntimeError("EventRunner has stopped")
        if self.broker_event_pump is None:
            return ()
        events = self.broker_event_pump.poll_events()
        if events:
            symbol = getattr(getattr(self.strategy, "data", None), "_name", None) or "data0"
            self._ensure_runtime(symbol)
            self.runtime.start()
        for event in events:
            self.runtime.step_broker(event)
        return events

    def stop(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()
        self._stopped = True


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


def _coerce_stream_bar(bar: StreamBar | dict[str, Any]) -> StreamBar:
    if isinstance(bar, StreamBar):
        return bar
    values = dict(bar)
    timestamp = values.get("ts", values.get("datetime"))
    if timestamp is None:
        raise ValueError("bar dict requires 'ts' or 'datetime'")
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return StreamBar(
        ts=timestamp,
        symbol=str(values.get("symbol", "data0")),
        open=float(values["open"]),
        high=float(values["high"]),
        low=float(values["low"]),
        close=float(values["close"]),
        volume=float(values.get("volume", 0.0)),
    )
