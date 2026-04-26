"""Minimal backtrader-style Strategy/Cerebro/Analyzer facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class Params:
    """Attribute access wrapper for strategy and analyzer params."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def asdict(self) -> dict[str, Any]:
        return dict(self._values)


def _build_params(owner: type[Any], overrides: dict[str, Any], label: str) -> Params:
    declared = getattr(owner, "params", ())
    if not isinstance(declared, tuple):
        declared_type = type(declared).__name__
        raise ValueError(
            f"{label}.params must be a tuple of (name, value) pairs, got {declared_type}."
        )

    values: dict[str, Any] = {}
    for item in declared:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError(f"{label}.params must be a tuple of (name, value) pairs.")
        values[item[0]] = item[1]
    values.update(overrides)
    return Params(values)


class LineSeries:
    """Backtrader-style line where index 0 is the current bar."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values
        self._cursor = -1

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def __getitem__(self, ago: int) -> Any:
        index = self._cursor + ago
        if index < 0 or index >= len(self._values):
            raise IndexError(f"tried to access line[{ago}] but current bar index is {self._cursor}")
        return self._values[index]

    def get(self, ago: int = 0, size: int = 1) -> list[Any]:
        end = self._cursor + ago + 1
        start = max(0, end - size)
        if end < 0:
            return []
        return self._values[start:end]

    def date(self, ago: int = 0) -> Any:
        value = self[ago]
        if hasattr(value, "date"):
            return value.date()
        return value


class DataFeed:
    """Column-based OHLCV feed exposed to strategies."""

    def __init__(self, data: pd.DataFrame, name: str | None = None) -> None:
        frame = data.copy()
        self._name = name
        self._frame = frame
        self.datetime = LineSeries(list(frame.index))
        self.open = LineSeries(frame["open"].tolist())
        self.high = LineSeries(frame["high"].tolist())
        self.low = LineSeries(frame["low"].tolist())
        self.close = LineSeries(frame["close"].tolist())
        self.volume = LineSeries(frame["volume"].tolist())
        self._lines = [self.datetime, self.open, self.high, self.low, self.close, self.volume]

    def __len__(self) -> int:
        return len(self._frame)

    def _advance(self, cursor: int) -> None:
        for line in self._lines:
            line._advance(cursor)


@dataclass
class Position:
    size: float = 0.0
    price: float = 0.0
    adjbase: float = 0.0

    def __bool__(self) -> bool:
        return self.size != 0.0


@dataclass
class BarSnapshot:
    datetime: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    data: DataFeed


class BrokerFacade:
    def __init__(self) -> None:
        self._cash = 10000.0

    def setcash(self, cash: float) -> None:
        self._cash = float(cash)

    def getcash(self, symbol: str | None = None) -> float:
        return self._cash

    def getvalue(self) -> float:
        return self._cash

    def setcommission(self, commission: float) -> None:
        self.commission = float(commission)


class Strategy:
    """Base strategy class with backtrader-style lifecycle hooks."""

    params: tuple[tuple[str, Any], ...] = ()

    def __init__(self) -> None:
        pass

    def start(self) -> None:
        pass

    def prenext(self) -> None:
        pass

    def next(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def notify_order(self, order: Any) -> None:
        pass

    def notify_trade(self, trade: Any) -> None:
        pass

    def notify_cashvalue(self, cash: float, value: float) -> None:
        pass

    def notify_timer(self, timer: Any, when: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def getposition(self, data: DataFeed | None = None) -> Position:
        return self._positions.setdefault(data or self.data, Position())

    @property
    def position(self) -> Position:
        return self.getposition(self.data)


class Analyzer:
    """Base analyzer class attached through Cerebro.addanalyzer."""

    params: tuple[tuple[str, Any], ...] = ()
    strategy: Strategy | None = None

    def __init__(self) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_bar(self, bar: BarSnapshot) -> None:
        pass

    def on_fill(self, fill: Any) -> None:
        pass

    def on_trade(self, trade: Any) -> None:
        pass

    def on_end(self, stats: dict[str, Any]) -> None:
        pass

    def get_analysis(self) -> dict[str, Any]:
        return {}


class Cerebro:
    """Small Cerebro facade for Stage 3 Python strategy API scaffolding."""

    def __init__(
        self,
        *,
        callback_batch: int = 1,
        trade_on_close: bool = False,
        exactbars: bool = False,
        stdstats: bool = True,
        **kwargs: Any,
    ) -> None:
        self.callback_batch = max(1, int(callback_batch))
        self.trade_on_close = trade_on_close
        self.exactbars = exactbars
        self.stdstats = stdstats
        self.options = dict(kwargs)
        self.datas: list[DataFeed] = []
        self._strategy_spec: tuple[type[Strategy], dict[str, Any]] | None = None
        self._analyzer_specs: list[tuple[str, type[Analyzer], dict[str, Any]]] = []
        self.broker = BrokerFacade()

    def adddata(self, data: pd.DataFrame | DataFeed, name: str | None = None) -> DataFeed:
        feed = data if isinstance(data, DataFeed) else DataFeed(data, name=name)
        if name is not None:
            feed._name = name
        self.datas.append(feed)
        return feed

    def addstrategy(self, strategy: type[Strategy], **params: Any) -> None:
        self._strategy_spec = (strategy, dict(params))

    def addanalyzer(self, analyzer: type[Analyzer], name: str | None = None, **params: Any) -> None:
        analyzer_name = name or analyzer.__name__
        self._analyzer_specs.append((analyzer_name, analyzer, dict(params)))

    def run(self) -> list[Strategy]:
        if self._strategy_spec is None:
            raise RuntimeError("Cerebro requires addstrategy() before run().")
        if not self.datas:
            raise RuntimeError("Cerebro requires at least one data feed.")

        strategy = self._instantiate_strategy()
        analyzers = self._instantiate_analyzers(strategy)
        strategy.analyzers = analyzers

        strategy.start()
        for analyzer in analyzers.values():
            analyzer.on_start()

        total_bars = min(len(data) for data in self.datas)
        for cursor in range(total_bars):
            for data in self.datas:
                data._advance(cursor)
            strategy.next()
            bar = self._snapshot(self.datas[0])
            for analyzer in analyzers.values():
                analyzer.on_bar(bar)

        stats = {"bars": total_bars}
        for analyzer in analyzers.values():
            analyzer.on_end(stats)
        strategy.stop()
        return [strategy]

    def _instantiate_strategy(self) -> Strategy:
        strategy_cls, params = self._strategy_spec or (Strategy, {})
        strategy = strategy_cls.__new__(strategy_cls)
        strategy.datas = self.datas
        strategy.data = self.datas[0]
        strategy.broker = self.broker
        strategy.p = _build_params(strategy_cls, params, "Strategy")
        strategy.params = strategy.p
        strategy._positions = {}
        strategy_cls.__init__(strategy)
        return strategy

    def _instantiate_analyzers(self, strategy: Strategy) -> dict[str, Analyzer]:
        analyzers: dict[str, Analyzer] = {}
        for name, analyzer_cls, params in self._analyzer_specs:
            analyzer = analyzer_cls.__new__(analyzer_cls)
            analyzer.strategy = strategy
            analyzer.p = _build_params(analyzer_cls, params, "Analyzer")
            analyzer.params = analyzer.p
            analyzer_cls.__init__(analyzer)
            analyzers[name] = analyzer
        return analyzers

    def _snapshot(self, data: DataFeed) -> BarSnapshot:
        return BarSnapshot(
            datetime=data.datetime[0],
            open=data.open[0],
            high=data.high[0],
            low=data.low[0],
            close=data.close[0],
            volume=data.volume[0],
            data=data,
        )
