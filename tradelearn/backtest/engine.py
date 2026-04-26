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

    def update(self, size_delta: float, price: float) -> None:
        new_size = self.size + size_delta
        if new_size == 0.0:
            self.size = 0.0
            self.price = 0.0
            return
        if self.size == 0.0 or self.size * size_delta > 0:
            total_abs = abs(self.size) + abs(size_delta)
            self.price = (abs(self.size) * self.price + abs(size_delta) * price) / total_abs
        elif self.size * new_size < 0:
            self.price = price
        self.size = new_size


@dataclass
class BarSnapshot:
    datetime: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    data: DataFeed


@dataclass
class ExecutedInfo:
    size: float = 0.0
    price: float = 0.0
    value: float = 0.0
    comm: float = 0.0
    pnl: float = 0.0


@dataclass
class Order:
    Submitted = 1
    Accepted = 2
    Partial = 3
    Completed = 4
    Canceled = 5
    Expired = 6
    Margin = 7
    Rejected = 8

    Buy = 1
    Sell = 2

    Market = 1
    Limit = 2
    Stop = 3
    StopLimit = 4
    Close = 5

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"

    ref: int
    data: DataFeed
    ordtype: int
    size: float
    price: float | None = None
    pricelimit: float | None = None
    exectype: int = Market
    time_in_force: str = GTC
    status: int = Submitted
    executed: ExecutedInfo = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.executed is None:
            self.executed = ExecutedInfo()


@dataclass
class Trade:
    ref: int
    data: DataFeed
    size: float
    price: float
    value: float
    commission: float
    pnl: float
    pnlcomm: float
    isopen: bool
    isclosed: bool
    status: int
    dtopen: Any
    dtclose: Any | None = None


class SimBroker:
    def __init__(self) -> None:
        self._cash = 10000.0
        self.commission = 0.0
        self.trade_on_close = False
        self._next_order_ref = 1
        self._next_trade_ref = 1
        self._pending: list[Order] = []
        self._last_strategy: Strategy | None = None

    def setcash(self, cash: float) -> None:
        self._cash = float(cash)

    def getcash(self, symbol: str | None = None) -> float:
        return self._cash

    def getvalue(self) -> float:
        if self._last_strategy is not None:
            return self._cash + sum(
                position.size * _current_close(data)
                for data, position in self._last_strategy._positions.items()
            )
        return self._cash

    def setcommission(self, commission: float) -> None:
        self.commission = float(commission)

    def buy(
        self,
        strategy: Strategy,
        data: DataFeed,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        *,
        pricelimit: float | None = None,
        time_in_force: str | None = None,
    ) -> Order:
        return self._submit(
            strategy,
            data,
            Order.Buy,
            size,
            price,
            exectype,
            pricelimit=pricelimit,
            time_in_force=time_in_force,
        )

    def sell(
        self,
        strategy: Strategy,
        data: DataFeed,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        *,
        pricelimit: float | None = None,
        time_in_force: str | None = None,
    ) -> Order:
        return self._submit(
            strategy,
            data,
            Order.Sell,
            size,
            price,
            exectype,
            pricelimit=pricelimit,
            time_in_force=time_in_force,
        )

    def process_bar(self, strategy: Strategy, analyzers: AnalyzerCollection) -> None:
        self._last_strategy = strategy
        pending, self._pending = self._pending, []
        for order in pending:
            executed = self._execute_order(strategy, order, analyzers)
            if not executed:
                self._handle_unfilled_order(strategy, order)

    def process_close(self, strategy: Strategy, analyzers: AnalyzerCollection) -> None:
        self._last_strategy = strategy
        pending, self._pending = self._pending, []
        for order in pending:
            if order.exectype != Order.Market:
                self._pending.append(order)
                continue
            executed = self._execute_order(strategy, order, analyzers, trade_on_close=True)
            if not executed:
                self._handle_unfilled_order(strategy, order)

    def _submit(
        self,
        strategy: Strategy,
        data: DataFeed,
        ordtype: int,
        size: float | None,
        price: float | None,
        exectype: int | None,
        *,
        pricelimit: float | None,
        time_in_force: str | None,
    ) -> Order:
        order = Order(
            ref=self._next_order_ref,
            data=data,
            ordtype=ordtype,
            size=float(1.0 if size is None else abs(size)),
            price=price,
            pricelimit=pricelimit,
            exectype=Order.Market if exectype is None else exectype,
            time_in_force=Order.GTC if time_in_force is None else time_in_force,
        )
        self._next_order_ref += 1
        strategy.notify_order(order)
        if order.time_in_force not in {Order.DAY, Order.GTC, Order.IOC}:
            order.status = Order.Rejected
            strategy.notify_order(order)
            return order
        if order.size <= 0:
            order.status = Order.Rejected
            strategy.notify_order(order)
            return order
        order.status = Order.Accepted
        self._pending.append(order)
        strategy.notify_order(order)
        return order

    def _execute_order(
        self,
        strategy: Strategy,
        order: Order,
        analyzers: AnalyzerCollection,
        *,
        trade_on_close: bool = False,
    ) -> bool:
        if order.size > float(order.data.volume[0]):
            order.status = Order.Rejected
            strategy.notify_order(order)
            return True
        fill = _match_order(order, self.commission, trade_on_close=trade_on_close)
        if fill is None:
            return False
        signed_size, price, commission, _slippage = fill
        position = strategy.getposition(order.data)
        old_size = position.size
        old_price = position.price
        realized = _realized_pnl(old_size, old_price, signed_size, price)

        self._cash -= signed_size * price + commission
        position.update(signed_size, price)

        order.status = Order.Completed
        order.executed = ExecutedInfo(
            size=signed_size,
            price=price,
            value=abs(signed_size) * price,
            comm=commission,
            pnl=realized,
        )
        trade = Trade(
            ref=self._next_trade_ref,
            data=order.data,
            size=position.size,
            price=price,
            value=abs(position.size) * price,
            commission=commission,
            pnl=realized,
            pnlcomm=realized - commission,
            isopen=position.size != 0.0,
            isclosed=position.size == 0.0 and old_size != 0.0,
            status=1 if position.size != 0.0 else 2,
            dtopen=order.data.datetime[0],
            dtclose=order.data.datetime[0] if position.size == 0.0 else None,
        )
        self._next_trade_ref += 1

        strategy.notify_order(order)
        strategy.notify_trade(trade)
        for analyzer in analyzers.values():
            analyzer.on_fill(order.executed)
            analyzer.on_trade(trade)
        return True

    def _handle_unfilled_order(self, strategy: Strategy, order: Order) -> None:
        if order.time_in_force == Order.IOC:
            order.status = Order.Canceled
            strategy.notify_order(order)
            return
        if order.time_in_force == Order.DAY:
            order.status = Order.Expired
            strategy.notify_order(order)
            return
        self._pending.append(order)


def _realized_pnl(old_size: float, old_price: float, fill_size: float, fill_price: float) -> float:
    if old_size == 0.0 or old_size * fill_size > 0:
        return 0.0
    closing_size = min(abs(old_size), abs(fill_size))
    return (fill_price - old_price) * closing_size * (1.0 if old_size > 0 else -1.0)


def _current_close(data: DataFeed) -> float:
    try:
        return float(data.close[0])
    except IndexError:
        return 0.0


def _match_order(
    order: Order,
    commission: float,
    *,
    trade_on_close: bool = False,
) -> tuple[float, float, float, float] | None:
    try:
        from tradelearn import _rust

        rust_match_order = _rust.match_order_fill
    except (ImportError, AttributeError):
        rust_match_order = None

    side = "buy" if order.ordtype == Order.Buy else "sell"
    order_type = _rust_order_type(order)
    limit_price = order.price if order.exectype == Order.Limit else order.pricelimit
    stop_price = order.price if order.exectype in {Order.Stop, Order.StopLimit} else None
    if rust_match_order is not None:
        return rust_match_order(
            order.ref,
            order.data._name or "data0",
            side,
            order_type,
            order.size,
            limit_price,
            stop_price,
            0,
            int(order.data.datetime[0].timestamp())
            if hasattr(order.data.datetime[0], "timestamp")
            else 0,
            float(order.data.open[0]),
            float(order.data.high[0]),
            float(order.data.low[0]),
            float(order.data.close[0]),
            float(order.data.volume[0]),
            trade_on_close,
            commission,
        )
    return _python_match_order(
        order,
        limit_price,
        stop_price,
        commission,
        trade_on_close=trade_on_close,
    )


def _rust_order_type(order: Order) -> str:
    if order.exectype == Order.Limit:
        return "limit"
    if order.exectype == Order.Stop:
        return "stop"
    if order.exectype == Order.StopLimit:
        return "stop_limit"
    return "market"


def _python_match_order(
    order: Order,
    limit_price: float | None,
    stop_price: float | None,
    commission: float,
    *,
    trade_on_close: bool = False,
) -> tuple[float, float, float, float] | None:
    if order.exectype == Order.Limit:
        price = _limit_fill_price(order, limit_price)
    elif order.exectype == Order.Stop:
        price = (
            _execution_price(order, trade_on_close)
            if _stop_triggered(order, stop_price)
            else None
        )
    elif order.exectype == Order.StopLimit:
        price = (
            _limit_fill_price(order, limit_price)
            if _stop_triggered(order, stop_price)
            else None
        )
    else:
        price = _execution_price(order, trade_on_close)
    if price is None:
        return None
    signed_size = order.size if order.ordtype == Order.Buy else -order.size
    return (
        signed_size,
        price,
        abs(signed_size) * price * commission,
        0.0,
    )


def _execution_price(order: Order, trade_on_close: bool) -> float:
    return float(order.data.close[0] if trade_on_close else order.data.open[0])


def _limit_fill_price(order: Order, limit_price: float | None) -> float | None:
    if limit_price is None:
        return None
    if order.ordtype == Order.Buy and order.data.low[0] <= limit_price:
        return min(limit_price, float(order.data.open[0]))
    if order.ordtype == Order.Sell and order.data.high[0] >= limit_price:
        return max(limit_price, float(order.data.open[0]))
    return None


def _stop_triggered(order: Order, stop_price: float | None) -> bool:
    if stop_price is None:
        return False
    if order.ordtype == Order.Buy:
        return bool(order.data.high[0] >= stop_price)
    return bool(order.data.low[0] <= stop_price)


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

    def buy(
        self,
        data: DataFeed | None = None,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs: Any,
    ) -> Order:
        return self.broker.buy(
            self,
            data or self.data,
            size,
            price,
            exectype,
            pricelimit=kwargs.get("pricelimit"),
            time_in_force=kwargs.get("time_in_force"),
        )

    def sell(
        self,
        data: DataFeed | None = None,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs: Any,
    ) -> Order:
        return self.broker.sell(
            self,
            data or self.data,
            size,
            price,
            exectype,
            pricelimit=kwargs.get("pricelimit"),
            time_in_force=kwargs.get("time_in_force"),
        )

    def close(self, data: DataFeed | None = None, **kwargs: Any) -> Order | None:
        target_data = data or self.data
        position = self.getposition(target_data)
        if position.size > 0:
            return self.sell(data=target_data, size=position.size)
        if position.size < 0:
            return self.buy(data=target_data, size=abs(position.size))
        return None

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


class AnalyzerCollection(dict[str, Analyzer]):
    """Named analyzers with backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Analyzer:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def getbyname(self, name: str) -> Analyzer:
        """Return a named analyzer."""
        return self[name]


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
        self.analyzer_results: dict[str, Any] = {}
        self.broker = SimBroker()
        self.broker.trade_on_close = self.trade_on_close

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
            self.broker.process_bar(strategy, analyzers)
            strategy.next()
            if self.trade_on_close:
                self.broker.process_close(strategy, analyzers)
            bar = self._snapshot(self.datas[0])
            for analyzer in analyzers.values():
                analyzer.on_bar(bar)

        stats = {"bars": total_bars}
        for analyzer in analyzers.values():
            analyzer.on_end(stats)
        self.analyzer_results = {
            name: analyzer.get_analysis() for name, analyzer in analyzers.items()
        }
        strategy.analyzer_results = dict(self.analyzer_results)
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

    def _instantiate_analyzers(self, strategy: Strategy) -> AnalyzerCollection:
        analyzers = AnalyzerCollection()
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
