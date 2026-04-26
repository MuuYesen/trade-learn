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


def _coerce_min_period(value: Any, label: str) -> int:
    try:
        period = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer.") from exc
    if period < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return period


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
    realized_pnl: float = 0.0

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
class Stats:
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    summary: dict[str, Any]
    analyzers: dict[str, Any]
    config: dict[str, Any]
    fills: pd.DataFrame


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


_ORDER_COLUMNS = [
    "ref",
    "datetime",
    "data",
    "side",
    "ordtype",
    "exectype",
    "status",
    "status_code",
    "size",
    "price",
    "pricelimit",
    "time_in_force",
    "executed_size",
    "executed_price",
    "executed_value",
    "commission",
    "pnl",
]
_FILL_COLUMNS = ["order_ref", "datetime", "data", "size", "price", "value", "commission", "pnl"]
_TRADE_COLUMNS = [
    "ref",
    "datetime",
    "data",
    "size",
    "price",
    "value",
    "commission",
    "pnl",
    "pnlcomm",
    "isopen",
    "isclosed",
    "status",
    "dtopen",
    "dtclose",
]
_POSITION_COLUMNS = [
    "datetime",
    "data",
    "size",
    "avg_price",
    "mark_price",
    "value",
    "unrealized_pnl",
    "realized_pnl",
    "margin_used",
]


class SimBroker:
    def __init__(self) -> None:
        self._cash = 10000.0
        self.commission = 0.0
        self.trade_on_close = False
        self._next_order_ref = 1
        self._next_trade_ref = 1
        self._pending: list[Order] = []
        self._last_strategy: Strategy | None = None
        self._order_records: list[dict[str, Any]] = []
        self._fill_records: list[dict[str, Any]] = []
        self._trade_records: list[dict[str, Any]] = []
        self._equity_records: list[dict[str, Any]] = []
        self._position_records: list[dict[str, Any]] = []

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
        self._record_order(order)
        _notify_order(strategy, order)
        if order.time_in_force not in {Order.DAY, Order.GTC, Order.IOC}:
            order.status = Order.Rejected
            self._record_order(order)
            _notify_order(strategy, order)
            return order
        if order.size <= 0:
            order.status = Order.Rejected
            self._record_order(order)
            _notify_order(strategy, order)
            return order
        order.status = Order.Accepted
        self._pending.append(order)
        self._record_order(order)
        _notify_order(strategy, order)
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
            self._record_order(order)
            _notify_order(strategy, order)
            return True
        fill = _match_order(order, self.commission, trade_on_close=trade_on_close)
        if fill is None:
            return False
        signed_size, price, commission, _slippage = fill
        if not self._has_required_cash(strategy, order, signed_size, price, commission):
            order.status = Order.Rejected
            self._record_order(order)
            _notify_order(strategy, order)
            return True
        position = strategy.getposition(order.data)
        old_size = position.size
        old_price = position.price
        realized = _realized_pnl(old_size, old_price, signed_size, price)

        self._cash -= signed_size * price + commission
        position.update(signed_size, price)
        position.realized_pnl += realized

        order.status = Order.Completed
        order.executed = ExecutedInfo(
            size=signed_size,
            price=price,
            value=abs(signed_size) * price,
            comm=commission,
            pnl=realized,
        )
        trades = self._build_trade_legs(
            order=order,
            old_size=old_size,
            new_size=position.size,
            price=price,
            commission=commission,
            realized=realized,
        )

        self._record_order(order)
        self._record_fill(order)
        for trade in trades:
            self._record_trade(trade)
        _notify_order(strategy, order)
        for trade in trades:
            strategy.notify_trade(trade)
        for analyzer in analyzers.values():
            analyzer.on_fill(order.executed)
            for trade in trades:
                analyzer.on_trade(trade)
        return True

    def _build_trade_legs(
        self,
        *,
        order: Order,
        old_size: float,
        new_size: float,
        price: float,
        commission: float,
        realized: float,
    ) -> list[Trade]:
        timestamp = order.data.datetime[0]
        if old_size != 0.0 and old_size * new_size < 0.0:
            closing_size = abs(old_size)
            opening_size = abs(new_size)
            total_size = closing_size + opening_size
            close_commission = commission * closing_size / total_size if total_size else 0.0
            open_commission = commission - close_commission
            return [
                self._make_trade(
                    data=order.data,
                    size=0.0,
                    price=price,
                    commission=close_commission,
                    pnl=realized,
                    isopen=False,
                    isclosed=True,
                    status=2,
                    dtopen=timestamp,
                    dtclose=timestamp,
                ),
                self._make_trade(
                    data=order.data,
                    size=new_size,
                    price=price,
                    commission=open_commission,
                    pnl=0.0,
                    isopen=True,
                    isclosed=False,
                    status=1,
                    dtopen=timestamp,
                    dtclose=None,
                ),
            ]
        return [
            self._make_trade(
                data=order.data,
                size=new_size,
                price=price,
                commission=commission,
                pnl=realized,
                isopen=new_size != 0.0,
                isclosed=new_size == 0.0 and old_size != 0.0,
                status=1 if new_size != 0.0 else 2,
                dtopen=timestamp,
                dtclose=timestamp if new_size == 0.0 else None,
            )
        ]

    def _make_trade(
        self,
        *,
        data: DataFeed,
        size: float,
        price: float,
        commission: float,
        pnl: float,
        isopen: bool,
        isclosed: bool,
        status: int,
        dtopen: Any,
        dtclose: Any | None,
    ) -> Trade:
        trade = Trade(
            ref=self._next_trade_ref,
            data=data,
            size=size,
            price=price,
            value=abs(size) * price,
            commission=commission,
            pnl=pnl,
            pnlcomm=pnl - commission,
            isopen=isopen,
            isclosed=isclosed,
            status=status,
            dtopen=dtopen,
            dtclose=dtclose,
        )
        self._next_trade_ref += 1
        return trade

    def _handle_unfilled_order(self, strategy: Strategy, order: Order) -> None:
        if order.time_in_force == Order.IOC:
            order.status = Order.Canceled
            self._record_order(order)
            _notify_order(strategy, order)
            return
        if order.time_in_force == Order.DAY:
            order.status = Order.Expired
            self._record_order(order)
            _notify_order(strategy, order)
            return
        self._pending.append(order)

    def _has_required_cash(
        self,
        strategy: Strategy,
        order: Order,
        signed_size: float,
        price: float,
        commission: float,
    ) -> bool:
        if signed_size > 0:
            return self._cash >= signed_size * price + commission
        position = strategy.getposition(order.data)
        short_open_size = max(0.0, abs(signed_size) - max(position.size, 0.0))
        if short_open_size == 0.0:
            return self._cash >= commission
        return self._cash >= short_open_size * price + commission

    def snapshot_portfolio(self, strategy: Strategy, timestamp: Any) -> None:
        self._last_strategy = strategy
        value = self.getvalue()
        self._equity_records.append({"datetime": timestamp, "cash": self._cash, "value": value})
        for data, position in strategy._positions.items():
            if not position and position.realized_pnl == 0.0:
                continue
            mark_price = _current_close(data)
            position_value = position.size * mark_price
            self._position_records.append(
                {
                    "datetime": timestamp,
                    "data": _data_name(data),
                    "size": position.size,
                    "avg_price": position.price,
                    "mark_price": mark_price,
                    "value": position_value,
                    "unrealized_pnl": (mark_price - position.price) * position.size,
                    "realized_pnl": position.realized_pnl,
                    "margin_used": abs(position_value),
                }
            )

    def orders_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._order_records, columns=_ORDER_COLUMNS)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._fill_records, columns=_FILL_COLUMNS)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_records, columns=_TRADE_COLUMNS)

    def positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._position_records, columns=_POSITION_COLUMNS)

    def equity_series(self) -> pd.Series:
        if not self._equity_records:
            return pd.Series(dtype="float64", name="equity")
        frame = pd.DataFrame(self._equity_records)
        series = pd.Series(frame["value"].to_numpy(), index=frame["datetime"], name="equity")
        series.index.name = None
        return series

    def realized_pnl(self) -> float:
        if self._last_strategy is None:
            return 0.0
        return sum(position.realized_pnl for position in self._last_strategy._positions.values())

    def unrealized_pnl(self) -> float:
        if self._last_strategy is None:
            return 0.0
        return sum(
            (_current_close(data) - position.price) * position.size
            for data, position in self._last_strategy._positions.items()
        )

    def margin_used(self) -> float:
        if self._last_strategy is None:
            return 0.0
        return sum(
            abs(position.size * _current_close(data))
            for data, position in self._last_strategy._positions.items()
        )

    def _record_order(self, order: Order) -> None:
        self._order_records.append(
            {
                "ref": order.ref,
                "datetime": _current_datetime(order.data),
                "data": _data_name(order.data),
                "side": _order_side_name(order.ordtype),
                "ordtype": order.ordtype,
                "exectype": _order_exectype_name(order.exectype),
                "status": _order_status_name(order.status),
                "status_code": order.status,
                "size": order.size,
                "price": order.price,
                "pricelimit": order.pricelimit,
                "time_in_force": order.time_in_force,
                "executed_size": order.executed.size,
                "executed_price": order.executed.price,
                "executed_value": order.executed.value,
                "commission": order.executed.comm,
                "pnl": order.executed.pnl,
            }
        )

    def _record_fill(self, order: Order) -> None:
        self._fill_records.append(
            {
                "order_ref": order.ref,
                "datetime": _current_datetime(order.data),
                "data": _data_name(order.data),
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
                "pnl": order.executed.pnl,
            }
        )

    def _record_trade(self, trade: Trade) -> None:
        self._trade_records.append(
            {
                "ref": trade.ref,
                "datetime": trade.dtclose or trade.dtopen,
                "data": _data_name(trade.data),
                "size": trade.size,
                "price": trade.price,
                "value": trade.value,
                "commission": trade.commission,
                "pnl": trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "isopen": trade.isopen,
                "isclosed": trade.isclosed,
                "status": trade.status,
                "dtopen": trade.dtopen,
                "dtclose": trade.dtclose,
            }
        )


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


def _current_datetime(data: DataFeed) -> Any:
    try:
        return data.datetime[0]
    except IndexError:
        return None


def _data_name(data: DataFeed) -> str:
    return data._name or "data0"


def _order_side_name(ordtype: int) -> str:
    return "buy" if ordtype == Order.Buy else "sell"


def _order_status_name(status: int) -> str:
    names = {
        Order.Submitted: "Submitted",
        Order.Accepted: "Accepted",
        Order.Partial: "Partial",
        Order.Completed: "Completed",
        Order.Canceled: "Canceled",
        Order.Expired: "Expired",
        Order.Margin: "Margin",
        Order.Rejected: "Rejected",
    }
    return names.get(status, f"Unknown({status})")


def _order_exectype_name(exectype: int) -> str:
    names = {
        Order.Market: "Market",
        Order.Limit: "Limit",
        Order.Stop: "Stop",
        Order.StopLimit: "StopLimit",
        Order.Close: "Close",
    }
    return names.get(exectype, f"Unknown({exectype})")


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


def _notify_order(strategy: Strategy, order: Order) -> None:
    strategy.notify_order(order)
    for analyzer in getattr(strategy, "analyzers", {}).values():
        analyzer.on_order(order)


class Strategy:
    """Base strategy class with backtrader-style lifecycle hooks."""

    params: tuple[tuple[str, Any], ...] = ()
    min_period = 0

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

    def addminperiod(self, period: int) -> None:
        self._min_period = max(self._min_period, _coerce_min_period(period, "period"))

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

    def on_order(self, order: Any) -> None:
        pass

    def on_bar(self, bar: BarSnapshot) -> None:
        pass

    def on_fill(self, fill: Any) -> None:
        pass

    def on_trade(self, trade: Any) -> None:
        pass

    def on_end(self, stats: Stats) -> None:
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
        self.stats: Stats | None = None
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

        total_bars = len(self.datas[0])
        data_cursors = [-1 for _ in self.datas]
        for cursor in range(total_bars):
            current_ts = self.datas[0]._frame.index[cursor]
            data_cursors[0] = cursor
            for index, data in enumerate(self.datas):
                if index > 0:
                    data_cursors[index] = _cursor_at_or_before(
                        data,
                        current_ts,
                        data_cursors[index],
                    )
                data._advance(data_cursors[index])
            self.broker.process_bar(strategy, analyzers)
            if cursor < strategy._min_period:
                strategy.prenext()
            else:
                strategy.next()
            if self.trade_on_close:
                self.broker.process_close(strategy, analyzers)
            bar = self._snapshot(self.datas[0])
            for analyzer in analyzers.values():
                analyzer.on_bar(bar)
            self.broker.snapshot_portfolio(strategy, bar.datetime)

        strategy.analyzer_results = {}
        self.stats = self._build_stats(strategy, total_bars)
        for analyzer in analyzers.values():
            analyzer.on_end(self.stats)
        self.analyzer_results = {
            name: analyzer.get_analysis() for name, analyzer in analyzers.items()
        }
        strategy.analyzer_results = dict(self.analyzer_results)
        self.stats.analyzers = dict(self.analyzer_results)
        strategy.stats = self.stats
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
        strategy._min_period = _coerce_min_period(
            getattr(strategy_cls, "min_period", 0), "Strategy.min_period"
        )
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

    def _build_stats(self, strategy: Strategy, total_bars: int) -> Stats:
        equity = self.broker.equity_series()
        returns = equity.pct_change().fillna(0.0).rename("returns")
        trades = self.broker.trades_frame()
        orders = self.broker.orders_frame()
        fills = self.broker.fills_frame()
        summary = {
            "bars": total_bars,
            "final_cash": self.broker.getcash(),
            "final_value": self.broker.getvalue(),
            "final_realized_pnl": self.broker.realized_pnl(),
            "final_unrealized_pnl": self.broker.unrealized_pnl(),
            "final_margin_used": self.broker.margin_used(),
            "total_trades": len(trades),
            "total_orders": len(orders),
            "total_fills": len(fills),
        }
        config = {
            "callback_batch": self.callback_batch,
            "trade_on_close": self.trade_on_close,
            "exactbars": self.exactbars,
            "stdstats": self.stdstats,
            "broker": {
                "cash": self.broker.getcash(),
                "commission": self.broker.commission,
            },
        }
        config.update(self.options)
        return Stats(
            returns=returns,
            equity=equity,
            trades=trades,
            positions=self.broker.positions_frame(),
            orders=orders,
            summary=summary,
            analyzers=dict(strategy.analyzer_results),
            config=config,
            fills=fills,
        )


def _cursor_at_or_before(data: DataFeed, timestamp: Any, current_cursor: int) -> int:
    cursor = current_cursor
    next_cursor = max(cursor + 1, 0)
    while next_cursor < len(data) and data._frame.index[next_cursor] <= timestamp:
        cursor = next_cursor
        next_cursor += 1
    return cursor
