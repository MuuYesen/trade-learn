"""Minimal backtrader-style Strategy/Cerebro/Analyzer facade."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.metrics import max_drawdown, sharpe


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

    def __init__(self, values: Any) -> None:
        if isinstance(values, np.ndarray):
            self._values = values
        else:
            self._values = np.asarray(values)
        
        # Force to raw numpy if it's a pandas-backed array
        if hasattr(self._values, "to_numpy"):
            self._values = self._values.to_numpy()
            
        self._cursor = -1
        self._is_datetime = False

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def __getitem__(self, ago: int) -> Any:
        # Ultra-hot path: no checks, direct index
        return self._values[self._cursor + ago]

    def __call__(self, ago: int = 0) -> Any:
        """Support bt-style call syntax: line(0) instead of line[0]."""
        if self._cursor < 0:
            return ShiftedLine(self, ago)
        return self[ago]

    def __eq__(self, other: Any) -> bool:
        try:
            return self[0] == other
        except Exception:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        return self[0] > other

    def __lt__(self, other: Any) -> bool:
        return self[0] < other

    def __ge__(self, other: Any) -> bool:
        return self[0] >= other

    def __le__(self, other: Any) -> bool:
        return self[0] <= other

    def __bool__(self) -> bool:
        """Match Backtrader: bool(line) returns True if current value is non-zero."""
        val = self[0]
        if isinstance(val, float) and np.isnan(val):
            return False
        return bool(val != 0)

    def get(self, ago: int = 0, size: int = 1) -> list[Any]:
        end = self._cursor + ago + 1
        start = max(0, end - size)
        if end <= 0:
            return []
        return self._values[start:end].tolist()

    def date(self, ago: int = 0) -> Any:
        # Only convert to datetime when explicitly requested via .date()
        val = self[ago]
        if isinstance(val, (np.int64, int)):
             return pd.Timestamp(val, unit='s')
        return val

    def datetime(self, ago: int = 0) -> Any:
        """Match Backtrader: return a datetime object for the bar at 'ago'."""
        return self.date(ago)


class ShiftedLine(LineSeries):
    """A view of a LineSeries shifted by a fixed amount."""
    def __init__(self, source: LineSeries, shift: int):
        import numpy as np
        self._source = source
        self._shift = shift
        
        source_vals = source._values
        if shift < 0:
            abs_shift = abs(shift)
            # Create object array to allow None padding if necessary, or NaN for float
            if source_vals.dtype == np.float64:
                pad = np.full(abs_shift, np.nan)
            else:
                pad = np.array([None] * abs_shift)
            vals = np.concatenate([pad, source_vals])[:len(source_vals)]
        else:
            if source_vals.dtype == np.float64:
                pad = np.full(shift, np.nan)
            else:
                pad = np.array([None] * shift)
            vals = np.concatenate([source_vals[shift:], pad])
            
        super().__init__(vals)
        self._is_datetime = source._is_datetime

    @property
    def _cursor(self) -> int:
        return self._source._cursor

    @_cursor.setter
    def _cursor(self, value: int):
        pass

    def _advance(self, cursor: int) -> None:
        # Cursor is tracked dynamically via property
        pass


        return self._values[start:end]

    def date(self, ago: int = 0) -> Any:
        value = self[ago]
        if hasattr(value, "date"):
            return value.date()
        return value


class DataFeed:
    """Column-based OHLCV feed exposed to strategies."""

    def __init__(self, data: pd.DataFrame, name: str | None = None) -> None:
        import numpy as np
        frame = data.copy()
        self._name = name
        self._frame = frame
        
        # Use raw numpy arrays for all lines
        # For datetime, store as Unix timestamp (int64) to avoid Pandas overhead
        if isinstance(frame.index, pd.DatetimeIndex):
            self.datetime = LineSeries(frame.index.values.astype('datetime64[s]').view(np.int64))
        else:
            self.datetime = LineSeries(frame.index.to_numpy())
            
        self.open = LineSeries(frame["open"].to_numpy(dtype=np.float64))
        self.high = LineSeries(frame["high"].to_numpy(dtype=np.float64))
        self.low = LineSeries(frame["low"].to_numpy(dtype=np.float64))
        self.close = LineSeries(frame["close"].to_numpy(dtype=np.float64))
        self.volume = LineSeries(frame["volume"].to_numpy(dtype=np.float64))
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


@dataclass(frozen=True)
class FixedSlippage:
    amount: float = 0.0

    def apply(self, price: float, side: int, order: Order | None = None) -> float:
        adjustment = float(self.amount)
        return price + adjustment if side == Order.Buy else price - adjustment


@dataclass(frozen=True)
class PercentSlippage:
    ratio: float = 0.0

    def apply(self, price: float, side: int, order: Order | None = None) -> float:
        multiplier = 1.0 + float(self.ratio) if side == Order.Buy else 1.0 - float(self.ratio)
        return price * multiplier


@dataclass(frozen=True)
class BarRangeSlippage:
    ratio: float = 0.0
    seed: int | None = None

    def apply(self, price: float, side: int, order: Order | None = None) -> float:
        if order is None:
            return price
        bar_range = max(0.0, float(order.data.high[0]) - float(order.data.low[0]))
        random_fraction = random.Random(0 if self.seed is None else self.seed).random()
        adjustment = bar_range * float(self.ratio) * random_fraction
        return price + adjustment if side == Order.Buy else price - adjustment


@dataclass(frozen=True)
class FixedCommission:
    amount: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        return float(self.amount)

    def as_config(self) -> float:
        return float(self.amount)


@dataclass(frozen=True)
class PercentCommission:
    ratio: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        return abs(size) * price * float(self.ratio)

    def as_config(self) -> float:
        return float(self.ratio)


@dataclass(frozen=True)
class TieredCommission:
    tiers: list[tuple[float, float]]

    def calculate(self, size: float, price: float, side: int) -> float:
        if not self.tiers:
            return 0.0
        notional = abs(size) * price
        threshold, ratio = max(
            ((float(threshold), float(ratio)) for threshold, ratio in self.tiers),
            key=lambda tier: (tier[0] <= notional, tier[0]),
        )
        if threshold > notional:
            ratio = 0.0
        return _round_execution(notional * ratio)

    def as_config(self) -> float:
        return float(self.tiers[0][1]) if self.tiers else 0.0


@dataclass(frozen=True)
class CNAStockCommission:
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002

    def calculate(self, size: float, price: float, side: int) -> float:
        notional = abs(size) * price
        broker_fee = max(notional * self.commission_rate, self.min_commission)
        stamp_tax = notional * self.stamp_tax_rate if side == Order.Sell else 0.0
        transfer_fee = notional * self.transfer_fee_rate
        return _round_execution(broker_fee + stamp_tax + transfer_fee)

    def as_config(self) -> float:
        return float(self.commission_rate)


SlippageModel = FixedSlippage | PercentSlippage | BarRangeSlippage
CommissionModel = FixedCommission | PercentCommission | TieredCommission | CNAStockCommission
SlippageConfig = SlippageModel
CommissionConfig = CommissionModel


@dataclass
class ExecutedInfo:
    size: float = 0.0
    price: float = 0.0
    value: float = 0.0
    comm: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0


@dataclass
class Order:
    Submitted = 1
    Accepted = 2
    Partial = 3
    Completed = 4
    Canceled = 5
    Cancelled = Canceled
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
    activation_bar: int = 0

    def __post_init__(self) -> None:
        if self.executed is None:
            self.executed = ExecutedInfo()

    def isbuy(self) -> bool:
        return self.ordtype == self.Buy

    def issell(self) -> bool:
        return self.ordtype == self.Sell

    def alive(self) -> bool:
        return self.status in {self.Submitted, self.Accepted, self.Partial}

    def getstatusname(self) -> str:
        return _order_status_name(self.status)


@dataclass
class Trade:
    Created = 0
    Open = 1
    Closed = 2

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

    def getstatusname(self) -> str:
        if self.status == self.Created:
            return "Created"
        if self.status == self.Open:
            return "Open"
        if self.status == self.Closed:
            return "Closed"
        return str(self.status)


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
_FILL_COLUMNS = [
    "order_ref",
    "datetime",
    "data",
    "size",
    "price",
    "value",
    "commission",
    "slippage",
    "pnl",
]
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
        self._slippage_model: SlippageConfig = FixedSlippage(0.0)
        self._commission_model: CommissionConfig = FixedCommission(0.0)
        self.trade_on_close = False
        self._next_order_ref = 1
        self._next_trade_ref = 1
        self._pending: list[Order] = []
        self._current_bar_index = 0
        self._submit_activation_bar = 1
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
        self._commission_model = PercentCommission(float(commission))

    @property
    def commission(self) -> float:
        return self._commission_model.as_config()

    def set_slippage_model(self, slippage: SlippageModel) -> None:
        self._slippage_model = slippage

    def set_commission_model(self, commission: CommissionModel) -> None:
        self._commission_model = commission

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
            if order.activation_bar > self._current_bar_index:
                self._pending.append(order)
                continue
            executed = self._execute_order(strategy, order, analyzers)
            if not executed:
                self._handle_unfilled_order(strategy, order)

    def process_close(self, strategy: Strategy, analyzers: AnalyzerCollection) -> None:
        self._last_strategy = strategy
        pending, self._pending = self._pending, []
        for order in pending:
            if order.activation_bar > self._current_bar_index:
                self._pending.append(order)
                continue
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
            activation_bar=self._submit_activation_bar,
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
        fill = _match_order(
            order,
            self._commission_model,
            self._slippage_model,
            trade_on_close=trade_on_close,
        )
        if fill is None:
            return False
        signed_size, price, commission, slippage = fill
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
            slippage=slippage,
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
                "slippage": order.executed.slippage,
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
    commission: CommissionModel,
    slippage: SlippageModel,
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
    if (
        rust_match_order is not None
        and isinstance(slippage, FixedSlippage)
        and slippage.amount == 0.0
        and isinstance(commission, PercentCommission)
    ):
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
            commission.ratio,
        )
    return _python_match_order(
        order,
        limit_price,
        stop_price,
        commission,
        slippage,
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
    commission: CommissionModel,
    slippage: SlippageModel,
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
    signed_size = _round_execution(order.size if order.ordtype == Order.Buy else -order.size)
    raw_price = price
    price = slippage.apply(price, order.ordtype, order)
    price = _round_execution(price)
    slippage_amount = _round_execution(price - raw_price)
    return (
        signed_size,
        price,
        _round_execution(commission.calculate(signed_size, price, order.ordtype)),
        slippage_amount,
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


def _round_execution(value: float) -> float:
    return round(float(value), 6)


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
        d = data or self.data
        # Use simple attribute if it's the main data (most common)
        if d is self.data:
            if not hasattr(self, "_main_pos_cache"):
                 self._main_pos_cache = self._positions.setdefault(d, Position())
            return self._main_pos_cache
        return self._positions.setdefault(d, Position())

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
        slippage: SlippageModel | None = None,
        commission: CommissionModel | float | None = None,
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
        if slippage is not None:
            self.broker.set_slippage_model(slippage)
        if commission is not None:
            if isinstance(commission, int | float):
                self.broker.setcommission(float(commission))
            else:
                self.broker.set_commission_model(commission)

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

        # Try Rust fast-path, fallback to Python
        try:
            self._run_rust(strategy, analyzers)
        except (ImportError, AttributeError, TypeError):
            self._run_python(strategy, analyzers)

        total_bars = len(self.datas[0])
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

    def _run_python(self, strategy: Strategy, analyzers: AnalyzerCollection) -> None:
        """Original Python-driven main loop."""
        total_bars = len(self.datas[0])
        data_cursors = [-1 for _ in self.datas]
        for cursor in range(total_bars):
            self.broker._current_bar_index = cursor
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
            self.broker._submit_activation_bar = (
                cursor
                if self.trade_on_close and self.callback_batch == 1
                else cursor + self.callback_batch
            )
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

    def _run_rust(self, strategy: Strategy, analyzers: AnalyzerCollection) -> None:
        """Rust-driven main loop: cursor, matching, snapshots all in Rust."""
        from tradelearn._rust import RustBacktestEngine
        from tradelearn.backtest.rust_broker import RustBrokerProxy

        data0 = self.datas[0]
        frame = data0._frame

        # Extract OHLCV as numpy arrays for efficient passing
        import numpy as np
        if isinstance(frame.index, pd.DatetimeIndex):
            timestamps = frame.index.values.astype('datetime64[s]').astype(np.int64)
        else:
            timestamps = frame.index.to_numpy(dtype=np.int64)
        
        opens = frame["open"].to_numpy(dtype=np.float64)
        highs = frame["high"].to_numpy(dtype=np.float64)
        lows = frame["low"].to_numpy(dtype=np.float64)
        closes = frame["close"].to_numpy(dtype=np.float64)
        volumes = frame["volume"].to_numpy(dtype=np.float64)

        # Get commission ratio, multiplier and margin from broker
        comm_ratio = 0.0
        mult = 1.0
        margin = 1.0
        if hasattr(self.broker._commission_model, 'ratio'):
            comm_ratio = self.broker._commission_model.ratio
            
        comminfo = getattr(self.broker, 'comminfo', None)
        if comminfo:
            p = getattr(comminfo, 'p', None)
            if p:
                comm_ratio = getattr(p, 'commission', comm_ratio)
                mult = getattr(p, 'mult', mult)
                margin = getattr(p, 'margin', margin)

        rust_engine = RustBacktestEngine(
            timestamps, opens, highs, lows, closes, volumes,
            self.broker._cash, comm_ratio, self.trade_on_close,
            mult, margin
        )

        # Replace broker with Rust proxy
        original_broker = self.broker
        rust_proxy = RustBrokerProxy(rust_engine, original_broker, mult=mult)
        strategy.broker = rust_proxy
        self.broker = rust_proxy

        total_bars = rust_engine.total_bars()
        for cursor in range(total_bars):
            # Advance Python-side LineSeries cursors (needed for strategy.next())
            for data in self.datas:
                data._advance(cursor)

            # Rust: process pending orders + snapshot portfolio
            fills = rust_engine.step(cursor)

            # Sync fills back to Python (notify_order, update positions)
            if fills:
                rust_proxy.process_fills(strategy, fills)

            # Python: call strategy
            if cursor >= strategy._min_period:
                strategy.next()
            else:
                strategy.prenext()
            
            # Optimization: Only snapshot if analyzers exist
            if analyzers:
                bar = self._snapshot(data0)
                for analyzer in analyzers.values():
                    analyzer.on_bar(bar)

        # Restore original broker for stats building, but populate equity records
        equity_ts, equity_cash, equity_value = rust_engine.get_equity_curve()
        original_broker._cash = rust_engine.get_cash()
        original_broker._last_strategy = strategy
        # Populate equity records for _build_stats
        original_broker._equity_records = [
            {"datetime": data0._frame.index[i] if i < len(data0._frame.index) else i,
             "cash": equity_cash[i],
             "value": equity_value[i]}
            for i in range(len(equity_ts))
        ]
        self.broker = original_broker

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
        # Set indicator context so indicators created during __init__
        # can auto-bind to data and register min_period
        from tradelearn.compat.backtrader.indicators import (
            set_current_data, set_current_strategy,
        )
        set_current_data(self.datas[0] if self.datas else None)
        set_current_strategy(strategy)
        strategy_cls.__init__(strategy)
        set_current_data(None)
        set_current_strategy(None)
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
            "sharpe": sharpe(returns, periods=252),
            "max_drawdown": max_drawdown(returns),
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
