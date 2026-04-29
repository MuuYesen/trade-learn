from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.core.broker_contracts import Fill, OrderRequest


class Params:
    """Simple parameter storage object."""

    def __init__(self, defaults: Any = (), **kwargs):
        self._keys: list[str] = []
        if isinstance(defaults, dict):
            for name, val in defaults.items():
                if name not in self._keys:
                    self._keys.append(name)
                setattr(self, name, val)
        elif isinstance(defaults, (list, tuple)):
            for name, val in defaults:
                if isinstance(name, (list, tuple)):  # tuple of (name, val)
                    if name[0] not in self._keys:
                        self._keys.append(name[0])
                    setattr(self, name[0], name[1])
                else:
                    if name not in self._keys:
                        self._keys.append(name)
                    setattr(self, name, val)
        for name, val in kwargs.items():
            if name not in self._keys:
                self._keys.append(name)
            setattr(self, name, val)

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and hasattr(self, "_keys") and name not in self._keys:
            self._keys.append(name)
        super().__setattr__(name, value)

    def __getitem__(self, index: int) -> Any:
        return getattr(self, self._keys[index])

    def asdict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._keys}


@dataclass
class Position:
    size: float = 0.0
    price: float = 0.0
    adjbase: float = 0.0
    realized_pnl: float = 0.0

    def __bool__(self) -> bool:
        return abs(self.size) > 1e-9

    def update(self, size_delta: float, price: float) -> None:
        new_size = self.size + size_delta
        if abs(new_size) < 1e-9:
            self.size = 0.0
            self.price = 0.0
            return
        if self.size == 0.0 or self.size * size_delta > 0:
            total_abs = abs(self.size) + abs(size_delta)
            self.price = (abs(self.size) * self.price + abs(size_delta) * price) / total_abs
        elif self.size * new_size < 0:
            self.price = price
        self.size = new_size


class TimeFrame:
    (NoTimeFrame, MicroSeconds, Seconds, Minutes, Days, Weeks, Months, Years) = range(8)
    Names = ["", "MicroSeconds", "Seconds", "Minutes", "Days", "Weeks", "Months", "Years"]

    @classmethod
    def getname(cls, tf: int, compression: int = 1) -> str:
        if tf < len(cls.Names):
            name = cls.Names[tf]
            if compression > 1:
                name = f"{compression}{name}"
            return name
        return ""


@dataclass
class BarSnapshot:
    datetime: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    data: Any


class Stats:
    """Backtest result container with optional lazy pandas artifact materialization."""

    def __init__(
        self,
        *,
        returns: pd.Series | Callable[[], pd.Series],
        equity: pd.Series | Callable[[], pd.Series],
        trades: pd.DataFrame | Callable[[], pd.DataFrame],
        positions: pd.DataFrame | Callable[[], pd.DataFrame],
        orders: pd.DataFrame | Callable[[], pd.DataFrame],
        summary: dict[str, Any],
        analyzers: dict[str, Any],
        config: dict[str, Any],
        fills: pd.DataFrame | Callable[[], pd.DataFrame],
    ) -> None:
        self._returns = returns
        self._equity = equity
        self._trades = trades
        self._positions = positions
        self._orders = orders
        self.summary = summary
        self.analyzers = analyzers
        self.config = config
        self._fills = fills

    @staticmethod
    def _materialize(value: Any) -> Any:
        return value() if callable(value) else value

    @property
    def returns(self) -> pd.Series:
        self._returns = self._materialize(self._returns)
        return self._returns

    @returns.setter
    def returns(self, value: pd.Series | Callable[[], pd.Series]) -> None:
        self._returns = value

    @property
    def equity(self) -> pd.Series:
        self._equity = self._materialize(self._equity)
        return self._equity

    @equity.setter
    def equity(self, value: pd.Series | Callable[[], pd.Series]) -> None:
        self._equity = value

    @property
    def trades(self) -> pd.DataFrame:
        self._trades = self._materialize(self._trades)
        return self._trades

    @trades.setter
    def trades(self, value: pd.DataFrame | Callable[[], pd.DataFrame]) -> None:
        self._trades = value

    @property
    def positions(self) -> pd.DataFrame:
        self._positions = self._materialize(self._positions)
        return self._positions

    @positions.setter
    def positions(self, value: pd.DataFrame | Callable[[], pd.DataFrame]) -> None:
        self._positions = value

    @property
    def orders(self) -> pd.DataFrame:
        self._orders = self._materialize(self._orders)
        return self._orders

    @orders.setter
    def orders(self, value: pd.DataFrame | Callable[[], pd.DataFrame]) -> None:
        self._orders = value

    @property
    def fills(self) -> pd.DataFrame:
        self._fills = self._materialize(self._fills)
        return self._fills

    @fills.setter
    def fills(self, value: pd.DataFrame | Callable[[], pd.DataFrame]) -> None:
        self._fills = value


@dataclass(frozen=True)
class FixedSlippage:
    amount: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        adj = float(self.amount)
        return price + adj if side == 1 else price - adj


@dataclass(frozen=True)
class PercentSlippage:
    ratio: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        adj = float(price) * float(self.ratio)
        return price + adj if side == 1 else price - adj


@dataclass
class BarRangeSlippage:
    ratio: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def apply(self, price: float, side: int, order: Any = None) -> float:
        data = getattr(order, "data", None)
        high = getattr(data, "high", None)
        low = getattr(data, "low", None)
        try:
            bar_range = float(high[0]) - float(low[0])
        except Exception:
            bar_range = 0.0
        adj = float(self._rng.random()) * bar_range * float(self.ratio)
        slipped = price + adj if side == 1 else price - adj
        return round(slipped, 6)


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
        return abs(float(size)) * float(price) * float(self.ratio)

    def as_config(self) -> float:
        return float(self.ratio)


@dataclass(frozen=True)
class TieredCommission:
    tiers: list[tuple[float, float]]

    def calculate(self, size: float, price: float, side: int) -> float:
        notional = abs(float(size)) * float(price)
        ratio = 0.0
        for threshold, tier_ratio in sorted(self.tiers, key=lambda item: item[0]):
            if notional >= threshold:
                ratio = float(tier_ratio)
        return notional * ratio


@dataclass(frozen=True)
class CNAStockCommission:
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002

    def calculate(self, size: float, price: float, side: int) -> float:
        notional = abs(float(size)) * float(price)
        commission = max(notional * self.commission_rate, self.min_commission)
        transfer_fee = notional * self.transfer_fee_rate
        stamp_tax = notional * self.stamp_tax_rate if side == Order.Sell else 0.0
        return round(commission + transfer_fee + stamp_tax, 6)


SlippageModel = FixedSlippage | PercentSlippage | BarRangeSlippage
CommissionModel = FixedCommission | PercentCommission | TieredCommission | CNAStockCommission


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
    Created, Submitted, Accepted, Partial, Completed, Canceled, Expired, Margin, Rejected = range(9)
    Cancelled = Canceled
    Buy, Sell = 1, 2
    Market, Limit, Stop, StopLimit, Close, StopTrail, StopTrailLimit = range(1, 8)
    DAY, GTC, IOC = "day", "gtc", "ioc"
    ref: int
    data: Any
    ordtype: int
    size: float
    price: float | None = None
    pricelimit: float | None = None
    exectype: int = Market
    time_in_force: str = GTC
    status: int = Created
    executed: ExecutedInfo = field(default_factory=ExecutedInfo)
    activation_bar: int = 0
    valid: Any = None
    oco: Any = None
    parent: Any = None
    transmit: bool = True
    trailamount: float | None = None
    trailpercent: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def getstatusname(self) -> str:
        return [
            "Created",
            "Submitted",
            "Accepted",
            "Partial",
            "Completed",
            "Canceled",
            "Expired",
            "Margin",
            "Rejected",
        ][self.status]

    def isbuy(self) -> bool:
        return self.ordtype == Order.Buy

    def issell(self) -> bool:
        return self.ordtype == Order.Sell

    def alive(self) -> bool:
        return self.status in (Order.Created, Order.Submitted, Order.Accepted, Order.Partial)

    @classmethod
    def from_request(cls, ref: int, request: OrderRequest, *, data: Any = None) -> Order:
        """Create a backtest runtime order from a broker-neutral request."""
        side = cls.Buy if request.side == "buy" else cls.Sell
        exectype = {
            "market": cls.Market,
            "limit": cls.Limit,
            "stop": cls.Stop,
            "stop_limit": cls.StopLimit,
        }.get(request.order_type, cls.Market)
        price = (
            request.stop_price
            if request.order_type in ("stop", "stop_limit")
            else request.limit_price
        )
        return cls(
            ref=ref,
            data=data,
            ordtype=side,
            size=float(request.qty),
            price=price,
            pricelimit=request.limit_price if request.order_type == "stop_limit" else None,
            exectype=exectype,
            time_in_force=request.tif,
            info={"client_oid": request.client_oid, "symbol": request.symbol},
        )

    def to_fill(
        self,
        *,
        qty: float,
        price: float,
        commission: float,
        ts: pd.Timestamp | None = None,
        broker_oid: str | None = None,
    ) -> Fill:
        """Emit a broker-neutral fill event for this backtest order."""
        return Fill(
            broker_oid=broker_oid or str(self.ref),
            symbol=str(self.info.get("symbol") or getattr(self.data, "_name", "data0")),
            qty=float(qty),
            price=float(price),
            commission=float(commission),
            ts=ts or pd.Timestamp.utcnow(),
        )


@dataclass
class Trade:
    Created, Open, Closed = range(3)
    ref: int = 0
    data: Any = None
    size: float = 0.0
    price: float = 0.0
    value: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0
    pnlcomm: float = 0.0
    isopen: bool = False
    isclosed: bool = False
    status: int = Created
    dtopen: Any = None
    dtclose: Any | None = None

    def getstatusname(self) -> str:
        return ["Created", "Open", "Closed"][self.status]


class BaseBroker:
    def __init__(self, **kwargs):
        pass

    def setcash(self, cash: float):
        pass

    def setcommission(self, commission: float):
        pass

    def getcash(self) -> float:
        return 0.0

    def getvalue(self) -> float:
        return 0.0

    def get_cash(self) -> float:
        return self.getcash()

    def get_value(self) -> float:
        return self.getvalue()


class BaseSizer:
    pass


class BaseAnalyzer:
    def __init__(self, **kwargs):
        self.strategy = None

    def on_order(self, order: Order):
        pass

    def on_trade(self, trade: Any):
        pass

    def stop(self):
        pass


def _notify_order(strategy: Any, order: Order) -> None:
    notify_order = getattr(strategy, "notify_order", None)
    if callable(notify_order):
        notify_order(order)
