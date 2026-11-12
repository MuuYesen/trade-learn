from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradelearn.core.broker_contracts import Fill, OrderRequest


@dataclass
class Position:
    size: float = 0
    price: float = 0.0
    adjbase: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        if not self.size:
            self.price = 0.0

    def __bool__(self) -> bool:
        return bool(self.size)

    def __len__(self):
        return abs(self.size)

    def update(self, size, price, dt=None):
        old = self.size
        self.datetime = dt
        new = old + size
        opened = closed = 0
        if new == 0:
            closed = size
            self.price = 0.0
        elif old == 0:
            opened = size
            self.price = price
        elif old * size > 0:
            opened = size
            self.price = (old*self.price + size*price)/new
        elif old * new > 0:
            closed = size
        else:
            opened, closed = new, -old
            self.price = price
        self.size = new
        self.upopened, self.upclosed = opened, closed
        return self.size, self.price, opened, closed


_SUMMARY_FIELDS: tuple[tuple[str | None, str, str], ...] = (
    ("start", "Start", "date"),
    ("end", "End", "date"),
    ("duration", "Duration", "timedelta"),
    ("exposure_pct", "Exposure Time [%]", "pct_plain"),
    ("final_value", "Equity Final [$]", "money"),
    ("peak_value", "Equity Peak [$]", "money"),
    ("return_pct", "Return [%]", "pct_plain"),
    ("buy_hold_return_pct", "Buy & Hold Return [%]", "pct_plain"),
    ("annual_return", "Return (Ann.) [%]", "frac"),
    ("volatility", "Volatility (Ann.) [%]", "frac"),
    ("cagr_pct", "CAGR [%]", "pct_plain"),
    ("sharpe", "Sharpe Ratio", "ratio"),
    ("sortino", "Sortino Ratio", "ratio"),
    ("calmar", "Calmar Ratio", "ratio"),
    ("alpha_pct", "Alpha [%]", "pct_plain"),
    ("beta", "Beta", "ratio"),
    ("max_drawdown", "Max. Drawdown [%]", "frac_abs"),
    ("avg_drawdown", "Avg. Drawdown [%]", "frac_abs"),
    ("max_dd_duration", "Max. Drawdown Duration", "timedelta"),
    ("avg_dd_duration", "Avg. Drawdown Duration", "timedelta"),
    ("total_trades", "# Trades", "int"),
    ("win_rate_pct", "Win Rate [%]", "pct_plain"),
    ("best_trade_pct", "Best Trade PnL / Initial Equity [%]", "pct_plain"),
    ("worst_trade_pct", "Worst Trade PnL / Initial Equity [%]", "pct_plain"),
    ("avg_trade_pct", "Avg. Trade PnL / Initial Equity [%]", "pct_plain"),
    ("max_trade_duration", "Max. Trade Duration", "timedelta"),
    ("avg_trade_duration", "Avg. Trade Duration", "timedelta"),
    ("profit_factor", "Profit Factor", "ratio"),
    ("expectancy", "Expectancy", "money_signed"),
    ("sqn", "SQN", "ratio"),
    ("kelly_criterion", "Kelly Criterion", "ratio"),
)


def _format_summary_value(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "date":
        if isinstance(value, pd.Timestamp):
            return str(value.replace(microsecond=0).tz_localize(None))
        return str(value)
    if kind == "timedelta":
        if isinstance(value, pd.Timedelta):
            return str(value.floor("s"))
        return str(value)
    
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    
    if math.isnan(x) or math.isinf(x):
        return "—"
    if kind == "int":
        return f"{int(round(x)):>15}"
    if kind == "money":
        return f"{x:>15,.2f}"
    if kind == "money_signed":
        return f"{x:>+15,.2f}"
    if kind == "pct":
        return f"{x:>+15.2f}"
    if kind == "pct_plain":
        return f"{x:>15.2f}"
    if kind == "frac":
        return f"{x * 100:>15.2f}"
    if kind == "frac_abs":
        # Usually drawdowns are negative, we show absolute or handled by engine
        return f"{abs(x * 100):>15.2f}"
    if kind == "ratio":
        return f"{x:>15.2f}"
    return str(value)


class SummaryDict(dict):
    """Dict subclass that prints as an aligned backtest summary table."""

    _TITLE = "Backtest Summary"

    def __str__(self) -> str:
        if not self:
            return f"{self._TITLE}\n(empty)"

        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        
        for key, label, kind in _SUMMARY_FIELDS:
            if key is not None and key in self:
                rows.append((label, _format_summary_value(self[key], kind)))
                seen.add(key)
        
        # Add dynamic strategy/metadata at the bottom
        for key, value in self.items():
            if key in seen or key in ("bars", "final_cash", "final_margin_used", "total_orders", "total_fills", "final_realized_pnl", "final_unrealized_pnl"):
                continue
            rows.append((str(key), str(value)))

        if not rows:
            return f"{self._TITLE}\n(empty)"
            
        label_w = 30
        lines = []
        for label, value in rows:
            lines.append(f"{label:<{label_w}}{value:>20}")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


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
        # Summary retains legacy key spellings, with canonical numeric units.
        # 252 is the existing per-bar summary profile, not inferred calendar proof.
        units = {"annual_return": "ratio", "volatility": "ratio",
                 "sharpe": "dimensionless", "sortino": "dimensionless",
                 "calmar": "dimensionless", "max_drawdown": "ratio",
                 "profit_factor": "dimensionless", "expectancy": "pnl_currency",
                 "win_rate_pct": "percent"}
        self.metric_metadata = {
            key: {"unit": unit,
                  "reason": "not_calculable" if value is None or not math.isfinite(float(value)) else "",
                  "periods": 252,
                  "period_source": "default_per_bar_profile"}
            for key, unit in units.items() if key in summary
            for value in (summary[key],)
        }
        for key in ("profit_factor", "expectancy", "win_rate_pct"):
            if key in self.metric_metadata:
                self.metric_metadata[key]["pnl_basis"] = "pnlcomm"

    @staticmethod
    def _materialize(value: Any) -> Any:
        return value() if callable(value) else value

    def is_materialized(self, name: str) -> bool:
        """Return whether a lazy pandas artifact has already been built."""
        if name not in {"returns", "equity", "trades", "positions", "orders", "fills"}:
            raise KeyError(f"unknown stats artifact {name!r}")
        return not callable(getattr(self, f"_{name}"))

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


@dataclass
class ExecutedInfo:
    size: float = 0.0
    price: float = 0.0
    value: float = 0.0
    comm: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    dt: Any = None
    remsize: float = 0.0


@dataclass
class Order:
    Created, Submitted, Accepted, Partial, Completed, Canceled, Expired, Margin, Rejected = range(9)
    Cancelled = Canceled
    Buy, Sell = range(2)
    Market, Close, Limit, Stop, StopLimit, StopTrail, StopTrailLimit, Historical = range(8)
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
    created_ts: Any = None

    def __post_init__(self):
        quantity = abs(float(self.size))
        self.size = quantity if self.ordtype == self.Buy else -quantity

    Status = ("Created", "Submitted", "Accepted", "Partial", "Completed",
              "Canceled", "Expired", "Margin", "Rejected")
    ExecTypes = ("Market", "Close", "Limit", "Stop", "StopLimit",
                 "StopTrail", "StopTrailLimit", "Historical")

    def _initialize_clock(self, owner, submitted_at=None):
        from types import SimpleNamespace
        self.created_ts = submitted_at
        self._bt_dates = bool(getattr(owner, "_bt_minperiods", False))
        clock = getattr(self.data, "datetime", None)
        date = clock.datetime() if clock is not None else None
        if self._bt_dates and date is not None:
            from tradelearn.backtest.clock import date2num
            date = date2num(date)
        close_line = getattr(self.data, "close", None)
        close = float(close_line[0]) if close_line is not None else float("nan")
        price = close if not self.price and not self.pricelimit else self.price
        self.created = SimpleNamespace(dt=date, size=self.size, price=price,
            pricelimit=self.pricelimit or price, pclose=close,
            trailamount=self.trailamount, trailpercent=self.trailpercent)
        if self.exectype in (self.StopTrail, self.StopTrailLimit):
            self._limitoffset = self.created.price - self.created.pricelimit
            reference = self.created.price
            self.created.price = float('inf') if self.isbuy() else float('-inf')
            self.trailadjust(reference)
        self.executed.remsize = self.size

    def _set_execution_clock(self, timestamp, *, unit="s"):
        self._executed_dt_raw = pd.Timestamp(timestamp, unit=unit, tz="UTC") if isinstance(timestamp, numbers.Real) else pd.Timestamp(timestamp)
        self.executed.dt = self._executed_dt_raw
        if getattr(self, "_bt_dates", False):
            from tradelearn.backtest.clock import date2num
            self.executed.dt = date2num(self._executed_dt_raw)

    def trailadjust(self, price):
        """Adjust the BT order projection; execution remains owned by its broker."""
        distance = self.trailamount or (price * self.trailpercent if self.trailpercent else 0.0)
        candidate = price + distance if self.isbuy() else price - distance
        improves = candidate < self.created.price if self.isbuy() else candidate > self.created.price
        if improves:
            self.created.price = candidate
            if self.exectype == self.StopTrailLimit:
                self.created.pricelimit = candidate - self._limitoffset

    def getstatusname(self, status=None) -> str:
        return self.Status[self.status if status is None else status]

    def getordername(self, exectype=None) -> str:
        return self.ExecTypes[self.exectype if exectype is None else exectype]

    def clone(self):
        from copy import copy
        result = copy(self)
        result.executed = copy(self.executed)
        return result

    def isbuy(self) -> bool:
        return self.ordtype == Order.Buy

    def issell(self) -> bool:
        return self.ordtype == Order.Sell

    def alive(self) -> bool:
        return self.status in (Order.Created, Order.Submitted, Order.Accepted, Order.Partial)

    @classmethod
    def _from_request(cls, ref: int, request: OrderRequest, *, data: Any = None) -> Order:
        """Create a backtest runtime order from a broker-neutral request."""
        side = cls.Buy if request.side == "buy" else cls.Sell
        exectype = {
            "market": cls.Market,
            "close": cls.Close,
            "limit": cls.Limit,
            "stop": cls.Stop,
            "stop_limit": cls.StopLimit,
            "stop_trail": cls.StopTrail,
            "stop_trail_limit": cls.StopTrailLimit,
        }.get(request.order_type, cls.Market)
        price = (
            request.stop_price
            if request.order_type in ("stop", "stop_limit", "stop_trail", "stop_trail_limit")
            else request.limit_price
        )
        return cls(
            ref=ref,
            data=data,
            ordtype=side,
            size=float(request.qty),
            price=price,
            pricelimit=request.limit_price if request.order_type in ("stop_limit", "stop_trail_limit") else None,
            exectype=exectype,
            time_in_force=request.tif,
            valid=(pd.Timestamp(request.expire_time_ns-1_000_000_000,unit='ns',tz='UTC')
                   if getattr(request,'expire_time_ns',None) is not None else None),
            trailamount=getattr(request, 'trail_amount', None),
            trailpercent=getattr(request, 'trail_percent', None),
            info={"client_oid": request.client_oid, "symbol": request.symbol},
        )

    def _to_fill(
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
            ts=ts or pd.Timestamp.now(tz="UTC"),
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


def _notify_order(strategy: Any, order: Order) -> None:
    from tradelearn.core.strategy_events import dispatch_callback
    if callable(getattr(strategy,"notify_order",None)):
        dispatch_callback(strategy,"notify_order",order)
