from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np

class Params:
    """Simple parameter storage object."""
    def __init__(self, defaults: Any, **kwargs):
        if isinstance(defaults, dict):
            for name, val in defaults.items(): setattr(self, name, val)
        elif isinstance(defaults, (list, tuple)):
            for name, val in defaults:
                if isinstance(name, (list, tuple)): # tuple of (name, val)
                    setattr(self, name[0], name[1])
                else:
                    setattr(self, name, val)
        for name, val in kwargs.items(): setattr(self, name, val)

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
            if compression > 1: name = f"{compression}{name}"
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
    def calculate(self, size: float, price: float, side: int) -> float: return float(self.amount)
    def as_config(self) -> float: return float(self.amount)

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
    Buy, Sell = 1, 2
    Market, Limit, Stop, StopLimit, Close = range(1, 6)
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
    def getstatusname(self) -> str: return ["Created", "Submitted", "Accepted", "Partial", "Completed", "Canceled", "Expired", "Margin", "Rejected"][self.status]
    
    def isbuy(self) -> bool:
        return self.ordtype == Order.Buy
        
    def issell(self) -> bool:
        return self.ordtype == Order.Sell

    def alive(self) -> bool:
        return self.status in (Order.Created, Order.Submitted, Order.Accepted, Order.Partial)

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

class BaseBroker:
    def __init__(self, **kwargs): pass
    def setcash(self, cash: float): pass
    def setcommission(self, commission: float): pass
    def getcash(self) -> float: return 0.0
    def getvalue(self) -> float: return 0.0
    def get_cash(self) -> float: return self.getcash()
    def get_value(self) -> float: return self.getvalue()

class BaseSizer: pass

class BaseAnalyzer:
    def __init__(self, **kwargs): self.strategy = None
    def on_order(self, order: Order): pass
    def on_trade(self, trade: Any): pass
    def stop(self): pass

def _notify_order(strategy: Any, order: Order) -> None:
    strategy.notify_order(order)
