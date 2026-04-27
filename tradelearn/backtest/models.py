from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

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
class FixedCommission:
    amount: float = 0.0
    def calculate(self, size: float, price: float, side: int) -> float: return float(self.amount)
    def as_config(self) -> float: return float(self.amount)

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
    ref: int
    data: Any
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
