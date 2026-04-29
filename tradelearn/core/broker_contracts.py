"""Broker-neutral order, fill, position, and account contracts.

These types are intentionally small immutable data shells. They do not encode
Backtrader, backtesting.py, or live broker state machines; adapters translate
between broker-specific runtime models and these contracts at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "ioc"]

__all__ = [
    "AccountSnapshot",
    "Fill",
    "OrderAck",
    "OrderRequest",
    "OrderSide",
    "OrderStatusUpdate",
    "OrderType",
    "PositionSnapshot",
    "TimeInForce",
]


@dataclass(frozen=True)
class OrderRequest:
    """Broker-neutral order submission request."""

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = "gtc"
    client_oid: str | None = None


@dataclass(frozen=True)
class OrderAck:
    """Broker-neutral order acknowledgement."""

    client_oid: str | None
    broker_oid: str
    accepted_ts: pd.Timestamp


@dataclass(frozen=True)
class Fill:
    """Broker-neutral fill event."""

    broker_oid: str
    symbol: str
    qty: float
    price: float
    commission: float
    ts: pd.Timestamp


@dataclass(frozen=True)
class PositionSnapshot:
    """Broker-neutral position snapshot."""

    symbol: str
    qty: float
    avg_price: float
    ts: pd.Timestamp


@dataclass(frozen=True)
class AccountSnapshot:
    """Broker-neutral account snapshot."""

    cash: float
    equity: float
    ts: pd.Timestamp


@dataclass(frozen=True)
class OrderStatusUpdate:
    """Broker-neutral order status update."""

    broker_oid: str
    status_str: str
    ts: pd.Timestamp
    replay: bool = False
