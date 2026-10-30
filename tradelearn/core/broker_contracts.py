"""Broker-neutral order, fill, position, and account contracts.

These types are intentionally small immutable data shells. They do not encode
Backtrader, backtesting.py, or live broker state machines; adapters translate
between broker-specific runtime models and these contracts at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "ioc"]

__all__ = [
    "AccountSnapshot",
    "Broker",
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
    market_value: float | None = None
    mark_price: float | None = None


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


@runtime_checkable
class Broker(Protocol):
    """Broker-neutral adapter contract for paper and live execution."""

    def place(self, req: OrderRequest) -> OrderAck:
        """Submit an order request and return the broker acknowledgement."""
        ...

    def cancel(self, broker_oid: str) -> None:
        """Request cancellation of an identified broker order."""
        ...

    def modify(self, broker_oid: str, **kwargs: Any) -> None:
        """Request supported changes to an identified broker order."""
        ...

    def positions(self) -> list[PositionSnapshot]:
        """Return the broker's current position snapshots."""
        ...

    def account(self) -> AccountSnapshot:
        """Return the broker's current account snapshot."""
        ...

    def order_status(self, broker_oid: str) -> OrderStatusUpdate:
        """Return the latest status of one broker order."""
        ...

    def order_statuses(
        self,
        broker_oids: Iterable[str],
    ) -> dict[str, OrderStatusUpdate]:
        """Return latest statuses keyed by broker order identifier."""
        ...

    def on_fill(self, cb: Callable[[Fill], None]) -> None:
        """Register a callback for broker fill events."""
        ...

    def on_cancel(self, cb: Callable[[Any], None]) -> None:
        """Register a callback for broker cancellation events."""
        ...

    def on_reject(self, cb: Callable[[Any, str], None]) -> None:
        """Register a callback for broker rejection events."""
        ...

    def connect(self) -> None:
        """Establish the broker connection."""
        ...

    def disconnect(self) -> None:
        """Close the broker connection."""
        ...

    def is_connected(self) -> bool:
        """Report whether the broker connection is active."""
        ...
