"""Lightweight broker proxy that forwards buy/sell to the Rust BacktestEngine.

When the Rust fast-path is active in ``Cerebro._run_rust()``, strategy.broker
is replaced with an instance of this class. It translates the Python-level
``buy()``/``sell()`` calls into ``rust_engine.submit_order()`` invocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradelearn.backtest.engine import (
    DataFeed,
    ExecutedInfo,
    Order,
    Position,
    Strategy,
    _notify_order,
)


class RustBrokerProxy:
    """Thin proxy that delegates order management to a Rust BacktestEngine."""

    def __init__(self, rust_engine: Any, original_broker: Any) -> None:
        self._engine = rust_engine
        self._original = original_broker
        self._cash: float = original_broker._cash
        self.trade_on_close: bool = original_broker.trade_on_close
        # Track orders by id for notify callbacks
        self._orders: dict[int, Order] = {}
        self._next_order_ref = 1

    # -- cash / value queries (delegate to Rust) --

    def setcash(self, cash: float) -> None:
        self._cash = float(cash)

    def getcash(self, symbol: str | None = None) -> float:
        return self._engine.get_cash()

    def getvalue(self) -> float:
        return self._engine.get_equity()

    def set_cash(self, cash: float) -> None:
        self.setcash(cash)

    @property
    def commission(self) -> float:
        return self._original.commission

    def setcommission(self, commission: float) -> None:
        pass  # Already configured in Rust engine at init

    def set_slippage_model(self, slippage: Any) -> None:
        pass

    def set_commission_model(self, commission: Any) -> None:
        pass

    # -- order submission (forward to Rust) --

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
        return self._submit(strategy, data, Order.Buy, size, price, exectype,
                            pricelimit=pricelimit)

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
        return self._submit(strategy, data, Order.Sell, size, price, exectype,
                            pricelimit=pricelimit)

    def _submit(
        self,
        strategy: Strategy,
        data: DataFeed,
        ordtype: int,
        size: float | None,
        price: float | None,
        exectype: int | None,
        *,
        pricelimit: float | None = None,
    ) -> Order:
        side_str = "buy" if ordtype == Order.Buy else "sell"
        etype = Order.Market if exectype is None else exectype
        order_type_str = _exectype_to_str(etype)
        actual_size = float(1.0 if size is None else abs(size))

        limit_price = price if etype == Order.Limit else pricelimit
        stop_price = price if etype in (Order.Stop, Order.StopLimit) else None

        rust_order_id = self._engine.submit_order(
            side_str, order_type_str, actual_size, limit_price, stop_price
        )

        # Build a Python Order object for notify_order compatibility
        order = Order(
            ref=self._next_order_ref,
            data=data,
            ordtype=ordtype,
            size=actual_size,
            price=price,
            pricelimit=pricelimit,
            exectype=etype,
        )
        order.status = Order.Accepted
        self._next_order_ref += 1
        self._orders[rust_order_id] = order

        # Notify submitted + accepted
        _notify_order(strategy, order)
        return order

    def process_fills(self, strategy: Strategy, fills: list) -> None:
        """Called after rust_engine.step() to sync fill events back to Python."""
        for fill_tuple in fills:
            order_id, side_str, fill_size, fill_price, commission, slippage, pnl = fill_tuple

            order = self._orders.get(order_id)
            if order is None:
                continue

            # Update position on Python side
            position = strategy.getposition(order.data)
            signed_size = fill_size  # Already signed from Rust
            position.update(signed_size, fill_price)
            position.realized_pnl += pnl

            # Mark order completed
            order.status = Order.Completed
            order.executed = ExecutedInfo(
                size=signed_size,
                price=fill_price,
                value=abs(signed_size) * fill_price,
                comm=commission,
                slippage=slippage,
                pnl=pnl,
            )
            _notify_order(strategy, order)

    # -- Broker interface stubs --

    def process_bar(self, strategy: Strategy, analyzers: Any) -> None:
        pass  # Handled by Rust engine

    def process_close(self, strategy: Strategy, analyzers: Any) -> None:
        pass

    def snapshot_portfolio(self, strategy: Strategy, timestamp: Any) -> None:
        pass  # Handled by Rust engine

    def _record_order(self, order: Order) -> None:
        pass

    def _record_fill(self, order: Order) -> None:
        pass

    def _record_trade(self, trade: Any) -> None:
        pass


def _exectype_to_str(exectype: int) -> str:
    if exectype == Order.Limit:
        return "limit"
    if exectype == Order.Stop:
        return "stop"
    if exectype == Order.StopLimit:
        return "stop_limit"
    return "market"
