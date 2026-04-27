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

    def equity_series(self):
        """Extract the full equity curve from Rust and return as a pandas Series."""
        import pandas as pd
        ts, cash, value = self._engine.get_equity_curve()
        return pd.Series(value, index=pd.to_datetime(ts, unit='s'))

    def trades_frame(self):
        """Extract all trade records from Rust and return as a pandas DataFrame."""
        import pandas as pd
        fills = self._engine.get_fills()
        if not fills:
            return pd.DataFrame()
            
        data = []
        for f in fills:
            # f is a tuple: (order_id, side_str, size, price, commission, slippage, pnl, ts)
            order_id, side_str, fill_size, fill_price, commission, slippage, pnl, ts = f
            data.append({
                'date': pd.to_datetime(ts, unit='s'),
                'symbol': 'data0',
                'side': side_str,
                'size': fill_size,
                'price': fill_price,
                'commission': commission,
                'pnl': pnl
            })
        return pd.DataFrame(data)

    # -- Broker interface stubs --

    def orders_frame(self):
        """Return orders as a DataFrame. Reuse fills data since Rust tracks at fill level."""
        return self.trades_frame()

    def fills_frame(self):
        """Return fills as a DataFrame."""
        return self.trades_frame()

    def positions_frame(self):
        """Return current positions as a DataFrame."""
        import pandas as pd
        size, avg_price = self._engine.get_position()
        if size == 0.0:
            return pd.DataFrame()
        return pd.DataFrame([{
            'symbol': 'data0',
            'size': size,
            'avg_price': avg_price,
            'value': size * avg_price,
        }])

    def realized_pnl(self):
        """Sum of all realized PnL from fills."""
        fills = self._engine.get_fills()
        return sum(f[6] for f in fills)  # pnl is index 6

    def unrealized_pnl(self):
        """Unrealized PnL based on current position vs last price."""
        return 0.0  # Simplified: mark-to-market handled by Rust

    def margin_used(self):
        """Margin used (not applicable for simple equity accounts)."""
        return 0.0

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
