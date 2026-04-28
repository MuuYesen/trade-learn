"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradelearn.backtest.core.models import (
    BaseBroker,
    ExecutedInfo,
    Order,
    Position,
    Trade,
    _notify_order,
)

if TYPE_CHECKING:
    from tradelearn.backtest.core.strategy import Strategy

class CommInfo:
    """Helper to simulate Backtrader's commission info."""
    def __init__(self, ratio: float):
        self.p = self.params = type('Params', (), {'commission': ratio})()
    
    def getcommission(self, size: float, price: float) -> float:
        return abs(size) * price * self.p.commission

class RustBroker(BaseBroker):
    """Proxy for the high-performance Rust backtesting engine."""

    _RUST_MATCH_MODES = {"exact", "smart"}

    def __init__(
        self,
        cash: float = 100000.0,
        commission: float = 0.0,
        mult: float = 1.0,
        match_mode: str = 'exact',
    ):
        super().__init__()
        if match_mode not in self._RUST_MATCH_MODES:
            raise ValueError(
                f"Unsupported match_mode={match_mode!r}; expected one of "
                f"{sorted(self._RUST_MATCH_MODES)}"
            )
        self._cash = cash
        self.commission_ratio = commission
        self._mult = mult
        self.match_mode = match_mode
        self._engine = None # Initialized in engine.py
        self._close_prices = None
        self._curr_idx = 0
        self._orders: list[Order] = []
        self._orders_by_ref: dict[int, Order] = {}
        self._pending_orders: list[Order] = []
        self._order_count = 0
        self._last_fill_idx = 0
        # For 'bt' mode, we maintain state in Python
        self._pos = Position(size=0.0, price=0.0)
        self._active_cash = cash
        self._comminfo: Any = None

    def _uses_rust_matching(self) -> bool:
        return self.match_mode in self._RUST_MATCH_MODES and self._engine is not None

    def set_comminfo(self, comminfo: Any) -> None:
        self._comminfo = comminfo
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'mult'):
            self._mult = comminfo.p.mult
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'commission'):
            self.commission_ratio = comminfo.p.commission

    def getcash(self) -> float:
        if self._uses_rust_matching():
            return self._engine.get_cash()
        return self._active_cash

    def getvalue(self) -> float:
        if self._uses_rust_matching():
            val = self.getcash()
            size, price = self._engine.get_position()
            if size != 0 and self._close_prices is not None:
                current_price = self._close_prices[self._curr_idx]
                val += size * current_price * self._mult
            return val
            
        # BT mode or fallback
        val = self._active_cash
        if self._pos.size != 0:
            if self._close_prices is not None:
                current_price = self._close_prices[self._curr_idx]
                val += self._pos.size * current_price * self._mult
        return val

    def getposition(self, data: Any = None) -> Position:
        if self._uses_rust_matching():
            size, price = self._engine.get_position()
            return Position(size=size, price=price)
        return self._pos

    def get_mult(self, data: Any = None) -> float:
        return self._mult

    def getcommissioninfo(self, data: Any) -> CommInfo:
        return CommInfo(self.commission_ratio)

    def _submit(
        self,
        owner: Strategy,
        data: Any,
        side: int,
        size: float,
        price: float | None = None,
        exectype: Any = None,
        **kwargs,
    ) -> Order:
        """Core submission entry point called by Strategy.buy/sell."""
        self._order_count += 1
        is_buy = (side == Order.Buy)
        actual_size = float(size if size is not None else 1.0)
        
        order = Order(
            ref=self._order_count,
            data=data,
            ordtype=side,
            size=actual_size,
            price=price,
            pricelimit=kwargs.get("pricelimit"),
            exectype=exectype or Order.Market
        )
        order.status = Order.Submitted
        self._orders.append(order)
        self._orders_by_ref[order.ref] = order
        
        if self._uses_rust_matching():
            side_str = "buy" if is_buy else "sell"
            otypes = {
                Order.Market: "market",
                Order.Limit: "limit",
                Order.Stop: "stop",
                Order.StopLimit: "stop_limit",
            }
            ot_str = otypes.get(order.exectype, "market")
            limit_price = None
            stop_price = None
            if order.exectype == Order.Limit:
                limit_price = price
            elif order.exectype == Order.Stop:
                stop_price = price
            elif order.exectype == Order.StopLimit:
                stop_price = price
                limit_price = order.pricelimit
            # Rust returns the assigned order_id
            order_id = self._engine.submit_order(
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
            )
            self._orders_by_ref.pop(order.ref, None)
            order.ref = order_id
            self._orders_by_ref[order.ref] = order
            order.status = Order.Accepted
        else:
            # BT Mode: Add to pending
            order.status = Order.Accepted
            self._pending_orders.append(order)
            
        return order

    def buy(
        self,
        owner: Strategy,
        data: Any,
        size: float,
        price: float | None = None,
        exectype: Any = None,
        **kwargs,
    ) -> Order:
        return self._submit(owner, data, Order.Buy, size, price, exectype, **kwargs)

    def sell(
        self,
        owner: Strategy,
        data: Any,
        size: float,
        price: float | None = None,
        exectype: Any = None,
        **kwargs,
    ) -> Order:
        return self._submit(owner, data, Order.Sell, size, price, exectype, **kwargs)

    def step(self, i: int) -> None:
        self._curr_idx = i
        if self._uses_rust_matching():
            self._engine.step_open(i)

    def process_fills(self, strategy: Strategy, i: int) -> None:
        """Synchronize filled orders back to Python."""
        if self._uses_rust_matching():
            new_fills = self._engine.get_new_fills(self._last_fill_idx)
            if new_fills:
                for fill in new_fills:
                    order_id, side_str, size, price, comm, _slippage, pnl = fill[:7]
                    matched_order = self._orders_by_ref.get(order_id)
                    if matched_order:
                        matched_order.status = Order.Completed
                        abs_size = abs(size)
                        matched_order.executed = ExecutedInfo(
                            price=price, size=abs_size, comm=comm,
                            value=abs_size * price * self._mult,
                        )
                        self._sync_python_fill_state(strategy, matched_order, size, price, pnl)
                        _notify_order(strategy, matched_order)
                self._last_fill_idx += len(new_fills)
        else:
            # BT Mode: Naive matching in Python
            if not self._pending_orders:
                return
            o = self._open_prices[i]
            h = self._high_prices[i]
            low = self._low_prices[i]
            remaining = []
            for order in self._pending_orders:
                executed = False
                exec_price = 0.0
                
                if order.exectype == Order.Market:
                    # Executed at Open of current bar
                    executed = True
                    exec_price = o
                elif order.exectype == Order.Limit:
                    is_buy = order.isbuy()
                    limit_price = order.price
                    if is_buy:
                        if low <= limit_price:
                            executed = True
                            exec_price = min(o, limit_price)
                    else: # Sell
                        if h >= limit_price:
                            executed = True
                            exec_price = max(o, limit_price)
                elif order.exectype == Order.Stop:
                    is_buy = order.isbuy()
                    stop_price = order.price
                    if is_buy:
                        if h >= stop_price:
                            executed = True
                            exec_price = max(o, stop_price)
                    else: # Sell
                        if low <= stop_price:
                            executed = True
                            exec_price = min(o, stop_price)

                if executed:
                    order.status = Order.Completed
                    abs_size = abs(order.size)
                    
                    if self._comminfo:
                        # Use custom commission logic if available
                        comm = self._comminfo.getcommission(abs_size, exec_price, False)
                    else:
                        comm = abs_size * exec_price * self.commission_ratio * self._mult
                        
                    order.executed = ExecutedInfo(
                        price=exec_price, size=abs_size, comm=comm,
                        value=abs_size * exec_price * self._mult
                    )
                    signed_size = order.size if order.isbuy() else -order.size
                    # self._pos.update(signed_size, exec_price) - Strategy._on_fill will do this
                    self._active_cash -= comm
                    if order.isbuy():
                        self._active_cash -= abs_size * exec_price
                    else:
                        self._active_cash += abs_size * exec_price
                    
                    strategy._on_fill(order.data, signed_size, exec_price)
                    _notify_order(strategy, order)
                else:
                    remaining.append(order)
            self._pending_orders = remaining

    def _sync_python_fill_state(
        self,
        strategy: Strategy,
        order: Order,
        signed_size: float,
        price: float,
        pnl: float,
    ) -> None:
        """Maintain Python-side fill notifications while Rust owns cash/position state."""
        data = order.data
        strategy._pending_size[data] = strategy._pending_size.get(data, 0.0) - signed_size
        if abs(strategy._pending_size[data]) < 1e-9:
            strategy._pending_size[data] = 0.0

        old_size = self._pos.size
        old_price = self._pos.price
        self._pos.update(signed_size, price)
        new_size = self._pos.size

        if old_size != 0 and (old_size * new_size <= 0):
            trade = Trade(data=data, size=old_size, price=old_price, status=Trade.Closed)
            trade.pnl = pnl
            trade.pnlcomm = pnl
            trade.isclosed = True
            strategy.notify_trade(trade)
