"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from tradelearn.backtest.core.models import (
    BaseBroker, _notify_order, Order, Position, ExecutedInfo
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
    def __init__(self, cash: float = 100000.0, commission: float = 0.0, mult: float = 1.0, match_mode: str = 'bt'):
        super().__init__()
        self._cash = cash
        self.commission_ratio = commission
        self._mult = mult
        self.match_mode = match_mode
        self._engine = None # Initialized in engine.py
        self._close_prices = None
        self._curr_idx = 0
        self._orders: List[Order] = []
        self._pending_orders: List[Order] = []
        self._order_count = 0
        self._last_fill_idx = 0
        # For 'bt' mode, we maintain state in Python
        self._pos = Position(size=0.0, price=0.0)
        self._active_cash = cash
        self._comminfo: Any = None

    def set_comminfo(self, comminfo: Any) -> None:
        self._comminfo = comminfo
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'mult'):
            self._mult = comminfo.p.mult
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'commission'):
            self.commission_ratio = comminfo.p.commission

    def getcash(self) -> float:
        if self.match_mode == 'smart' and self._engine:
            return self._engine.get_cash()
        return self._active_cash

    def getvalue(self) -> float:
        if self.match_mode == 'smart' and self._engine:
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
        if self.match_mode == 'smart' and self._engine:
            size, price = self._engine.get_position()
            return Position(size=size, price=price)
        return self._pos

    def get_mult(self, data: Any = None) -> float:
        return self._mult

    def getcommissioninfo(self, data: Any) -> CommInfo:
        return CommInfo(self.commission_ratio)

    def _submit(self, owner: Strategy, data: Any, side: int, size: float, 
                price: Optional[float] = None, exectype: Any = None, **kwargs) -> Order:
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
            exectype=exectype or Order.Market
        )
        order.status = Order.Submitted
        self._orders.append(order)
        
        if self.match_mode == 'smart' and self._engine:
            # signature: submit_order(self, /, side, order_type, size, limit_price=None, stop_price=None)
            side_str = "buy" if is_buy else "sell"
            otypes = {Order.Market: "market", Order.Limit: "limit", Order.Stop: "stop", Order.StopLimit: "stoplimit"}
            ot_str = otypes.get(order.exectype, "market")
            # Rust returns the assigned order_id
            order_id = self._engine.submit_order(side_str, ot_str, actual_size)
            order.ref = order_id
            order.status = Order.Accepted
        else:
            # BT Mode: Add to pending
            order.status = Order.Accepted
            self._pending_orders.append(order)
            
        return order

    def buy(self, owner: Strategy, data: Any, size: float, price: Optional[float] = None, 
            exectype: Any = None, **kwargs) -> Order:
        return self._submit(owner, data, Order.Buy, size, price, exectype, **kwargs)

    def sell(self, owner: Strategy, data: Any, size: float, price: Optional[float] = None, 
             exectype: Any = None, **kwargs) -> Order:
        return self._submit(owner, data, Order.Sell, size, price, exectype, **kwargs)

    def step(self, i: int) -> None:
        self._curr_idx = i
        if self.match_mode == 'smart' and self._engine:
            self._engine.step_open(i)
            self._engine.step_close(i)

    def process_fills(self, strategy: Strategy, i: int) -> None:
        """Synchronize filled orders back to Python."""
        if self.match_mode == 'smart' and self._engine:
            all_fills = self._engine.get_fills()
            if all_fills and len(all_fills) > self._last_fill_idx:
                new_fills = all_fills[self._last_fill_idx:]
                for fill in new_fills:
                    order_id, side_str, size, price, comm = fill[:5]
                    matched_order = next((o for o in self._orders if o.ref == order_id), None)
                    if matched_order:
                        matched_order.status = Order.Completed
                        abs_size = abs(size)
                        matched_order.executed = ExecutedInfo(
                            price=price, size=abs_size, comm=comm, 
                            value=abs_size * price * self._mult
                        )
                        strategy._on_fill(matched_order.data, size, price)
                        _notify_order(strategy, matched_order)
                self._last_fill_idx = len(all_fills)
        else:
            # BT Mode: Naive matching in Python
            if not self._pending_orders: return
            o, h, l, c = self._open_prices[i], self._high_prices[i], self._low_prices[i], self._close_prices[i]
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
                        if l <= limit_price:
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
                        if l <= stop_price:
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
                    if order.isbuy(): self._active_cash -= abs_size * exec_price
                    else: self._active_cash += abs_size * exec_price
                    
                    strategy._on_fill(order.data, signed_size, exec_price)
                    _notify_order(strategy, order)
                else:
                    remaining.append(order)
            self._pending_orders = remaining
