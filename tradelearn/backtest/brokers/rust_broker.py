"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from tradelearn.backtest.base import BaseBroker, _notify_order
from tradelearn.backtest.models import Order, Position, ExecutedInfo

if TYPE_CHECKING:
    from tradelearn.backtest.strategy import Strategy

class CommInfo:
    """Helper to simulate Backtrader's commission info."""
    def __init__(self, ratio: float):
        self.p = self.params = type('Params', (), {'commission': ratio})()
    
    def getcommission(self, size: float, price: float) -> float:
        return abs(size) * price * self.p.commission

class RustBroker(BaseBroker):
    """Proxy for the high-performance Rust backtesting engine."""
    def __init__(self, cash: float = 100000.0, commission: float = 0.0, mult: float = 1.0):
        super().__init__()
        self._cash = cash
        self.commission_ratio = commission
        self._mult = mult
        self._engine = None # Initialized in engine.py
        self._orders: List[Order] = []
        self._order_count = 0
        self._last_fill_idx = 0

    def getcash(self) -> float:
        return self._engine.get_cash() if self._engine else self._cash

    def getvalue(self) -> float:
        if not self._engine: return self._cash
        val = self._engine.get_equity()
        return val

    def getposition(self, data: Any = None) -> Position:
        if self._engine:
            size, price = self._engine.get_position()
            return Position(size=size, price=price)
        return Position(size=0.0, price=0.0)

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
        
        if self._engine:
            # signature: submit_order(self, /, side, order_type, size, limit_price=None, stop_price=None)
            side_str = "buy" if is_buy else "sell"
            otypes = {Order.Market: "market", Order.Limit: "limit", Order.Stop: "stop", Order.StopLimit: "stoplimit"}
            ot_str = otypes.get(order.exectype, "market")
            # Rust returns the assigned order_id
            order_id = self._engine.submit_order(side_str, ot_str, actual_size)
            order.ref = order_id
            order.status = Order.Accepted
            
        return order

    def buy(self, owner: Strategy, data: Any, size: float, price: Optional[float] = None, 
            exectype: Any = None, **kwargs) -> Order:
        return self._submit(owner, data, Order.Buy, size, price, exectype, **kwargs)

    def sell(self, owner: Strategy, data: Any, size: float, price: Optional[float] = None, 
             exectype: Any = None, **kwargs) -> Order:
        return self._submit(owner, data, Order.Sell, size, price, exectype, **kwargs)

    def step(self, i: int) -> None:
        if self._engine:
            self._engine.step_open(i)
            self._engine.step_close(i)

    def process_fills(self, strategy: Strategy, i: int) -> None:
        """Synchronize filled orders from Rust back to Python."""
        if not self._engine: return
        
        all_fills = self._engine.get_fills()
        if all_fills and len(all_fills) > self._last_fill_idx:
            new_fills = all_fills[self._last_fill_idx:]
            for fill in new_fills:
                order_id, side_str, size, price, comm = fill[:5]
                is_buy_fill = (side_str == "buy")
                
                matched_order = None
                for order in self._orders:
                    if order.status < Order.Completed and order.isbuy() == is_buy_fill:
                        matched_order = order
                        if order.ref == order_id:
                            break
                
                if matched_order:
                    matched_order.status = Order.Completed
                    matched_order.executed = ExecutedInfo(price=price, size=size, comm=comm, value=size*price)
                    strategy._on_fill(matched_order.data, size if is_buy_fill else -size, price)
                    _notify_order(strategy, matched_order)
            
            self._last_fill_idx = len(all_fills)
