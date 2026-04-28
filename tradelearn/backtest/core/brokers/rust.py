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
        self._rust_state_cache: tuple[int, float, float, float] | None = None
        self._step_fills_from_collect: list[Any] | None = None
        self._buffer_order_submissions = False
        self._order_submit_buffer: list[tuple[int, str, str, float, float | None, float | None]] = []
        # For 'bt' mode, we maintain state in Python
        self._pos = Position(size=0.0, price=0.0)
        self._active_cash = cash
        self._comminfo: Any = None

    def _uses_rust_matching(self) -> bool:
        return self._engine is not None

    def set_comminfo(self, comminfo: Any) -> None:
        self._comminfo = comminfo
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'mult'):
            self._mult = comminfo.p.mult
        if hasattr(comminfo, 'p') and hasattr(comminfo.p, 'commission'):
            self.commission_ratio = comminfo.p.commission

    def getcash(self) -> float:
        if self._engine is not None:
            _, cash, _, _ = self._get_rust_state()
            return cash
        return self._active_cash

    def getvalue(self) -> float:
        if self._engine is not None:
            _, cash, size, _price = self._get_rust_state()
            val = cash
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
        if self._engine is not None:
            _, _cash, size, price = self._get_rust_state()
            return Position(size=size, price=price)
        return self._pos

    def get_position_size(self) -> float:
        if self._engine is not None:
            _, _cash, size, _price = self._get_rust_state()
            return size
        return self._pos.size

    def _get_rust_state(self) -> tuple[int, float, float, float]:
        """Return cached Rust cash/position for the current bar."""
        if self._rust_state_cache is not None and self._rust_state_cache[0] == self._curr_idx:
            return self._rust_state_cache
        cash = self._engine.get_cash()
        size, price = self._engine.get_position()
        self._rust_state_cache = (self._curr_idx, cash, size, price)
        return self._rust_state_cache

    def get_mult(self, data: Any = None) -> float:
        return self._mult

    def getcommissioninfo(self, data: Any) -> CommInfo:
        return CommInfo(self.commission_ratio)

    def begin_order_buffering(self) -> None:
        """Delay Rust order submission until callbacks return."""
        self._buffer_order_submissions = True

    def flush_order_buffer(self) -> None:
        """Submit buffered orders to Rust and update Python order refs."""
        for provisional_ref, side_str, ot_str, actual_size, limit_price, stop_price in self.drain_order_buffer():
            order_id = self._engine.submit_order(
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
            )
            self.bind_rust_order_ref(provisional_ref, order_id)

    def drain_order_buffer(self) -> list[tuple[int, str, str, float, float | None, float | None]]:
        """Return buffered order payloads without calling back into Rust."""
        buffered = self._order_submit_buffer
        self._order_submit_buffer = []
        self._buffer_order_submissions = False
        return buffered

    def bind_rust_order_ref(self, provisional_ref: int, rust_ref: int) -> None:
        """Replace a provisional Python order ref with the Rust-assigned ref."""
        order = self._orders_by_ref.pop(provisional_ref)
        order.ref = rust_ref
        self._orders_by_ref[order.ref] = order

    def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
        """Replace multiple provisional Python order refs with Rust-assigned refs."""
        for provisional_ref, rust_ref in bindings:
            self.bind_rust_order_ref(provisional_ref, rust_ref)

    def submit_drained_order(
        self,
        provisional_ref: int,
        side_str: str,
        ot_str: str,
        actual_size: float,
        limit_price: float | None,
        stop_price: float | None,
    ) -> int:
        """Submit one drained order payload through Python's engine handle."""
        order_id = self._engine.submit_order(
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
        )
        self.bind_rust_order_ref(provisional_ref, order_id)
        return order_id

    def _submit_to_rust_engine(
        self,
        order: Order,
        side_str: str,
        ot_str: str,
        actual_size: float,
        limit_price: float | None,
        stop_price: float | None,
    ) -> None:
        order_id = self._engine.submit_order(
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
            )
        self.bind_rust_order_ref(order.ref, order_id)

    def _rust_order_payload(
        self,
        order: Order,
        side_str: str,
        actual_size: float,
        price: float | None,
    ) -> tuple[str, str, float, float | None, float | None]:
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
        return side_str, ot_str, actual_size, limit_price, stop_price

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
        
        if self._engine is not None:
            side_str = "buy" if is_buy else "sell"
            payload = self._rust_order_payload(order, side_str, actual_size, price)
            if self._buffer_order_submissions:
                self._order_submit_buffer.append((order.ref, *payload))
            else:
                self._submit_to_rust_engine(order, *payload)
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
        self._rust_state_cache = None
        self._step_fills_from_collect = None
        if self._engine is not None:
            if hasattr(self._engine, "step_open_collect"):
                fills, cash, size, price = self._engine.step_open_collect(i, self._last_fill_idx)
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            else:
                self._engine.step_open(i)
                self._rust_state_cache = None

    def process_fills(self, strategy: Strategy, i: int) -> None:
        """Synchronize filled orders back to Python."""
        if self._engine is not None:
            if self._step_fills_from_collect is None:
                new_fills = self._engine.get_new_fills(self._last_fill_idx)
            else:
                new_fills = self._step_fills_from_collect
                self._step_fills_from_collect = []
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
