"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from tradelearn.backtest.models import ExecutedInfo, Order, Position, Trade, _notify_order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.core import BrokerEventPump

if TYPE_CHECKING:
    from tradelearn.backtest.strategy import Strategy

OrderPayload = tuple[int, str, str, str, float, float | None, float | None]
_EMPTY_ORDER_BUFFER: tuple[()] = ()
CompactFillBatch = tuple[list[int], list[float], list[float], list[float], list[float], list[float]]
_ORDER_TYPE_TO_RUST = {
    1: "market",
    2: "limit",
    3: "stop",
    4: "stop_limit",
    6: "stop",
    7: "stop_limit",
}


class _CommissionInfoView:
    """Facade-only fallback for Backtrader-style commission info."""

    def __init__(self, ratio: float):
        self.p = self.params = type("Params", (), {"commission": ratio})()

    def getcommission(self, size: float, price: float) -> float:
        return abs(size) * price * self.p.commission


class RustBroker:
    """Proxy for the high-performance Rust backtesting engine."""

    _RUST_MATCH_MODES = {"exact", "smart"}

    def __init__(
        self,
        cash: float = 100000.0,
        commission: float = 0.0,
        mult: float = 1.0,
        match_mode: str = "exact",
    ):
        if match_mode not in self._RUST_MATCH_MODES:
            raise ValueError(
                f"Unsupported match_mode={match_mode!r}; expected one of "
                f"{sorted(self._RUST_MATCH_MODES)}"
            )
        self._cash = cash
        self.commission_ratio = commission
        self._mult = mult
        self.match_mode = match_mode
        self._engine = None  # Initialized in engine.py
        self._step_open_collect = None
        self._step_open_collect_compact = None
        self._step_close = None
        self._step_close_compact = None
        self._step_open_bars_compact = None
        self._step_close_bars_compact = None
        self._get_new_fills = None
        self._get_new_fills_compact = None
        self._close_prices = None
        self._curr_idx = 0
        self._orders: list[Order] = []
        self._orders_by_ref: dict[int, Order] = {}
        self._fills: list[dict[str, Any]] = []
        self._fills_frame_cache: Any = None
        self._fills_frame_cache_len = -1
        self._pending_orders: list[Order] = []
        self._deferred_child_orders: dict[int, list[tuple[Order, bool, float, float | None]]] = {}
        self._oco_order_count = 0
        self._order_count = 0
        self._closed_trade_count = 0
        self._winning_trade_count = 0
        self._last_fill_idx = 0
        self._rust_state_cache: tuple[int, float, float, float] | None = None
        self._position_state_cache: dict[str, tuple[int, float, float]] = {}
        self._step_fills_from_collect: list[Any] | CompactFillBatch | None = None
        self._active_datas: list[Any] = []
        self._buffer_order_submissions = False
        self._terminal_order_suppression = False
        self._order_submit_buffer: list[OrderPayload] = []
        self._proxy_events: list[Any] = []
        self._trade_on_close = False
        # For 'bt' mode, we maintain state in Python
        self._pos = Position(size=0.0, price=0.0)
        self._positions: dict[Any, Position] = {}
        self._active_cash = cash
        self._comminfo: Any = None
        self._slippage_model: Any = None
        self._commission_model: Any = None
        self._storage: dict[str, Any] | None = None

    def configure_matching(
        self,
        *,
        trade_on_close: bool | None = None,
        slippage: Any | None = None,
        commission: Any | None = None,
    ) -> None:
        """Configure matching options exposed to facade runners."""
        if trade_on_close is not None:
            self._trade_on_close = bool(trade_on_close)
        if slippage is not None:
            self._slippage_model = slippage
        if commission is not None:
            self._commission_model = commission

    def set_storage(self, storage: dict[str, Any] | None) -> None:
        """Attach facade-level run storage without exposing broker internals."""
        self._storage = storage

    @property
    def storage(self) -> dict[str, Any] | None:
        return self._storage

    def _uses_rust_matching(self) -> bool:
        return self._engine is not None

    def bind_engine(self, engine: Any) -> None:
        """Bind a Rust engine and cache optional FFI methods once."""
        self._engine = engine
        self._step_open_collect = getattr(engine, "step_open_collect", None)
        self._step_open_collect_compact = getattr(engine, "step_open_collect_compact", None)
        self._step_close = getattr(engine, "step_close", None)
        self._step_close_compact = getattr(engine, "step_close_compact", None)
        self._step_open_bars_compact = getattr(engine, "step_open_bars_compact", None)
        self._step_close_bars_compact = getattr(engine, "step_close_bars_compact", None)
        self._get_new_fills = getattr(engine, "get_new_fills", None)
        self._get_new_fills_compact = getattr(engine, "get_new_fills_compact", None)
        self._clear_state_caches()

    def setcash(self, cash: float) -> None:
        self._cash = float(cash)
        self._active_cash = float(cash)

    def set_cash(self, cash: float) -> None:
        self.setcash(cash)

    def setcommission(
        self, commission: float = 0.0, margin: float = 0.0, mult: float = 1.0
    ) -> None:
        self.commission_ratio = float(commission)
        self._mult = float(mult)

    def set_comminfo(self, comminfo: Any) -> None:
        self._comminfo = comminfo
        if hasattr(comminfo, "p") and hasattr(comminfo.p, "mult"):
            self._mult = comminfo.p.mult
        if hasattr(comminfo, "p") and hasattr(comminfo.p, "commission"):
            self.commission_ratio = comminfo.p.commission

    def addcommissioninfo(self, comminfo: Any, name: str | None = None) -> None:
        self.set_comminfo(comminfo)

    def getcash(self) -> float:
        if self._engine is not None:
            if (
                self._buffer_order_submissions
                and self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
            ):
                return float(self._rust_state_cache[1])
            _, cash, _, _ = self._get_rust_state()
            return cash
        return self._active_cash

    def get_cash(self) -> float:
        return self.getcash()

    def getvalue(self, datas: list[Any] | tuple[Any, ...] | None = None) -> float:
        if datas is not None:
            values = []
            for data in datas:
                position = self.getposition(data)
                if position.size == 0:
                    values.append(0.0)
                    continue
                values.append(position.size * self._current_close(data) * self._mult)
            return float(sum(values))

        if self._engine is not None:
            if (
                self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
            ):
                _, cash, size, _price = self._rust_state_cache
                if len(self._active_datas) <= 1 and size != 0 and self._close_prices is not None:
                    return float(cash + size * self._close_prices[self._curr_idx] * self._mult)
                if len(self._active_datas) > 1:
                    value = float(cash)
                    for data, position in self._positions.items():
                        if position.size:
                            value += position.size * self._current_close(data) * self._mult
                    return value
                return float(cash)
            if (
                self._buffer_order_submissions
                and self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
            ):
                value = float(self._rust_state_cache[1])
                for data, position in self._positions.items():
                    if position.size:
                        value += position.size * self._current_close(data) * self._mult
                return value
            get_equity = getattr(self._engine, "get_equity", None)
            if callable(get_equity):
                return float(get_equity())
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

    def get_value(self) -> float:
        return self.getvalue()

    def getposition(self, data: Any = None) -> Position:
        if self._engine is not None:
            if (
                self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
                and len(self._active_datas) <= 1
            ):
                _, _cash, size, price = self._rust_state_cache
                return Position(size=size, price=price)
            if (
                self._buffer_order_submissions
                and self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
                and len(self._active_datas) > 1
            ):
                position = self._positions.get(data)
                if position is not None:
                    return Position(size=position.size, price=position.price)
                return Position(size=0.0, price=0.0)
            get_position_for_symbol = getattr(self._engine, "get_position_for_symbol", None)
            if callable(get_position_for_symbol):
                symbol = self._rust_symbol(data)
                cached = self._position_state_cache.get(symbol)
                if cached is not None and cached[0] == self._curr_idx:
                    _, size, price = cached
                    return Position(size=size, price=price)
                size, price = get_position_for_symbol(symbol)
                self._position_state_cache[symbol] = (self._curr_idx, size, price)
                return Position(size=size, price=price)
            _, _cash, size, price = self._get_rust_state()
            return Position(size=size, price=price)
        if data is None:
            return self._pos
        return self._positions.setdefault(data, Position(size=0.0, price=0.0))

    def get_position_size(self) -> float:
        if self._engine is not None:
            _, _cash, size, _price = self._get_rust_state()
            return size
        return self._pos.size

    def current_position_size(self) -> float:
        return self.get_position_size()

    def bind_datas(self, datas: list[Any]) -> None:
        self._active_datas = list(datas)

    def _rust_symbol(self, data: Any = None) -> str:
        if data is None or len(self._active_datas) <= 1:
            return "data0"
        return str(getattr(data, "_name", None) or "data0")

    @staticmethod
    def _current_close(data: Any) -> float:
        close = getattr(data, "close", None)
        if close is not None:
            return float(close[0])
        return float(data.get_array("close")[data._cursor])

    @staticmethod
    def _current_bar_vectors(
        datas: list[Any],
    ) -> tuple[
        list[str],
        list[int],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        symbols: list[str] = []
        timestamps: list[int] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []
        single_data = len(datas) <= 1
        for index, data in enumerate(datas):
            cursor = getattr(data, "_cursor", -1)
            if cursor < 0:
                continue
            symbol = "data0" if single_data else str(getattr(data, "_name", None) or f"data{index}")
            symbols.append(symbol)
            timestamps.append(int(data._datetime[cursor]))
            opens.append(float(data._open[cursor]))
            highs.append(float(data._high[cursor]))
            lows.append(float(data._low[cursor]))
            closes.append(float(data._close[cursor]))
            volumes.append(float(data._volume[cursor]))
        return symbols, timestamps, opens, highs, lows, closes, volumes

    def drain_proxy_events(self) -> list[Any]:
        events = self._proxy_events
        self._proxy_events = []
        return events

    def event_pump(self) -> BrokerEventPump:
        return BrokerEventPump(self.drain_proxy_events)

    def fills_frame(self):
        import pandas as pd

        if self._fills_frame_cache is None or self._fills_frame_cache_len != len(self._fills):
            frame = pd.DataFrame(self._fills)
            if not frame.empty and "datetime" in frame.columns:
                datetimes = frame["datetime"]
                if pd.api.types.is_numeric_dtype(datetimes):
                    frame["datetime"] = pd.to_datetime(datetimes, unit="s", utc=True)
                else:
                    frame["datetime"] = pd.to_datetime(datetimes, utc=True)
            self._fills_frame_cache = frame
            self._fills_frame_cache_len = len(self._fills)
        return self._fills_frame_cache

    def trade_summary(self) -> tuple[int, int]:
        return self._closed_trade_count, self._winning_trade_count

    def _fill_datetime(self, data: Any) -> Any:
        timestamps = getattr(data, "_datetime", None)
        if timestamps is None:
            return None
        try:
            if self._curr_idx >= len(timestamps):
                return None
            value = timestamps[self._curr_idx]
        except (IndexError, TypeError):
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return value

    def _record_fill(
        self,
        order: Order,
        signed_size: float,
        price: float,
        comm: float,
        *,
        pnl: float = 0.0,
        trade_closed: bool = False,
    ) -> None:
        self._fills.append(
            {
                "datetime": self._fill_datetime(order.data),
                "ref": order.ref,
                "data": getattr(order.data, "_name", None),
                "side": "buy" if order.isbuy() else "sell",
                "size": signed_size,
                "price": price,
                "commission": comm,
                "value": abs(signed_size) * price * self._mult,
                "pnl": pnl,
                "trade_closed": trade_closed,
            }
        )

    def _notify_order_event(self, owner: Strategy, order: Order) -> None:
        if getattr(type(owner), "notify_order", None) is not CoreStrategy.notify_order:
            _notify_order(owner, order)
        for analyzer in getattr(owner, "analyzers", {}).values():
            on_order = getattr(analyzer, "on_order", None)
            if callable(on_order):
                on_order(order)

    def _notify_fill_event(
        self, owner: Strategy, order: Order, signed_size: float, price: float, comm: float
    ) -> None:
        callbacks = [
            on_fill
            for analyzer in getattr(owner, "analyzers", {}).values()
            if callable(on_fill := getattr(analyzer, "on_fill", None))
        ]
        if not callbacks:
            return
        fill = SimpleNamespace(
            order=order,
            ref=order.ref,
            data=order.data,
            size=signed_size,
            price=price,
            commission=comm,
        )
        for on_fill in callbacks:
            on_fill(fill)

    def _notify_trade_event(self, owner: Strategy, trade: Trade) -> None:
        notify_trade = getattr(owner, "notify_trade", None)
        if callable(notify_trade):
            notify_trade(trade)
        for on_trade in self._trade_callbacks(owner):
            on_trade(trade)

    @staticmethod
    def _trade_callbacks(owner: Strategy) -> tuple[Any, ...]:
        return tuple(
            on_trade
            for analyzer in getattr(owner, "analyzers", {}).values()
            if "on_trade" in type(analyzer).__dict__
            and callable(on_trade := getattr(analyzer, "on_trade", None))
        )

    @staticmethod
    def _strategy_wants_trade(owner: Strategy) -> bool:
        notify_trade = getattr(type(owner), "notify_trade", None)
        return notify_trade is not None and notify_trade is not CoreStrategy.notify_trade

    def _wants_trade_event(self, owner: Strategy) -> bool:
        return self._strategy_wants_trade(owner) or bool(self._trade_callbacks(owner))

    def _get_rust_state(self) -> tuple[int, float, float, float]:
        """Return cached Rust cash/position for the current bar."""
        if self._rust_state_cache is not None and self._rust_state_cache[0] == self._curr_idx:
            return self._rust_state_cache
        cash = self._engine.get_cash()
        size, price = self._engine.get_position()
        self._rust_state_cache = (self._curr_idx, cash, size, price)
        return self._rust_state_cache

    def _clear_state_caches(self) -> None:
        """Drop same-bar Rust state mirrors after a state-changing operation."""
        self._rust_state_cache = None
        self._position_state_cache.clear()

    def get_mult(self, data: Any = None) -> float:
        return self._mult

    def getcommissioninfo(self, data: Any) -> _CommissionInfoView:
        if self._comminfo is not None:
            return self._comminfo
        return _CommissionInfoView(self.commission_ratio)

    def get_orders_history(self) -> list[Order]:
        return list(self._orders)

    def get_orders_open(self, safe: bool = False) -> list[Order]:
        open_statuses = {Order.Submitted, Order.Accepted, Order.Partial}
        orders = [order for order in self._orders if order.status in open_statuses]
        return list(orders) if safe else orders

    def begin_order_buffering(self) -> None:
        """Delay Rust order submission until callbacks return."""
        self._buffer_order_submissions = True

    def begin_terminal_order_suppression(self) -> None:
        """Return terminal-bar order objects without routing them to the broker."""
        self._terminal_order_suppression = True

    def end_terminal_order_suppression(self) -> None:
        """Restore normal order routing after a terminal-bar strategy callback."""
        self._terminal_order_suppression = False

    def flush_order_buffer(self) -> None:
        """Submit buffered orders to Rust and update Python order refs."""
        submitted = False
        for (
            provisional_ref,
            symbol,
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
        ) in self.drain_order_buffer():
            order_id = self._submit_payload_to_engine(
                symbol, side_str, ot_str, actual_size, limit_price, stop_price
            )
            self.bind_rust_order_ref(provisional_ref, order_id)
            submitted = True
        if (
            submitted
            and self._trade_on_close
            and self._engine is not None
        ):
            self._clear_state_caches()
            if self._active_datas and self._step_close_bars_compact is not None:
                fills, cash, size, price = self._step_close_bars_compact(
                    *self._current_bar_vectors(self._active_datas)
                )
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            elif self._step_close_compact is not None:
                self._step_fills_from_collect = self._step_close_compact(self._curr_idx)
            elif self._step_close is not None:
                self._step_fills_from_collect = self._step_close(self._curr_idx)

    def drain_order_buffer(self) -> list[OrderPayload] | tuple[()]:
        """Return buffered order payloads without calling back into Rust."""
        buffered = self._order_submit_buffer
        self._buffer_order_submissions = False
        if not buffered:
            return _EMPTY_ORDER_BUFFER
        self._order_submit_buffer = []
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
        order_id = self._submit_payload_to_engine(
            "data0", side_str, ot_str, actual_size, limit_price, stop_price
        )
        self.bind_rust_order_ref(provisional_ref, order_id)
        return order_id

    def _submit_payload_to_engine(
        self,
        symbol: str,
        side_str: str,
        ot_str: str,
        actual_size: float,
        limit_price: float | None,
        stop_price: float | None,
    ) -> int:
        submit_for_symbol = getattr(self._engine, "submit_order_for_symbol", None)
        if callable(submit_for_symbol):
            return submit_for_symbol(
                symbol,
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
            )
        return self._engine.submit_order(
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
        )

    def _submit_to_rust_engine(
        self,
        order: Order,
        symbol: str,
        side_str: str,
        ot_str: str,
        actual_size: float,
        limit_price: float | None,
        stop_price: float | None,
    ) -> None:
        order_id = self._submit_payload_to_engine(
            symbol,
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
    ) -> tuple[str, str, str, float, float | None, float | None]:
        symbol = self._rust_symbol(order.data)
        ot_str = _ORDER_TYPE_TO_RUST.get(order.exectype, "market")
        limit_price = None
        stop_price = None
        if order.exectype == Order.Limit:
            limit_price = price
        elif order.exectype == Order.Stop:
            stop_price = price
        elif order.exectype == Order.StopLimit:
            stop_price = price
            limit_price = order.pricelimit
        return symbol, side_str, ot_str, actual_size, limit_price, stop_price

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
        is_buy = side == Order.Buy
        actual_size = float(size if size is not None else 1.0)

        order = Order(
            ref=self._order_count,
            data=data,
            ordtype=side,
            size=actual_size,
            price=price,
            pricelimit=kwargs.get("pricelimit"),
            exectype=exectype or Order.Market,
            valid=kwargs.get("valid"),
            oco=kwargs.get("oco"),
            parent=kwargs.get("parent"),
            transmit=bool(kwargs.get("transmit", True)),
            trailamount=kwargs.get("trailamount"),
            trailpercent=kwargs.get("trailpercent"),
            info=dict(kwargs.get("info", {})),
        )
        if self._terminal_order_suppression:
            return order
        self._register_and_route_order(owner, order, is_buy, actual_size, price)

        return order

    def _submit_basic(
        self,
        owner: Strategy,
        data: Any,
        side: int,
        size: float,
        price: float | None,
        exectype: Any,
    ) -> Order:
        """Fast path for facade orders without optional Backtrader metadata."""
        self._order_count += 1
        is_buy = side == Order.Buy
        actual_size = float(size if size is not None else 1.0)
        exectype = exectype or Order.Market

        order = Order(
            ref=self._order_count,
            data=data,
            ordtype=side,
            size=actual_size,
            price=price,
            exectype=exectype,
        )
        if self._terminal_order_suppression:
            return order
        self._register_and_route_order(owner, order, is_buy, actual_size, price)

        return order

    def submit_basic(
        self,
        owner: Strategy,
        data: Any,
        side: int,
        size: float,
        price: float | None,
        exectype: Any,
    ) -> Order:
        """Public fast path for facade orders without optional metadata."""
        return self._submit_basic(owner, data, side, size, price, exectype)

    def _register_and_route_order(
        self,
        owner: Strategy,
        order: Order,
        is_buy: bool,
        actual_size: float,
        price: float | None,
    ) -> None:
        order.status = Order.Submitted
        self._orders.append(order)
        self._orders_by_ref[order.ref] = order
        if order.oco is not None:
            self._oco_order_count += 1
        self._notify_order_event(owner, order)

        if order.parent is not None and order.parent.status != Order.Completed:
            self._deferred_child_orders.setdefault(id(order.parent), []).append(
                (order, is_buy, actual_size, price)
            )
            order.status = Order.Accepted
            self._notify_order_event(owner, order)
            return

        self._route_accepted_order_to_matcher(order, is_buy, actual_size, price)
        self._notify_order_event(owner, order)

    def _route_accepted_order_to_matcher(
        self,
        order: Order,
        is_buy: bool,
        actual_size: float,
        price: float | None,
    ) -> None:
        if self._engine is not None:
            side_str = "buy" if is_buy else "sell"
            payload = self._rust_order_payload(order, side_str, actual_size, price)
            if self._buffer_order_submissions:
                self._order_submit_buffer.append((order.ref, *payload))
            else:
                self._submit_to_rust_engine(order, *payload)
            order.status = Order.Accepted
        else:
            order.status = Order.Accepted
            self._pending_orders.append(order)

    def cancel(self, order: Order) -> None:
        """Cancel an order in the Python mirror and pending queue."""
        self._cancel_order_mirror(order)

    def _cancel_order_mirror(self, order: Order, owner: Strategy | None = None) -> bool:
        """Cancel one Python-side order mirror and notify when an owner is available."""
        if order.status in (Order.Completed, Order.Canceled, Order.Expired):
            return False
        order.status = Order.Canceled
        self._pending_orders = [pending for pending in self._pending_orders if pending is not order]
        if owner is not None:
            self._notify_order_event(owner, order)
        return True

    def _cancel_oco_siblings(self, owner: Strategy, completed: Order) -> None:
        """Cancel live OCO siblings after one order in the OCO pair fills."""
        if completed.oco is None and self._oco_order_count == 0:
            return
        for candidate in self._orders:
            if candidate is completed or not candidate.alive():
                continue
            if completed.oco is candidate or candidate.oco is completed:
                self._cancel_order_mirror(candidate, owner)

    def _activate_child_orders(self, owner: Strategy, parent: Order) -> None:
        """Route deferred bracket child orders once the parent has filled."""
        children = self._deferred_child_orders.pop(id(parent), ())
        for child, is_buy, actual_size, price in children:
            if child.alive():
                if self._engine is not None and hasattr(self._engine, "run_bar_loop"):
                    side_str = "buy" if is_buy else "sell"
                    payload = self._rust_order_payload(child, side_str, actual_size, price)
                    self._order_submit_buffer.append((child.ref, *payload))
                    child.status = Order.Accepted
                else:
                    self._route_accepted_order_to_matcher(child, is_buy, actual_size, price)
                self._notify_order_event(owner, child)

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
        self._clear_state_caches()
        self._step_fills_from_collect = None
        if self._engine is not None:
            if self._active_datas and self._step_open_bars_compact is not None:
                fills, cash, size, price = self._step_open_bars_compact(
                    *self._current_bar_vectors(self._active_datas)
                )
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            elif self._step_open_collect_compact is not None:
                fills, cash, size, price = self._step_open_collect_compact(
                    i,
                    self._last_fill_idx,
                )
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            elif self._step_open_collect is not None:
                fills, cash, size, price = self._step_open_collect(i, self._last_fill_idx)
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            else:
                self._engine.step_open(i)
                self._rust_state_cache = None

    @staticmethod
    def _fill_batch_len(fills: Any) -> int:
        if fills is None:
            return 0
        if RustBroker._is_compact_fill_batch(fills):
            return len(fills[0])
        return len(fills)

    @staticmethod
    def _is_compact_fill_batch(fills: Any) -> bool:
        return isinstance(fills, tuple) and len(fills) == 6

    def process_fills(self, strategy: Strategy, i: int) -> None:
        """Synchronize filled orders back to Python."""
        if self._engine is not None:
            if self._step_fills_from_collect is None:
                if self._get_new_fills_compact is not None:
                    new_fills = self._get_new_fills_compact(self._last_fill_idx)
                else:
                    new_fills = self._get_new_fills(self._last_fill_idx)
            else:
                new_fills = self._step_fills_from_collect
                self._step_fills_from_collect = None
            fill_count = self._fill_batch_len(new_fills)
            if fill_count:
                self._process_rust_fills_batch(strategy, new_fills)
                self._last_fill_idx += fill_count
        elif self._pending_orders:
            raise RuntimeError("RustBroker requires a Rust engine for order matching")

    def _iter_rust_fills(self, fills: Any):
        if self._is_compact_fill_batch(fills):
            order_ids, sizes, prices, comms, _slippages, pnls = fills
            return zip(order_ids, sizes, prices, comms, pnls, strict=True)
        return (
            (fill[0], fill[2], fill[3], fill[4], fill[6])
            for fill in fills
        )

    def _process_rust_fills_batch(self, strategy: Strategy, fills: Any) -> None:
        """Synchronize a batch of Rust fills while preserving notification order."""
        self._position_state_cache.clear()
        orders_by_ref_get = self._orders_by_ref.get
        pending_size = strategy._pending_size
        mult = self._mult
        wants_trade_event = self._wants_trade_event(strategy)

        for order_id, signed_size, price, comm, pnl in self._iter_rust_fills(fills):
            order = orders_by_ref_get(order_id)
            if order is None:
                continue

            order.status = Order.Completed
            abs_size = abs(signed_size)
            order.executed = ExecutedInfo(
                price=price,
                size=abs_size,
                comm=comm,
                value=abs_size * price * mult,
            )

            data = order.data
            pending = pending_size.get(data, 0.0) - signed_size
            pending_size[data] = 0.0 if abs(pending) < 1e-9 else pending

            pos = self._positions.setdefault(data, Position(size=0.0, price=0.0))
            old_size = pos.size
            pos.update(signed_size, price)
            if data is None:
                self._pos = pos
            new_size = pos.size

            trade_closed = old_size != 0 and old_size * new_size <= 0
            if trade_closed:
                self._closed_trade_count += 1
                if pnl > 0:
                    self._winning_trade_count += 1
            if wants_trade_event and old_size == 0 and new_size != 0:
                trade = Trade(data=data, size=new_size, price=price, status=Trade.Open)
                trade.pnl = 0.0
                trade.pnlcomm = 0.0
                trade.isopen = True
                self._notify_trade_event(strategy, trade)
            elif wants_trade_event and trade_closed:
                trade = Trade(data=data, size=new_size, price=price, status=Trade.Closed)
                trade.pnl = pnl
                trade.pnlcomm = pnl
                trade.isclosed = True
                self._notify_trade_event(strategy, trade)

            self._record_fill(
                order,
                signed_size,
                price,
                comm,
                pnl=pnl,
                trade_closed=trade_closed,
            )
            self._notify_fill_event(strategy, order, signed_size, price, comm)
            self._notify_order_event(strategy, order)
            self._activate_child_orders(strategy, order)
            self._cancel_oco_siblings(strategy, order)
