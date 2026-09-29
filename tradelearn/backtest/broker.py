"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from tradelearn.backtest.models import ExecutedInfo, Order, Position, Trade, _notify_order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.core.broker_events import BrokerEventPump

if TYPE_CHECKING:
    from tradelearn.backtest.strategy import Strategy

OrderPayload = tuple[int, str, str, str, float, float | None, float | None]
_MISSING = object()
_EMPTY_ORDER_BUFFER: tuple[()] = ()
CompactFillBatch = tuple[list[int], list[float], list[float], list[float], list[float], list[float]]
_ORDER_TYPE_TO_RUST = {
    Order.Market: "market", Order.Close: "market", Order.Limit: "limit",
    Order.Stop: "stop", Order.StopLimit: "stop_limit",
    Order.StopTrail: "stop_trail", Order.StopTrailLimit: "stop_trail_limit",
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
        # Bracket 子单组注册表：按父单 id 记录该父单下的全部子单（含已激活的），
        # 对齐 backtrader ``bbroker._pchildren``。任一子单成交/失效时通过
        # ``_bracketize(cancel=True)`` 连带撤销同组其余子单，避免孤儿 bracket 子单残留。
        self._pchildren: dict[int, list[Order]] = {}
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
        # 缓冲旁路：按 order.ref 携带 trail 参数，保持 OrderPayload 7 元组契约不变
        self._trail_params_by_ref: dict[int, tuple[float | None, float | None]] = {}
        self._cancel_buffer: list[int] = []
        self._bound_orders: set[int] = set()
        self._control_updates: dict[int, Order] = {}
        self._order_owners: dict[int, Strategy] = {}
        self._order_deadlines: dict[int, int | None] = {}
        self._valid_deadline_by_ref: dict[int, Any] = {}
        self._native_order_events: list[tuple[int, str]] = []
        self._authoritative_terminal_refs: set[int] = set()
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
            trail_amount, trail_percent = self._trail_params_by_ref.pop(
                provisional_ref, (None, None)
            )
            order_id = self._submit_payload_to_engine(
                symbol,
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
                trail_amount,
                trail_percent,
            )
            self.bind_rust_order_ref(provisional_ref, order_id)
            submitted = True
        self._flush_order_controls()
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

        drain_events = getattr(self._engine, "drain_order_events", None)
        if drain_events is not None:
            self.receive_order_events(drain_events())

    def drain_order_buffer(self) -> list[OrderPayload] | tuple[()]:
        """Return buffered order payloads without calling back into Rust.

        缓冲契约保持 7 元组不变（OrderPayload）；跟踪止损的 trail 参数不在元组内，
        由 Rust 侧在消化订单时经 `pop_trail_params(ref)` 旁路取出。
        """
        buffered = self._order_submit_buffer
        self._buffer_order_submissions = False
        if not buffered:
            return _EMPTY_ORDER_BUFFER
        self._order_submit_buffer = []
        return [payload for payload in buffered
                if (order := self._orders_by_ref.get(payload[0])) is not None and order.alive()]

    def pop_trail_params(self, provisional_ref: int) -> tuple[float | None, float | None]:
        """取出并移除某个（临时）订单 ref 的 trail 参数（供 Rust 订单消化时调用）。

        仅在存在 trail 参数时返回非空，使无跟踪订单保持与历史一致的行为。
        """
        return self._trail_params_by_ref.pop(provisional_ref, (None, None))

    def bind_rust_order_ref(self, provisional_ref: int, rust_ref: int) -> None:
        """Replace a provisional Python order ref with the Rust-assigned ref."""
        order = self._orders_by_ref.pop(provisional_ref)
        order.ref = rust_ref
        self._orders_by_ref[order.ref] = order
        self._bound_orders.add(id(order))
        self._control_updates[id(order)] = order
        for other in self._orders if self._oco_order_count else ():
            if other.oco is order and id(other) in self._bound_orders:
                self._control_updates[id(other)] = other

    def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
        """Replace multiple provisional Python order refs with Rust-assigned refs."""
        orders = [(self._orders_by_ref.pop(provisional), native) for provisional, native in bindings]
        for order, native in orders:
            order.ref = native
            self._orders_by_ref[native] = order
            self._bound_orders.add(id(order))
            self._control_updates[id(order)] = order
        for order in self._orders if self._oco_order_count else ():
            if order.oco is not None and id(order) in self._bound_orders:
                self._control_updates[id(order)] = order

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
        trail_amount, trail_percent = self._trail_params_by_ref.pop(
            provisional_ref, (None, None)
        )
        order_id = self._submit_payload_to_engine(
            "data0",
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
            trail_amount,
            trail_percent,
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
        trail_amount: float | None = None,
        trail_percent: float | None = None,
    ) -> int:
        submit_for_symbol = getattr(self._engine, "submit_order_for_symbol", None)
        if callable(submit_for_symbol):
            if trail_amount is None and trail_percent is None:
                # 无跟踪参数时保持历史 6 参签名，避免第三方/测试 FakeEngine 破契约
                return submit_for_symbol(
                    symbol, side_str, ot_str, actual_size, limit_price, stop_price
                )
            return submit_for_symbol(
                symbol,
                side_str,
                ot_str,
                actual_size,
                limit_price,
                stop_price,
                trail_amount,
                trail_percent,
            )
        if trail_amount is None and trail_percent is None:
            return self._engine.submit_order(
                side_str, ot_str, actual_size, limit_price, stop_price
            )
        return self._engine.submit_order(
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
            trail_amount,
            trail_percent,
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
        trail_amount: float | None = None,
        trail_percent: float | None = None,
    ) -> None:
        order_id = self._submit_payload_to_engine(
            symbol,
            side_str,
            ot_str,
            actual_size,
            limit_price,
            stop_price,
            trail_amount,
            trail_percent,
        )
        self.bind_rust_order_ref(order.ref, order_id)
        self._flush_order_controls()

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
        elif order.exectype in (Order.StopTrail, Order.StopTrailLimit):
            # 跟踪止损：初始止损价作为首帧水位基准；回撤量由 trail* 传入
            stop_price = price if price is not None else float(order.data.close[0])
            if order.exectype == Order.StopTrailLimit:
                limit_price = order.pricelimit
        return symbol, side_str, ot_str, actual_size, limit_price, stop_price

    def _trail_params(self, order: Order) -> tuple[float | None, float | None]:
        """提取跟踪止损的 trailamount / trailpercent（仅跟踪类订单需要）。"""
        if order.exectype in (Order.StopTrail, Order.StopTrailLimit):
            return order.trailamount, order.trailpercent
        return None, None

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
            ref=self._order_count + ((1 << 32) if hasattr(self._engine, "configure_order") else 0),
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
            info={**dict(kwargs.get("info", {})), **({"tag": kwargs["tag"]} if "tag" in kwargs else {})},
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
            ref=self._order_count + ((1 << 32) if hasattr(self._engine, "configure_order") else 0),
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
        self._order_owners[id(order)] = owner
        self._order_deadlines[id(order)] = self._normalize_deadline(order)
        order.status = Order.Submitted
        # 回填订单创建时刻的 bar 时间戳（用于回测落库映射「委托日期」，未成交取消单同样可得）
        if getattr(order, "created_ts", None) is None:
            try:
                order.created_ts = self._fill_datetime(order.data)
            except Exception:
                pass
        self._orders.append(order)
        self._orders_by_ref[order.ref] = order
        if order.oco is not None:
            self._oco_order_count += 1
        self._notify_order_event(owner, order)
        # Submitted notifications are actionable; do not resurrect an intent
        # canceled before it reached either the native or deferred queue.
        if not order.alive():
            return

        # 对齐 backtrader ``_pchildren``：把子单登记进其父单的子单组，
        # 供任一子单成交/失效时连带撤销同组其余子单（见 ``_bracketize``）。
        if order.parent is not None:
            self._pchildren.setdefault(id(order.parent), []).append(order)

        # 父单未成交时，bracket 子单只是被登记并挂起，**尚未递交市场**。
        if order.parent is not None and order.parent.status != Order.Completed:
            if not order.parent.alive():
                self.cancel(order)
                return
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
                # 缓冲契约保持 7 元组不变；trail 参数经 _trail_params_by_ref 旁路携带
                self._trail_params_by_ref[order.ref] = self._trail_params(order)
                self._order_submit_buffer.append((order.ref, *payload))
            else:
                trail_amount, trail_percent = self._trail_params(order)
                self._submit_to_rust_engine(order, *payload, trail_amount, trail_percent)
            order.status = Order.Accepted
        else:
            order.status = Order.Accepted
            self._pending_orders.append(order)

    def cancel(self, order: Order) -> None:
        """Cancel an intent, including children that have not reached the kernel."""
        if order.ref in self._authoritative_terminal_refs:
            return
        if not self._cancel_order_mirror(order):
            return
        if id(order) in self._bound_orders:
            self._cancel_buffer.append(order.ref)
        for child, *_ in self._deferred_child_orders.pop(id(order), ()):
            self.cancel(child)

    def _cancel_order_mirror(self, order: Order, owner: Strategy | None = None) -> bool:
        if not order.alive():
            return False
        self._finish_order(order, Order.Canceled, owner)
        return True

    def _finish_order(self, order: Order, status: int, owner: Strategy | None = None) -> None:
        if not order.alive():
            return
        order.status = status
        self._pending_orders = [pending for pending in self._pending_orders if pending is not order]
        # 注意：不能用 ``owner or ...`` —— Strategy 定义了 ``__len__``，对其做布尔
        # 判断会触发 ``len(owner)``（在无数据/哑数据的测试适配器上会抛错），
        # 必须显式判 None。
        if owner is None:
            owner = self._order_owners.get(id(order))
        if owner is not None:
            pending = owner._pending_size.get(order.data, 0.0)
            size = order.size if order.isbuy() else -order.size
            owner._pending_size[order.data] = pending - size
            self._notify_order_event(owner, order)

    def receive_order_events(self, events: list[tuple[int, str]]) -> None:
        self._native_order_events.extend(events)

    def _cancel_oco_siblings(self, owner: Strategy, completed: Order) -> None:
        """Cancel live OCO siblings after one order in the OCO pair fills.

        被 OCO 连带撤销的兄弟单若本身是 bracket 子单，还需级联撤销其 bracket 同组
        子单（如 ref2 止损子单被 OCO 撤下后，同组的 ref3 止盈子单也应一并撤销），
        避免孤儿 bracket 子单残留。
        """
        if completed.oco is None and self._oco_order_count == 0:
            return
        for candidate in list(self._orders):
            if candidate is completed or not candidate.alive():
                continue
            if completed.oco is candidate or candidate.oco is completed:
                self._cancel_order_mirror(candidate, owner)
                if self._engine is not None:
                    self._cancel_buffer.append(candidate.ref)
                # 级联：被 OCO 撤下的候选单若属于某 bracket 组，连带撤销其同组兄弟
                self._bracketize(owner, candidate, cancel=True)

    def _bracketize(self, owner: Strategy, order: Order, cancel: bool = False) -> None:
        """Bracket 父子链连带撤销，对齐 backtrader ``bbroker._bracketize``。

        - ``cancel=True``：订单（子单成交/失效/被 OCO 撤下、或父单被撤）离场时，
          连带撤销其同组兄弟子单，杜绝孤儿挂单残留；
        - ``cancel=False``：父单成交时清理其子单组里的父单登记（子单激活由
          ``_activate_child_orders`` 负责），组内仅保留子单以便后续连带撤销。

        组以父单 ``id(parent)`` 为键。子单成交时通过 ``order.parent`` 反查所在组，
        撤销同组其余存活子单。
        """
        parent = getattr(order, "parent", None)
        if parent is None:
            # 本单无父单（顶层/父单自身）：
            #  - cancel=True（父单离场，如 Expired/Canceled）→ 连带撤销其所有子单；
            #  - cancel=False（父单成交）→ 仅解除父单在组中的登记，子单交由
            #    _activate_child_orders 激活。
            group = self._pchildren.get(id(order))
            if group is not None:
                if cancel:
                    for child in list(group):
                        if child is order or not child.alive():
                            continue
                        if self._cancel_order_mirror(child, owner) and self._engine is not None:
                            self._cancel_buffer.append(child.ref)
                    self._pchildren.pop(id(order), None)
                else:
                    remaining = [o for o in group if o is not order]
                    if remaining:
                        self._pchildren[id(order)] = remaining
                    else:
                        self._pchildren.pop(id(order), None)
            return

        group_key = id(parent)
        group = self._pchildren.get(group_key, [])
        if cancel:
            # 子单离场 → 撤销同组其余存活子单（父单被撤时同样连带所有子单）
            for sibling in list(group):
                if sibling is order or not sibling.alive() or sibling is parent:
                    continue
                if self._cancel_order_mirror(sibling, owner) and self._engine is not None:
                    self._cancel_buffer.append(sibling.ref)
        # 组内除本单外已无存活子单 → 清理登记，避免字典无限增长
        if not any(o.alive() for o in group if o is not order and o is not parent):
            self._pchildren.pop(group_key, None)
        self._deferred_child_orders.pop(group_key, None)

    def _order_created_datetime(self, order: Order) -> Any:
        """订单创建时刻的时间基准（相对 order.valid 以此为起算点）。

        主运行时时钟来自策略主数据（owner.data），而非最后可用的次数据 bar；
        优先复用已回填的 ``order.created_ts``，缺失时按 owner/order 数据取当前 bar 时间戳。
        """
        created_ts = getattr(order, "created_ts", None)
        if created_ts is not None:
            return created_ts
        owner = self._order_owners.get(id(order))
        data = getattr(owner, "data", None) or order.data
        if data is None:
            return None
        try:
            return self._fill_datetime(data)
        except Exception:
            return None

    def _order_valid_deadline(self, order: Order) -> Any:
        """把 order.valid 解析为该订单的绝对失效时间（datetime）；无有效值返回 None。

        支持的类型（与 backtrader 语义对齐）：
          - timedelta：自「创建 bar 时间」起相对 N 天/秒；
          - datetime/date：绝对失效时间；
          - 数字：视为相对秒数（向后兼容）。

        相对 valid 以「订单创建时刻」为基准，且只算一次并缓存，否则每 bar 用当前
        时间重算会让 deadline 永远后移、永不触发。

        关键：缓存键必须用对象身份 id(order) 而非 order.ref。
        order.ref 会被复用（Python provisional ref 与 Rust 分配的 ref 空间不一致，
        撤销/未递交的订单消耗 provisional 号却不占 Rust 号），若按 ref 缓存，
        新订单会命中旧订单遗留的 deadline，导致订单被错误地提前 Expired，
        进而使真实成交无法同步（Rust 引擎已成交、Python 镜像停在 Expired），
        最终持仓/交易/订单三表数据相互矛盾。
        """
        valid = getattr(order, "valid", None)
        if valid is None:
            return None
        cache_key = id(order)
        cached = self._valid_deadline_by_ref.get(cache_key, _MISSING)
        if cached is not _MISSING:
            return cached
        base = self._order_created_datetime(order)
        if isinstance(base, (int, float)):
            try:
                base = datetime.datetime.fromtimestamp(int(base), tz=datetime.timezone.utc)
            except (OverflowError, OSError, ValueError):
                base = None
        try:
            if isinstance(valid, datetime.timedelta):
                if base is None:
                    return None
                deadline = base + valid
                self._valid_deadline_by_ref[cache_key] = deadline
                return deadline
            if isinstance(valid, datetime.datetime):
                self._valid_deadline_by_ref[cache_key] = valid
                return valid
            if isinstance(valid, datetime.date):
                deadline = datetime.datetime(
                    valid.year,
                    valid.month,
                    valid.day,
                    tzinfo=getattr(base, "tzinfo", datetime.timezone.utc),
                )
                self._valid_deadline_by_ref[cache_key] = deadline
                return deadline
            if isinstance(valid, (int, float)):
                if base is None:
                    return None
                deadline = base + datetime.timedelta(seconds=float(valid))
                self._valid_deadline_by_ref[cache_key] = deadline
                return deadline
        except Exception:
            return None
        return None

    def _normalize_deadline(self, order: Order) -> int | None:
        """Rust ``configure_order`` 所需的失效时间戳（int epoch 秒）；无有效值返回 None。"""
        deadline = self._order_valid_deadline(order)
        if deadline is None:
            return None
        try:
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=datetime.timezone.utc)
            return int(deadline.timestamp())
        except (AttributeError, OverflowError, OSError, ValueError):
            return None

    def drain_order_controls(self) -> tuple[list[int], list[tuple[int, int | None, int | None]]]:
        updates = []
        for order in self._control_updates.values():
            peer = order.oco
            peer_ref = peer.ref if peer is not None and id(peer) in self._bound_orders else None
            updates.append((order.ref, self._order_deadlines.get(id(order)), peer_ref))
        self._control_updates.clear()
        cancels, self._cancel_buffer = self._cancel_buffer, []
        return cancels, updates

    def _flush_order_controls(self) -> None:
        if self._engine is None:
            return
        cancels, updates = self.drain_order_controls()
        configure = getattr(self._engine, "configure_order", None)
        if configure is not None:
            for update in updates:
                configure(*update)
        cancel = getattr(self._engine, "cancel_order", None)
        if cancel is not None:
            for ref in cancels:
                cancel(ref)

    def _process_order_events(self, strategy: Strategy) -> None:
        events, self._native_order_events = self._native_order_events, []
        for ref, status in events:
            order = self._orders_by_ref.get(ref)
            if order is not None:
                self._finish_order(order, {
                    "expired": Order.Expired, "canceled": Order.Canceled,
                    "margin": Order.Margin, "rejected": Order.Rejected,
                }[status], strategy)
                for child, *_ in self._deferred_child_orders.pop(id(order), ()):
                    self.cancel(child)

    def _activate_child_orders(self, owner: Strategy, parent: Order) -> None:
        """Route deferred bracket child orders once the parent has filled."""
        children = self._deferred_child_orders.pop(id(parent), ())
        for child, is_buy, actual_size, price in children:
            if child.alive():
                if self._engine is not None and hasattr(self._engine, "run_bar_loop"):
                    side_str = "buy" if is_buy else "sell"
                    payload = self._rust_order_payload(child, side_str, actual_size, price)
                    # 缓冲契约保持 7 元组不变；trail 参数旁路携带
                    self._trail_params_by_ref[child.ref] = self._trail_params(child)
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
        self._flush_order_controls()
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

        drain_events = getattr(self._engine, "drain_order_events", None)
        if drain_events is not None:
            self.receive_order_events(drain_events())

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
        """Synchronize a native batch without permitting retroactive cancellation."""
        new_fills = []
        if self._engine is not None:
            if self._step_fills_from_collect is None:
                if self._get_new_fills_compact is not None:
                    new_fills = self._get_new_fills_compact(self._last_fill_idx)
                else:
                    new_fills = self._get_new_fills(self._last_fill_idx)
            else:
                new_fills = self._step_fills_from_collect
                self._step_fills_from_collect = None

        if new_fills is None:
            new_fills = []
        # The kernel has already resolved these orders, even though Python
        # notifications are delivered sequentially. Preserve outer batch guards
        # if a callback causes nested processing.
        previous_terminal_refs = self._authoritative_terminal_refs
        self._authoritative_terminal_refs = previous_terminal_refs | {
            ref for ref, _ in self._native_order_events
        } | {int(fill[0]) for fill in self._iter_rust_fills(new_fills)}
        try:
            self._process_order_events(strategy)
            if self._engine is not None:
                fill_count = self._fill_batch_len(new_fills)
                if fill_count:
                    filled_order_refs = self._process_rust_fills_batch(strategy, new_fills)
                    self._last_fill_idx += fill_count
                else:
                    filled_order_refs = set()
                # Modern kernels report actual rejections. An Accepted market
                # order may be deferred, buffered, or awaiting its symbol's bar.
                if not callable(getattr(self._engine, "drain_order_events", None)):
                    self._reconcile_unfilled_market_orders(strategy, filled_order_refs)
            elif self._pending_orders:
                raise RuntimeError("RustBroker requires a Rust engine for order matching")
        finally:
            self._authoritative_terminal_refs = previous_terminal_refs

    def _iter_rust_fills(self, fills: Any):
        if self._is_compact_fill_batch(fills):
            order_ids, sizes, prices, comms, _slippages, pnls = fills
            return zip(order_ids, sizes, prices, comms, pnls, strict=True)
        return (
            (fill[0], fill[2], fill[3], fill[4], fill[6])
            for fill in fills
        )

    def _process_rust_fills_batch(self, strategy: Strategy, fills: Any) -> set[int]:
        """Synchronize a batch of Rust fills while preserving notification order."""
        self._position_state_cache.clear()
        orders_by_ref_get = self._orders_by_ref.get
        pending_size = strategy._pending_size
        mult = self._mult
        wants_trade_event = self._wants_trade_event(strategy)
        filled_order_refs: set[int] = set()

        for order_id, signed_size, price, comm, pnl in self._iter_rust_fills(fills):
            order = orders_by_ref_get(order_id)
            if order is None:
                continue
            filled_order_refs.add(int(order_id))

            order.status = Order.Completed
            abs_size = abs(signed_size)
            order.executed = ExecutedInfo(
                price=price,
                size=abs_size,
                comm=comm,
                value=abs_size * price * mult,
            )

            data = order.data
            requested_size = float(order.size if order.isbuy() else -order.size)
            pending = pending_size.get(data, 0.0) - requested_size
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
            # Bracket 连带撤销（对齐 backtrader bbroker._bracketize）：
            #  - 子单成交（有 parent）→ cancel=True，撤销同组其余存活子单；
            #  - 父单成交（无 parent）→ cancel=False，仅解除父单在组中的登记，
            #    子单已由 _activate_child_orders 激活，保持独立生命周期。
            self._bracketize(strategy, order, cancel=order.parent is not None)
        return filled_order_refs

    def _reconcile_unfilled_market_orders(
        self,
        strategy: Strategy,
        filled_order_refs: set[int],
    ) -> None:
        """Mirror market orders the Rust matcher dropped because they could not fill."""
        pending_size = strategy._pending_size
        for order in self._orders:
            if order.status != Order.Accepted or order.exectype != Order.Market:
                continue
            if int(order.ref) in filled_order_refs:
                continue
            requested_size = float(order.size if order.isbuy() else -order.size)
            pending = pending_size.get(order.data, 0.0) - requested_size
            pending_size[order.data] = 0.0 if abs(pending) < 1e-9 else pending
            order.status = Order.Margin
            self._notify_order_event(strategy, order)
