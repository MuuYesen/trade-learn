"""Rust-backed broker proxy for high-performance backtesting."""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from tradelearn.backtest.models import ExecutedInfo, Order, Position, Trade, _notify_order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.core.broker_events import BrokerEventPump
from tradelearn.core.strategy_events import (
    dispatch_callback, has_observers, invoke_component, observer_mutation_guard,
)

if TYPE_CHECKING:
    from tradelearn.backtest.strategy import Strategy

OrderPayload = tuple[int, str, str, str, float, float | None, float | None]
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

    def getsize(self, price, cash):
        return int(cash // price)


class RustBroker:
    """Proxy for the high-performance Rust backtesting engine."""

    _RUST_MATCH_MODES = {"exact", "smart"}
    _supports_event_clock = True

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
        self._fundmode = False
        self._fundstartval = self._fundval = 100.0
        self._fundshares = cash / self._fundval
        self._fund_epoch = None
        self._valuation_epoch=0
        self._valuation_view_override=None
        self._setcash_lever_override=None
        self._cash_control_buffer = []
        self._pending_cash_requests = {}
        self._next_cash_command = 1
        self._setcash_value_override = None
        self._clock_unit = 's'
        self.commission_ratio = commission
        self._mult = mult
        self.match_mode = match_mode
        self._engine = None  # Initialized in engine.py
        self._step_open_collect = None
        self._step_open_collect_compact = None
        self._step_close = None
        self._step_close_compact = None
        self._step_open_bars_compact = None
        self._step_bars_with_quotes_compact = None
        self._step_close_bars_compact = None
        self._get_new_fills = None
        self._get_new_fills_compact = None
        self._close_prices = None
        self._curr_idx = 0
        self._orders: list[Order] = []
        self._active_orders: dict[int,Order] = {}
        self._orders_by_ref: dict[int, Order] = {}
        self._orders_by_native_ref: dict[int, Order] = {}
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
        self._bt_notifications = False
        self._order_notifications = []
        self._trade_notifications = []
        self._trade_states = {}
        self._realized_trade_pnl = {}
        self._next_trade_ref = 1
        self._notification_cycle = None
        self._order_submit_buffer: list[OrderPayload] = []
        # 缓冲旁路：按 order.ref 携带 trail 参数，保持 OrderPayload 7 元组契约不变
        self._trail_params_by_ref: dict[int, tuple[float | None, float | None]] = {}
        self._cancel_buffer: list[int] = []
        self._bound_orders: set[int] = set()
        self._control_updates: dict[int, Order] = {}
        self._market_eligibility_updates = {}
        self._market_close_updates = {}
        self._submission_checks = {}
        self._target_expiry_updates = set()
        self._order_owners: dict[int, Strategy] = {}
        self._order_deadlines: dict[int, int | None] = {}
        self._native_order_events: list[tuple[int, str]] = []
        self._authoritative_terminal_refs: set[int] = set()
        self._proxy_events: list[Any] = []
        self._trade_on_close = False
        # For 'bt' mode, we maintain state in Python
        self._pos = Position(size=0.0, price=0.0)
        self._positions: dict[Any, Position] = {}
        self._position_views: dict[Any, Position] = {}
        self._active_cash = cash
        self._comminfo: Any = None
        self._comminfo_factory = None
        self._slippage_model: Any = None
        self._commission_model: Any = None
        self._storage: dict[str, Any] | None = None

    @observer_mutation_guard
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

    @observer_mutation_guard
    def set_storage(self, storage: dict[str, Any] | None) -> None:
        """Attach facade-level run storage without exposing broker internals."""
        self._storage = storage

    @property
    def storage(self) -> dict[str, Any] | None:
        return self._storage

    def _uses_rust_matching(self) -> bool:
        return self._engine is not None

    @observer_mutation_guard
    def bind_engine(self, engine: Any) -> None:
        """Bind a Rust engine and cache optional FFI methods once."""
        precision = getattr(engine, '_configure_bt_precision', None)
        if self._bt_notifications and not callable(precision):
            raise RuntimeError('BT input precision requires the matching candidate TL native kernel')
        if callable(precision):
            precision(bool(self._bt_notifications))
        self._engine = engine
        # The one runtime constructs this native account from the same cash
        # configuration; do not add a second FFI account read during binding.
        self._fundval = self._fundstartval
        self._fundshares = self._cash / self._fundval
        self._fund_epoch = None
        self._step_open_collect = getattr(engine, "step_open_collect", None)
        self._step_open_collect_compact = getattr(engine, "step_open_collect_compact", None)
        self._step_close = getattr(engine, "step_close", None)
        self._step_close_compact = getattr(engine, "step_close_compact", None)
        self._step_open_bars_compact = getattr(engine, "step_open_bars_compact", None)
        self._step_bars_with_quotes_compact = getattr(engine, "step_bars_with_quotes_compact", None)
        self._step_bars_with_quote_views_compact = getattr(engine, "step_bars_with_quote_views_compact", None)
        self._step_close_bars_compact = getattr(engine, "step_close_bars_compact", None)
        self._get_new_fills = getattr(engine, "get_new_fills", None)
        self._get_new_fills_compact = getattr(engine, "get_new_fills_compact", None)
        self._clear_state_caches()

    @observer_mutation_guard
    def set_fundmode(self, fundmode, fundstartval=None):
        self._fundmode = fundmode
        if fundstartval is not None:
            self.set_fundstartval(fundstartval)

    def get_fundmode(self):
        return self._fundmode

    fundmode = property(get_fundmode, set_fundmode)

    @observer_mutation_guard
    def set_fundstartval(self, fundstartval):
        self._fundstartval = fundstartval

    def get_fundshares(self):
        return self._fundshares

    fundshares = property(get_fundshares)

    def get_fundvalue(self):
        if self._engine is None:
            return self._fundval
        # Unit metadata is sampled once per actual native bar. Only applied
        # native deposit/withdrawal acknowledgements change the share count.
        if self._rust_state_cache is not None and self._fund_epoch != self._curr_idx:
            self._fundval = RustBroker.getvalue(self) / self._fundshares
            self._fund_epoch = self._curr_idx
        return self._fundval

    fundvalue = property(get_fundvalue)

    @observer_mutation_guard
    def setcash(self, cash: float) -> None:
        cash=float(cash)
        if not math.isfinite(cash):raise ValueError('finite cash required')
        if self._engine is not None:
            self._setcash_lever_override=(self._valuation_epoch,self.getvalue(lever=True))
            self._queue_cash_control('set',cash)
            self._setcash_value_override=(self._curr_idx,cash)
            if self._rust_state_cache is not None:
                i,_,size,price=self._rust_state_cache
                self._rust_state_cache=(i,cash,size,price)
        self._cash = cash
        self._active_cash = cash

    @observer_mutation_guard
    def add_cash(self, cash: float) -> None:
        cash=float(cash)
        if not math.isfinite(cash):raise ValueError('finite cash addition required')
        self._queue_cash_control('add',cash)

    def _queue_cash_control(self,kind,amount):
        if len(self._pending_cash_requests)>=4096:raise ValueError('cash command capacity exceeded')
        if self._engine is not None and not callable(getattr(self._engine,'apply_cash_controls',None)):
            raise RuntimeError('cash mutation requires the matching candidate TL native kernel')
        cid=self._next_cash_command;self._next_cash_command+=1
        self._pending_cash_requests[cid]=(kind,amount)
        self._cash_control_buffer.append((cid,kind,amount))
        if self._engine is not None and not self._buffer_order_submissions:
            self._flush_cash_controls()

    def drain_cash_controls(self):
        commands,self._cash_control_buffer=self._cash_control_buffer,[]
        return commands

    def receive_cash_events(self,events):
        shares=self._fundshares;seen=set()
        for cid,kind,amount,balance in events:
            if (type(cid) is not int or cid in seen or cid<=0
                    or self._pending_cash_requests.get(cid)!=(kind,amount)
                    or kind not in ('set','add') or isinstance(balance,bool)
                    or not math.isfinite(balance)):
                raise ValueError('unknown, repeated or conflicting native cash acknowledgement')
            seen.add(cid)
            if kind=='add':
                if not self._fundval:raise ValueError('native cash flow requires a nonzero prior fund unit')
                shares+=amount/self._fundval
                if not math.isfinite(shares) or shares==0:
                    raise ValueError('native cash flow cannot qualify finite nonzero fund units')
        self._fundshares=shares
        for cid in seen:del self._pending_cash_requests[cid]

    def _flush_cash_controls(self):
        commands=self.drain_cash_controls()
        if commands:
            apply=getattr(self._engine,'apply_cash_controls',None)
            if not callable(apply):raise RuntimeError('cash mutation requires the matching candidate TL native kernel')
            apply(commands)
        drain=getattr(self._engine,'drain_cash_events',None)
        if callable(drain):self.receive_cash_events(drain())

    @observer_mutation_guard
    def set_cash(self, cash: float) -> None:
        self.setcash(cash)

    @observer_mutation_guard
    def set_coc(self, coc):
        self.configure_matching(trade_on_close=bool(coc))

    @observer_mutation_guard
    def setcommission(self, commission=0.0, margin=None, mult=1.0, commtype=None,
                      percabs=True, stocklike=False, interest=0.0,
                      interest_long=False, leverage=1.0, automargin=False, name=None):
        factory = self._comminfo_factory
        if callable(factory):
            if self._engine is not None:
                raise NotImplementedError("commission changes after native initialization are not implemented")
            info = factory(commission=commission, margin=margin, mult=mult,
                           commtype=commtype, percabs=percabs, stocklike=stocklike,
                           interest=interest, interest_long=interest_long,
                           leverage=leverage, automargin=automargin)
            self.addcommissioninfo(info,name=name)
            return
        if (margin not in (None,0) or commtype is not None or not percabs or stocklike
                or interest or interest_long or leverage != 1 or automargin or name is not None):
            raise NotImplementedError("commission economics require a qualified broker profile")
        self._configure_commission(commission=commission, mult=mult)

    def _configure_commission(self, commission=0.0, margin=0.0, mult=1.0):
        """Internal legacy Rust economics; not a BT broker configuration facade."""
        self.commission_ratio = float(commission)
        self._mult = float(mult)
        self._comminfo = None

    @observer_mutation_guard
    def _set_comminfo(self, comminfo: Any) -> None:
        self._comminfo = comminfo
        if hasattr(comminfo, "p") and hasattr(comminfo.p, "mult"):
            self._mult = comminfo.p.mult
        if hasattr(comminfo, "p") and hasattr(comminfo.p, "commission"):
            self.commission_ratio = comminfo.p.commission

    @observer_mutation_guard
    def addcommissioninfo(self, comminfo, name=None):
        if callable(self._comminfo_factory):
            factory = self._comminfo_factory
            custom = any(getattr(type(comminfo), method) is not getattr(factory, method)
                         for method in ("getcommission", "_getcommission", "confirmexec",
                                        "getoperationcost", "getvaluesize", "getvalue",
                                        "profitandloss", "cashadjust"))
            if (custom or self._engine is not None or name is not None or not comminfo.stocklike
                    or comminfo._commtype != comminfo.COMM_PERC or comminfo.p.mult != 1
                    or comminfo.p.leverage != 1 or comminfo.p.interest
                    or comminfo.p.automargin):
                raise NotImplementedError("commission economics are not qualified for this native backtest broker")
        elif name is not None:
            raise NotImplementedError("named commission economics are not implemented")
        self._set_comminfo(comminfo)

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

    def getvalue(self, datas: list[Any] | tuple[Any, ...] | None = None, mkt=False, lever=False) -> float:
        if datas is not None:
            selected=list(datas)
            for data in selected:
                if not any(data is bound for bound in self._active_datas):
                    raise ValueError('valuation data does not belong to this broker')
            if self._engine is None:
                return self._active_cash if len(selected)!=1 else 0.0
            observe=getattr(self._engine,'observe_valuation',None)
            if not callable(observe):raise RuntimeError('subset valuation requires the matching candidate TL native kernel')
            symbols=[self._rust_symbol(data) for data in selected] if selected else None
            # A single-data BT query settles cash but returns before refreshing
            # cached total/market valuation or the fund unit value.
            prior=(self.getvalue(),self.getvalue(mkt=True)) if len(selected)==1 else None
            self._flush_cash_controls()
            cash,value,market=observe(symbols,len(selected)!=1,True)
            self.receive_cash_events(self._engine.drain_cash_events())
            self._clear_state_caches()
            if len(selected)!=1:
                # This is BT's cached valuation view, not a second account.
                self._valuation_view_override=(self._valuation_epoch,value,market)
                self._setcash_value_override=None;self._setcash_lever_override=None
                self._fundval=value/self._fundshares;self._fund_epoch=self._curr_idx
            elif prior is not None:
                self._valuation_view_override=(self._valuation_epoch,*prior)
            return float(value)

        cached=self._valuation_view_override
        view=cached if cached is not None and cached[0]==self._valuation_epoch else None
        if mkt:
            if view is not None:return float(view[2])
            if self._engine is None:return 0.0
            observe=getattr(self._engine,'observe_valuation',None)
            if not callable(observe):raise RuntimeError('market valuation requires the matching candidate TL native kernel')
            return float(observe(None,False,False)[2])
        if lever:
            prior=self._setcash_lever_override
            if prior is not None and prior[0]==self._valuation_epoch:return float(prior[1])
        if view is not None and self._setcash_value_override is None:return float(view[1])

        override=self._setcash_value_override
        if override is not None and override[0]==self._curr_idx:return override[1]
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

    def get_value(self, datas=None, mkt=False, lever=False) -> float:
        return self.getvalue(datas,mkt=mkt,lever=lever)

    def _position_view(self,data,size,price):
        if data is None and self._active_datas:
            data=self._active_datas[0]
        view=self._position_views.setdefault(data,Position())
        view.size,view.price=size,price
        return view

    def getposition(self, data: Any = None) -> Position:
        if self._engine is not None:
            if (
                self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
                and len(self._active_datas) <= 1
            ):
                _, _cash, size, price = self._rust_state_cache
                return self._position_view(data,size,price)
            if (
                self._buffer_order_submissions
                and self._rust_state_cache is not None
                and self._rust_state_cache[0] == self._curr_idx
                and len(self._active_datas) > 1
            ):
                position = self._positions.get(data)
                if position is not None:
                    return self._position_view(data,position.size,position.price)
                return self._position_view(data,0.0,0.0)
            get_position_for_symbol = getattr(self._engine, "get_position_for_symbol", None)
            if callable(get_position_for_symbol):
                symbol = self._rust_symbol(data)
                cached = self._position_state_cache.get(symbol)
                if cached is not None and cached[0] == self._curr_idx:
                    _, size, price = cached
                    return self._position_view(data,size,price)
                size, price = get_position_for_symbol(symbol)
                self._position_state_cache[symbol] = (self._curr_idx, size, price)
                return self._position_view(data,size,price)
            _, _cash, size, price = self._get_rust_state()
            return self._position_view(data,size,price)
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

    @observer_mutation_guard
    @property
    def positions(self):
        for data in self._active_datas:
            self.getposition(data)
        return self._position_views if self._engine is not None else self._positions

    def bind_datas(self, datas: list[Any]) -> None:
        self._active_datas = list(datas)

    def _rust_symbol(self, data: Any = None) -> str:
        if data is None or len(self._active_datas) <= 1:
            return "data0"
        from .data import native_data_symbols
        return next((symbol for candidate, symbol in zip(self._active_datas, native_data_symbols(self._active_datas))
                     if candidate is data), "data0")

    @staticmethod
    def _current_close(data: Any) -> float:
        close = getattr(data, "close", None)
        if close is not None:
            return float(close[0])
        return float(data.get_array("close")[data._cursor])

    @staticmethod
    def _current_bar_vectors(
        datas: list[Any], *, unit='s',
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
        from .data import native_data_symbols
        execution_symbols = native_data_symbols(datas)
        for index, data in enumerate(datas):
            cursor = getattr(data, "_cursor", -1)
            if cursor < 0:
                continue
            symbol = execution_symbols[index]
            symbols.append(symbol)
            from .clock import native_timestamps
            timestamps.append(int(native_timestamps(data, unit)[cursor]))
            opens.append(float(data._open[cursor]))
            highs.append(float(data._high[cursor]))
            lows.append(float(data._low[cursor]))
            closes.append(float(data._close[cursor]))
            volumes.append(float(data._volume[cursor]))
        return symbols, timestamps, opens, highs, lows, closes, volumes

    @observer_mutation_guard
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

    def _event_bar_vectors(self):
        vectors = (self._current_bar_vectors(self._active_datas) if self._clock_unit == 's' else
                   self._current_bar_vectors(self._active_datas, unit=self._clock_unit))
        clock = getattr(self, "_event_clock_timestamp", None)
        if clock is None or not vectors[1] or vectors[1][0] == clock:
            return vectors
        if clock not in vectors[1]:
            return vectors  # Replay's BT view is distinct from original quote timestamps.
        active = vectors[1].index(clock)
        order = [active, *range(active), *range(active + 1, len(vectors[1]))]
        return tuple([[column[i] for i in order] for column in vectors])

    def _step_event_bars(self, *, trade_on_close=False):
        vectors = self._event_bar_vectors()
        quoted = any(hasattr(data, '_execution_quote_values') for data in self._active_datas)
        if not quoted:
            step = self._step_close_bars_compact if trade_on_close else self._step_open_bars_compact
            return step(*vectors)
        if self._step_bars_with_quotes_compact is None:
            raise RuntimeError('resample execution ticks require the matching candidate TL native kernel')
        from .data import native_data_symbols
        by_symbol = dict(zip(native_data_symbols(self._active_datas), self._active_datas))
        quotes = [[], [], [], []]
        for index, symbol in enumerate(vectors[0]):
            data = by_symbol[symbol]
            values = getattr(data, '_execution_quote_values', None)
            row = values[data._cursor] if values is not None else [vectors[field][index] for field in (2, 3, 4, 5)]
            for column, value in zip(quotes, row):
                column.append(float(value))
        views=[]
        for symbol,data in by_symbol.items():
            if getattr(data,'_native_delivery_views',False) and data._cursor>=0 and symbol in vectors[0]:
                # A BT broker turn can revisit a held/withdrawn clone quote
                # after another feed advances. Identity belongs to this unique
                # runtime turn; source row identity remains data._cursor.
                views.append((symbol,self._curr_idx+1,int(data._delivery_date_ranks[data._cursor])))
        if views:
            if self._step_bars_with_quote_views_compact is None:
                raise RuntimeError('configured replay requires the candidate native delivery-view ABI')
            return self._step_bars_with_quote_views_compact(*vectors,*quotes,trade_on_close,views)
        return self._step_bars_with_quotes_compact(*vectors, *quotes, trade_on_close)

    def _fill_datetime(self, data: Any) -> Any:
        timestamps = getattr(data, "_datetime", None)
        if timestamps is None:
            return None
        try:
            cursor = getattr(data, "_cursor", self._curr_idx)
            if cursor < 0 or cursor >= len(timestamps):
                return None
            value = timestamps[cursor]
            if getattr(data,'_native_delivery_views',False):
                import pandas as pd
                return pd.Timestamp(int(data._delivery_datetime[cursor]),unit='ns',tz='UTC')
            if self._clock_unit == 'ns':
                import pandas as pd
                from .clock import native_timestamps
                return pd.Timestamp(int(native_timestamps(data, 'ns')[cursor]), unit='ns', tz='UTC')
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
        if getattr(order.data,'_native_delivery_views',False):
            import pandas as pd
            self._fills[-1]['source_quote_datetime']=pd.Timestamp(int(order.data._clock_timestamps[order.data._cursor]),unit='ns',tz='UTC')

    def _notify_order_event(self, owner: Strategy, order: Order) -> None:
        if order.alive():self._active_orders[order.ref]=order
        else:self._active_orders.pop(order.ref,None)
        if self._bt_notifications:
            self._order_notifications.append((owner, order, order.clone()))
            return
        self._deliver_order_notification(owner, order)

    def _deliver_order_notification(self, owner, order):
        for observer in getattr(owner, 'observers', {}).values():
            callback = getattr(observer, '_notify_order', None)
            if callable(callback):invoke_component(owner, callback, order)
        if (getattr(type(owner), "notify_order", None) is not CoreStrategy.notify_order
                or has_observers(owner,"business.notify_order")):
            _notify_order(owner, order)
        for analyzer in getattr(owner, "analyzers", {}).values():
            callback = getattr(analyzer,"notify_order",None)
            if callable(callback):
                invoke_component(owner,callback,order)
            on_order = getattr(analyzer, "on_order", None)
            if callable(on_order):
                invoke_component(owner,on_order,order)

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
            invoke_component(owner,on_fill,fill)

    def _notify_trade_event(self, owner: Strategy, trade: Trade) -> None:
        if self._bt_notifications:
            from copy import copy
            self._trade_notifications.append((owner, copy(trade)))
            return
        self._deliver_trade_notification(owner, trade)

    def _deliver_trade_notification(self, owner, trade):
        for observer in getattr(owner, 'observers', {}).values():
            callback = getattr(observer, '_notify_trade', None)
            if callable(callback):invoke_component(owner, callback, trade)
        notify_trade = getattr(owner, "notify_trade", None)
        if callable(notify_trade):
            dispatch_callback(owner,"notify_trade",trade)
        for on_trade in self._trade_callbacks(owner):
            invoke_component(owner,on_trade,trade)

    def _dispatch_cycle_notifications(self):
        if not self._bt_notifications or self._notification_cycle == self._curr_idx:
            return
        self._notification_cycle = self._curr_idx
        orders, self._order_notifications = self._order_notifications, []
        trades, self._trade_notifications = self._trade_notifications, []
        # Admission is one broker phase in BT: all submitted intents precede
        # their accepted snapshots, even though the existing native route is
        # called synchronously for each intent. Keep terminal/control event
        # boundaries and native fill order intact; this only projects the
        # queued admission phase, not another admission or matching loop.
        phased = []
        admission = []
        def flush_admission():
            phased.extend(item for item in admission if item[2].status == Order.Submitted)
            phased.extend(item for item in admission if item[2].status == Order.Accepted)
            admission.clear()
        for item in orders:
            if item[2].status in (Order.Submitted, Order.Accepted):
                admission.append(item)
            else:
                flush_admission()
                phased.append(item)
        flush_admission()
        for owner, original, snapshot in phased:
            snapshot.ref = original.ref
            self._deliver_order_notification(owner, snapshot)
        for owner, trade in trades:
            self._deliver_trade_notification(owner, trade)

    @staticmethod
    def _trade_callbacks(owner: Strategy) -> tuple[Any, ...]:
        return tuple(callback for analyzer in getattr(owner,"analyzers",{}).values()
                     for name in ("notify_trade","on_trade")
                     if callable(callback := getattr(analyzer,name,None)))

    @staticmethod
    def _strategy_wants_trade(owner: Strategy) -> bool:
        notify_trade = getattr(type(owner), "notify_trade", None)
        return ((notify_trade is not None and notify_trade is not CoreStrategy.notify_trade)
                or has_observers(owner,"business.notify_trade"))

    def _wants_trade_event(self, owner: Strategy) -> bool:
        return (self._strategy_wants_trade(owner) or bool(self._trade_callbacks(owner))
                or any(getattr(observer, '_observes_trades', False)
                       for observer in getattr(owner, 'observers', {}).values()))

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

    _supports_order_owner = True

    def get_orders_open(self, safe: bool = False, *, owner=None):
        orders = [order for order in self._active_orders.values() if order.alive()
                  and (self._order_owners.get(id(order)) is owner if owner is not None
                       else order.status in (Order.Accepted, Order.Partial))]
        if safe:
            orders = [order.clone() for order in orders]
        return tuple(orders) if owner is not None else orders

    def begin_order_buffering(self) -> None:
        """Delay Rust order submission until callbacks return."""
        self._buffer_order_submissions = True

    def begin_terminal_order_suppression(self) -> None:
        """Return terminal-bar order objects without routing them to the broker."""
        self._terminal_order_suppression = True

    def end_terminal_order_suppression(self) -> None:
        """Restore normal order routing after a terminal-bar strategy callback."""
        self._terminal_order_suppression = False

    @observer_mutation_guard
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
            and not self._bt_notifications  # BT applies COC at the next broker turn.
        ):
            self._clear_state_caches()
            if self._active_datas and self._step_close_bars_compact is not None:
                fills, cash, size, price = self._step_event_bars(trade_on_close=True)
                self._step_fills_from_collect = fills
                self._rust_state_cache = (self._curr_idx, cash, size, price)
            elif self._step_close_compact is not None:
                self._step_fills_from_collect = self._step_close_compact(self._curr_idx)
            elif self._step_close is not None:
                self._step_fills_from_collect = self._step_close(self._curr_idx)

        drain_cash=getattr(self._engine,'drain_cash_events',None)
        if callable(drain_cash):self.receive_cash_events(drain_cash())
        drain_events = getattr(self._engine, "drain_order_events", None)
        if drain_events is not None:
            self.receive_order_events(drain_events())

    @observer_mutation_guard
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
        """Bind native identity without changing the public order reference."""
        order = self._orders_by_ref[provisional_ref]
        order._native_ref = rust_ref
        self._orders_by_native_ref[rust_ref] = order
        self._bound_orders.add(id(order))
        owner=self._order_owners.get(id(order))
        reference=float(order.created.price)
        if self._bt_notifications and getattr(owner,'_bt_minperiods',False) and math.isfinite(reference):
            self._submission_checks[rust_ref]=reference

        self._control_updates[id(order)] = order
        if getattr(order, '_bt_dates', False) and self._order_deadlines.get(id(order)) is not None:
            self._target_expiry_updates.add(rust_ref)
        if (order.exectype == Order.Market and getattr(order, '_bt_dates', False)
                and order.created.dt is not None):
            from .clock import bt_date_upper_tick,bt_date_rank
            self._market_eligibility_updates[rust_ref] = (bt_date_rank(order.created.dt) if getattr(order.data,'_native_delivery_views',False)
                                                        else bt_date_upper_tick(order.created.dt, self._clock_unit))
        if (order.exectype==Order.Market and self._trade_on_close
                and getattr(order,'_bt_dates',False)):
            order._bt_coc=True
            self._market_close_updates[rust_ref]=float(order.created.pclose)
        for other in self._orders if self._oco_order_count else ():
            if other.oco is order and id(other) in self._bound_orders:
                self._control_updates[id(other)] = other

    def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
        for public_ref, native_ref in bindings:
            self.bind_rust_order_ref(public_ref, native_ref)

    def _public_order_ref(self, native_ref):
        order = self._orders_by_native_ref.get(int(native_ref))
        return order.ref if order is not None else int(native_ref)

    @observer_mutation_guard
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
                limit_price = order.created.pricelimit
        return symbol, side_str, ot_str, actual_size, limit_price, stop_price

    def _trail_params(self, order: Order) -> tuple[float | None, float | None]:
        """提取跟踪止损的 trailamount / trailpercent（仅跟踪类订单需要）。"""
        if order.exectype in (Order.StopTrail, Order.StopTrailLimit):
            return order.trailamount, order.trailpercent
        return None, None

    @observer_mutation_guard
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
        if 'post_only' in kwargs:
            if type(kwargs['post_only']) is not bool:
                raise TypeError('post_only must be boolean')
            if kwargs['post_only']:
                raise NotImplementedError('post-only venue policy is not simulated by the OHLCV backend')
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
            info={**dict(kwargs.get("info", {})), **({"tag": kwargs["tag"]} if "tag" in kwargs else {})},
        )
        if self._terminal_order_suppression:
            self._stage_terminal_order(owner, order)
            return order
        self._register_and_route_order(owner, order, is_buy, actual_size, price)

        return order

    @observer_mutation_guard
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
            self._stage_terminal_order(owner, order)
            return order
        self._register_and_route_order(owner, order, is_buy, actual_size, price)

        return order

    @observer_mutation_guard
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

    def _submission_datetime(self, owner, data):
        runtime = getattr(owner, "_runtime", None)
        if runtime is not None and runtime.profile == "batch" and runtime._cursor >= 0:
            timeline = (runtime._clock._deliveries if getattr(runtime._clock,'has_delivery_views',False)
                        else runtime._clock.timestamps)
            value = int(timeline[runtime._cursor])
            if runtime._clock.unit == 'ns':
                import pandas as pd
                return pd.Timestamp(value, unit='ns', tz='UTC')
            return value
        clock = getattr(owner, "data", None)
        return self._fill_datetime(clock if clock is not None else data)

    def _stage_terminal_order(self, owner, order):
        """Keep a submitted intent when no next broker cycle remains."""
        order._initialize_clock(owner, self._submission_datetime(owner, order.data))
        order.status = Order.Submitted
        self._order_owners[id(order)] = owner
        self._order_deadlines[id(order)] = self._normalize_deadline(order)
        self._orders.append(order)
        self._orders_by_ref[order.ref] = order
        self._active_orders[order.ref] = order

    def _register_and_route_order(
        self,
        owner: Strategy,
        order: Order,
        is_buy: bool,
        actual_size: float,
        price: float | None,
    ) -> None:
        order._initialize_clock(owner, self._submission_datetime(owner, order.data))
        self._order_owners[id(order)] = owner
        self._order_deadlines[id(order)] = self._normalize_deadline(order)
        order.status = Order.Submitted
        self._orders.append(order)
        self._orders_by_ref[order.ref] = order
        if order.oco is not None:
            self._oco_order_count += 1
        self._notify_order_event(owner, order)
        # Submitted notifications are actionable; do not resurrect an intent
        # canceled before it reached either the native or deferred queue.
        if not order.alive():
            return

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

    @observer_mutation_guard
    def cancel(self, order: Order) -> None:
        """Cancel an intent, including children that have not reached the kernel."""
        order = self._orders_by_ref.get(order.ref, order)
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
        owner = owner or self._order_owners.get(id(order))
        if owner is not None:
            pending = owner._pending_size.get(order.data, 0.0)
            size = order.size
            owner._pending_size[order.data] = pending - size
            self._notify_order_event(owner, order)

    def _cancel_oco_siblings(self, owner: Strategy, completed: Order) -> None:
        # Native matching enforces exclusion atomically. Mirror peers, including
        # deferred orders; this also supports existing broker test adapters.
        if completed.oco is None and self._oco_order_count == 0:
            return
        group = {id(completed)}
        changed = True
        while changed:
            changed = False
            for order in self._orders:
                if order.oco is not None and (id(order) in group or id(order.oco) in group):
                    before = len(group)
                    group.update((id(order), id(order.oco)))
                    changed |= len(group) != before
        for order in self._orders:
            if order is not completed and id(order) in group and order.alive():
                self.cancel(order)

    def _normalize_deadline(self, order: Order) -> int | None:
        valid = order.valid
        if valid is None:
            return None
        owner = self._order_owners.get(id(order))
        if getattr(owner, "_bt_minperiods", False):
            # The Engine front normalized BT dates/durations to ordinal days.
            # Market never expires, and numeric0 is distinct from timedelta DAY.
            if order.exectype == Order.Market or not valid:
                return None
            from .clock import bt_date_upper_tick,bt_date_rank
            return bt_date_rank(valid) if getattr(order.data,'_native_delivery_views',False) else bt_date_upper_tick(valid, self._clock_unit)
        import numbers

        import pandas as pd
        if isinstance(valid, (datetime.timedelta, numbers.Real)):
            # The primary runtime clock defines submission time, not the last
            # available secondary-feed bar.
            data = getattr(self._order_owners.get(id(order)), "data", None) or order.data
            ts = order.created_ts if order.created_ts is not None else self._fill_datetime(data)
            if ts is None:
                raise ValueError("relative order.valid requires a submission timestamp")
            base = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, numbers.Real) else pd.Timestamp(ts)
            delta = valid if isinstance(valid, datetime.timedelta) else datetime.timedelta(seconds=float(valid))
            deadline = base + delta
        elif isinstance(valid, (datetime.datetime, datetime.date)):
            deadline = pd.Timestamp(valid)
        else:
            raise ValueError("order.valid must be a date, datetime, timedelta or relative seconds")
        deadline = deadline.tz_localize("UTC") if deadline.tzinfo is None else deadline.tz_convert("UTC")
        return int(deadline.value) if self._clock_unit == 'ns' else int(deadline.timestamp())

    def drain_order_controls(self) -> tuple[list[int], list[tuple[int, int | None, int | None]]]:
        updates = []
        for order in self._control_updates.values():
            peer = order.oco
            peer_ref = peer._native_ref if peer is not None and id(peer) in self._bound_orders else None
            updates.append((order._native_ref, self._order_deadlines.get(id(order)), peer_ref))
        self._control_updates.clear()
        cancels, self._cancel_buffer = self._cancel_buffer, []
        return [getattr(self._orders_by_ref[ref], "_native_ref", ref) for ref in cancels], updates

    def drain_target_expiry_clocks(self):
        updates, self._target_expiry_updates = self._target_expiry_updates, set()
        return list(updates)

    def drain_market_eligibility(self):
        updates, self._market_eligibility_updates = self._market_eligibility_updates, {}
        return list(updates.items())

    def drain_submission_checks(self):
        updates,self._submission_checks=self._submission_checks,{}
        return list(updates.items())

    def _flush_order_controls(self) -> None:
        if self._engine is None:
            return
        self._flush_cash_controls()
        checks=self.drain_submission_checks()
        configure_submission=getattr(self._engine,'configure_submission_check',None)
        if callable(configure_submission):
            for ref,price in checks:configure_submission(ref,price)
        cancels, updates = self.drain_order_controls()
        configure = getattr(self._engine, "configure_order", None)
        if configure is not None:
            for update in updates:
                configure(*update)
        expiry_updates = self.drain_target_expiry_clocks()
        configure_expiry_clock = getattr(self._engine, 'configure_order_expiry_clock', None)
        if expiry_updates and configure_expiry_clock is None:
            raise RuntimeError('BT ordered-feed expiry requires the matching candidate TL native kernel')
        for ref in expiry_updates:
            configure_expiry_clock(ref, True)
        market_updates = self.drain_market_eligibility()
        configure_market = getattr(self._engine, 'configure_market_eligibility', None)
        if market_updates and configure_market is None:
            raise RuntimeError('BT market eligibility requires the matching candidate TL native kernel')
        for update in market_updates:
            configure_market(*update)
        close_updates,self._market_close_updates=self._market_close_updates,{}
        configure_close=getattr(self._engine,'configure_market_close_reference',None)
        if close_updates and configure_close is None:
            raise RuntimeError('adjusted BT replay COC requires candidate native close references')
        for ref,price in close_updates.items():configure_close(ref,price)
        cancel = getattr(self._engine, "cancel_order", None)
        if cancel is not None:
            for ref in cancels:
                cancel(ref)

    def receive_order_events(self, events: list[tuple[int, str]]) -> None:
        self._native_order_events.extend((self._public_order_ref(ref), status) for ref, status in events)

    def _process_order_events(self, strategy: Strategy) -> None:
        events, self._native_order_events = self._native_order_events, []
        for ref, status in events:
            order = self._orders_by_ref.get(ref)
            if order is not None:
                if status=='submission_margin':
                    self._order_notifications=[item for item in self._order_notifications
                                               if not (item[1] is order and item[2].status==Order.Accepted)]
                self._finish_order(order, {
                    "expired": Order.Expired, "canceled": Order.Canceled,
                    "margin": Order.Margin, "submission_margin":Order.Margin, "rejected": Order.Rejected,
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

    @observer_mutation_guard
    def step(self, i: int) -> None:
        self._curr_idx = i
        self._clear_state_caches()
        self._step_fills_from_collect = None
        self._flush_order_controls()
        if self._engine is not None:
            if self._active_datas and self._step_open_bars_compact is not None:
                fills, cash, size, price = self._step_event_bars()
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

        drain_cash=getattr(self._engine,'drain_cash_events',None)
        if callable(drain_cash):self.receive_cash_events(drain_cash())
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

    @observer_mutation_guard
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
            self._dispatch_cycle_notifications()
        finally:
            self._authoritative_terminal_refs = previous_terminal_refs

    def _iter_rust_fills(self, fills: Any):
        if self._is_compact_fill_batch(fills):
            order_ids, sizes, prices, comms, _slippages, pnls = fills
            return ((self._public_order_ref(ref), size, price, comm, pnl)
                    for ref, size, price, comm, pnl in zip(order_ids, sizes, prices, comms, pnls, strict=True))
        return (
            (self._public_order_ref(fill[0]), fill[2], fill[3], fill[4], fill[6])
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
                size=signed_size,
                comm=comm,
                value=abs_size * price * mult,
                remsize=0.0,
            )

            order._set_execution_clock(self._fill_datetime(order.data))
            if getattr(order,'_bt_coc',False):order.executed.dt=order.created.dt
            data = order.data
            requested_size = float(order.size)
            pending = pending_size.get(data, 0.0) - requested_size
            pending_size[data] = 0.0 if abs(pending) < 1e-9 else pending

            pos = self._positions.setdefault(data, Position(size=0.0, price=0.0))
            old_size = pos.size
            old_price = pos.price
            pos.update(signed_size, price)
            if data is None:
                self._pos = pos
            new_size = pos.size

            trade_closed = old_size != 0 and old_size * new_size <= 0
            accumulated_pnl = self._realized_trade_pnl.get(data, 0.0) + pnl
            if trade_closed:
                self._closed_trade_count += 1
                if accumulated_pnl > 0:
                    self._winning_trade_count += 1
                self._realized_trade_pnl.pop(data, None)
            elif new_size:
                self._realized_trade_pnl[data] = accumulated_pnl
            if wants_trade_event:
                self._project_fill_trade(strategy, order, old_size, old_price,
                                         signed_size, price, comm, pnl, True)

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
        return filled_order_refs

    def _project_fill_trade(self, owner, order, old_size, old_price,
                            signed_size, price, commission, pnl, notify):
        """Accumulate views of authoritative fills; never update account cash.

        A reversal closes the previous Trade and opens a separate Trade. Fees
        from its one native fill are apportioned by the actual closed/open qty.
        """
        from copy import copy
        data = order.data
        trade = self._trade_states.get(data)
        total = abs(signed_size)
        closing = min(abs(old_size), total) if old_size * signed_size < 0 else 0.0
        opening = total - closing
        direction = 1 if signed_size > 0 else -1
        bar = len(data) if hasattr(data, "__len__") else 0
        info = self.getcommissioninfo(data)
        value = getattr(info, "getvaluesize", lambda size, cost: size * cost)
        close_fee = commission * closing / total if closing else 0.0
        if closing:
            if trade is None:
                raise RuntimeError("native fill closes a position without its trade projection")
            trade.justopened = False
            trade.size = old_size + direction * closing
            trade.commission += close_fee
            trade.pnl += pnl
            trade.pnlcomm = trade.pnl - trade.commission
            trade.value = value(trade.size, trade.price)
            trade.barlen = bar - trade.baropen
            trade.isopen = bool(trade.size)
            trade.isclosed = not trade.isopen
            if trade.isclosed:
                trade.status = Trade.Closed
                trade.dtclose = order.executed.dt
                trade.barclose = bar
                self._trade_states.pop(data)
                if notify:
                    self._notify_trade_event(owner, copy(trade))
                trade = None
        if opening:
            justopened = trade is None
            if justopened:
                trade = Trade(ref=self._next_trade_ref, data=data, status=Trade.Open,
                              isopen=True, dtopen=order.executed.dt,
                              long=direction > 0, baropen=bar)
                self._next_trade_ref += 1
                self._trade_states[data] = trade
            old_trade_size = trade.size
            trade.size += direction * opening
            trade.price = (old_trade_size * trade.price + direction * opening * price) / trade.size
            trade.value = value(trade.size, trade.price)
            trade.commission += commission - close_fee
            trade.pnlcomm = trade.pnl - trade.commission
            trade.justopened = justopened
            trade.barlen = bar - trade.baropen
            if justopened and notify:
                self._notify_trade_event(owner, copy(trade))

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
            requested_size = float(order.size)
            pending = pending_size.get(order.data, 0.0) - requested_size
            pending_size[order.data] = 0.0 if abs(pending) < 1e-9 else pending
            order.status = Order.Margin
            self._notify_order_event(strategy, order)
