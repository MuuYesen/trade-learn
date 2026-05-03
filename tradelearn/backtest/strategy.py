from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tradelearn.backtest.models import Order, Position
from tradelearn.backtest.targets import TargetWeightSnapshot, build_target_weight_intents


class Strategy:
    """Minimalist strategy base class for core execution."""

    def __init__(self, *args, **kwargs) -> None:
        if not hasattr(self, "_sizers"):
            self._sizers = {}
        if not hasattr(self, "_sizer"):
            self._sizer = None
        if not hasattr(self, "_positions"):
            self._positions = {}
        if not hasattr(self, "_pending_size"):
            self._pending_size = {}
        if not hasattr(self, "datas"):
            self.datas = []
        if not hasattr(self, "data"):
            self.data = None
        if not hasattr(self, "broker"):
            self.broker = None
        if not hasattr(self, "analyzers"):
            self.analyzers = {}
        if not hasattr(self, "_indicators"):
            self._indicators = []
        if not hasattr(self, "_manual_min_period"):
            self._manual_min_period = 0
        if not hasattr(self, "_research_result"):
            self._research_result = None

    @property
    def research_result(self) -> Any:
        """Return the current strategy-bound research result."""
        return getattr(self, "_research_result", None)

    @research_result.setter
    def research_result(self, result: Any) -> None:
        bound = self._bind_research_result(result) if result is not None else None
        self._research_result = bound
        if bound is not None:
            self._append_research_result(bound)

    def _bind_research_result(self, result: Any) -> Any:
        """Adapt research results for strategy-time access.

        The core backtest runtime deliberately does not import the research
        layer. User facades that support ResearchResult binding override this
        hook.
        """
        return result

    def start(self):
        pass

    def init(self):
        pass

    def prenext(self):
        pass

    def next(self):
        pass

    def stop(self):
        pass

    def record_research_result(self, result: Any) -> Any:
        """Expose the research result executed on the current bar to analyzers."""
        self.research_result = result
        return self.research_result

    def _append_research_result(self, result: Any) -> None:
        history = getattr(self, "research_results_history", None)
        if history is None:
            history = []
            self.research_results_history = history
        history.append(result)

    def notify_order(self, order: Any):
        pass

    def notify_trade(self, trade: Any):
        pass

    def notify_cashvalue(self, cash: float, value: float):
        pass

    def _set_bar_advancers(self, advancers: tuple[Any, ...]) -> None:
        """Install the internal per-bar cursor advance plan."""
        self._bar_advancers = advancers

    def _pre_next(self, cursor: int) -> None:
        """Advance data and indicator cursors before broker/strategy callbacks."""
        for advance in getattr(self, "_bar_advancers", ()):
            advance(cursor)

    @property
    def position(self) -> Position:
        return self.getposition()

    def _resolve_data(self, data: Any = None) -> Any:
        resolved = self.data if data is None else data
        if resolved is None:
            raise ValueError("strategy has no data feed bound")
        return resolved

    def _fallback_position(self, data: Any) -> Position:
        if data not in self._positions:
            self._positions[data] = Position()
        return self._positions[data]

    def getposition(self, data: Any = None) -> Position:
        data = self._resolve_data(data)
        if self.broker and hasattr(self.broker, "getposition"):
            return self.broker.getposition(data)
        return self._fallback_position(data)

    def getdatabyname(self, name: str) -> Any:
        for data in self.datas:
            if getattr(data, "_name", None) == name:
                return data
        raise KeyError(f"data feed {name!r} not found")

    def getpositionbyname(self, name: str) -> Position:
        return self.getposition(self.getdatabyname(name))

    def setsizer(self, sizer: Any, name: Any = None) -> Any:
        if name is None:
            self._sizer = sizer
        else:
            self._sizers[name] = sizer
        if self.broker:
            sizer._set(self, self.broker)
        return sizer

    def getsizer(self) -> Any:
        return self._sizer

    def getsizing(self, data: Any = None, isbuy: bool = True) -> float:
        data = self._resolve_data(data)
        sizer = self._sizers.get(data, self._sizer)
        if sizer is None:
            return 1.0
        return sizer.getsizing(data, isbuy)

    def submit_order(
        self,
        side: int,
        data: Any = None,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs,
    ):
        """Submit a shared event-driven order through the bound broker."""
        data = self._resolve_data(data)
        if self.broker is None:
            raise RuntimeError("strategy has no broker bound")
        isbuy = side == Order.Buy
        if size is None:
            size = self.getsizing(data, isbuy=isbuy)
        actual_size = float(abs(size))
        pending_delta = actual_size if isbuy else -actual_size
        self._pending_size[data] = self._pending_size.get(data, 0.0) + pending_delta
        return self.broker._submit(self, data, side, actual_size, price, exectype, **kwargs)

    def buy(
        self,
        data: Any = None,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs,
    ):
        return self.submit_order(Order.Buy, data, size, price, exectype, **kwargs)

    def sell(
        self,
        data: Any = None,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs,
    ):
        return self.submit_order(Order.Sell, data, size, price, exectype, **kwargs)

    def _buy_data(
        self,
        data: Any,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs,
    ):
        return self.buy(data=data, size=size, price=price, exectype=exectype, **kwargs)

    def _sell_data(
        self,
        data: Any,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs,
    ):
        return self.sell(data=data, size=size, price=price, exectype=exectype, **kwargs)

    def close(self, data: Any = None, size: float | None = None, **kwargs):
        data = self._resolve_data(data)
        pos = self.getposition(data)
        pending = self._pending_size.get(data, 0.0)
        effective_size = pos.size + pending
        if effective_size > 0:
            return self.sell(data=data, size=size or abs(effective_size), **kwargs)
        elif effective_size < 0:
            return self.buy(data=data, size=size or abs(effective_size), **kwargs)
        return None

    def cancel(self, order: Any):
        return self.broker.cancel(order)

    def _current_price(self, data: Any) -> float:
        close = getattr(data, "close", None)
        if close is not None:
            return float(close[0])
        return float(data.get_array("close")[data._cursor])

    def _position_mult(self, data: Any) -> float:
        getcommissioninfo = getattr(self.broker, "getcommissioninfo", None)
        if callable(getcommissioninfo):
            comminfo = getcommissioninfo(data)
            return float(getattr(getattr(comminfo, "p", comminfo), "mult", 1.0))
        return float(getattr(self.broker, "_mult", 1.0))

    def order_target_size(self, data: Any = None, target: float = 0, **kwargs):
        data = self._resolve_data(data)
        possize = self.getposition(data).size + self._pending_size.get(data, 0.0)
        delta = float(target) - float(possize)
        if delta > 0:
            return self._buy_data(data=data, size=delta, **kwargs)
        if delta < 0:
            return self._sell_data(data=data, size=abs(delta), **kwargs)
        return None

    def order_target_value(
        self,
        data: Any = None,
        target: float = 0.0,
        price: float | None = None,
        **kwargs,
    ):
        data = self._resolve_data(data)
        possize = self.getposition(data).size + self._pending_size.get(data, 0.0)
        if not target and possize:
            return self.close(data=data, price=price, **kwargs)
        price = float(price if price is not None else self._current_price(data))
        mult = self._position_mult(data)
        current_value = float(self.broker.getvalue(datas=[data]))
        delta = float(target) - current_value
        if abs(delta) < 1e-12:
            return None
        size = int(abs(delta) / (price * mult))
        if not size:
            return None
        if delta > 0:
            return self._buy_data(data=data, size=size, **kwargs)
        return self._sell_data(data=data, size=size, **kwargs)

    def order_target_percent(self, data: Any = None, target: float = 0.0, **kwargs):
        data = self._resolve_data(data)
        return self.order_target_value(
            data=data,
            target=float(target) * float(self.broker.getvalue()),
            **kwargs,
        )

    def target_weights(
        self,
        weights: Mapping[str, float],
        *,
        close_missing: bool = True,
    ) -> list[Any]:
        """Move all data feeds toward symbol target weights."""
        data_by_name = self._target_weight_data_map()
        snapshots = {
            name: TargetWeightSnapshot(
                price=self._current_price(data),
                size=float(self.getposition(data).size + self._pending_size.get(data, 0.0)),
                mult=self._position_mult(data),
            )
            for name, data in data_by_name.items()
        }
        intents = build_target_weight_intents(
            weights,
            data_by_symbol=data_by_name,
            snapshots=snapshots,
            equity=float(self.broker.getvalue()) if self.broker is not None else 0.0,
            close_missing=close_missing,
            unknown_label="symbol(s)",
        )
        orders: list[Any] = []
        for intent in intents:
            if intent.side == "buy":
                order = self._buy_data(data=intent.data, size=intent.qty)
            else:
                order = self._sell_data(data=intent.data, size=intent.qty)
            if order is not None:
                orders.append(order)
        return orders

    def _target_weight_data_map(self) -> dict[str, Any]:
        return {
            str(getattr(data, "_name", None) or f"data{i}"): data
            for i, data in enumerate(self.datas)
        }

    def buy_bracket(
        self,
        data: Any = None,
        size: float | None = None,
        price: float | None = None,
        stopprice: float | None = None,
        limitprice: float | None = None,
        pricelimit: float | None = None,
        exectype: int = Order.Limit,
        stopexec: int = Order.Stop,
        limitexec: int = Order.Limit,
        oargs: dict[str, Any] | None = None,
        stopargs: dict[str, Any] | None = None,
        limitargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Any]:
        data = self._resolve_data(data)
        main = self._buy_data(
            data=data,
            size=size,
            price=price,
            pricelimit=pricelimit,
            exectype=exectype,
            transmit=False,
            **kwargs,
            **dict(oargs or {}),
        )
        stop = self._sell_data(
            data=data,
            size=size,
            price=stopprice,
            exectype=stopexec,
            parent=main,
            transmit=False,
            **dict(stopargs or {}),
        )
        limit = self._sell_data(
            data=data,
            size=size,
            price=limitprice,
            exectype=limitexec,
            parent=main,
            oco=stop,
            transmit=True,
            **dict(limitargs or {}),
        )
        return [main, stop, limit]

    def sell_bracket(
        self,
        data: Any = None,
        size: float | None = None,
        price: float | None = None,
        stopprice: float | None = None,
        limitprice: float | None = None,
        pricelimit: float | None = None,
        exectype: int = Order.Limit,
        stopexec: int = Order.Stop,
        limitexec: int = Order.Limit,
        oargs: dict[str, Any] | None = None,
        stopargs: dict[str, Any] | None = None,
        limitargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Any]:
        data = self._resolve_data(data)
        main = self._sell_data(
            data=data,
            size=size,
            price=price,
            pricelimit=pricelimit,
            exectype=exectype,
            transmit=False,
            **kwargs,
            **dict(oargs or {}),
        )
        stop = self._buy_data(
            data=data,
            size=size,
            price=stopprice,
            exectype=stopexec,
            parent=main,
            transmit=False,
            **dict(stopargs or {}),
        )
        limit = self._buy_data(
            data=data,
            size=size,
            price=limitprice,
            exectype=limitexec,
            parent=main,
            oco=stop,
            transmit=True,
            **dict(limitargs or {}),
        )
        return [main, stop, limit]

    def _on_fill(self, data: Any, signed_size: float, price: float) -> None:
        """Handle fills produced by the Python fallback path.

        Rust-backed runs synchronize fills in ``RustBroker._process_rust_fills_batch``;
        this hook is intentionally not called from that path.
        """
        self._pending_size[data] = self._pending_size.get(data, 0.0) - signed_size
        if abs(self._pending_size[data]) < 1e-9:
            self._pending_size[data] = 0.0

        pos = self.getposition(data)
        old_size = pos.size
        old_price = pos.price

        pos.update(signed_size, price)

        new_size = pos.size

        # Simple Trade tracking for notification
        if old_size == 0 and new_size != 0:
            from tradelearn.backtest.models import Trade

            trade = Trade(data=data, size=new_size, price=price, status=Trade.Open)
            trade.pnl = 0.0
            trade.pnlcomm = 0.0
            trade.isopen = True
            self.notify_trade(trade)
        elif old_size != 0 and (old_size * new_size <= 0):
            # Trade closed or flipped
            from tradelearn.backtest.models import Trade

            trade = Trade(data=data, size=new_size, price=price, status=Trade.Closed)
            trade.pnl = (price - old_price) * old_size * getattr(self.broker, "_mult", 1.0)
            trade.pnlcomm = trade.pnl  # Simplified
            trade.isclosed = True
            self.notify_trade(trade)

    def _register_indicator(self, indicator: Any):
        if indicator not in self._indicators:
            self._indicators.append(indicator)

    def addminperiod(self, minperiod: int) -> None:
        """Extend the strategy warmup period used before calling ``next``."""
        self._manual_min_period = max(int(minperiod), int(self._manual_min_period))

    def __len__(self) -> int:
        if self.data is not None:
            return len(self.data)
        return 0
