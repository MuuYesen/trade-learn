from __future__ import annotations

from typing import Any

from tradelearn.backtest.models import Order, Position


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

    def setsizer(self, sizer: Any, name: Any = None) -> Any:
        if name is None:
            self._sizer = sizer
        else:
            self._sizers[name] = sizer
        if self.broker:
            sizer._set(self, self.broker)
        return sizer

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
