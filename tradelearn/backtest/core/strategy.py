from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
from tradelearn.backtest.core.models import Order, Position

class Strategy:
    """Minimalist strategy base class for core execution."""
    def __init__(self, *args, **kwargs) -> None:
        self._sizers: Dict[Any, Any] = {}
        self._sizer: Any = None # Primary/default sizer
        self._positions: Dict[Any, Any] = {}
        self._pending_size: Dict[Any, float] = {}
        self.datas: List[Any] = []
        self.data: Any = None
        self.broker: Any = None
        self.analyzers: Dict[str, Any] = {}
        self._indicators: List[Any] = []
        self._manual_min_period = 0

    def start(self): pass
    def prenext(self): pass
    def next(self): pass
    def stop(self): pass

    def notify_order(self, order: Any): pass
    def notify_trade(self, trade: Any): pass
    def notify_cashvalue(self, cash: float, value: float): pass

    @property
    def position(self) -> Position:
        return self.getposition()

    def getposition(self, data: Any = None) -> Position:
        data = data or self.data
        if data not in self._positions:
            self._positions[data] = Position()
        return self._positions[data]

    def setsizer(self, sizer: Any, name: Any = None) -> Any:
        if name is None:
            self._sizer = sizer
        else:
            self._sizers[name] = sizer
        if self.broker:
            sizer._set(self, self.broker)
        return sizer

    def getsizing(self, data: Any = None, isbuy: bool = True) -> float:
        data = data or self.data
        sizer = self._sizers.get(data, self._sizer)
        if sizer is None: return 1.0
        return sizer.getsizing(data, isbuy)

    def buy(self, data: Any = None, size: float | None = None, price: float | None = None, exectype: int | None = None, **kwargs):
        data = data or self.data
        if size is None: size = self.getsizing(data, isbuy=True)
        actual_size = float(abs(size))
        print(f"DEBUG: CoreStrategy.buy size={actual_size}")
        self._pending_size[data] = self._pending_size.get(data, 0.0) + actual_size
        return self.broker._submit(self, data, Order.Buy, actual_size, price, exectype, **kwargs)

    def sell(self, data: Any = None, size: float | None = None, price: float | None = None, exectype: int | None = None, **kwargs):
        data = data or self.data
        if size is None: size = self.getsizing(data, isbuy=False)
        actual_size = float(abs(size))
        print(f"DEBUG: CoreStrategy.sell size={actual_size}")
        self._pending_size[data] = self._pending_size.get(data, 0.0) - actual_size
        return self.broker._submit(self, data, Order.Sell, actual_size, price, exectype, **kwargs)

    def close(self, data: Any = None, size: float | None = None, **kwargs):
        data = data or self.data
        pos = self.getposition(data)
        pending = self._pending_size.get(data, 0.0)
        effective_size = pos.size + pending
        print(f"DEBUG: CoreStrategy.close pos={pos.size}, pending={pending}")
        if effective_size > 0:
            return self.sell(data=data, size=size or abs(effective_size), **kwargs)
        elif effective_size < 0:
            return self.buy(data=data, size=size or abs(effective_size), **kwargs)
        return None

    def _on_fill(self, data: Any, signed_size: float, price: float) -> None:
        self._pending_size[data] = self._pending_size.get(data, 0.0) - signed_size
        if abs(self._pending_size[data]) < 1e-9:
            self._pending_size[data] = 0.0
            
        pos = self.getposition(data)
        old_size = pos.size
        old_price = pos.price
        
        # Avoid double-update if strategy.getposition returns the same object as broker.getposition
        # (which is the case in the Backtrader compatibility facade)
        broker_pos = getattr(self.broker, 'getposition', lambda d: None)(data)
        if pos is not broker_pos:
            pos.update(signed_size, price)
        
        new_size = pos.size
        
        # Simple Trade tracking for notification
        if old_size != 0 and (old_size * new_size <= 0):
            # Trade closed or flipped
            from tradelearn.backtest.models import Trade
            trade = Trade(data=data, size=old_size, price=old_price, status=Trade.Closed)
            # In a real engine we would calculate PnL correctly here.
            # For now we just notify so benchmark can capture it.
            trade.pnl = (price - old_price) * old_size * getattr(self.broker, '_mult', 1.0)
            trade.pnlcomm = trade.pnl # Simplified
            trade.isclosed = True
            self.notify_trade(trade)

    def __len__(self) -> int:
        if self.data is not None:
            return len(self.data)
        return 0
