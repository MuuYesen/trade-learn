from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
from tradelearn.backtest.base import LineRoot

if TYPE_CHECKING:
    from tradelearn.backtest.datafeed import DataFeed
    from tradelearn.backtest.base import BaseBroker
    from tradelearn.backtest.sizer import BaseSizer

class Strategy(LineRoot):
    """Base strategy class with backtrader-style lifecycle hooks."""
    params = ()
    min_period = 0

    def _setup(self) -> None:
        from tradelearn.backtest.base import set_current_strategy, _CURRENT_DATA, _CURRENT_DATAS
        set_current_strategy(self)
        self._manual_min_period = getattr(self.__class__, "min_period", 0)
        self._sizers: Dict[Any, Any] = {}
        self._sizer: Any = None # Primary/default sizer
        self._positions: Dict[Any, Any] = {}
        # Track pending (submitted but not yet filled) size per data feed.
        # Mirrors Backtrader's behavior: position reflects pending orders immediately
        self._pending_size: Dict[Any, float] = {}
        self.datas: List[Any] = _CURRENT_DATAS
        self.data: Any = _CURRENT_DATA
        self.broker: Any = None
        self.analyzers: Any = None
        self._indicators: List[Any] = []

    def _register_indicator(self, indicator: Any) -> None:
        if indicator not in self._indicators:
            self._indicators.append(indicator)
            warmup = getattr(indicator, "min_period", 0)
            self.addminperiod(warmup)

    def __init__(self, *args, **kwargs) -> None:
        # Metaclass already handled _base_init and _setup
        pass

    @property
    def position(self) -> Any:
        """Returns the current filled position."""
        return self.getposition()

    def addminperiod(self, period: int) -> None:
        self._manual_min_period = max(self._manual_min_period, period)

    @property
    def _min_period(self) -> int:
        """Returns the aggregate minimum period from all indicators and lines."""
        m = self._manual_min_period
        # Scan registered indicators
        for ind in self._indicators:
            m = max(m, getattr(ind, 'min_period', 0))
        
        # Scan attributes for delayed lines or other series
        # Use __dict__ to avoid infinite recursion with __getattr__
        for attr, val in self.__dict__.items():
            if attr.startswith('_'): continue
            if hasattr(val, 'min_period'):
                m = max(m, val.min_period)
        return m

    def setsizer(self, sizer: Any, name: Any = None) -> Any:
        if name is None:
            self._sizer = sizer
        else:
            # Map name/data to sizer
            self._sizers[name] = sizer
            
        if hasattr(self, 'broker') and self.broker:
            sizer._set(self, self.broker)
        return sizer

    def start(self): pass
    def prenext(self): pass
    def next(self): pass
    def stop(self): pass

    def notify_order(self, order: Any): pass
    def notify_trade(self, trade: Any): pass
    def notify_cashvalue(self, cash: float, value: float): pass
    
    def getsizing(self, data=None, isbuy=True) -> float:
        data = data or self.datas[0]
        # Try specific sizer first, then default sizer
        sizer = self._sizers.get(data, self._sizer)
        if sizer is None: return 1.0
        return sizer.getsizing(data, isbuy)

    def buy(self, data=None, size=None, price=None, exectype=None, **kwargs):
        from tradelearn.backtest.models import Order
        data = data or self.datas[0]
        if size is None: size = self.getsizing(data, isbuy=True)
        actual_size = float(abs(size))
        # Optimistically add to pending so position reflects intent immediately
        self._pending_size[data] = self._pending_size.get(data, 0.0) + actual_size
        return self.broker._submit(self, data, Order.Buy, actual_size, price, exectype, **kwargs)

    def sell(self, data=None, size=None, price=None, exectype=None, **kwargs):
        from tradelearn.backtest.models import Order
        data = data or self.datas[0]
        if size is None: size = self.getsizing(data, isbuy=False)
        actual_size = float(abs(size))
        # Optimistically subtract from pending so position reflects intent immediately
        self._pending_size[data] = self._pending_size.get(data, 0.0) - actual_size
        return self.broker._submit(self, data, Order.Sell, actual_size, price, exectype, **kwargs)

    def order_target_percent(self, data=None, target=0.0, **kwargs):
        data = data or self.datas[0]
        value = self.broker.getvalue()
        target_value = value * target
        price = data.close[0]
        if price <= 0: return None
        
        mult = getattr(self.broker, "get_mult", lambda d: 1.0)(data)
        pos = self.getposition(data)
        current_value = pos.size * price * mult
        needed_value = target_value - current_value
        needed_size = needed_value / (price * mult)
        
        if abs(needed_size) < 1e-6: return None # Avoid tiny orders
        
        if needed_size > 0:
            return self.buy(data=data, size=needed_size, **kwargs)
        elif needed_size < 0:
            return self.sell(data=data, size=abs(needed_size), **kwargs)
        return None

    def _on_fill(self, data: Any, signed_size: float, price: float) -> None:
        """Called by broker when a fill arrives. Updates position and removes pending intent."""
        # Use a small epsilon to avoid floating point issues with zero
        self._pending_size[data] = self._pending_size.get(data, 0.0) - signed_size
        if abs(self._pending_size[data]) < 1e-9:
            self._pending_size[data] = 0.0
            
        pos = self.getposition(data)
        pos.update(signed_size, price)

    def close(self, data=None, size=None, **kwargs):
        """Close the current position for the given data."""
        data = data or self.datas[0]
        pos = self.getposition(data)
        # Account for pending orders to avoid double-closing
        pending = self._pending_size.get(data, 0.0)
        effective_size = pos.size + pending
        
        if effective_size > 0:
            return self.sell(data=data, size=size or abs(effective_size), **kwargs)
        elif effective_size < 0:
            return self.buy(data=data, size=size or abs(effective_size), **kwargs)
        return None

    def getposition(self, data=None) -> Any:
        from tradelearn.backtest.models import Position
        data = data or self.datas[0]
        if data not in self._positions:
            self._positions[data] = Position()
        return self._positions[data]

    def __len__(self) -> int:
        """Returns the length of the primary data feed, indicating the current bar."""
        if hasattr(self, 'data') and self.data is not None:
            return len(self.data)
        return 0

    def __getattr__(self, name: str) -> Any:
        # dataX -> self.datas[X]
        if name.startswith("data") and name[4:].isdigit():
            idx = int(name[4:])
            if idx < len(self.datas): return self.datas[idx]
        
        # data_close -> self.data.close
        if name.startswith("data_"):
            line_name = name[5:]
            d = self.datas[0] if self.datas else None
            if d and hasattr(d, line_name): return getattr(d, line_name)
                
        return super().__getattr__(name)
