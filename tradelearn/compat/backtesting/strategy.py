from __future__ import annotations
from typing import Any, Callable, List, Dict
import numpy as np
import pandas as pd
from tradelearn.backtest.core.strategy import Strategy as CoreStrategy
from tradelearn.backtest.core.models import Order

class BacktestingDataProxy:
    """Proxy for data to support backtesting.py style capitalized attributes and indexing."""
    def __init__(self, data_feed: Any):
        self._feed = data_feed

    def __getattr__(self, name: str) -> Any:
        mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
        core_name = mapping.get(name, name.lower())
        arr = self._feed.get_array(core_name)
        if getattr(self._feed, '_cursor', -1) < 0:
            return arr
        return IndicatorProxy(arr, self._feed)

    def __len__(self) -> int:
        cursor = getattr(self._feed, '_cursor', 0)
        return cursor + 1

class PositionProxy:
    """Proxy for strategy.position."""
    def __init__(self, strategy: Strategy):
        self._strategy = strategy

    def __bool__(self) -> bool:
        return self.size != 0

    @property
    def size(self) -> float:
        return self._strategy.getposition().size

    def close(self):
        # In backtesting.py, position.close() closes the position
        self._strategy.close()

class Strategy(CoreStrategy):
    """Facade for backtesting.py style strategies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bt_data = None
        self._bt_position = PositionProxy(self)
        self._trades = []

    def notify_trade(self, trade: Any):
        if trade.isclosed:
            self._trades.append(trade)

    def _setup(self):
        """Called by engine before init()."""
        self._bt_data = BacktestingDataProxy(self.datas[0])
        if hasattr(self, 'init'):
            self.init()

    @property
    def data(self) -> Any:
        res = self._bt_data if self._bt_data is not None else self.__dict__.get('data')
        return res

    @data.setter
    def data(self, value: Any):
        self.__dict__['data'] = value

    @property
    def position(self) -> Any:
        return self._bt_position

    def I(self, func: Callable, *args, **kwargs) -> Any:
        """Indicator wrapper."""
        # backtesting.py indicators are computed ONCE on the full data in init()
        # and then they advance.
        res = func(*args, **kwargs)
        # We need to return something that, when indexed with [-1], 
        # returns the value at the CURRENT cursor.
        return IndicatorProxy(res, self.datas[0])

    def buy(self, *, data: Any = None, size: float = 0.9999, limit: float = None, stop: float = None, sl: float = None, tp: float = None):
        # backtesting.py size can be pct (0.0 to 1.0)
        data = data or self.datas[0]
        if 0 < size < 1:
            equity = self.broker.getvalue()
            # Use current price
            price = data.get_array('close')[data._cursor]
            size = (equity * size) / price
        
        return super().buy(data=data, size=size, price=limit or stop, 
                           exectype=Order.Limit if limit else Order.Stop if stop else Order.Market)

    def sell(self, *, data: Any = None, size: float = 0.9999, limit: float = None, stop: float = None, sl: float = None, tp: float = None):
        data = data or self.datas[0]
        if 0 < size < 1:
            equity = self.broker.getvalue()
            price = data.get_array('close')[data._cursor]
            size = (equity * size) / price
            
        return super().sell(data=data, size=size, price=limit or stop,
                            exectype=Order.Limit if limit else Order.Stop if stop else Order.Market)

class IndicatorProxy:
    """Wraps an indicator array to support cursor-based indexing."""
    def __init__(self, data: Any, feed: Any):
        self._data = np.array(data)
        self._feed = feed

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        # For pd.Series() and np.array() conversion
        # During init(), cursor might be -1, return full array
        # During next(), return only up to cursor
        cursor = self._feed._cursor
        if cursor < 0: return self._data
        return self._data[:cursor + 1]

    def __getitem__(self, key: int | slice) -> Any:
        cursor = self._feed._cursor
        if isinstance(key, int):
            if key < 0:
                # [-1] means current cursor
                return self._data[cursor + 1 + key]
            return self._data[key]
        elif isinstance(key, slice):
            # Handle slices if needed, but usually it's just [-1] or [-2]
            return self._data[:cursor + 1][key]

    def __len__(self) -> int:
        return self._feed._cursor + 1
