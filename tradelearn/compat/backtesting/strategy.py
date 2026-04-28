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
        self._open_array = data_feed.get_array("open")
        self._high_array = data_feed.get_array("high")
        self._low_array = data_feed.get_array("low")
        self._close_array = data_feed.get_array("close")
        self._volume_array = data_feed.get_array("volume")
        self._open_proxy = IndicatorProxy(self._open_array, data_feed)
        self._high_proxy = IndicatorProxy(self._high_array, data_feed)
        self._low_proxy = IndicatorProxy(self._low_array, data_feed)
        self._close_proxy = IndicatorProxy(self._close_array, data_feed)
        self._volume_proxy = IndicatorProxy(self._volume_array, data_feed)

    @property
    def Open(self) -> Any:
        return self._open_array if self._feed._cursor < 0 else self._open_proxy

    @property
    def High(self) -> Any:
        return self._high_array if self._feed._cursor < 0 else self._high_proxy

    @property
    def Low(self) -> Any:
        return self._low_array if self._feed._cursor < 0 else self._low_proxy

    @property
    def Close(self) -> Any:
        return self._close_array if self._feed._cursor < 0 else self._close_proxy

    @property
    def Volume(self) -> Any:
        return self._volume_array if self._feed._cursor < 0 else self._volume_proxy

    def __getattr__(self, name: str) -> Any:
        mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
        core_name = mapping.get(name, name.lower())
        return self._line_or_array(core_name)

    def _line_or_array(self, core_name: str) -> Any:
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
        broker = self._strategy.broker
        if hasattr(broker, "get_position_size"):
            return broker.get_position_size() != 0
        return self.size != 0

    @property
    def size(self) -> float:
        broker = self._strategy.broker
        if hasattr(broker, "get_position_size"):
            return broker.get_position_size()
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
            # Use current price adjusted for commission to match original logic
            price = data.get_array('close')[data._cursor]
            comm_ratio = getattr(self.broker, 'commission_ratio', 0.0)
            adjusted_price = price * (1 + comm_ratio)
            size = int((equity * size) / adjusted_price)
        
        return super().buy(data=data, size=size, price=limit or stop, 
                           exectype=Order.Limit if limit else Order.Stop if stop else Order.Market)

    def sell(self, *, data: Any = None, size: float = 0.9999, limit: float = None, stop: float = None, sl: float = None, tp: float = None):
        data = data or self.datas[0]
        if 0 < size < 1:
            equity = self.broker.getvalue()
            price = data.get_array('close')[data._cursor]
            comm_ratio = getattr(self.broker, 'commission_ratio', 0.0)
            # For short, price is effectively lower due to commission
            adjusted_price = price * (1 - comm_ratio)
            size = int((equity * size) / adjusted_price)
            
        return super().sell(data=data, size=size, price=limit or stop,
                            exectype=Order.Limit if limit else Order.Stop if stop else Order.Market)

class IndicatorProxy:
    """Proxy for indicators and data to support backtesting.py syntax."""
    def __init__(self, data: np.ndarray, feed: Any):
        # We store as numpy array for speed
        self._data = np.asarray(data)
        self._feed = feed
        self._cursor_ptr = None # Cache for feed._cursor reference

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        cursor = self._feed._cursor
        if cursor < 0: return self._data
        return self._data[:cursor + 1]

    def __getitem__(self, key: int | slice) -> Any:
        cursor = self._feed._cursor
        if key == -1:
            return self._data[cursor]
        if key == -2:
            idx = cursor - 1
            if idx < 0:
                raise IndexError("Index out of bounds")
            return self._data[idx]
        if isinstance(key, int):
            if key < 0:
                # Relative indexing from CURRENT cursor
                idx = cursor + 1 + key
                if idx < 0: raise IndexError("Index out of bounds")
                return self._data[idx]
            else:
                return self._data[key]
        elif isinstance(key, slice):
            # Handle slices up to cursor
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else cursor + 1
            if stop > cursor + 1: stop = cursor + 1
            return self._data[start:stop:key.step]
        return self._data[key]

    def __len__(self) -> int:
        return self._feed._cursor + 1
