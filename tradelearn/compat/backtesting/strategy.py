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
        self.Open = self._open_proxy
        self.High = self._high_proxy
        self.Low = self._low_proxy
        self.Close = self._close_proxy
        self.Volume = self._volume_proxy
        self._extra_line_cache: dict[str, tuple[Any, IndicatorProxy]] = {}

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
        cached = self._extra_line_cache.get(core_name)
        if cached is not None and cached[0] is arr:
            return cached[1]
        line = IndicatorProxy(arr, self._feed)
        self._extra_line_cache[core_name] = (arr, line)
        return line

    def __len__(self) -> int:
        return self._feed._cursor + 1

class PositionProxy:
    """Proxy for strategy.position."""
    def __init__(self, strategy: Strategy):
        self._strategy = strategy
        self._size_getter_broker = None
        self._rust_state_getter = None
        self._size_getter = None

    def _bind_broker_size_getters(self, broker):
        self._size_getter_broker = broker
        self._rust_state_getter = getattr(broker, "_get_rust_state", None)
        self._size_getter = None if self._rust_state_getter is not None else getattr(broker, "get_position_size", None)

    def __bool__(self) -> bool:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        rust_state_getter = self._rust_state_getter
        if rust_state_getter is not None:
            return rust_state_getter()[2] != 0
        size_getter = self._size_getter
        if size_getter is not None:
            return size_getter() != 0
        return self.size != 0

    @property
    def size(self) -> float:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        rust_state_getter = self._rust_state_getter
        if rust_state_getter is not None:
            return rust_state_getter()[2]
        size_getter = self._size_getter
        if size_getter is not None:
            return size_getter()
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
        """Bind backtesting.py compatibility proxies before init()."""
        if self._bt_data is None:
            self._bt_data = BacktestingDataProxy(self.datas[0])
            self.data = self._bt_data

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

    def __iter__(self):
        return iter(np.asarray(self))

    def __getitem__(self, key: int | slice) -> Any:
        cursor = self._feed._cursor
        data = self._data
        if key == -1:
            return data[cursor]
        if key == -2:
            idx = cursor - 1
            if idx < 0:
                raise IndexError("Index out of bounds")
            return data[idx]
        if isinstance(key, int):
            if key < 0:
                # Relative indexing from CURRENT cursor
                idx = cursor + 1 + key
                if idx < 0: raise IndexError("Index out of bounds")
                return data[idx]
            else:
                return data[key]
        elif isinstance(key, slice):
            # Handle slices up to cursor
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else cursor + 1
            if stop > cursor + 1: stop = cursor + 1
            return data[start:stop:key.step]
        return data[key]

    def __len__(self) -> int:
        cursor = self._feed._cursor
        if cursor < 0:
            return len(self._data)
        return cursor + 1
