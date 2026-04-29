from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.indicator_cache import BatchIndicatorCache
from tradelearn.backtest.models import Order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.compat.backtesting.util import _TA


class BacktestingDataProxy:
    """Proxy for data to support backtesting.py style capitalized attributes and indexing."""

    __slots__ = (
        "_feed",
        "_open_array",
        "_high_array",
        "_low_array",
        "_close_array",
        "_volume_array",
        "_open_proxy",
        "_high_proxy",
        "_low_proxy",
        "_close_proxy",
        "_volume_proxy",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ta",
        "_extra_line_cache",
    )

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
        self.ta = _TA(_ta_frame(data_feed))
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


def _ta_frame(data_feed: Any) -> pd.DataFrame:
    frame = getattr(data_feed, "_frame", None)
    if frame is not None:
        return frame
    return pd.DataFrame(
        {
            "open": data_feed.get_array("open"),
            "high": data_feed.get_array("high"),
            "low": data_feed.get_array("low"),
            "close": data_feed.get_array("close"),
            "volume": data_feed.get_array("volume"),
        }
    )

class PositionProxy:
    """Proxy for strategy.position."""

    __slots__ = ("_strategy", "_size_getter_broker", "_size_getter")

    def __init__(self, strategy: Strategy):
        self._strategy = strategy
        self._size_getter_broker = None
        self._size_getter = None

    def _bind_broker_size_getters(self, broker):
        self._size_getter_broker = broker
        self._size_getter = getattr(
            broker,
            "current_position_size",
            getattr(broker, "get_position_size", None),
        )

    def __bool__(self) -> bool:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        if size_getter is not None:
            return size_getter() != 0
        return self.size != 0

    @property
    def size(self) -> float:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        if size_getter is not None:
            return size_getter()
        return self._strategy.getposition().size

    def close(self):
        # In backtesting.py, position.close() closes the position
        strategy = self._strategy
        data = strategy.datas[0]
        effective_size = self.size + strategy._pending_size.get(data, 0.0)
        if effective_size > 0:
            return strategy.sell(data=data, size=abs(effective_size))
        if effective_size < 0:
            return strategy.buy(data=data, size=abs(effective_size))
        return None

class Strategy(CoreStrategy):
    """Facade for backtesting.py style strategies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bt_data = None
        self._bt_position = PositionProxy(self)
        self._indicator_cache = {}
        self._batch_indicator_cache = None
        self._bt_close_array = None
        self._bt_primary_data = None

    def _setup(self):
        """Bind backtesting.py compatibility proxies before init()."""
        if self._bt_data is None:
            data = self.datas[0]
            self._bt_primary_data = data
            self._bt_data = BacktestingDataProxy(data)
            self.data = self._bt_data
            self._bt_close_array = data.get_array("close")

    @property
    def position(self) -> Any:
        return self._bt_position

    def I(self, func: Callable, *args, **kwargs) -> Any:  # noqa: E743
        """Indicator wrapper."""
        # backtesting.py indicators are computed ONCE on the full data in init()
        # and then they advance.
        cache_key = self._indicator_cache_key(func, args, kwargs)
        cached = self._indicator_cache.get(cache_key)
        if cached is not None:
            return cached
        batch_cache = self._get_batch_indicator_cache()
        indicator_name = getattr(func, "name", None) or getattr(func, "__name__", None)
        if indicator_name is None:
            indicator_name = func.__class__.__name__
        elif indicator_name == "<lambda>":
            indicator_name = f"{indicator_name}:{id(func)}"
        line = batch_cache.precompute(
            indicator_name,
            getattr(func, "compute", func),
            *args,
            **kwargs,
        )
        # We need to return something that, when indexed with [-1], 
        # returns the value at the CURRENT cursor.
        proxy = IndicatorProxy(line._values, self.datas[0])
        self._indicator_cache[cache_key] = proxy
        return proxy

    def _get_batch_indicator_cache(self) -> BatchIndicatorCache:
        if self._batch_indicator_cache is None:
            self._batch_indicator_cache = BatchIndicatorCache(self.datas[0])
        return self._batch_indicator_cache

    def _indicator_cache_key(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        return (
            self._cache_value_key(func),
            tuple(self._cache_value_key(arg) for arg in args),
            tuple((key, self._cache_value_key(value)) for key, value in sorted(kwargs.items())),
        )

    @staticmethod
    def _cache_value_key(value: Any) -> tuple[str, Any]:
        try:
            hash(value)
        except TypeError:
            return ("id", id(value))
        return ("value", value)

    def buy(
        self,
        *,
        data: Any = None,
        size: float = 0.9999,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        # backtesting.py size can be pct (0.0 to 1.0)
        data = data or self._bt_primary_data
        broker = self.broker
        if 0 < size < 1:
            equity = broker.getvalue()
            # Use current price adjusted for commission to match original logic
            price = (
                self._bt_close_array[data._cursor]
                if data is self._bt_primary_data
                else data.get_array("close")[data._cursor]
            )
            comm_ratio = broker.commission_ratio
            adjusted_price = price * (1 + comm_ratio)
            size = int((equity * size) / adjusted_price)
        
        actual_size = float(abs(size))
        pending = self._pending_size
        pending[data] = pending.get(data, 0.0) + actual_size
        submit = getattr(broker, "submit_basic", broker._submit)
        return submit(
            self,
            data,
            Order.Buy,
            actual_size,
            limit or stop,
            Order.Limit if limit else Order.Stop if stop else Order.Market,
        )

    def sell(
        self,
        *,
        data: Any = None,
        size: float = 0.9999,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        data = data or self._bt_primary_data
        broker = self.broker
        if 0 < size < 1:
            equity = broker.getvalue()
            price = (
                self._bt_close_array[data._cursor]
                if data is self._bt_primary_data
                else data.get_array("close")[data._cursor]
            )
            comm_ratio = broker.commission_ratio
            # For short, price is effectively lower due to commission
            adjusted_price = price * (1 - comm_ratio)
            size = int((equity * size) / adjusted_price)
            
        actual_size = float(abs(size))
        pending = self._pending_size
        pending[data] = pending.get(data, 0.0) - actual_size
        submit = getattr(broker, "submit_basic", broker._submit)
        return submit(
            self,
            data,
            Order.Sell,
            actual_size,
            limit or stop,
            Order.Limit if limit else Order.Stop if stop else Order.Market,
        )

class IndicatorProxy:
    """Proxy for indicators and data to support backtesting.py syntax."""

    __slots__ = ("_data", "_feed", "_length")

    def __init__(self, data: np.ndarray, feed: Any):
        # We store as numpy array for speed
        self._data = np.asarray(data)
        self._feed = feed
        self._length = len(self._data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        cursor = self._feed._cursor
        if cursor < 0:
            return self._data
        return self._data[:cursor + 1]

    def __iter__(self):
        return iter(np.asarray(self))

    def current(self) -> Any:
        return self._data[self._feed._cursor]

    def previous(self) -> Any:
        idx = self._feed._cursor - 1
        if idx < 0:
            raise IndexError("Index out of bounds")
        return self._data[idx]

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
                if idx < 0:
                    raise IndexError("Index out of bounds")
                return data[idx]
            else:
                return data[key]
        elif isinstance(key, slice):
            # Handle slices up to cursor
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else cursor + 1
            if stop > cursor + 1:
                stop = cursor + 1
            return data[start:stop:key.step]
        return data[key]

    def __len__(self) -> int:
        cursor = self._feed._cursor
        if cursor < 0:
            return self._length
        return cursor + 1
