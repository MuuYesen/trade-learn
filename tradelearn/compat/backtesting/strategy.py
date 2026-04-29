from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.indicator_cache import BatchIndicatorCache
from tradelearn.backtest.models import Order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.compat.backtesting.util import _TA


class BacktestingDataProxy:
    """Tradelearn 1.x data view with gradually revealed lower-case OHLCV lines."""

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
        "open",
        "high",
        "low",
        "close",
        "volume",
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
        frame = _ta_frame(data_feed)
        self._open_proxy = IndicatorProxy(
            self._open_array, data_feed, index=frame.index, name="open"
        )
        self._high_proxy = IndicatorProxy(
            self._high_array, data_feed, index=frame.index, name="high"
        )
        self._low_proxy = IndicatorProxy(self._low_array, data_feed, index=frame.index, name="low")
        self._close_proxy = IndicatorProxy(
            self._close_array, data_feed, index=frame.index, name="close"
        )
        self._volume_proxy = IndicatorProxy(
            self._volume_array, data_feed, index=frame.index, name="volume"
        )
        self.open = self._open_proxy
        self.high = self._high_proxy
        self.low = self._low_proxy
        self.close = self._close_proxy
        self.volume = self._volume_proxy
        self.ta = _TA(_ta_frame(data_feed))
        self._extra_line_cache: dict[str, tuple[Any, IndicatorProxy]] = {}

    def __getattr__(self, name: str) -> Any:
        if name[:1].isupper():
            raise AttributeError(
                f"Column '{name}' is not available in the Tradelearn 1.x facade; "
                f"use '{name.lower()}' instead."
            )
        line = self._line_or_array(name)
        if len(line._data) == 0:
            raise AttributeError(f"Column '{name}' not in data")
        return line

    def _line_or_array(self, core_name: str) -> Any:
        arr = self._feed.get_array(core_name)
        cached = self._extra_line_cache.get(core_name)
        if cached is not None and cached[0] is arr:
            return cached[1]
        line = IndicatorProxy(arr, self._feed, index=_ta_frame(self._feed).index, name=core_name)
        self._extra_line_cache[core_name] = (arr, line)
        return line

    def __len__(self) -> int:
        return self._feed._cursor + 1

    @property
    def df(self) -> pd.DataFrame:
        frame = _ta_frame(self._feed)
        cursor = self._feed._cursor
        if cursor < 0:
            return frame
        return frame.iloc[: cursor + 1]

    @property
    def index(self) -> pd.Index:
        return self.df.index

    @property
    def now(self) -> Any:
        return self.index[-1]

    @property
    def tickers(self) -> list[str]:
        frame = _ta_frame(self._feed)
        if isinstance(frame.columns, pd.MultiIndex):
            return [str(value) for value in frame.columns.get_level_values(0).unique()]
        return [str(getattr(self._feed, "_name", None) or "Asset")]

    @property
    def the_ticker(self) -> str:
        tickers = self.tickers
        if len(tickers) != 1:
            raise ValueError("Ticker must explicitly specified for multi-asset backtesting")
        return tickers[0]

    @property
    def pip(self) -> float:
        close = np.asarray(self._close_array, dtype=float)
        if close.size == 0:
            return 0.01
        decimals = [
            len(str(value).partition(".")[-1].rstrip("0"))
            for value in close
            if np.isfinite(value)
        ]
        if not decimals:
            return 0.01
        return float(10 ** -int(np.median(decimals)))


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
    """Tradelearn 1.x position view returned by ``strategy.position()``."""

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

    def __call__(self, ticker: str | None = None) -> PositionProxy:
        if ticker is not None and ticker != self._strategy.data.the_ticker:
            raise ValueError("multi-asset position lookup is not supported by this facade yet")
        return self

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

    def close(self, portion: float = 1.0):
        strategy = self._strategy
        data = strategy.datas[0]
        effective_size = self.size + strategy._pending_size.get(data, 0.0)
        if effective_size > 0:
            return strategy._submit_1x_order(
                Order.Sell,
                data,
                abs(effective_size) * float(portion),
                None,
                None,
                None,
            )
        if effective_size < 0:
            return strategy._submit_1x_order(
                Order.Buy,
                data,
                abs(effective_size) * float(portion),
                None,
                None,
                None,
            )
        return None

    @property
    def pl(self) -> float:
        pos = self._strategy.getposition(self._strategy.datas[0])
        if pos.size == 0:
            return 0.0
        price = self._strategy._bt_close_array[self._strategy.datas[0]._cursor]
        return (
            (float(price) - float(pos.price))
            * float(pos.size)
            * getattr(self._strategy.broker, "_mult", 1.0)
        )

    @property
    def pl_pct(self) -> float:
        pos = self._strategy.getposition(self._strategy.datas[0])
        if pos.size == 0 or pos.price == 0:
            return 0.0
        return (self.pl / (abs(pos.size) * float(pos.price))) * 100.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

class Strategy(CoreStrategy):
    """Tradelearn 1.x-style strategy facade."""

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self) -> str:
            return ".9999"

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = self._check_params(kwargs)
        self._bt_data = None
        self._bt_position = PositionProxy(self)
        self._indicator_cache = {}
        self._batch_indicator_cache = None
        self._bt_close_array = None
        self._bt_primary_data = None
        self._records: dict[str, pd.Series | pd.DataFrame] = {}
        self._storage: dict[str, Any] | None = None
        self._start_on_day = 0

    def _check_params(self, params: dict[str, Any]) -> dict[str, Any]:
        for key, value in params.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{key}'."
                    " Strategy class should define parameters as class variables before they "
                    "can be optimized or run with."
                )
            setattr(self, key, value)
        return params

    def _setup(self):
        """Bind Tradelearn 1.x facade views before init()."""
        if self._bt_data is None:
            data = self.datas[0]
            self._bt_primary_data = data
            self._bt_data = BacktestingDataProxy(data)
            self.data = self._bt_data
            self._bt_close_array = data.get_array("close")
        self._storage = getattr(self.broker, "_storage", None)

    def position(self, ticker: str | None = None) -> Any:
        """Return the Tradelearn 1.x position view for a ticker."""
        if ticker is not None and ticker != self.data.the_ticker:
            raise ValueError("multi-asset position lookup is not supported by this facade yet")
        return self._bt_position

    def I(  # noqa: E743
        self,
        funcval: Callable | pd.DataFrame | pd.Series | Any,
        *args,
        name: str | None = None,
        plot: bool = True,
        overlay: bool | None = None,
        color: str | None = None,
        scatter: bool = False,
        **kwargs,
    ) -> Any:
        """Declare a Tradelearn 1.x-style gradually revealed indicator."""
        cache_key = self._indicator_cache_key(funcval, args, kwargs)
        cached = self._indicator_cache.get(cache_key)
        if cached is not None:
            return cached

        if callable(funcval):
            batch_cache = self._get_batch_indicator_cache()
            indicator_name = self._indicator_name(funcval, name, args, kwargs)
            line = batch_cache.precompute(
                indicator_name,
                getattr(funcval, "compute", funcval),
                *args,
                **kwargs,
            )
            values = line._values
        else:
            indicator_name = name or getattr(funcval, "name", None) or funcval.__class__.__name__
            values = self._coerce_indicator_values(funcval, indicator_name)

        proxy = IndicatorProxy(values, self.datas[0], index=self.data.index, name=indicator_name)
        proxy.attrs.update(
            {
                "name": indicator_name,
                "plot": plot,
                "overlay": overlay,
                "color": color,
                "scatter": scatter,
                **kwargs,
            }
        )
        self._indicator_cache[cache_key] = proxy
        return proxy

    def _indicator_name(
        self,
        func: Callable,
        name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        if name is not None:
            return name
        func_name = getattr(func, "name", None) or getattr(func, "__name__", None)
        if func_name is None:
            func_name = func.__class__.__name__
        if func_name == "<lambda>":
            func_name = f"{func_name}:{id(func)}"
        return str(func_name)

    def _coerce_indicator_values(self, value: Any, name: str) -> np.ndarray:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            if not value.index.equals(self.data.index):
                raise ValueError(
                    "Indicators of pd.DataFrame or pd.Series must have the same index as `data`"
                )
            values = value.to_numpy()
        else:
            values = np.asarray(value, order="C")
        if values.ndim == 2 and min(values.shape) == 1:
            values = values.reshape(-1)
        if values.ndim not in (1, 2) or values.shape[0] != len(self.data.index):
            raise ValueError(
                "Indicators must have the same length as `data` "
                f'(data shape: {len(self.data.index)}; '
                f'indicator "{name}" shape: {getattr(values, "shape", "")})'
            )
        if values.ndim == 2:
            raise ValueError(
                "DataFrame-style multi-column indicators are not supported by this facade yet"
            )
        return values

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
        ticker: str = None,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ):
        assert 0 < size < 1 or round(size) == size, (
            "size must be a positive fraction of equity, or a positive whole number of units"
        )
        data = self._resolve_ticker_data(ticker)
        return self._submit_1x_order(Order.Buy, data, size, limit, stop, tag)

    def _submit_1x_order(
        self,
        side: int,
        data: Any,
        size: float,
        limit: float | None,
        stop: float | None,
        tag: object | None,
    ) -> Any:
        broker = self.broker
        if 0 < size < 1:
            equity = broker.getvalue()
            price = (
                self._bt_close_array[data._cursor]
                if data is self._bt_primary_data
                else data.get_array("close")[data._cursor]
            )
            comm_ratio = broker.commission_ratio
            adjusted_price = price * (1 + comm_ratio if side == Order.Buy else 1 - comm_ratio)
            size = int((equity * size) / adjusted_price)

        actual_size = float(abs(size))
        pending = self._pending_size
        pending_delta = actual_size if side == Order.Buy else -actual_size
        pending[data] = pending.get(data, 0.0) + pending_delta
        exectype = Order.Limit if limit else Order.Stop if stop else Order.Market
        if tag is None:
            submit = getattr(broker, "submit_basic", broker._submit)
            return submit(self, data, side, actual_size, limit or stop, exectype)
        return broker._submit(
            self, data, side, actual_size, limit or stop, exectype, info={"tag": tag}
        )

    def sell(
        self,
        *,
        ticker: str = None,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ):
        assert 0 < size < 1 or round(size) == size, (
            "size must be a positive fraction of equity, or a positive whole number of units"
        )
        data = self._resolve_ticker_data(ticker)
        return self._submit_1x_order(Order.Sell, data, size, limit, stop, tag)

    def _resolve_ticker_data(self, ticker: str | None = None) -> Any:
        if ticker is None:
            return self._bt_primary_data
        for data in self.datas:
            if getattr(data, "_name", None) == ticker:
                return data
        if self.data is not None and ticker == self.data.the_ticker:
            return self._bt_primary_data
        raise ValueError(f"Unknown ticker {ticker!r}")

    def record(
        self,
        name: str = None,
        plot: bool = True,
        overlay: bool = None,
        color: str = None,
        scatter: bool = False,
        **kwargs,
    ) -> None:
        index = self.data.index
        row = max(0, len(self.data) - 1)
        for key, value in kwargs.items():
            if isinstance(value, (dict, pd.Series)):
                values = dict(value)
                frame = self._records.get(key)
                if not isinstance(frame, pd.DataFrame):
                    frame = pd.DataFrame(
                        index=_ta_frame(self._bt_primary_data).index,
                        columns=values.keys(),
                    )
                    self._records[key] = frame
                frame.loc[index[-1], list(values.keys())] = list(values.values())
                frame.name = name or key
                frame.attrs.update(
                    {
                        "name": name or key,
                        "plot": plot,
                        "overlay": overlay,
                        "color": color,
                        "scatter": scatter,
                    }
                )
            else:
                series = self._records.get(key)
                if not isinstance(series, pd.Series):
                    series = pd.Series(
                        index=_ta_frame(self._bt_primary_data).index,
                        dtype="float64",
                    )
                    self._records[key] = series
                series.iloc[row] = value
                series.name = name or key
                series.attrs.update(
                    {
                        "name": name or key,
                        "plot": plot,
                        "overlay": overlay,
                        "color": color,
                        "scatter": scatter,
                    }
                )

    @property
    def equity(self) -> float:
        return float(self.broker.getvalue())

    @property
    def storage(self) -> dict[str, Any] | None:
        return self._storage

    @property
    def orders(self) -> tuple[Any, ...]:
        return tuple(getattr(self.broker, "_orders", ()))

    def trades(self, ticker: str = None) -> tuple[Any, ...]:
        return tuple(getattr(self, "_trades", ()))

    @property
    def closed_trades(self) -> tuple[Any, ...]:
        return tuple(getattr(self, "_closed_trades", ()))

    @property
    def alloc(self) -> Any:
        if not hasattr(self, "_alloc"):
            self._alloc = Allocation(self.data.tickers)
        return self._alloc

    def rebalance(
        self,
        force: bool = False,
        rtol: float = 0.01,
        atol: int = 0,
        cash_reserve: float = 0.1,
    ):
        raise NotImplementedError(
            "Allocation.rebalance is reserved for the multi-asset facade path"
        )

    def start_on_day(self, n: int) -> None:
        assert 0 <= n < len(self.data.index), f"day must be within [0, {len(self.data.index)-1}]"
        self._start_on_day = int(n)
        self.addminperiod(int(n) + 1)

    @classmethod
    def prepare_data(cls, tickers: list[str], start: str) -> pd.DataFrame | None:
        return None

class IndicatorProxy:
    """Gradually revealed indicator/data line used by the 1.x facade."""

    __slots__ = ("_data", "_feed", "_length", "_index", "_name", "attrs")

    def __init__(
        self,
        data: np.ndarray,
        feed: Any,
        index: pd.Index | None = None,
        name: str | None = None,
    ):
        # We store as numpy array for speed
        self._data = np.asarray(data)
        self._feed = feed
        self._length = len(self._data)
        self._index = index
        self._name = name
        self.attrs: dict[str, Any] = {}

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

    @property
    def df(self) -> pd.Series:
        cursor = self._feed._cursor
        stop = self._length if cursor < 0 else cursor + 1
        index = self._index
        if index is not None:
            index = index[:stop]
        series = pd.Series(self._data[:stop], index=index, name=self._name)
        series.attrs.update(self.attrs)
        return series


class Allocation:
    """Minimal 1.x allocation object placeholder for strategy code that configures weights."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = list(tickers)
        self.weights = pd.Series(0.0, index=self.tickers, dtype="float64")

    @property
    def unallocated(self) -> float:
        return float(1.0 - self.weights.sum())

    def assume_zero(self) -> Allocation:
        self.weights.loc[:] = 0.0
        return self

    def assume_previous(self) -> Allocation:
        return self
