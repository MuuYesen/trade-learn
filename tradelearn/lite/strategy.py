from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as pta

from tradelearn.backtest.indicator_cache import BatchIndicatorCache
from tradelearn.backtest.models import Order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.lite.util import _TA


class LiteDataProxy:
    """Tradelearn Lite data view with lower-case OHLCV lines."""

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
        self.ta = _TA(
            _ta_frame(data_feed),
            wrapper=lambda result, name: _wrap_indicator_result(
                result,
                data_feed,
                frame.index,
                name,
            ),
        )
        self._extra_line_cache: dict[str, tuple[Any, IndicatorProxy]] = {}

    def __getattr__(self, name: str) -> Any:
        if name[:1].isupper():
            raise AttributeError(
                f"Column '{name}' is not available in the Tradelearn Lite facade; "
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
            raise ValueError("Ticker must explicitly specified for multi-asset Lite backtests")
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


BacktestingDataProxy = LiteDataProxy


class Signal:
    """Lite signal wrapper with line-like indexing."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def __getitem__(self, index: int) -> float:
        try:
            return float(self._values[index])
        except (IndexError, TypeError, ValueError):
            return 0.0


class PositionProxy:
    """Tradelearn Lite position view returned by ``strategy.position()``."""

    __slots__ = ("_strategy", "_ticker", "_data", "_size_getter_broker", "_size_getter")

    def __init__(self, strategy: Strategy, ticker: str | None = None, data: Any = None):
        self._strategy = strategy
        self._ticker = ticker
        self._data = data
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
        return self._strategy.position(ticker)

    def __bool__(self) -> bool:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        if size_getter is not None and self._data is self._strategy._bt_primary_data:
            return size_getter() != 0
        return self.size != 0

    @property
    def size(self) -> float:
        data = self._data or self._strategy._bt_primary_data
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        if size_getter is not None and data is self._strategy._bt_primary_data:
            return size_getter()
        return self._strategy.getposition(data).size

    def close(self, portion: float = 1.0):
        strategy = self._strategy
        data = self._data or strategy.datas[0]
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
        data = self._data or self._strategy.datas[0]
        pos = self._strategy.getposition(data)
        if pos.size == 0:
            return 0.0
        price = data.get_array("close")[data._cursor]
        return (
            (float(price) - float(pos.price))
            * float(pos.size)
            * getattr(self._strategy.broker, "_mult", 1.0)
        )

    @property
    def pl_pct(self) -> float:
        data = self._data or self._strategy.datas[0]
        pos = self._strategy.getposition(data)
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
    """Tradelearn Lite strategy facade."""

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self) -> str:
            return ".9999"

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        user_next = cls.__dict__.get("next")
        if user_next is None or user_next is Strategy.next:
            return
        if "_next_lite_custom" not in cls.__dict__:
            cls._next_lite_custom = user_next
        cls.next = Strategy.next

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = self._check_params(kwargs)
        self._bt_data = None
        self._bt_position = PositionProxy(self)
        self._bt_positions: dict[str | None, PositionProxy] = {}
        self._bt_data_by_ticker: dict[str, Any] = {}
        self._indicator_cache = {}
        self._batch_indicator_cache = None
        self._bt_close_array = None
        self._bt_primary_data = None
        self._records: dict[str, pd.Series | pd.DataFrame] = {}
        self._storage: dict[str, Any] | None = None
        self._start_on_day = 0
        self._lite_signals: list[tuple[str, Any, str | None]] = []
        self._lite_signal_sentinel: Any = None
        self._lite_signal_state: dict[str | None, int] = {}

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
        """Bind Tradelearn Lite facade views before init()."""
        if self._bt_data is None:
            data = self.datas[0]
            self._bt_primary_data = data
            self._bt_data = LiteDataProxy(data)
            self.data = self._bt_data
            self._bt_close_array = data.get_array("close")
            self._bt_data_by_ticker = {
                str(getattr(feed, "_name", None) or f"data{i}"): feed
                for i, feed in enumerate(self.datas)
            }
            for i, feed in enumerate(self.datas):
                setattr(self, f"data{i}", LiteDataProxy(feed))
        self._storage = getattr(self.broker, "_storage", None)

    def position(self, ticker: str | None = None) -> Any:
        """Return the Tradelearn Lite position view for a ticker."""
        if ticker is None:
            return self._bt_position
        data = self._resolve_ticker_data(ticker)
        proxy = self._bt_positions.get(ticker)
        if proxy is None or proxy._data is not data:
            proxy = PositionProxy(self, ticker=ticker, data=data)
            self._bt_positions[ticker] = proxy
        return proxy

    def signal(self, line: Any, *, kind: str = "long", ticker: str | None = None) -> Any:
        """Register a Lite signal line as strategy syntax sugar."""
        kind = kind.lower()
        if kind not in {"long", "short", "longshort"}:
            raise ValueError("kind must be one of 'long', 'short', or 'longshort'")
        self._lite_signals.append((kind, line, ticker))
        return line

    def next(self) -> None:
        self._process_lite_signals()
        self._next_lite_custom()

    def _next_lite_custom(self) -> None:
        pass

    def _signal_value(self, line: Any) -> float:
        try:
            return float(line[0])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _process_lite_signals(self) -> None:
        if not self._lite_signals:
            return
        for kind, line, ticker in self._lite_signals:
            value = self._signal_value(line)
            state = self._lite_signal_state.get(ticker, 0)
            if kind == "long":
                if value > 0.0 and state <= 0:
                    self.buy(ticker=ticker, size=1)
                    self._lite_signal_state[ticker] = 1
            elif kind == "short":
                if value < 0.0 and state >= 0:
                    self.sell(ticker=ticker, size=1)
                    self._lite_signal_state[ticker] = -1
            elif value > 0.0 and state <= 0:
                self.buy(ticker=ticker, size=1)
                self._lite_signal_state[ticker] = 1
            elif value < 0.0 and state >= 0:
                self.sell(ticker=ticker, size=1)
                self._lite_signal_state[ticker] = -1

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
        """Declare a Tradelearn Lite gradually revealed indicator."""
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
            if isinstance(funcval, pd.DataFrame):
                if not funcval.index.equals(self.data.index):
                    raise ValueError(
                        "Indicators of pd.DataFrame or pd.Series must have the same index as `data`"
                    )
                proxy = IndicatorBundle(funcval, self.datas[0], indicator_name)
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
        if sl is not None or tp is not None:
            return self.buy_bracket(ticker=ticker, size=size, sl=sl, tp=tp, tag=tag)
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
        **order_kwargs: Any,
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
        if tag is None and not order_kwargs:
            submit = getattr(broker, "submit_basic", broker._submit)
            return submit(self, data, side, actual_size, limit or stop, exectype)
        info = dict(order_kwargs.pop("info", {}))
        if tag is not None:
            info["tag"] = tag
        return broker._submit(
            self,
            data,
            side,
            actual_size,
            limit or stop,
            exectype,
            info=info,
            **order_kwargs,
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
        if sl is not None or tp is not None:
            return self.sell_bracket(ticker=ticker, size=size, sl=sl, tp=tp, tag=tag)
        data = self._resolve_ticker_data(ticker)
        return self._submit_1x_order(Order.Sell, data, size, limit, stop, tag)

    def cancel(self, order: Any) -> None:
        return self.broker.cancel(order)

    def order_target_size(self, *, ticker: str = None, target: float = 0, **kwargs: Any):
        data = self._resolve_ticker_data(ticker)
        current = self.getposition(data).size + self._pending_size.get(data, 0.0)
        delta = float(target) - float(current)
        if delta > 0:
            return self.buy(ticker=ticker, size=delta, **kwargs)
        if delta < 0:
            return self.sell(ticker=ticker, size=abs(delta), **kwargs)
        return None

    def order_target_value(
        self,
        *,
        ticker: str = None,
        target: float = 0.0,
        price: float | None = None,
        **kwargs: Any,
    ):
        data = self._resolve_ticker_data(ticker)
        current = self.getposition(data).size + self._pending_size.get(data, 0.0)
        price = float(price if price is not None else data.get_array("close")[data._cursor])
        mult = getattr(self.broker, "_mult", 1.0)
        current_value = float(current) * price * mult
        delta_value = float(target) - current_value
        if abs(delta_value) < 1e-12:
            return None
        size = int(abs(delta_value) / (price * mult))
        if not size:
            return None
        if delta_value > 0:
            return self.buy(ticker=ticker, size=size, **kwargs)
        return self.sell(ticker=ticker, size=size, **kwargs)

    def order_target_percent(self, *, ticker: str = None, target: float = 0.0, **kwargs: Any):
        return self.order_target_value(
            ticker=ticker,
            target=float(target) * float(self.broker.getvalue()),
            **kwargs,
        )

    def buy_bracket(
        self,
        *,
        ticker: str = None,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ) -> list[Any]:
        data = self._resolve_ticker_data(ticker)
        main = self._submit_1x_order(Order.Buy, data, size, limit, stop, tag, transmit=False)
        stop_order = None
        if sl is not None:
            stop_order = self._submit_1x_order(
                Order.Sell,
                data,
                size,
                None,
                sl,
                tag,
                parent=main,
                transmit=False,
            )
        limit_order = None
        if tp is not None:
            limit_order = self._submit_1x_order(
                Order.Sell,
                data,
                size,
                tp,
                None,
                tag,
                parent=main,
                oco=stop_order,
                transmit=True,
            )
        return [order for order in (main, stop_order, limit_order) if order is not None]

    def sell_bracket(
        self,
        *,
        ticker: str = None,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ) -> list[Any]:
        data = self._resolve_ticker_data(ticker)
        main = self._submit_1x_order(Order.Sell, data, size, limit, stop, tag, transmit=False)
        stop_order = None
        if sl is not None:
            stop_order = self._submit_1x_order(
                Order.Buy,
                data,
                size,
                None,
                sl,
                tag,
                parent=main,
                transmit=False,
            )
        limit_order = None
        if tp is not None:
            limit_order = self._submit_1x_order(
                Order.Buy,
                data,
                size,
                tp,
                None,
                tag,
                parent=main,
                oco=stop_order,
                transmit=True,
            )
        return [order for order in (main, stop_order, limit_order) if order is not None]

    def _resolve_ticker_data(self, ticker: str | None = None) -> Any:
        if ticker is None:
            return self._bt_primary_data
        if ticker in self._bt_data_by_ticker:
            return self._bt_data_by_ticker[ticker]
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

    def start_on_bar(self, n: int) -> None:
        """Start ``next`` no earlier than bar index ``n``."""
        self.start_on_day(n)

    @classmethod
    def prepare_data(cls, tickers: list[str], start: str) -> pd.DataFrame | None:
        return None

class IndicatorProxy:
    """Gradually revealed indicator/data line used by the Lite facade."""

    __slots__ = ("_data", "_feed", "_length", "_index", "_name", "attrs", "ta")

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
        self.ta = _LineTA(self)

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
        if isinstance(key, int):
            idx = cursor + int(key)
            if idx < 0 or idx >= len(data):
                raise IndexError("Index out of bounds")
            return data[idx]
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


class IndicatorBundle:
    """Gradually revealed multi-column indicator bundle used by ``data.ta``."""

    __slots__ = ("_lines", "_frame", "attrs")

    def __init__(self, frame: pd.DataFrame, feed: Any, name: str):
        self._frame = frame
        self.attrs = {"name": name}
        self._lines: dict[str, IndicatorProxy] = {}
        for column in frame.columns:
            proxy = IndicatorProxy(
                frame[column].to_numpy(),
                feed,
                index=frame.index,
                name=str(column),
            )
            for alias in _indicator_column_aliases(column):
                self._lines.setdefault(alias, proxy)

    def __getattr__(self, name: str) -> IndicatorProxy:
        try:
            return self._lines[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str | tuple[slice, int]) -> IndicatorProxy:
        if isinstance(key, tuple):
            rows, column = key
            if rows != slice(None):
                raise TypeError("Lite indicator bundles only support full-row column slicing")
            return self._lines[str(self._frame.columns[int(column)]).lower()]
        return self._lines[key]

    def __len__(self) -> int:
        first = next(iter(self._lines.values()))
        return len(first)

    @property
    def df(self) -> pd.DataFrame:
        first = next(iter(self._lines.values()))
        stop = len(first)
        return self._frame.iloc[:stop]


def _wrap_indicator_result(result: Any, feed: Any, index: pd.Index, name: str) -> Any:
    if isinstance(result, pd.Series):
        return _series_to_proxy(result, feed, index, name)
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            return _series_to_proxy(result.iloc[:, 0], feed, index, name)
        _validate_indicator_frame(result, index, name)
        return IndicatorBundle(result, feed, name)
    return _series_to_proxy(pd.Series(result, index=index, name=name), feed, index, name)


def _series_to_proxy(series: pd.Series, feed: Any, index: pd.Index, name: str) -> IndicatorProxy:
    if not series.index.equals(index):
        if len(series) != len(index):
            raise ValueError(
                "Indicators must have the same length as data "
                f'(indicator "{name}" shape: {getattr(series, "shape", "")})'
            )
        series = pd.Series(series.to_numpy(), index=index, name=series.name)
    proxy = IndicatorProxy(
        series.to_numpy(),
        feed,
        index=index,
        name=getattr(series, "name", None) or name,
    )
    proxy.attrs.update({"name": name})
    return proxy


def _validate_indicator_frame(frame: pd.DataFrame, index: pd.Index, name: str) -> None:
    if not frame.index.equals(index):
        if len(frame) != len(index):
            raise ValueError(
                "Indicators must have the same length as data "
                f'(indicator "{name}" shape: {getattr(frame, "shape", "")})'
            )


def _indicator_column_aliases(column: object) -> tuple[str, ...]:
    text = str(column)
    lowered = text.lower()
    base = lowered.split("_", 1)[0]
    aliases = {lowered, base}
    if base == "macds":
        aliases.add("signal")
    elif base == "macdh":
        aliases.update({"hist", "histogram"})
    elif base == "macd":
        aliases.add("macd")
    return tuple(aliases)


def _reject_bracket_args(sl: float | None, tp: float | None) -> None:
    if sl is not None or tp is not None:
        raise NotImplementedError(
            "Lite sl/tp bracket orders are not implemented yet; "
            "manage exits explicitly or use the engine facade."
        )


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


class _LineTA:
    """pandas-ta-classic shortcut accessor bound to one Lite line."""

    def __init__(self, line: IndicatorProxy) -> None:
        self._line = line

    def __getattr__(self, name: str) -> Callable[..., IndicatorProxy]:
        indicator = getattr(pta, name)

        def call(*args: Any, **kwargs: Any) -> IndicatorProxy:
            result = indicator(self._line.df, *args, **kwargs)
            if isinstance(result, pd.DataFrame):
                if result.shape[1] != 1:
                    raise ValueError(
                        f"Line-level ta.{name} returned multiple columns; use data.ta.{name}"
                    )
                result = result.iloc[:, 0]
            values = np.asarray(result, dtype=float)
            proxy = IndicatorProxy(
                values,
                self._line._feed,
                index=self._line._index,
                name=getattr(result, "name", None) or name,
            )
            proxy.attrs.update({"name": name})
            return proxy

        return call
