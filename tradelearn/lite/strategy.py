from __future__ import annotations

import sys
from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.indicator_cache import BatchIndicatorCache
from tradelearn.backtest.models import Order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.lite.data import LiteDataProxy, _ta_frame
from tradelearn.lite.indicator import IndicatorBundle, IndicatorProxy
from tradelearn.lite.position import PositionProxy


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
        self._storage = getattr(self.broker, "storage", None)

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

    def _buy_data(
        self,
        data: Any,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs: Any,
    ):
        limit = price if exectype == Order.Limit else None
        stop = price if exectype == Order.Stop else None
        tag = kwargs.pop("tag", None)
        info = kwargs.get("info")
        if tag is None and isinstance(info, dict):
            tag = info.get("tag")
        return self._submit_1x_order(Order.Buy, data, size or 1, limit, stop, tag, **kwargs)

    def _sell_data(
        self,
        data: Any,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs: Any,
    ):
        limit = price if exectype == Order.Limit else None
        stop = price if exectype == Order.Stop else None
        tag = kwargs.pop("tag", None)
        info = kwargs.get("info")
        if tag is None and isinstance(info, dict):
            tag = info.get("tag")
        return self._submit_1x_order(Order.Sell, data, size or 1, limit, stop, tag, **kwargs)

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

    def _current_price(self, data: Any) -> float:
        return float(data.get_array("close")[data._cursor])

    def order_target_size(self, *, ticker: str = None, target: float = 0, **kwargs: Any):
        return super().order_target_size(
            data=self._resolve_ticker_data(ticker),
            target=target,
            **kwargs,
        )

    def order_target_value(
        self,
        *,
        ticker: str = None,
        target: float = 0.0,
        price: float | None = None,
        **kwargs: Any,
    ):
        return super().order_target_value(
            data=self._resolve_ticker_data(ticker),
            target=target,
            price=price,
            **kwargs,
        )

    def order_target_percent(self, *, ticker: str = None, target: float = 0.0, **kwargs: Any):
        return super().order_target_value(
            data=self._resolve_ticker_data(ticker),
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
        kwargs = {"info": {"tag": tag}} if tag is not None else {}
        return super().buy_bracket(
            data=self._resolve_ticker_data(ticker),
            size=size,
            price=limit or stop,
            stopprice=sl,
            limitprice=tp,
            exectype=Order.Limit if limit else Order.Stop if stop else Order.Market,
            **kwargs,
        )

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
        kwargs = {"info": {"tag": tag}} if tag is not None else {}
        return super().sell_bracket(
            data=self._resolve_ticker_data(ticker),
            size=size,
            price=limit or stop,
            stopprice=sl,
            limitprice=tp,
            exectype=Order.Limit if limit else Order.Stop if stop else Order.Market,
            **kwargs,
        )

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
            tickers = list(self._bt_data_by_ticker) or self.data.tickers
            self._alloc = Allocation(tickers)
        return self._alloc

    def rebalance(
        self,
        force: bool = False,
        rtol: float = 0.01,
        atol: int = 0,
        cash_reserve: float = 0.1,
    ):
        """Rebalance positions according to ``self.alloc`` target weights."""
        if not 0 <= cash_reserve < 1:
            raise AssertionError("cash_reserve should be between 0 and 1")
        if not 0 <= rtol < 1:
            raise AssertionError("rtol should be between 0 and 1")
        if atol < 0:
            raise AssertionError("atol should be non-negative")

        alloc = self.alloc
        if not force and not alloc.modified:
            alloc._next()
            return []

        total_equity = float(self.broker.getvalue())
        if total_equity <= 0:
            alloc._next()
            return []

        weights = alloc.weights
        target_values = weights * total_equity * (1.0 - float(cash_reserve))
        current_values = pd.Series(
            {
                ticker: self.position(ticker).size
                * self._current_price(self._resolve_ticker_data(ticker))
                for ticker in alloc.tickers
            },
            dtype="float64",
        )
        value_diff = target_values - current_values
        value_diff_abs = float(value_diff.abs().sum())
        value_diff_rel = value_diff_abs / total_equity
        should_trade = (
            force
            or (bool(atol) and value_diff_abs > float(atol))
            or value_diff_rel > rtol
        )

        orders = []
        for ticker in value_diff.sort_values().index:
            target_weight = float(weights.loc[ticker])
            if target_weight == 0.0:
                order = self.order_target_percent(ticker=ticker, target=0.0)
            elif should_trade:
                order = self.order_target_value(
                    ticker=ticker,
                    target=float(target_values.loc[ticker]),
                    price=self._current_price(self._resolve_ticker_data(ticker)),
                )
            else:
                order = None
            if order is not None:
                orders.append(order)

        alloc._next()
        return orders

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


class Allocation:
    """1.x-style allocation plan used by ``Strategy.rebalance``."""

    class Bucket:
        """Ranked subset of assets with helper weighting methods."""

        def __init__(self, alloc: Allocation) -> None:
            self._alloc = alloc
            self._tickers: list[str] = []
            self._weights: pd.Series | None = None

        @property
        def tickers(self) -> list[str]:
            return self._tickers.copy()

        @property
        def weights(self) -> pd.Series:
            if self._weights is None:
                raise RuntimeError("Bucket.weight_*() should be called before reading weights")
            if (self._weights < 0).any():
                raise AssertionError("Weight should be non-negative.")
            if self._weights.sum() > 1.000000000000001:
                raise AssertionError(
                    f"Total weight should be less than or equal to 1. Got {self._weights.sum()}"
                )
            return self._weights.copy()

        def append(
            self,
            ranked_list: list | pd.Series,
            *conditions: list | pd.Series,
        ) -> Allocation.Bucket:
            candidates = self._candidate_counts(ranked_list, *conditions)
            selected = [
                ticker for ticker, count in candidates.items() if count == len(conditions) + 1
            ]
            self._tickers.extend(ticker for ticker in selected if ticker not in self._tickers)
            return self

        def remove(self, *conditions: list | pd.Series) -> Allocation.Bucket:
            if not conditions:
                return self
            candidates = self._candidate_counts(*conditions)
            self._tickers = [
                ticker for ticker in self._tickers if candidates.get(ticker, 0) < len(conditions)
            ]
            return self

        def trim(self, limit: int) -> Allocation.Bucket:
            self._tickers = self._tickers[: int(limit)]
            return self

        def weight_explicitly(self, weight: float | list | pd.Series) -> Allocation.Bucket:
            if not self._tickers:
                self._weights = pd.Series(dtype="float64")
            elif isinstance(weight, Number):
                if not 0 <= float(weight) * len(self._tickers) <= 1.000000000000001:
                    raise AssertionError("Total weight should be within [0, 1].")
                self._weights = pd.Series(float(weight), index=self._tickers, dtype="float64")
            elif isinstance(weight, list):
                if any(float(value) < 0 for value in weight) or sum(weight) > 1.000000000000001:
                    raise AssertionError("Weight should be non-negative and sum to <= 1.")
                values = list(weight[: len(self._tickers)])
                values.extend([0.0] * (len(self._tickers) - len(values)))
                self._weights = pd.Series(values, index=self._tickers, dtype="float64")
            elif isinstance(weight, pd.Series):
                if (weight < 0).any() or weight.sum() > 1.000000000000001:
                    raise AssertionError("Weight should be non-negative and sum to <= 1.")
                self._weights = pd.Series(0.0, index=self._tickers, dtype="float64")
                selected = weight[weight.index.isin(self._tickers)]
                self._weights.loc[selected.index] = selected.astype(float)
            else:
                raise ValueError("Weight should be a number, list, or Series.")
            return self

        def weight_equally(self, sum_: float | None = None) -> Allocation.Bucket:
            if sum_ is not None and not 0 <= float(sum_) <= 1.000000000000001:
                raise AssertionError("Total weight should be within [0, 1].")
            total = self._alloc.unallocated if sum_ is None else float(sum_)
            if not self._tickers:
                self._weights = pd.Series(dtype="float64")
            else:
                self._weights = pd.Series(
                    total / len(self._tickers),
                    index=self._tickers,
                    dtype="float64",
                )
            return self

        def weight_proportionally(
            self,
            relative_weights: list,
            sum_: float | None = None,
        ) -> Allocation.Bucket:
            if len(relative_weights) != len(self._tickers):
                raise AssertionError(
                    f"Length of relative_weight {len(relative_weights)} does not match "
                    f"number of assets {len(self._tickers)}"
                )
            if any(float(value) < 0 for value in relative_weights):
                raise AssertionError("Relative weights should be non-negative.")
            if sum_ is not None and not 0 <= float(sum_) <= 1.000000000000001:
                raise AssertionError("Total weight should be within [0, 1].")
            total = self._alloc.unallocated if sum_ is None else float(sum_)
            denom = float(sum(relative_weights))
            if not self._tickers or denom == 0:
                self._weights = pd.Series(dtype="float64")
            else:
                self._weights = pd.Series(relative_weights, index=self._tickers, dtype="float64")
                self._weights = self._weights / denom * total
            return self

        def apply(self, method: str = "update") -> Allocation.Bucket:
            if self._weights is None:
                raise RuntimeError("Bucket.weight_*() should be called before apply()")
            if self.weights.empty:
                return self
            index = self.weights.index
            if method == "update":
                self._alloc.weights.loc[index] = self.weights
            elif method == "overwrite":
                self._alloc.weights.loc[:] = 0.0
                self._alloc.weights.loc[index] = self.weights
            elif method == "accumulate":
                self._alloc.weights.loc[index] = self._alloc.weights.loc[index] + self.weights
            else:
                raise ValueError(f"Invalid method {method!r}")
            return self

        @staticmethod
        def _as_candidates(item: list | pd.Series) -> list[str]:
            if isinstance(item, pd.Series):
                return [
                    str(index)
                    for index, value in item.items()
                    if not isinstance(value, (bool, np.bool_)) or bool(value)
                ]
            return [str(value) for value in item]

        @classmethod
        def _candidate_counts(cls, *items: list | pd.Series) -> dict[str, int]:
            candidates: dict[str, int] = {}
            for item in items:
                for ticker in cls._as_candidates(item):
                    candidates[ticker] = candidates.get(ticker, 0) + 1
            return candidates

        def __len__(self) -> int:
            return len(self._tickers)

        def __iter__(self):
            return iter(self._tickers)

    class BucketGroup:
        """Dictionary-like lazy bucket collection."""

        def __init__(self, alloc: Allocation) -> None:
            self._alloc = alloc
            self._buckets: dict[str, Allocation.Bucket] = {}

        def __getitem__(self, name: str) -> Allocation.Bucket:
            if name not in self._buckets:
                self._buckets[name] = Allocation.Bucket(self._alloc)
            return self._buckets[name]

        def clear(self) -> None:
            self._buckets.clear()

        def __iter__(self):
            return iter(self._buckets)

        def __len__(self) -> int:
            return len(self._buckets)

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = list(tickers)
        self._previous_weights = pd.Series(0.0, index=self.tickers, dtype="float64")
        self._weights: pd.Series | None = None
        self._bucket_group = Allocation.BucketGroup(self)

    def _require_weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError('"Allocation.assume_*()" must be called first.')
        return self._weights

    @property
    def bucket(self) -> BucketGroup:
        self._require_weights()
        return self._bucket_group

    @property
    def weights(self) -> pd.Series:
        weights = self._require_weights()
        if weights.index.to_list() != self.tickers:
            raise AssertionError("Weight index should be the same as the asset space.")
        if (weights < 0).any():
            raise AssertionError("Weight should be non-negative.")
        if weights.sum() > 1.000000000000001:
            raise AssertionError(
                f"Total weight should be less than or equal to 1. Got {weights.sum()}"
            )
        return weights

    @weights.setter
    def weights(self, value: pd.Series) -> None:
        weights = self._require_weights()
        value = pd.Series(value, dtype="float64")
        if (value < 0).any():
            raise AssertionError("Weight should be non-negative.")
        if value.sum() > 1.000000000000001:
            raise AssertionError(
                f"Total weight should be less than or equal to 1. Got {value.sum()}"
            )
        unknown = [str(index) for index in value.index if str(index) not in self.tickers]
        if unknown:
            raise KeyError(f"Unknown allocation ticker(s): {unknown}")
        weights.loc[:] = 0.0
        weights.loc[[str(index) for index in value.index]] = value.to_numpy(dtype=float)

    @property
    def previous_weights(self) -> pd.Series:
        return self._previous_weights.copy()

    @property
    def unallocated(self) -> float:
        allocated = float(self.weights.abs().sum())
        if allocated > 1.000000000000001:
            raise AssertionError(f"Total weight should be less than or equal to 1. Got {allocated}")
        return float(1.0 - allocated)

    def assume_zero(self) -> Allocation:
        self._weights = pd.Series(0.0, index=self.tickers, dtype="float64")
        return self

    def assume_previous(self) -> Allocation:
        self._weights = self.previous_weights
        return self

    def normalize(self) -> pd.Series:
        total = float(self.weights.abs().sum())
        if total:
            self._weights = self.weights / total
        return self.weights

    @property
    def modified(self) -> bool:
        return not self.weights.equals(self.previous_weights)

    def _next(self) -> None:
        self._previous_weights = self.weights.copy()
        self._weights = None
        self._bucket_group.clear()

    def _clear(self) -> None:
        self._previous_weights = pd.Series(0.0, index=self.tickers, dtype="float64")
        self._weights = None
        self._bucket_group.clear()
