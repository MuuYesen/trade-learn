from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
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
        self._lite_target_size_by_data: dict[Any, float] = {}

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
        next_target_size = self._lite_position_size(data) + pending_delta
        pending[data] = pending.get(data, 0.0) + pending_delta
        self._lite_target_size_by_data[data] = next_target_size
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

    def _lite_position_size(self, data: Any) -> float:
        if data in self._lite_target_size_by_data:
            return float(self._lite_target_size_by_data[data])
        return float(self.getposition(data).size + self._pending_size.get(data, 0.0))

    def order_target_size(self, *, ticker: str = None, target: float = 0, **kwargs: Any):
        data = self._resolve_ticker_data(ticker)
        delta = float(target) - self._lite_position_size(data)
        if delta > 0:
            return self._buy_data(data=data, size=delta, **kwargs)
        if delta < 0:
            return self._sell_data(data=data, size=abs(delta), **kwargs)
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
        price = float(price if price is not None else self._current_price(data))
        mult = self._position_mult(data)
        current_value = self._lite_position_size(data) * price * mult
        delta = float(target) - current_value
        if abs(delta) < 1e-12:
            return None
        size = int(abs(delta) / (price * mult))
        if not size:
            return None
        if delta > 0:
            return self._buy_data(data=data, size=size, **kwargs)
        return self._sell_data(data=data, size=size, **kwargs)

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

    def target_percent(self, ticker: str, target: float):
        """Move one ticker toward a target portfolio weight."""
        return self.order_target_percent(ticker=ticker, target=float(target))

    def target_weights(
        self,
        weights: Mapping[str, float] | pd.Series,
        *,
        close_missing: bool = True,
    ) -> list[Any]:
        """Move the portfolio toward the requested ticker weights.

        ``cash`` is accepted as a reserved key and is not translated into an order.
        """
        targets = pd.Series(dict(weights), dtype="float64")
        if (targets < 0).any():
            raise ValueError("target weights must be non-negative")

        cash_weight = float(targets.pop("cash")) if "cash" in targets else 0.0
        if cash_weight < 0:
            raise ValueError("cash target weight must be non-negative")
        if float(targets.sum() + cash_weight) > 1.000000000000001:
            raise ValueError("target weights plus cash must sum to <= 1")

        known = set(self._bt_data_by_ticker) or set(self.data.tickers)
        unknown = sorted(str(ticker) for ticker in targets.index if str(ticker) not in known)
        if unknown:
            raise ValueError(f"Unknown ticker(s): {unknown}")

        orders: list[Any] = []
        if close_missing:
            for ticker in sorted(known.difference(str(ticker) for ticker in targets.index)):
                order = self.target_percent(ticker, 0.0)
                if order is not None:
                    orders.append(order)

        for ticker, target in targets.sort_values().items():
            order = self.target_percent(str(ticker), float(target))
            if order is not None:
                orders.append(order)
        return orders

    def target_equal(
        self,
        tickers: Sequence[str],
        *,
        weight: float = 1.0,
        close_missing: bool = True,
    ) -> list[Any]:
        """Assign an equal combined target weight to the selected tickers."""
        tickers = [str(ticker) for ticker in tickers]
        if not tickers:
            return self.close_all() if close_missing else []
        per_ticker = float(weight) / len(tickers)
        return self.target_weights(
            {ticker: per_ticker for ticker in tickers},
            close_missing=close_missing,
        )

    def close_all(self) -> list[Any]:
        """Close all known Lite data-feed positions."""
        orders = []
        for ticker in self._bt_data_by_ticker or {self.data.the_ticker: self._bt_primary_data}:
            order = self.target_percent(str(ticker), 0.0)
            if order is not None:
                orders.append(order)
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
