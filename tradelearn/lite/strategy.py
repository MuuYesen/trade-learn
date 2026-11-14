from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradelearn.backtest.history import window as _window
from tradelearn.backtest.models import Order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.backtest.targets import TargetWeightSnapshot, build_target_weight_intents
from tradelearn.lite.data import LiteDataProxy
from tradelearn.lite.position import PositionProxy


@dataclass(frozen=True)
class _TargetWeightSnapshots:
    prices: dict[str, float]
    sizes: dict[str, float]
    mults: dict[str, float]


class Strategy(CoreStrategy):
    """Tradelearn Lite strategy facade."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = self._check_params(kwargs)
        self._bt_data = None
        self._bt_position = PositionProxy(self)
        self._bt_positions: dict[str | None, PositionProxy] = {}
        self._bt_data_by_ticker: dict[str, Any] = {}
        self._bt_primary_data = None
        self._records: dict[str, pd.Series | pd.DataFrame] = {}
        self._storage: dict[str, Any] | None = None
        self._start_on_day = 0

    def init(self):
        """Initialize shared streaming lines and strategy-owned state."""
        pass

    @classmethod
    def _trial_param_defaults(cls):
        """Supply Lite's declared class-attribute defaults to historical trials."""
        framework_bases = set(Strategy.__mro__)
        defaults = {}
        for base in reversed(cls.__mro__):
            if base in framework_bases:
                continue
            defaults.update({name: value for name, value in vars(base).items()
                if not name.startswith('_') and not callable(value)
                and not isinstance(value, (property, classmethod, staticmethod))})
        return defaults

    def _check_params(self, params: dict[str, Any]) -> dict[str, Any]:
        missing = object()
        for key, value in params.items():
            declared = next((vars(cls)[key] for cls in type(self).__mro__
                             if cls is not Strategy and cls is not CoreStrategy and key in vars(cls)), missing)
            if key.startswith("_") or callable(declared) or isinstance(declared, property) or declared is missing:
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

    def buy(
        self,
        *,
        ticker: str = None,
        size: float = 1.0,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ):
        if not isinstance(size, (int, float)) or not __import__("math").isfinite(size) or size <= 0:
            raise ValueError("size must be a positive finite order quantity")
        if sl is not None or tp is not None:
            return self.buy_bracket(ticker=ticker, size=size, sl=sl, tp=tp, tag=tag)
        data = self._resolve_ticker_data(ticker)
        return self._submit_lite_order(Order.Buy, data, size, limit, stop, tag)

    def _submit_lite_order(
        self,
        side: int,
        data: Any,
        size: float,
        limit: float | None,
        stop: float | None,
        tag: object | None,
        **order_kwargs: Any,
    ) -> Any:
        if limit is not None and stop is not None:
            exectype, price = Order.StopLimit, stop
            order_kwargs["pricelimit"] = limit
        else:
            exectype = Order.Limit if limit is not None else Order.Stop if stop is not None else Order.Market
            price = limit if limit is not None else stop
        info = dict(order_kwargs.pop("info", {}))
        if tag is not None:
            info["tag"] = tag
        if info:
            order_kwargs["info"] = info
        return super()._submit_order(side, data, size, price, exectype, **order_kwargs)

    def _buy_data(
        self,
        data: Any,
        size: float | None = None,
        price: float | None = None,
        exectype: int | None = None,
        **kwargs: Any,
    ):
        return super()._submit_order(Order.Buy, data, size, price, exectype, **kwargs)

    def _sell_data(self, data, size=None, price=None, exectype=None, **kwargs):
        return super()._submit_order(Order.Sell, data, size, price, exectype, **kwargs)

    def sell(
        self,
        *,
        ticker: str = None,
        size: float = 1.0,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        tag: object = None,
    ):
        if not isinstance(size, (int, float)) or not __import__("math").isfinite(size) or size <= 0:
            raise ValueError("size must be a positive finite order quantity")
        if sl is not None or tp is not None:
            return self.sell_bracket(ticker=ticker, size=size, sl=sl, tp=tp, tag=tag)
        data = self._resolve_ticker_data(ticker)
        return self._submit_lite_order(Order.Sell, data, size, limit, stop, tag)

    def _current_price(self, data: Any) -> float:
        return float(data.close[0])

    def _lite_position_size(self, data: Any) -> float:
        """Filled quantity only; pending orders are separate broker facts."""
        return float(self.getposition(data).size)

    def _resolve_data(self, data=None):
        resolved = self._bt_primary_data if data is None else data
        return super()._resolve_data(getattr(resolved, "_feed", resolved))

    def close(self, *, ticker=None, size=None, **kwargs):
        data = self._resolve_ticker_data(ticker)
        native_close = getattr(self.broker,'_close_position',None)
        if callable(native_close):
            result = native_close(self,data,size=size,**kwargs)
            if result is not NotImplemented:
                return result
        position = self.getposition(data).size
        if position > 0:
            return self._sell_data(data, abs(position) if size is None else size, **kwargs)
        if position < 0:
            return self._buy_data(data, abs(position) if size is None else size, **kwargs)
        return None

    def order_target_size(self, *, ticker: str = None, target: float = 0, **kwargs: Any):
        return super().order_target_size(data=self._resolve_ticker_data(ticker), target=target, **kwargs)

    def order_target_value(self, *, ticker=None, target=0.0, price=None, **kwargs):
        return super().order_target_value(data=self._resolve_ticker_data(ticker), target=target,
                                          price=price, **kwargs)

    def order_target_percent(self, *, ticker=None, target=0.0, **kwargs):
        return self.order_target_value(ticker=ticker, target=float(target)*self.equity, **kwargs)

    def buy_bracket(
        self,
        *,
        ticker: str = None,
        size: float = 1.0,
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
            price=stop if stop is not None else limit,
            stopprice=sl,
            limitprice=tp,
            exectype=Order.StopLimit if limit is not None and stop is not None else Order.Limit if limit is not None else Order.Stop if stop is not None else Order.Market,
            pricelimit=limit if stop is not None and limit is not None else None,
            stopexec=Order.Stop if sl is not None else None,
            limitexec=Order.Limit if tp is not None else None,
            **kwargs,
        )

    def sell_bracket(
        self,
        *,
        ticker: str = None,
        size: float = 1.0,
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
            price=stop if stop is not None else limit,
            stopprice=sl,
            limitprice=tp,
            exectype=Order.StopLimit if limit is not None and stop is not None else Order.Limit if limit is not None else Order.Stop if stop is not None else Order.Market,
            pricelimit=limit if stop is not None and limit is not None else None,
            stopexec=Order.Stop if sl is not None else None,
            limitexec=Order.Limit if tp is not None else None,
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

    def record(self, name=None, plot=True, overlay=None, color=None, scatter=False, **kwargs):
        """Record observations at the delivered clock, without copying future bars."""
        if not len(self.data):
            raise RuntimeError("record requires a delivered bar")
        timestamp = self.data.now
        runtime = getattr(self,"_runtime",None)
        stream = runtime is not None and runtime.profile == "stream"
        if stream and len(set(self._records) | set(kwargs)) > 128:
            raise ValueError("stream record supports at most 128 named series")
        for key, value in kwargs.items():
            if isinstance(value, (dict, pd.Series)):
                values = dict(value)
                frame = self._records.get(key)
                if not isinstance(frame, pd.DataFrame):
                    frame = pd.DataFrame(columns=list(values), dtype=float)
                for column in values:
                    if column not in frame:
                        frame[column] = float("nan")
                frame.loc[timestamp, list(values)] = list(values.values())
                self._records[key] = frame
            else:
                series = self._records.get(key)
                if not isinstance(series, pd.Series):
                    series = pd.Series(dtype=float)
                series.loc[timestamp] = value
                self._records[key] = series
            if stream:
                capacity = runtime._feed_for(self._bt_primary_data)._buffer.capacity
                self._records[key] = self._records[key].iloc[-capacity:]
            result = self._records[key]
            result.name = name or key
            result.attrs.update(name=name or key, plot=plot, overlay=overlay, color=color, scatter=scatter)

    @property
    def equity(self) -> float:
        return float(self.broker.getvalue())

    @property
    def storage(self) -> dict[str, Any] | None:
        return self._storage

    @property
    def orders(self) -> tuple[Any, ...]:
        return self.broker.get_orders_open(owner=self) if getattr(self.broker, "_supports_order_owner", False) else tuple(
            order for order in self.broker.get_orders_open()
            if getattr(order, "owner", self) is self
        )

    def trades(self, ticker: str = None) -> tuple[Any, ...]:
        return tuple(getattr(self, "_trades", ()))

    @property
    def closed_trades(self) -> tuple[Any, ...]:
        return tuple(getattr(self, "_closed_trades", ()))

    def window(self, size: int, data=None, ago: int = 0):
        """Copy the delivered OHLCV window using the shared research converter."""
        return _window(self, size=size, data=data, ago=ago)

    def target_weights(
        self,
        weights: Mapping[str, float] | pd.Series,
        *,
        close_missing: bool = True,
        constraints: Any | None = None,
    ) -> list[Any]:
        """Move the portfolio toward the requested ticker weights.

        ``cash`` is accepted as a reserved key and is not translated into an order.
        """
        return super()._target_weights(weights, close_missing=close_missing, constraints=constraints)

    def _target_weight_data_map(self) -> dict[str, Any]:
        if self._bt_data_by_ticker:
            return self._bt_data_by_ticker
        return {self.data.the_ticker: self._bt_primary_data}

    def _target_weight_snapshots(self, data_by_ticker: dict[str, Any]) -> _TargetWeightSnapshots:
        prices: dict[str, float] = {}
        sizes: dict[str, float] = {}
        mults: dict[str, float] = {}
        pending = self._pending_size
        comminfo = getattr(self.broker, "_comminfo", None)
        default_mult = float(getattr(self.broker, "_mult", 1.0))
        for ticker, data in data_by_ticker.items():
            prices[ticker] = float(data.close[0])
            mults[ticker] = self._position_mult(data) if comminfo is not None else default_mult
            sizes[ticker] = float(self.getposition(data).size + self._target_pending_size(data))
        return _TargetWeightSnapshots(prices=prices, sizes=sizes, mults=mults)

    def close_all(self) -> list[Any]:
        """Close all known Lite data-feed positions."""
        orders = []
        for ticker in self._bt_data_by_ticker or {self.data.the_ticker: self._bt_primary_data}:
            order = self.close(ticker=str(ticker))
            if order is not None:
                orders.append(order)
        return orders

    def start_on_day(self, n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("bar index must be a non-negative integer")
        self._start_on_day = int(n)
        self.addminperiod(int(n) + 1)

    def start_on_bar(self, n: int) -> None:
        """Start ``next`` no earlier than bar index ``n``."""
        self.start_on_day(n)
