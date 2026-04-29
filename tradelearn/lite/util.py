"""Utility helpers for the Tradelearn Lite facade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pandas_ta_classic as pta


class _TA:
    """Small pandas-ta-classic accessor for Lite ``data.ta`` usage."""

    _CLOSE_ONLY = {"bbands", "ema", "macd", "roc", "rsi", "sma"}
    _OHLC = {"adx", "atr", "ichimoku", "supertrend"}
    _OHLCV = {"vwap"}

    def __init__(self, data: pd.DataFrame, wrapper: Callable[[Any, str], Any] | None = None) -> None:
        self._data = data
        self._wrapper = wrapper

    def __getattr__(self, name: str) -> Callable[..., Any]:
        indicator = getattr(pta, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            if args and _looks_like_series(args[0]):
                result = indicator(*args, **kwargs)
                return self._wrap(result, name)
            if name in self._CLOSE_ONLY:
                result = indicator(self._data["close"], *args, **kwargs)
                return self._wrap(result, name)
            if name in self._OHLC:
                result = indicator(
                    self._data["high"],
                    self._data["low"],
                    self._data["close"],
                    *args,
                    **kwargs,
                )
                return self._wrap(result, name)
            if name in self._OHLCV:
                result = indicator(
                    self._data["high"],
                    self._data["low"],
                    self._data["close"],
                    self._data["volume"],
                    *args,
                    **kwargs,
                )
                return self._wrap(result, name)
            result = indicator(*args, **kwargs)
            return self._wrap(result, name)

        return call

    def _wrap(self, result: Any, name: str) -> Any:
        if self._wrapper is None:
            return result
        return self._wrapper(result, name)


def _looks_like_series(value: Any) -> bool:
    return isinstance(value, pd.Series) or hasattr(value, "__array__")
