from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.compat.backtrader.base import LineSeries


class IndicatorCache:
    """Precompute event-loop indicator arrays and expose cursor-aware proxies."""

    def __init__(self, frame: pd.DataFrame | Any) -> None:
        if hasattr(frame, "_frame"):
            frame = frame._frame
        self._frame = frame
        self._cursor = -1
        self._cache: dict[tuple[Any, ...], LineSeries] = {}

    def advance(self, cursor: int) -> None:
        self._cursor = cursor
        for proxy in self._cache.values():
            proxy._advance(cursor)

    def precompute(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> LineSeries:
        key = self._key(name, args, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resolved_args = tuple(self._resolve_arg(arg) for arg in args)
        result = func(*resolved_args, **kwargs)
        proxy = LineSeries(self._to_array(result))
        self._cache[key] = proxy
        return proxy

    def get(self, name: str, **kwargs: Any) -> LineSeries | None:
        prefix = (name,)
        kw_items = tuple((key, self._value_key(value)) for key, value in sorted(kwargs.items()))
        for key, proxy in self._cache.items():
            if key[0:1] == prefix and key[2] == kw_items:
                return proxy
        return None

    def _resolve_arg(self, arg: Any) -> Any:
        if isinstance(arg, str) and arg in self._frame.columns:
            return self._frame[arg]
        return arg

    @staticmethod
    def _to_array(result: Any) -> np.ndarray:
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].to_numpy()
        if isinstance(result, pd.Series):
            return result.to_numpy()
        return np.asarray(result)

    def _key(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return (
            name,
            tuple(self._value_key(arg) for arg in args),
            tuple((key, self._value_key(value)) for key, value in sorted(kwargs.items())),
        )

    @staticmethod
    def _value_key(value: Any) -> tuple[str, Any]:
        if isinstance(value, str):
            return ("column", value)
        try:
            hash(value)
        except TypeError:
            return ("id", id(value))
        return ("value", value)
