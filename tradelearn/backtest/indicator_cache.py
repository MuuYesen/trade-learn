from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.lines import LineSeries


class IndicatorBundle:
    """Named collection of indicator lines returned by multi-output indicators."""

    def __init__(self, lines: dict[str, LineSeries]) -> None:
        self._lines = dict(lines)
        self.min_period = max(
            (getattr(line, "min_period", 0) for line in self._lines.values()),
            default=0,
        )

    def __getitem__(self, name: str) -> LineSeries:
        return self._lines[name]

    def __getattr__(self, name: str) -> LineSeries:
        try:
            return self._lines[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __iter__(self):
        return iter(self._lines.values())

    def keys(self):
        return self._lines.keys()

    def values(self):
        return self._lines.values()

    def items(self):
        return self._lines.items()

    def _advance(self, cursor: int) -> None:
        for line in self._lines.values():
            line._advance(cursor)


class BatchIndicatorCache:
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

    def precompute_many(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, LineSeries]:
        key = self._key(name, args, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            return {name: cached}
        resolved_args = tuple(self._resolve_arg(arg) for arg in args)
        result = func(*resolved_args, **kwargs)
        if not isinstance(result, pd.DataFrame):
            proxy = LineSeries(self._to_array(result))
            self._cache[key] = proxy
            return {name: proxy}
        lines: dict[str, LineSeries] = {}
        for column in result.columns:
            line_name = str(column)
            key = self._key(f"{name}.{line_name}", args, kwargs)
            proxy = self._cache.get(key)
            if proxy is None:
                proxy = LineSeries(result[column].to_numpy())
                self._cache[key] = proxy
            lines[line_name] = proxy
        return lines

    def get(self, name: str, **kwargs: Any) -> LineSeries | None:
        prefix = (name,)
        kw_items = tuple((key, self._value_key(value)) for key, value in sorted(kwargs.items()))
        for key, proxy in self._cache.items():
            if key[0:1] == prefix and key[2] == kw_items:
                return proxy
        return None

    def _resolve_arg(self, arg: Any) -> Any:
        line_name = self._line_name(arg)
        if line_name is not None:
            if line_name in self._frame.columns:
                return self._frame[line_name]
            lower_name = line_name.lower()
            if lower_name in self._frame.columns:
                return self._frame[lower_name]
            title_name = line_name.capitalize()
            if title_name in self._frame.columns:
                return self._frame[title_name]
        if isinstance(arg, str) and arg in self._frame.columns:
            return self._frame[arg]
        if hasattr(arg, "_values"):
            return pd.Series(arg._values)
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
        line_name = BatchIndicatorCache._line_name(value)
        if line_name is not None:
            return ("line", line_name)
        if isinstance(value, str):
            return ("column", value)
        try:
            hash(value)
        except TypeError:
            return ("id", id(value))
        return ("value", value)

    @staticmethod
    def _line_name(value: Any) -> str | None:
        buffer_name = getattr(value, "_buffer_name", None)
        if isinstance(buffer_name, str):
            return buffer_name
        return None


class RollingIndicatorCache:
    """Rolling-window indicator cache for live/paper event runners."""

    def __init__(self, window: int = 256) -> None:
        self.window = max(1, int(window))
        self._bars: list[dict[str, Any]] = []
        self._registrations: list[
            tuple[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []
        self._cache: dict[tuple[Any, ...], LineSeries] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> LineSeries:
        key = self._key(name, args, kwargs)
        proxy = self._cache.get(key)
        if proxy is None:
            proxy = LineSeries([])
            self._cache[key] = proxy
            self._registrations.append((name, func, args, kwargs))
        return proxy

    def append_bar(self, bar: dict[str, Any]) -> None:
        self._bars.append(dict(bar))
        if len(self._bars) > self.window:
            self._bars = self._bars[-self.window :]
        frame = pd.DataFrame(self._bars)
        for name, func, args, kwargs in self._registrations:
            key = self._key(name, args, kwargs)
            proxy = self._cache[key]
            resolved_args = tuple(
                frame[arg] if isinstance(arg, str) and arg in frame else arg for arg in args
            )
            result = func(*resolved_args, **kwargs)
            value = self._last_value(result)
            proxy._values = np.append(proxy._values, value)
            proxy._advance(len(proxy._values) - 1)

    @staticmethod
    def _last_value(result: Any) -> float:
        if result is None:
            return np.nan
        if isinstance(result, pd.DataFrame):
            value = result.iloc[-1, 0]
        elif isinstance(result, pd.Series):
            value = result.iloc[-1]
        else:
            arr = np.asarray(result)
            value = arr[-1]
        return float(value) if not pd.isna(value) else np.nan

    def _key(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return (
            name,
            tuple(BatchIndicatorCache._value_key(arg) for arg in args),
            tuple(
                (key, BatchIndicatorCache._value_key(value))
                for key, value in sorted(kwargs.items())
            ),
        )


IndicatorCache = BatchIndicatorCache
