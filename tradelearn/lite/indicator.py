from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Signal:
    """Lite signal wrapper with line-like indexing."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def __getitem__(self, index: int) -> float:
        try:
            return float(self._values[index])
        except (IndexError, TypeError, ValueError):
            return 0.0


class IndicatorProxy:
    """Gradually revealed indicator/data line used by the Lite facade."""

    __slots__ = ("_data", "_feed", "_length", "_index", "_name", "attrs")

    def __init__(
        self,
        data: np.ndarray,
        feed: Any,
        index: pd.Index | None = None,
        name: str | None = None,
    ):
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
        return self._data[: cursor + 1]

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
        if isinstance(key, slice):
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

    def to_series(self) -> pd.Series:
        """Return the full line as a pandas Series for vector indicators."""
        return pd.Series(self._data, index=self._index, name=self._name)

    def wrap_indicator(self, values: Any, name: str | None = None) -> Any:
        """Wrap vector indicator output back into Lite indicator proxies."""
        return _wrap_indicator_result(
            values,
            self._feed,
            self._index if self._index is not None else pd.RangeIndex(self._length),
            name or self._name or "indicator",
        )


class IndicatorBundle:
    """Gradually revealed multi-column indicator bundle used by Lite indicator results."""

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
    if not frame.index.equals(index) and len(frame) != len(index):
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


__all__ = ["IndicatorBundle", "IndicatorProxy", "Signal", "_wrap_indicator_result"]
