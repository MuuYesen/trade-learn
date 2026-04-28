from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Lines:
    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._lines = []

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value
            cls_lines = getattr(self._owner.__class__, "lines", [])
            if isinstance(cls_lines, (list, tuple)) and name in cls_lines:
                idx = list(cls_lines).index(name)
                while len(self._lines) <= idx:
                    self._lines.append(None)
                self._lines[idx] = value
            elif value not in self._lines:
                self._lines.append(value)

    def __getitem__(self, i: int) -> Any:
        return self._lines[i]

    def __iter__(self):
        return iter(self._lines)

    def __len__(self):
        return len(self._lines)


class LineSeries:
    def __init__(
        self,
        values: Any,
        is_datetime: bool = False,
        buffer: Any | None = None,
        buffer_name: str | None = None,
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._cursor = 0
        self._is_datetime = is_datetime
        self._buffer = buffer
        self._buffer_name = buffer_name
        self._series_cache = None
        self.min_period = 0

    def date(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.date() if dt else None

    def time(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.time() if dt else None

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def __len__(self) -> int:
        buffer = self._buffer
        if buffer is not None:
            return buffer.cursor + 1
        return self._cursor + 1

    def __getitem__(self, ago: Any) -> Any:
        values = self._values
        buffer = self._buffer
        if ago == 0:
            if not self._is_datetime:
                if buffer is not None and self._buffer_name is not None:
                    return buffer.value(self._buffer_name, ago=0)
                source = getattr(self, "_cursor_source", None)
                cursor = source._cursor if source is not None else self._cursor
                if cursor < 0 or cursor >= len(values):
                    return np.nan
                return values[cursor]
            return self._format_value(self._current_value(0))
        if ago == -1:
            if not self._is_datetime:
                if buffer is not None and self._buffer_name is not None:
                    return buffer.value(self._buffer_name, ago=1)
                source = getattr(self, "_cursor_source", None)
                cursor = source._cursor if source is not None else self._cursor
                idx = cursor - 1
                if idx < 0 or idx >= len(values):
                    return np.nan
                return values[idx]
            return self._format_value(self._current_value(1))
        if not isinstance(ago, (int, slice, np.integer)):
            return self
        if isinstance(ago, slice):
            return values[ago]
        cursor = buffer.cursor if buffer is not None else self._cursor
        idx = cursor + int(ago)
        if idx < 0 or idx >= len(values):
            return np.nan
        value = values[idx]
        if not self._is_datetime:
            return value
        return self._format_value(value)

    def _current_value(self, ago: int) -> Any:
        if self._buffer is not None and self._buffer_name is not None:
            return self._buffer.value(self._buffer_name, ago=ago)
        idx = self._cursor - ago
        if idx < 0 or idx >= len(self._values):
            return np.nan
        return self._values[idx]

    def _format_value(self, value: Any) -> Any:
        if not self._is_datetime:
            return value
        if pd.isna(value):
            return None
        return pd.to_datetime(value, unit="s" if abs(value) < 1e11 else "ms", utc=True)

    def __call__(self, ago: int = 0) -> LineSeries:
        return DelayedLine(self, ago)

    def datetime(self, ago: int = 0) -> Any:
        val = self[ago]
        if pd.isna(val):
            return None
        if self._is_datetime:
            return pd.to_datetime(val)
        return pd.to_datetime(val, unit="s" if val < 1e11 else "ms")

    def __bool__(self) -> bool:
        val = self[0]
        return bool(val) and not np.isnan(val)

    def __lt__(self, other):
        return self[0] < (other[0] if hasattr(other, "__getitem__") else other)

    def __gt__(self, other):
        return self[0] > (other[0] if hasattr(other, "__getitem__") else other)

    def __le__(self, other):
        return self[0] <= (other[0] if hasattr(other, "__getitem__") else other)

    def __ge__(self, other):
        return self[0] >= (other[0] if hasattr(other, "__getitem__") else other)

    def __eq__(self, other):
        return self[0] == (other[0] if hasattr(other, "__getitem__") else other)

    def __add__(self, other):
        return self._math_op(other, np.add)

    def __sub__(self, other):
        return self._math_op(other, np.subtract)

    def __mul__(self, other):
        return self._math_op(other, np.multiply)

    def __truediv__(self, other):
        return self._math_op(other, np.divide)

    def _math_op(self, other, op):
        v1 = self._values
        target = other.lines[0] if hasattr(other, "lines") else other
        res = None
        if hasattr(target, "_values"):
            v2 = target._values
            if v1.shape != v2.shape:
                max_len = max(len(v1), len(v2))
                v1_ext = np.full(max_len, np.nan)
                v1_ext[: len(v1)] = v1
                v2_ext = np.full(max_len, np.nan)
                v2_ext[: len(v2)] = v2
                res = LineSeries(op(v1_ext, v2_ext))
            else:
                res = LineSeries(op(v1, v2))
            res.min_period = max(getattr(self, "min_period", 0), getattr(target, "min_period", 0))
        else:
            res = LineSeries(op(v1, target))
            res.min_period = getattr(self, "min_period", 0)
        source = getattr(self, "_cursor_source", None)
        if source is not None:
            res._cursor_source = source
        return res


class DelayedLine(LineSeries):
    def __init__(self, source: LineSeries, ago: int) -> None:
        self._source = source
        self._ago = ago
        shifted = pd.Series(source._values).shift(-ago).values
        super().__init__(shifted)
        self._is_datetime = source._is_datetime
        cursor_source = getattr(source, "_cursor_source", None)
        if cursor_source is not None:
            self._cursor_source = cursor_source
        self.min_period = source.min_period + abs(ago)


class IndicatorLine(LineSeries):
    def __init__(self, source: LineSeries, shift: int):
        super().__init__(pd.Series(source._values).shift(-shift).values)
