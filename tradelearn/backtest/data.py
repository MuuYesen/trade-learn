from __future__ import annotations

import numpy as np
import pandas as pd

from tradelearn.core import StreamBar


class SharedBarBuffer:
    """Zero-copy OHLCV array view with a mutable cursor source."""

    __slots__ = ("_source", "arrays")

    def __init__(self, source: DataContainer, arrays: dict[str, np.ndarray]) -> None:
        self._source = source
        self.arrays = arrays

    @property
    def cursor(self) -> int:
        return self._source._cursor

    def value(self, name: str, ago: int = 0) -> float:
        arr = self.arrays.get(name)
        if arr is None:
            return np.nan
        idx = self.cursor - ago
        if idx < 0 or idx >= len(arr):
            return np.nan
        return float(arr[idx])


class RollingBarBuffer:
    """Append-only rolling OHLCV buffer for live/paper event runners."""

    __slots__ = ("arrays", "capacity", "_cursor", "_size")

    def __init__(self, capacity: int, fields: tuple[str, ...] | None = None) -> None:
        self.capacity = max(1, int(capacity))
        fields = fields or ("datetime", "open", "high", "low", "close", "volume")
        self.arrays = {name: np.full(self.capacity, np.nan, dtype=np.float64) for name in fields}
        self._cursor = -1
        self._size = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    def append(self, bar: StreamBar | dict[str, object]) -> None:
        values = self._bar_values(bar)
        if self._size < self.capacity:
            self._cursor = self._size
            self._size += 1
        else:
            for arr in self.arrays.values():
                arr[:-1] = arr[1:]
            self._cursor = self.capacity - 1
        for name, value in values.items():
            if name in self.arrays:
                self.arrays[name][self._cursor] = value

    def value(self, name: str, ago: int = 0) -> float:
        arr = self.arrays.get(name)
        if arr is None:
            return np.nan
        idx = self.cursor - ago
        if idx < 0 or idx >= self._size:
            return np.nan
        return float(arr[idx])

    @staticmethod
    def _bar_values(bar: StreamBar | dict[str, object]) -> dict[str, float]:
        if isinstance(bar, StreamBar):
            return {
                "datetime": float(bar.ts.timestamp()),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        values = dict(bar)
        dt = values.get("datetime", values.get("ts", np.nan))
        if isinstance(dt, pd.Timestamp):
            values["datetime"] = float(dt.timestamp())
        return {key: float(value) for key, value in values.items() if value is not None}


class DataContainer:
    """Core data storage for OHLCV and extra columns."""

    def __init__(self, data: pd.DataFrame, name: str | None = None) -> None:
        self._name = name
        self._frame = data.copy()
        self._cursor = -1

        # Common OHLCV arrays for fast access
        def _get_col(df, names, default_val=0.0):
            for n in names:
                if n in df.columns:
                    return df[n].to_numpy(dtype=np.float64)
                if n.lower() in df.columns:
                    return df[n.lower()].to_numpy(dtype=np.float64)
                if n.capitalize() in df.columns:
                    return df[n.capitalize()].to_numpy(dtype=np.float64)
            return np.full(len(df), default_val, dtype=np.float64)

        if isinstance(data.index, pd.DatetimeIndex):
            self._datetime = data.index.values.astype("datetime64[s]").view(np.int64)
        else:
            self._datetime = data.index.to_numpy()

        self._open = _get_col(data, ["open", "Open"])
        self._high = _get_col(data, ["high", "High"])
        self._low = _get_col(data, ["low", "Low"])
        self._close = _get_col(data, ["close", "Close"])
        self._volume = _get_col(data, ["volume", "Volume"])

        # Store all columns in a dict for flexible access
        self._arrays: dict[str, np.ndarray] = {
            "datetime": self._datetime,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
        }

        # Add extra columns
        for col in data.columns:
            if col not in self._arrays:
                self._arrays[col] = data[col].to_numpy(dtype=np.float64)
        self._bar_buffer = SharedBarBuffer(self, self._arrays)

    def __len__(self) -> int:
        return self._cursor + 1

    def buflen(self) -> int:
        return len(self._close)

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def get_value(self, name: str, ago: int = 0) -> float:
        arr = self._arrays.get(name)
        if arr is None:
            return np.nan
        idx = self._cursor - ago
        if idx < 0 or idx >= len(arr):
            return np.nan
        return float(arr[idx])

    def get_array(self, name: str) -> np.ndarray:
        return self._arrays.get(name, np.array([], dtype=np.float64))

    def shared_bar_buffer(self) -> SharedBarBuffer:
        """Return the zero-copy OHLCV buffer used by line accessors."""
        return self._bar_buffer

    def rolling_bar_buffer(self, capacity: int) -> RollingBarBuffer:
        """Return a live-compatible rolling buffer seeded from this data."""
        buffer = RollingBarBuffer(capacity=capacity)
        start = max(0, self.buflen() - buffer.capacity)
        for i in range(start, self.buflen()):
            buffer.append(
                {
                    "datetime": float(self._datetime[i]),
                    "open": float(self._open[i]),
                    "high": float(self._high[i]),
                    "low": float(self._low[i]),
                    "close": float(self._close[i]),
                    "volume": float(self._volume[i]),
                }
            )
        return buffer
