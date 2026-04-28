from __future__ import annotations

import numpy as np
import pandas as pd


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
            self._datetime = data.index.values.astype('datetime64[s]').view(np.int64)
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
            "volume": self._volume
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
