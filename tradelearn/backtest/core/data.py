from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict

class DataContainer:
    """Core data storage for OHLCV and extra columns."""
    def __init__(self, data: pd.DataFrame, name: str | None = None) -> None:
        self._name = name
        self._frame = data.copy()
        self._cursor = -1
        
        # Common OHLCV arrays for fast access
        if isinstance(data.index, pd.DatetimeIndex):
            self._datetime = data.index.values.astype('datetime64[s]').view(np.int64)
        else:
            self._datetime = data.index.to_numpy()
            
        self._open = data["open"].to_numpy(dtype=np.float64)
        self._high = data["high"].to_numpy(dtype=np.float64)
        self._low = data["low"].to_numpy(dtype=np.float64)
        self._close = data["close"].to_numpy(dtype=np.float64)
        self._volume = data["volume"].to_numpy(dtype=np.float64)
        
        # Store all columns in a dict for flexible access
        self._arrays: Dict[str, np.ndarray] = {
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

    def __len__(self) -> int:
        return self._cursor + 1

    def buflen(self) -> int:
        return len(self._close)

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def get_value(self, name: str, ago: int = 0) -> float:
        arr = self._arrays.get(name)
        if arr is None: return np.nan
        idx = self._cursor - ago
        if idx < 0 or idx >= len(arr): return np.nan
        return float(arr[idx])

    def get_array(self, name: str) -> np.ndarray:
        return self._arrays.get(name, np.array([], dtype=np.float64))
