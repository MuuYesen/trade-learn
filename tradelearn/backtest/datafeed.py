from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, List
from tradelearn.backtest.base import LineRoot, LineSeries

class DataFeed(LineRoot):
    """Column-based OHLCV feed exposed to strategies."""
    lines = ("datetime", "open", "high", "low", "close", "volume")

    def __init__(self, data: pd.DataFrame, name: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        frame = data.copy()
        self._name = name
        self._frame = frame
        
        if isinstance(frame.index, pd.DatetimeIndex):
            self.lines.datetime = LineSeries(frame.index.values.astype('datetime64[s]').view(np.int64))
        else:
            self.lines.datetime = LineSeries(frame.index.to_numpy())
            
        self.lines.open = LineSeries(frame["open"].to_numpy(dtype=np.float64))
        self.lines.high = LineSeries(frame["high"].to_numpy(dtype=np.float64))
        self.lines.low = LineSeries(frame["low"].to_numpy(dtype=np.float64))
        self.lines.close = LineSeries(frame["close"].to_numpy(dtype=np.float64))
        self.lines.volume = LineSeries(frame["volume"].to_numpy(dtype=np.float64))
        
        self._lines = [self.datetime, self.open, self.high, self.low, self.close, self.volume]

    def __len__(self) -> int:
        return self.close._cursor + 1

    def buflen(self) -> int:
        return len(self._frame)

    def _advance(self, cursor: int) -> None:
        for line in self._lines:
            line._advance(cursor)
            
    def _find_cursor(self, timestamp: int, current_cursor: int) -> int:
        # Re-implement cursor search logic
        cursor = current_cursor
        limit = len(self._frame)
        ts_values = self.datetime._values
        while cursor + 1 < limit and ts_values[cursor + 1] <= timestamp:
            cursor += 1
        return cursor
