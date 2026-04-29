from __future__ import annotations

import pandas as pd

from tradelearn.backtest.data import DataContainer
from tradelearn.compat.backtrader.base import LineRoot, LineSeries


class DataFeed(DataContainer, LineRoot):
    """Backtrader-style DataFeed with LineSeries magic."""
    lines = ("datetime", "open", "high", "low", "close", "volume")

    def __init__(self, data: pd.DataFrame, name: str | None = None, **kwargs) -> None:
        # Initialize Core DataContainer first
        DataContainer.__init__(self, data, name=name)
        # Initialize LineRoot (handles params and lines)
        LineRoot._base_init(self, **kwargs)
        buffer = self.shared_bar_buffer()
        
        # Wrap raw arrays in LineSeries
        self.lines.datetime = LineSeries(
            self._datetime,
            is_datetime=True,
            buffer=buffer,
            buffer_name="datetime",
        )
        self.lines.open = LineSeries(self._open, buffer=buffer, buffer_name="open")
        self.lines.high = LineSeries(self._high, buffer=buffer, buffer_name="high")
        self.lines.low = LineSeries(self._low, buffer=buffer, buffer_name="low")
        self.lines.close = LineSeries(self._close, buffer=buffer, buffer_name="close")
        self.lines.volume = LineSeries(self._volume, buffer=buffer, buffer_name="volume")

        self.datetime = self.lines.datetime
        self.open = self.lines.open
        self.high = self.lines.high
        self.low = self.lines.low
        self.close = self.lines.close
        self.volume = self.lines.volume
        
        self._lines_list = [self.datetime, self.open, self.high, self.low, self.close, self.volume]

    def __len__(self) -> int:
        return self._cursor + 1

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor
        for line in self._lines_list:
            line._cursor = cursor
