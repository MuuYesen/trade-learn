from __future__ import annotations

import pandas as pd

from tradelearn.backtest.feed import RuntimeDataFeed
from tradelearn.backtest._bt_datetime import BTDateLine
from tradelearn.engine.base import LineRoot, LineSeries


class _DateLine(BTDateLine, LineSeries):
    """BT view on unchanged Unix storage."""


class DataFeed(RuntimeDataFeed, LineRoot):
    """Backtrader-style DataFeed with LineSeries magic."""
    _reset_line_aliases = True
    _bt_line_views = True
    lines = ("close", "low", "high", "open", "volume", "openinterest", "datetime")

    @property
    def _indicator_line(self):
        return self.close

    def __getitem__(self, ago):
        return self.close[ago]

    def __init__(self, data: pd.DataFrame, name: str | None = None, **kwargs) -> None:
        RuntimeDataFeed.__init__(self, data, name=name)
        self.datetime = _DateLine(self._datetime, is_datetime=True,
                                  buffer=self._bar_buffer, buffer_name="datetime")
        self.datetime._cursor_source = self
        self._lines_list[0] = self.datetime
        LineRoot._base_init(self, **kwargs)
        self.lines.datetime = self.datetime
        self.lines.open = self.open
        self.lines.high = self.high
        self.lines.low = self.low
        self.lines.close = self.close
        self.lines.volume = self.volume

        self.datetime = self.lines.datetime
        self.open = self.lines.open
        self.high = self.lines.high
        self.low = self.lines.low
        self.close = self.lines.close
        self.volume = self.lines.volume
