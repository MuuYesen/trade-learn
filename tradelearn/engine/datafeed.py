from __future__ import annotations

import pandas as pd

from tradelearn.backtest.feed import RuntimeDataFeed
from tradelearn.engine.base import LineRoot, LineSeries


class DataFeed(RuntimeDataFeed, LineRoot):
    """Backtrader-style DataFeed with LineSeries magic."""
    lines = ("datetime", "open", "high", "low", "close", "volume")

    def __init__(self, data: pd.DataFrame, name: str | None = None, **kwargs) -> None:
        RuntimeDataFeed.__init__(self, data, name=name)
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
