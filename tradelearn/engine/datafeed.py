from __future__ import annotations

import pandas as pd

from tradelearn.backtest.feed import RuntimeDataFeed
from tradelearn.engine.base import LineRoot, LineSeries


class _DateLine(LineSeries):
    """Convert only the BT public surface; raw storage remains Unix seconds."""
    def __getitem__(self, ago):
        value = super().__getitem__(ago)
        from tradelearn.engine import date2num
        import numpy as np
        if isinstance(ago, slice):
            return np.asarray([date2num(pd.Timestamp(v, unit="s")) for v in value])
        return float("nan") if value is None else date2num(value)

    def get(self, ago=0, size=1):
        from tradelearn.engine import date2num
        import numpy as np
        return np.asarray([date2num(pd.Timestamp(v,unit="s")) for v in super().get(ago,size)])

    def datetime(self, ago=0, tz=None, naive=True):
        from tradelearn.engine import num2date
        value = self[ago]
        return None if pd.isna(value) else num2date(value,tz=tz,naive=naive)


class DataFeed(RuntimeDataFeed, LineRoot):
    """Backtrader-style DataFeed with LineSeries magic."""
    _reset_line_aliases = True
    lines = ("close", "low", "high", "open", "volume", "openinterest", "datetime")

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
