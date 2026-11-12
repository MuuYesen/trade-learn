"""BT datetime read surface shared by preloaded and incremental Engine feeds."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BTDateLine:
    """Convert the user view only; backing arrays and event clocks remain Unix."""

    def __getitem__(self, ago):
        from tradelearn.backtest.clock import date2num

        value = super().__getitem__(ago)
        if isinstance(ago, slice):
            return np.asarray([date2num(pd.Timestamp(v, unit="s")) for v in value])
        return float("nan") if value is None or pd.isna(value) else date2num(value)

    def get(self, ago=0, size=1):
        from tradelearn.backtest.clock import date2num

        values = super().get(ago, size)
        return np.asarray([date2num(pd.Timestamp(v, unit="s")) for v in values])

    def datetime(self, ago=0, tz=None, naive=True):
        from tradelearn.backtest.clock import num2date

        value = self[ago]
        return None if pd.isna(value) else num2date(value, tz=tz, naive=naive)

    def date(self, ago=0, tz=None, naive=True):
        value = self.datetime(ago=ago, tz=tz, naive=naive)
        return None if value is None else value.date()

    def time(self, ago=0, tz=None, naive=True):
        value = self.datetime(ago=ago, tz=tz, naive=naive)
        return None if value is None else value.time()
