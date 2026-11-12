"""BT datetime read surface shared by preloaded and incremental Engine feeds."""

from __future__ import annotations

from array import array

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
        return array("d", (date2num(pd.Timestamp(v, unit="s")) for v in values))

    def _numeric_line(self):
        """Project BT ordinal days for Line algebra without changing raw clocks."""
        from tradelearn.backtest.clock import date2num
        from tradelearn.backtest.lines import LineSeries
        from tradelearn.backtest.event_runtime import RuntimeLine

        epoch = date2num(pd.Timestamp(0, unit="s"))
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            line = LineSeries(self._values / 86400.0 + epoch)
            line.min_period = self.min_period
            line._cursor_source = getattr(self, "_cursor_source", self)
            return line
        cached = getattr(self, "_bt_numeric_projection", None)
        if cached is not None and cached._runtime is runtime:
            return cached
        source = self if isinstance(self, RuntimeLine) else runtime.causal_line(self)
        line = RuntimeLine(
            source.feed,
            compute=lambda: ((source.full_series() if line.history == "full"
                              else source.to_series()) / 86400.0 + epoch),
            dependencies=(source,), min_period=source.min_period,
        )
        self._bt_numeric_projection = line
        return line

    def _math_op(self, other, op):
        target = other._numeric_line() if isinstance(other, BTDateLine) else other
        return self._numeric_line()._math_op(target, op)

    def __call__(self, ago=0):
        return self._numeric_line()(ago)

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
