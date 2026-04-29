"""Quickstart Example: Minimal SMA strategy."""

from __future__ import annotations

import tradelearn.engine as bt


class SMA(bt.Indicator):
    """Simple moving average defined locally for this example."""

    lines = ("sma",)
    params = (("period", 30),)

    def __init__(self):
        if hasattr(self.data, "to_series"):
            values = self.data.to_series().rolling(self.p.period).mean()
            self.lines.sma = self.data.wrap_indicator(values, name="sma")
        else:
            self.addminperiod(self.p.period)

    def next(self):
        if not hasattr(self.data, "to_series"):
            self.lines.sma[0] = sum(self.data.get(size=self.p.period)) / self.p.period


class QuickstartSmaCross(bt.Strategy):
    """Minimal moving-average crossover strategy."""

    def __init__(self) -> None:
        self.fast = SMA(self.data.close, period=3)
        self.slow = SMA(self.data.close, period=6)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=10)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()
