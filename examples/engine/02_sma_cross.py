"""Standard SMA Crossover Strategy."""

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


class SmaCross(bt.Strategy):
    """
    Moving-average crossover strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Close when fast SMA crosses below slow SMA.
    """

    params = (
        ("fast", 10),
        ("slow", 30),
        ("size", 100),
    )

    def __init__(self) -> None:
        self.ma_fast = SMA(self.data.close, period=self.p.fast)
        self.ma_slow = SMA(self.data.close, period=self.p.slow)

    def next(self) -> None:
        # Skip if indicators are not yet ready (NaN)
        if self.ma_fast[0] != self.ma_fast[0] or self.ma_slow[0] != self.ma_slow[0]:
            return

        if not self.position:
            if self.ma_fast[0] > self.ma_slow[0]:
                self.buy(size=self.p.size)
        elif self.ma_fast[0] < self.ma_slow[0]:
            self.close()
