"""Migration Example: Legacy SMA strategy migrated to 2.0 API."""

from __future__ import annotations
import tradelearn.engine as bt

class MigratedSmaCross(bt.Strategy):
    """Backtrader-style replacement for a legacy moving-average strategy."""

    params = (("size", 1),)

    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=3)
        self.slow = bt.indicators.SMA(self.data.close, period=5)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=self.p.size)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()
