"""Standard SMA Crossover Strategy."""

from __future__ import annotations
import tradelearn.compat.backtrader as bt

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
        # Use compat indicators which mirror backtrader API
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow)

    def next(self) -> None:
        # Skip if indicators are not yet ready (NaN)
        if self.ma_fast[0] != self.ma_fast[0] or self.ma_slow[0] != self.ma_slow[0]:
            return

        if not self.position:
            if self.ma_fast[0] > self.ma_slow[0]:
                self.buy(size=self.p.size)
        elif self.ma_fast[0] < self.ma_slow[0]:
            self.close()
