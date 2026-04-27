"""Quickstart Example: Minimal SMA strategy."""

from __future__ import annotations
import tradelearn.compat.backtrader as bt

class QuickstartSmaCross(bt.Strategy):
    """Minimal moving-average crossover strategy."""

    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=3)
        self.slow = bt.indicators.SMA(self.data.close, period=6)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=10)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()
