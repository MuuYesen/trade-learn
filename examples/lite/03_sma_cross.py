"""Lite SMA crossover strategy for the Lite API."""

from __future__ import annotations

import tradelearn.lite as tl
from examples.lite._indicators import Rolling


def crossover(series1, series2) -> bool:
    return series1[-1] < series2[-1] and series1[0] > series2[0]


class LiteSmaCross(tl.Strategy):
    fast = 10
    slow = 20

    def init(self) -> None:
        self.ma1 = Rolling(self.data.close, period=self.fast).mean
        self.ma2 = Rolling(self.data.close, period=self.slow).mean

    def next(self) -> None:
        if len(self.ma1) < 2:
            return
        if crossover(self.ma1, self.ma2):
            self.position().close()
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position().close()
            self.sell()
