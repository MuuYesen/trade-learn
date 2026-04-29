"""Tradelearn 1.x-style SMA crossover strategy for the Lite API."""

from __future__ import annotations

from tradelearn.lite import Strategy


def crossover(series1, series2) -> bool:
    return series1[-1] < series2[-1] and series1[0] > series2[0]


class OneXSmaCross(Strategy):
    fast = 10
    slow = 20

    def init(self) -> None:
        price = self.data.close.df
        self.ma1 = self.I(lambda arr, n: arr.rolling(n).mean(), price, self.fast, overlay=True)
        self.ma2 = self.I(lambda arr, n: arr.rolling(n).mean(), price, self.slow, overlay=True)

    def next(self) -> None:
        if len(self.ma1) < 2:
            return
        if crossover(self.ma1, self.ma2):
            self.position().close()
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position().close()
            self.sell()
