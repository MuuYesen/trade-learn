"""Lite Bollinger band strategy for the Lite API."""

from __future__ import annotations

import tradelearn.lite as tl


def crossover(series1, series2) -> bool:
    return series1[-1] < series2[-1] and series1[0] > series2[0]


class LiteBollBandCross(tl.Strategy):
    title = "Long"
    n = 20
    p = 2

    def init(self) -> None:
        price = self.data.close.df
        mid = price.rolling(self.n).mean()
        std = price.rolling(self.n).std(ddof=0)
        upper = mid + self.p * std
        lower = mid - self.p * std

        self.upper = self.I(upper, name="upper", overlay=True)
        self.mid = self.I(mid, name="mid", overlay=True)
        self.lower = self.I(lower, name="lower", overlay=True)

    def next(self) -> None:
        if len(self.upper) < 2:
            return

        position = self.position()
        if self.title in ("Long", "Long&Short"):
            if crossover(self.data.close, self.upper) and position.size == 0:
                self.buy()
            if crossover(self.mid, self.data.close) and position.size > 0:
                position.close()

        if self.title in ("Short", "Long&Short"):
            if crossover(self.lower, self.data.close) and position.size == 0:
                self.sell()
            if crossover(self.data.close, self.mid) and position.size < 0:
                position.close()
