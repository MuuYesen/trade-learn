"""Lite MACD crossover strategy for the Lite API."""

from __future__ import annotations

import pandas as pd

import tradelearn.lite as tl


def crossover(series1, series2) -> bool:
    return series1[-1] < series2[-1] and series1[0] > series2[0]


class LiteMACDCross(tl.Strategy):
    title = "Long"
    s = 12
    l = 26
    m = 9

    def init(self) -> None:
        price = self.data.close.df
        fast = price.ewm(span=self.s, adjust=False).mean()
        slow = price.ewm(span=self.l, adjust=False).mean()
        dif = fast - slow
        dea = dif.ewm(span=self.m, adjust=False).mean()
        macd = pd.DataFrame({"DIF": dif, "DEA": dea}, index=self.data.index)
        self.macd = self.I(macd, name="macd", overlay=False)

    def next(self) -> None:
        if len(self.macd) < 2:
            return

        dif = self.macd[:, 0]
        dea = self.macd[:, 1]

        if self.title == "Long":
            if crossover(dea, dif):
                self.position().close()
            if crossover(dif, dea):
                self.buy()

        if self.title == "Short":
            if crossover(dif, dea):
                self.position().close()
            if crossover(dea, dif):
                self.sell()

        if self.title == "Long&Short":
            if crossover(dea, dif):
                self.position().close()
                self.sell()
            if crossover(dif, dea):
                self.position().close()
                self.buy()
