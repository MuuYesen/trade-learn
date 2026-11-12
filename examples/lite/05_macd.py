"""Lite MACD crossover strategy for the Lite API."""

from __future__ import annotations

import tradelearn.lite as tl
from examples.lite._indicators import MACD


def crossover(series1, series2) -> bool:
    return series1[-1] < series2[-1] and series1[0] > series2[0]


class LiteMACDCross(tl.Strategy):
    title = "Long"
    s = 12
    l = 26  # noqa: E741 - retain the documented strategy parameter
    m = 9

    def init(self) -> None:
        self.macd = MACD(self.data.close, fast=self.s, slow=self.l, signal=self.m)

    def next(self) -> None:
        if len(self.macd) < 2:
            return

        dif = self.macd.dif
        dea = self.macd.dea

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
