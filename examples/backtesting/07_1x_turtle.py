"""Tradelearn 1.x-style turtle strategy for the Lite API."""

from __future__ import annotations

import pandas as pd

from tradelearn.lite import Strategy


class OneXAdvancedTurtle(Strategy):
    title = "Long"
    entry_window = 20
    exit_window = 10
    atr_window = 20
    max_units = 4

    def init(self) -> None:
        self.high_20 = self.I(self.data.high.df.rolling(self.entry_window).max())
        self.low_20 = self.I(self.data.low.df.rolling(self.entry_window).min())
        self.high_10 = self.I(self.data.high.df.rolling(self.exit_window).max())
        self.low_10 = self.I(self.data.low.df.rolling(self.exit_window).min())
        self.atr = self.I(_atr(self.data.high.df, self.data.low.df, self.data.close.df, self.atr_window))
        self.units = 0
        self.last_price = 0.0

    def _execute_trade(self, is_buy: bool, close: float, atr: float, size_param: float) -> None:
        if is_buy:
            self.buy(size=size_param)
        else:
            self.sell(size=size_param)
        self.last_price = close
        self.units = self.units + 1 if self.position() else 1

    def next(self) -> None:
        if len(self.atr) < 2 or pd.isna(self.atr[0]) or self.atr[0] <= 0:
            return
        close = self.data.close[0]
        atr = self.atr[0]
        size_param = min(0.95, (self.equity * 0.01 / atr) * close / max(self.equity, 1.0))
        if size_param <= 0.005:
            return

        pos = self.position()
        if self.title == "Long":
            if pos:
                if close < self.low_10[-1]:
                    pos.close()
                    self.units = 0
                elif self.units < self.max_units and close > self.last_price + 0.5 * atr:
                    self._execute_trade(True, close, atr, size_param)
            elif close > self.high_20[-1]:
                self._execute_trade(True, close, atr, size_param)

        if self.title == "Short":
            if pos:
                if close > self.high_10[-1]:
                    pos.close()
                    self.units = 0
                elif self.units < self.max_units and close < self.last_price - 0.5 * atr:
                    self._execute_trade(False, close, atr, size_param)
            elif close < self.low_20[-1]:
                self._execute_trade(False, close, atr, size_param)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()
