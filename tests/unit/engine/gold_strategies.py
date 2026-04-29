from __future__ import annotations

import tradelearn.engine as bt


class SmaCross(bt.Strategy):
    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=3)
        self.slow = bt.indicators.SMA(self.data.close, period=6)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=1)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


class EmaTrend(bt.Strategy):
    def __init__(self) -> None:
        self.trend = bt.indicators.EMA(self.data.close, period=5)

    def next(self) -> None:
        if self.trend[0] != self.trend[0]:
            return
        if not self.position and self.data.close[0] > self.trend[0]:
            self.buy(size=1)
        elif self.position and self.data.close[0] < self.trend[0]:
            self.close()


class WeightedMaPullback(bt.Strategy):
    def __init__(self) -> None:
        self.wma = bt.indicators.WMA(self.data.close, period=4)

    def next(self) -> None:
        if self.wma[0] != self.wma[0]:
            return
        if not self.position and self.data.close[0] > self.wma[0]:
            self.buy(size=1)
        elif self.position and self.data.close[0] < self.wma[0]:
            self.close()


class RsiOversold(bt.Strategy):
    def __init__(self) -> None:
        self.rsi = bt.indicators.RSI(self.data.close, period=3)

    def next(self) -> None:
        if self.rsi[0] != self.rsi[0]:
            return
        if not self.position and self.rsi[0] < 45:
            self.buy(size=1)
        elif self.position and self.rsi[0] > 55:
            self.close()


class MacdCross(bt.Strategy):
    def __init__(self) -> None:
        self.macd = bt.indicators.MACD(self.data.close, fast=3, slow=6, signal=3)

    def next(self) -> None:
        if self.macd.macd[0] != self.macd.macd[0] or self.macd.signal[0] != self.macd.signal[0]:
            return
        if not self.position and self.macd.macd[0] > self.macd.signal[0]:
            self.buy(size=1)
        elif self.position and self.macd.macd[0] < self.macd.signal[0]:
            self.close()


class BollingerBreakout(bt.Strategy):
    def __init__(self) -> None:
        self.bands = bt.indicators.BollingerBands(self.data.close, period=5, devfactor=1.0)

    def next(self) -> None:
        if self.bands.top[0] != self.bands.top[0]:
            return
        if not self.position and self.data.close[0] > self.bands.mid[0]:
            self.buy(size=1)
        elif self.position and self.data.close[0] < self.bands.mid[0]:
            self.close()


class DonchianBreakout(bt.Strategy):
    def __init__(self) -> None:
        self.highest = bt.indicators.Highest(self.data.high, period=4)
        self.lowest = bt.indicators.Lowest(self.data.low, period=4)

    def next(self) -> None:
        try:
            previous_high = self.highest[-1]
            previous_low = self.lowest[-1]
        except IndexError:
            return
        if previous_high != previous_high or previous_low != previous_low:
            return
        if not self.position and self.data.close[0] > previous_high:
            self.buy(size=1)
        elif self.position and self.data.close[0] < previous_low:
            self.close()


class AtrRange(bt.Strategy):
    def __init__(self) -> None:
        self.atr = bt.indicators.ATR(self.data, period=3)
        self.sma = bt.indicators.SMA(self.data.close, period=3)

    def next(self) -> None:
        if self.atr[0] != self.atr[0] or self.sma[0] != self.sma[0]:
            return
        if not self.position and self.data.close[0] > self.sma[0] + self.atr[0] * 0.1:
            self.buy(size=1)
        elif self.position and self.data.close[0] < self.sma[0]:
            self.close()


class StochasticReversal(bt.Strategy):
    def __init__(self) -> None:
        self.stoch = bt.indicators.Stochastic(self.data, period=4, period_dfast=2)

    def next(self) -> None:
        if self.stoch.percK[0] != self.stoch.percK[0]:
            return
        if not self.position and self.stoch.percK[0] < 35:
            self.buy(size=1)
        elif self.position and self.stoch.percK[0] > 65:
            self.close()


class CrossOverSignal(bt.Strategy):
    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=2)
        self.slow = bt.indicators.SMA(self.data.close, period=5)
        self.cross = bt.indicators.CrossOver(self.fast, self.slow)

    def next(self) -> None:
        if not self.position and self.cross[0] > 0:
            self.buy(size=1)
        elif self.position and self.cross[0] < 0:
            self.close()


STRATEGIES = [
    SmaCross,
    EmaTrend,
    WeightedMaPullback,
    RsiOversold,
    MacdCross,
    BollingerBreakout,
    DonchianBreakout,
    AtrRange,
    StochasticReversal,
    CrossOverSignal,
]
