"""Bounded example indicators preserving the previous pandas formulas.

EMA seeds from the first close (pandas adjust=False), rather than silently
switching the examples to TA-Lib's different EMA seed. TR uses a simple rolling
mean as before, not Wilder ATR. All inputs are delivered bars only.
"""

from collections import deque
from statistics import fmean, pstdev

from tradelearn.engine import Indicator


class EMA(Indicator):
    lines = ("ema",)
    params = (("period", 20),)

    def __init__(self):
        self.value = None
        self.alpha = 2.0 / (self.p.period + 1)

    def next(self):
        close = self.data[0]
        self.value = (
            close if self.value is None else self.alpha * close + (1 - self.alpha) * self.value
        )
        self.ema[0] = self.value


class Rolling(Indicator):
    lines = ("mean", "upper", "lower", "highest", "lowest")
    params = (("period", 20), ("deviation", 2.0))

    def __init__(self):
        self.values = deque(maxlen=self.p.period)

    def next(self):
        self.values.append(self.data[0])
        if len(self.values) < self.p.period:
            return
        mean = fmean(self.values)
        std = pstdev(self.values)
        self.mean[0] = mean
        self.upper[0] = mean + self.p.deviation * std
        self.lower[0] = mean - self.p.deviation * std
        self.highest[0] = max(self.values)
        self.lowest[0] = min(self.values)


class TrueRange(Indicator):
    lines = ("tr",)

    def next(self):
        previous = self.data.close[-1] if len(self.data.close) > 1 else self.data.close[0]
        self.tr[0] = max(
            self.data.high[0] - self.data.low[0],
            abs(self.data.high[0] - previous),
            abs(self.data.low[0] - previous),
        )


class MACD(Indicator):
    lines = ("dif", "dea")
    params = (("fast", 12), ("slow", 26), ("signal", 9))

    def __init__(self):
        self.fast = EMA(self.data, period=self.p.fast)
        self.slow = EMA(self.data, period=self.p.slow)
        self.dif_line = self.fast - self.slow
        self.signal = EMA(self.dif_line, period=self.p.signal)
        self.lines.dif = self.dif_line
        self.lines.dea = self.signal.ema
