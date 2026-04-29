"""backtesting.py facade feature-surface strategy.

This file is intentionally a strategy-only reference.  It does not run a
backtest or load data; use it to inspect the currently supported
``tradelearn.compat.backtesting`` calling style.
"""

from __future__ import annotations

import numpy as np

from tradelearn import ta
from tradelearn.compat.backtesting import Strategy


class BacktestingFeatureSurfaceStrategy(Strategy):
    """Reference strategy covering the main backtesting.py-compatible calls."""

    fast = 10
    slow = 30
    rsi_length = 14
    risk_size = 0.50
    limit_offset = 0.01
    stop_offset = 0.01

    def init(self) -> None:
        # OHLCV proxies use backtesting.py's capitalized names.
        self.open = self.data.Open
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.Close
        self.volume = self.data.Volume

        # Strategy.I(...) is the backtesting.py facade indicator entry.
        # FunctionIndicator objects from tradelearn.ta are accepted directly.
        self.fast_sma = self.I(ta.sma, self.close, period=self.fast)
        self.slow_sma = self.I(ta.sma, self.close, period=self.slow)
        self.ema = self.I(ta.ema, self.close, length=self.fast)
        self.rsi = self.I(ta.rsi, self.close, length=self.rsi_length)
        self.macd = self.I(lambda close: ta.macd(close)["macd"], self.close)
        self.atr = self.I(ta.atr, self.high, self.low, self.close, length=14)
        self.vwap = self.I(ta.vwap, self.high, self.low, self.close, self.volume)

        # data.ta mirrors the backtesting.py data accessor and delegates to
        # pandas-ta-classic through the current compat facade.
        self.roc = self.I(lambda: self.data.ta.roc(5))

    def next(self) -> None:
        price = float(self.close[-1])
        fast = float(self.fast_sma[-1])
        slow = float(self.slow_sma[-1])
        rsi = float(self.rsi[-1])
        atr = float(self.atr[-1])

        if any(np.isnan(value) for value in (fast, slow, rsi, atr)):
            return

        # Position proxy supports truthiness, .size, and .close().
        if self.position:
            if fast < slow or rsi > 75:
                self.position.close()
            return

        # Market order using fractional sizing (0 < size < 1 means equity pct).
        if fast > slow and rsi < 65:
            self.buy(size=self.risk_size)
            return

        # Limit / stop order shapes are supported by the facade.
        if fast > self.ema[-1]:
            self.buy(size=0.25, limit=price * (1 - self.limit_offset))
        elif fast < self.ema[-1]:
            self.sell(size=0.25, stop=price * (1 - self.stop_offset))
