"""Runnable quickstart smoke example for the public Engine API."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class QuickstartSmaCross(bt.Strategy):
    """Minimal moving-average crossover strategy."""

    def __init__(self) -> None:
        self.fast = bt.talib.SMA(self.data.close, timeperiod=3)
        self.slow = bt.talib.SMA(self.data.close, timeperiod=6)

    def next(self) -> None:
        fast = float(self.fast[0])
        slow = float(self.slow[0])
        if math.isnan(fast) or math.isnan(slow):
            return
        if not self.position and fast > slow:
            self.buy(size=10)
        elif self.position and fast < slow:
            self.close()


def _sample_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    base = np.linspace(10.0, 14.0, len(index))
    wave = np.sin(np.linspace(0.0, 8.0, len(index))) * 0.8
    close = base + wave
    open_ = close - 0.05
    high = close + 0.4
    low = close - 0.4
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )


def run_quickstart() -> dict[str, float | int | str]:
    """Run the quickstart strategy and return a compact result summary."""

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_sample_bars(), name="quickstart")
    cerebro.addstrategy(QuickstartSmaCross)
    [strategy] = cerebro.run()
    summary = strategy.stats.summary
    final_value = float(summary["final_value"])
    return {
        "strategy": strategy.__class__.__name__,
        "bars": int(summary["bars"]),
        "fills": int(summary["total_fills"]),
        "final_value": final_value,
        "return_pct": (final_value / 100_000.0 - 1.0) * 100.0,
    }


if __name__ == "__main__":
    print(run_quickstart())
