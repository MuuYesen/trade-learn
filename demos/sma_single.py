"""Single-symbol moving-average demo using the Engine API."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class DemoSmaCross(bt.Strategy):
    """Small custom-indicator SMA crossover demo."""

    def __init__(self) -> None:
        self.fast = bt.talib.SMA(self.data.close, timeperiod=4)
        self.slow = bt.talib.SMA(self.data.close, timeperiod=9)

    def next(self) -> None:
        fast = float(self.fast[0])
        slow = float(self.slow[0])
        if math.isnan(fast) or math.isnan(slow):
            return
        if not self.position and fast > slow:
            self.buy(size=5)
        elif self.position and fast < slow:
            self.close()


def _bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = 15.0 + np.linspace(0.0, 4.0, len(index)) + np.sin(np.linspace(0, 10, len(index)))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.full(len(index), 3_000.0),
        },
        index=index,
    )


def run_demo() -> dict[str, float | int | str]:
    """Run the single-symbol demo and return a compact summary."""

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(DemoSmaCross)
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
    print(run_demo())
