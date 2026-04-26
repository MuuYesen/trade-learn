"""Single-asset SMA crossover demo using the Stage 9 public API."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

import tradelearn.compat.backtrader as bt


class DemoSmaCross(bt.Strategy):
    """Small moving-average crossover strategy for the release demo."""

    params = (("fast", 4), ("slow", 9), ("size", 10))

    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=self.p.size)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


def run_demo() -> dict[str, Any]:
    """Run the single-asset SMA demo and return a compact summary."""
    bars = sample_bars()
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars, name="single-demo"))
    cerebro.addstrategy(DemoSmaCross)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("SMA demo did not produce stats")

    return {
        "strategy": DemoSmaCross.__name__,
        "bars": len(bars),
        "fills": len(strategy.stats.fills),
        "final_value": float(strategy.stats.summary["final_value"]),
        "return_pct": float(strategy.stats.returns.add(1.0).prod() - 1.0),
    }


def sample_bars() -> pd.DataFrame:
    """Deterministic OHLCV bars used by the offline demo and tests."""
    close = [
        20.0,
        20.4,
        20.2,
        20.9,
        21.5,
        22.1,
        22.6,
        21.9,
        21.1,
        20.5,
        21.0,
        21.8,
        22.7,
        23.6,
        24.4,
        23.5,
        22.6,
        21.7,
        22.4,
        23.3,
        24.5,
        25.6,
        26.4,
        25.2,
        24.0,
        23.1,
        23.8,
        24.9,
        26.1,
        27.0,
        25.9,
        24.7,
        23.8,
        24.4,
        25.5,
        26.8,
    ]
    return pd.DataFrame(
        {
            "open": [value - 0.2 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.6 for value in close],
            "close": close,
            "volume": [2000.0 + index * 30.0 for index in range(len(close))],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SMA crossover demo.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    result = run_demo()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "DemoSmaCross "
            f"bars={result['bars']} fills={result['fills']} "
            f"final_value={result['final_value']:.2f} "
            f"return_pct={result['return_pct']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
