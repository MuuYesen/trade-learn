"""Five-minute quickstart example for a backtrader-style strategy."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

import tradelearn.compat.backtrader as bt


class QuickstartSmaCross(bt.Strategy):
    """Minimal moving-average crossover strategy."""

    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(self.data.close, period=3)
        self.slow = bt.indicators.SMA(self.data.close, period=6)

    def next(self) -> None:
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=10)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


def run_quickstart() -> dict[str, Any]:
    """Run the quickstart strategy and return a compact result summary."""
    bars = _sample_bars()
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars, name="demo"))
    cerebro.addstrategy(QuickstartSmaCross)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("quickstart strategy did not produce stats")

    summary = strategy.stats.summary
    return {
        "strategy": QuickstartSmaCross.__name__,
        "bars": len(bars),
        "fills": len(strategy.stats.fills),
        "final_value": float(summary["final_value"]),
        "return_pct": float(strategy.stats.returns.add(1.0).prod() - 1.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the trade-learn quickstart strategy.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    result = run_quickstart()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "QuickstartSmaCross "
            f"bars={result['bars']} fills={result['fills']} "
            f"final_value={result['final_value']:.2f} return_pct={result['return_pct']:.4f}"
        )
    return 0


def _sample_bars() -> pd.DataFrame:
    close = [
        10.0,
        10.3,
        10.1,
        10.8,
        11.4,
        10.7,
        9.9,
        9.4,
        10.2,
        11.1,
        12.0,
        11.2,
        10.4,
        9.8,
        10.5,
        11.5,
        12.6,
        11.7,
        10.8,
        10.0,
        10.9,
        12.0,
        13.1,
        12.2,
        11.1,
        10.3,
        11.3,
        12.5,
        13.8,
        12.6,
    ]
    return pd.DataFrame(
        {
            "open": [value - 0.2 for value in close],
            "high": [value + 0.6 for value in close],
            "low": [value - 0.7 for value in close],
            "close": close,
            "volume": [1000.0 + index * 25.0 for index in range(len(close))],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
