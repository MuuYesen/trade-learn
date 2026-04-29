"""Throughput benchmark for Tradelearn user-facing APIs.

This runner measures end-to-end bars/s for:

- ``tradelearn.engine``: Backtrader-style high-level API.
- ``tradelearn.lite``: Tradelearn 1.x-style lightweight API.
- ``backtrader``: external reference API.

It intentionally does not import or compare against ``backtesting.py``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

REFERENCE_BASELINE_BARS_PER_SEC = 1_682
REFERENCE_TARGET_BARS_PER_SEC = 419_552


@dataclass(frozen=True)
class ThroughputResult:
    name: str
    elapsed_s: float
    bars_per_sec: float
    final_value: float
    fills: int
    closed_trades: int

    @property
    def vs_reference_baseline(self) -> float:
        return self.bars_per_sec / REFERENCE_BASELINE_BARS_PER_SEC

    @property
    def target_pct(self) -> float:
        return self.bars_per_sec / REFERENCE_TARGET_BARS_PER_SEC * 100


def make_data(n_bars: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n_bars, freq="1min", tz="UTC")
    x = np.arange(n_bars, dtype=float)
    close = 100.0 + np.sin(x / 17.0) * 2.0 + x * 0.0001
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": np.full(n_bars, 1000.0),
        },
        index=index,
    )


def _timed(
    factory: Callable[[], tuple[float, int, int]],
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, int, int, float]:
    for _ in range(max(0, warmup)):
        factory()
    timings: list[float] = []
    final_value = 0.0
    fills = 0
    closed_trades = 0
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        final_value, fills, closed_trades = factory()
        timings.append(time.perf_counter() - start)
    return final_value, fills, closed_trades, statistics.median(timings)


def run_engine(data: pd.DataFrame, *, trade_on_close: bool = False) -> tuple[float, int, int]:
    import tradelearn.engine as bt

    class SMA(bt.Indicator):
        lines = ("sma",)
        params = (("period", 30),)

        def __init__(self) -> None:
            line = self.data.close if hasattr(self.data, "close") else self.data
            self.lines.sma = bt.talib.SMA(line, timeperiod=self.p.period)

    class EngineSmaCross(bt.Strategy):
        params = (("period", 20),)

        def __init__(self) -> None:
            self.sma = SMA(self.data.close, period=self.p.period)

        def next(self) -> None:
            if self.sma[0] != self.sma[0]:
                return
            if not self.position and self.data.close[0] > self.sma[0]:
                self.buy(size=1)
            elif self.position and self.data.close[0] < self.sma[0]:
                self.close()

    cerebro = bt.Cerebro(trade_on_close=trade_on_close)
    cerebro.adddata(bt.feeds.PandasData(dataname=data, name="throughput"))
    cerebro.addstrategy(EngineSmaCross)
    cerebro.broker.setcash(100_000.0)
    [strategy] = cerebro.run()
    fills = len(strategy.stats.fills) if strategy.stats is not None else 0
    closed_trades, _wins = cerebro.broker.trade_summary()
    return float(cerebro.broker.getvalue()), fills, int(closed_trades)


def run_lite(data: pd.DataFrame, *, trade_on_close: bool = False) -> tuple[float, int, int]:
    import tradelearn as tl
    from tradelearn.lite import Backtest, Strategy

    class LiteSmaCross(Strategy):
        period = 20

        def init(self) -> None:
            self.sma = tl.talib.SMA(self.data.close, timeperiod=self.period)

        def next(self) -> None:
            if self.sma[0] != self.sma[0]:
                return
            if not self.position() and self.data.close[0] > self.sma[0]:
                self.buy(size=1)
            elif self.position() and self.data.close[0] < self.sma[0]:
                self.position().close()

    backtest = Backtest(data, LiteSmaCross, cash=100_000.0, trade_on_close=trade_on_close)
    stats = backtest.run()
    fills = len(backtest.broker.fills_frame())
    return float(stats["Equity Final [$]"]), fills, int(stats["# Trades"])


def run_backtrader(data: pd.DataFrame) -> tuple[float, int, int]:
    import backtrader as bt

    class PandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", None),
        )

    class BacktraderSmaCross(bt.Strategy):
        params = (("period", 20),)

        def __init__(self) -> None:
            self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)
            self.fills = 0
            self.closed_trades = 0

        def notify_order(self, order) -> None:
            if order.status == order.Completed:
                self.fills += 1

        def notify_trade(self, trade) -> None:
            if trade.isclosed:
                self.closed_trades += 1

        def next(self) -> None:
            if self.sma[0] != self.sma[0]:
                return
            if not self.position and self.data.close[0] > self.sma[0]:
                self.buy(size=1)
            elif self.position and self.data.close[0] < self.sma[0]:
                self.close()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.adddata(PandasData(dataname=data))
    cerebro.addstrategy(BacktraderSmaCross)
    [strategy] = cerebro.run()
    return float(cerebro.broker.getvalue()), int(strategy.fills), int(strategy.closed_trades)


def run_benchmark(
    *,
    n_bars: int = 50_000,
    repeat: int = 1,
    warmup: int = 0,
    include_backtrader: bool = True,
) -> list[ThroughputResult]:
    data = make_data(n_bars)
    runners: list[tuple[str, Callable[[pd.DataFrame], tuple[float, int, int]]]] = [
        ("Tradelearn Engine", run_engine),
        ("Tradelearn Lite", run_lite),
    ]
    if include_backtrader:
        runners.append(("Backtrader", run_backtrader))

    results: list[ThroughputResult] = []
    for name, runner in runners:
        final_value, fills, closed_trades, elapsed_s = _timed(
            lambda runner=runner: runner(data.copy()),
            warmup=warmup,
            repeat=repeat,
        )
        results.append(
            ThroughputResult(
                name=name,
                elapsed_s=elapsed_s,
                bars_per_sec=n_bars / elapsed_s if elapsed_s > 0 else 0.0,
                final_value=final_value,
                fills=fills,
                closed_trades=closed_trades,
            )
        )
    return results


def print_results(
    results: list[ThroughputResult],
    *,
    n_bars: int,
    repeat: int,
    warmup: int,
) -> None:
    print("\nThroughput Benchmark")
    print(f"Bars: {n_bars:,} | repeat={repeat} | warmup={warmup}")
    print(
        f"Reference: {REFERENCE_BASELINE_BARS_PER_SEC:,} bars/s -> "
        f"{REFERENCE_TARGET_BARS_PER_SEC:,} bars/s "
        f"({REFERENCE_TARGET_BARS_PER_SEC / REFERENCE_BASELINE_BARS_PER_SEC:.1f}x)"
    )
    print("=" * 116)
    print(
        f"{'Engine':<18} | {'Time [s]':>9} | {'Bars/s':>12} | "
        f"{'vs 1,682':>10} | {'vs 419,552':>11} | {'Final Value':>12} | "
        f"{'Fills':>7} | {'Closed Trades':>13}"
    )
    print("-" * 116)
    for result in results:
        print(
            f"{result.name:<18} | {result.elapsed_s:>9.4f} | "
            f"{result.bars_per_sec:>12,.0f} | "
            f"{result.vs_reference_baseline:>9.1f}x | "
            f"{result.target_pct:>10.1f}% | "
            f"{result.final_value:>12.2f} | "
            f"{result.fills:>7} | {result.closed_trades:>13}"
        )
    print("=" * 116)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-backtrader", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_benchmark(
        n_bars=args.bars,
        repeat=args.repeat,
        warmup=args.warmup,
        include_backtrader=not args.no_backtrader,
    )
    print_results(results, n_bars=args.bars, repeat=args.repeat, warmup=args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
