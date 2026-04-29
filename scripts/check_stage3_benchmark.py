from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradelearn.engine import Cerebro, Strategy


@dataclass(frozen=True)
class BenchmarkResult:
    bars: int
    symbols: int
    elapsed_ms: float
    max_ms: float

    @property
    def ok(self) -> bool:
        return self.elapsed_ms <= self.max_ms

    def asdict(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "symbols": self.symbols,
            "elapsed_ms": self.elapsed_ms,
            "max_ms": self.max_ms,
            "ok": self.ok,
        }


class NoopStrategy(Strategy):
    def next(self) -> None:
        _ = self.data.close[0]


def build_bars(count: int, *, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2016-01-01", periods=count, freq="B", tz="UTC")
    base = pd.Series(range(count), dtype="float64") + 100.0 + offset
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": 1000.0 + base,
        },
        index=index,
    )


def run_case(*, bars: int, symbols: int, max_ms: float) -> BenchmarkResult:
    cerebro = Cerebro(stdstats=False)
    for symbol_index in range(symbols):
        cerebro.adddata(build_bars(bars, offset=float(symbol_index)), name=f"S{symbol_index:04d}")
    cerebro.addstrategy(NoopStrategy)

    start = time.perf_counter()
    cerebro.run()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BenchmarkResult(bars=bars, symbols=symbols, elapsed_ms=elapsed_ms, max_ms=max_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 backtest benchmark gates.")
    parser.add_argument("--single-bars", type=int, default=2520)
    parser.add_argument("--portfolio-bars", type=int, default=2520)
    parser.add_argument("--portfolio-symbols", type=int, default=500)
    parser.add_argument("--max-single-ms", type=float, default=50.0)
    parser.add_argument("--max-portfolio-ms", type=float, default=5000.0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    single = run_case(bars=args.single_bars, symbols=1, max_ms=args.max_single_ms)
    portfolio = run_case(
        bars=args.portfolio_bars,
        symbols=args.portfolio_symbols,
        max_ms=args.max_portfolio_ms,
    )
    payload = {
        "ok": single.ok and portfolio.ok,
        "single": single.asdict(),
        "portfolio": portfolio.asdict(),
    }

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "stage3-benchmark:"
            f" single={single.elapsed_ms:.3f}ms/{single.max_ms:.3f}ms"
            f" portfolio={portfolio.elapsed_ms:.3f}ms/{portfolio.max_ms:.3f}ms"
            f" ok={payload['ok']}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
