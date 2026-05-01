from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradelearn.lite import Backtest, Strategy


@dataclass(frozen=True)
class StageTiming:
    name: str
    seconds: float


@dataclass
class TargetWeightsProfile:
    signal_rank_seconds: float = 0.0
    target_weights_seconds: float = 0.0
    order_submit_seconds: float = 0.0
    stats_read_seconds: float = 0.0
    rebalance_count: int = 0
    order_count: int = 0


def _make_panel(symbols: int, bars: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=bars, freq="D", tz="UTC")
    panel: dict[str, pd.DataFrame] = {}
    for i in range(symbols):
        returns = rng.normal(loc=0.0002, scale=0.01, size=bars)
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = close * (1.0 + rng.normal(0.0, 0.001, size=bars))
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        volume = rng.integers(10_000, 100_000, size=bars).astype(float)
        panel[f"S{i:04d}"] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
    return panel


def _top_equal_weights(strategy: Strategy, count: int) -> dict[str, float]:
    scores: list[tuple[float, str]] = []
    tickers = getattr(strategy, "_bt_data_by_ticker", None) or {
        strategy.data.the_ticker: strategy._bt_primary_data
    }
    for ticker in tickers:
        data = strategy._resolve_ticker_data(ticker)
        close = data.get_array("close")
        cursor = data._cursor
        if cursor < 20:
            continue
        momentum = float(close[cursor] / close[cursor - 20] - 1.0)
        scores.append((momentum, ticker))
    selected = [ticker for _score, ticker in sorted(scores, reverse=True)[:count]]
    if not selected:
        return {}
    weight = 0.95 / len(selected)
    return {ticker: weight for ticker in selected}


def _measure_order_submit(strategy: Strategy, profile: TargetWeightsProfile) -> dict[str, object]:
    broker = strategy.broker
    originals: dict[str, object] = {}

    for name in ("submit_basic", "_submit"):
        method = getattr(broker, name, None)
        if not callable(method):
            continue
        originals[name] = method

        def _wrapped(*args, _method=method, **kwargs):
            start = time.perf_counter()
            try:
                return _method(*args, **kwargs)
            finally:
                profile.order_submit_seconds += time.perf_counter() - start
                profile.order_count += 1

        setattr(broker, name, _wrapped)

    return originals


def _restore_order_submit(strategy: Strategy, originals: dict[str, object]) -> None:
    for name, method in originals.items():
        setattr(strategy.broker, name, method)


def _strategy_cls(
    rebalance_every: int,
    holdings: int,
    profile: TargetWeightsProfile,
) -> type[Strategy]:
    class TargetWeightsBenchmarkStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(20)

        def next(self) -> None:
            if (len(self.data) - 1) % rebalance_every:
                return
            profile.rebalance_count += 1

            start = time.perf_counter()
            weights = _top_equal_weights(self, holdings)
            profile.signal_rank_seconds += time.perf_counter() - start

            originals = _measure_order_submit(self, profile)
            start = time.perf_counter()
            try:
                self.target_weights(
                    weights,
                    close_missing=True,
                )
            finally:
                profile.target_weights_seconds += time.perf_counter() - start
                _restore_order_submit(self, originals)

    return TargetWeightsBenchmarkStrategy


def run_benchmark(
    *,
    symbols: int,
    bars: int,
    holdings: int,
    rebalance_every: int,
    cash: float,
    seed: int,
) -> tuple[list[StageTiming], pd.Series, TargetWeightsProfile]:
    profile = TargetWeightsProfile()
    start = time.perf_counter()
    panel = _make_panel(symbols, bars, seed)
    data_seconds = time.perf_counter() - start

    start = time.perf_counter()
    backtest = Backtest(
        panel,
        _strategy_cls(rebalance_every=rebalance_every, holdings=holdings, profile=profile),
        cash=cash,
        match_mode="exact",
    )
    init_seconds = time.perf_counter() - start

    start = time.perf_counter()
    stats = backtest.run()
    run_seconds = time.perf_counter() - start

    start = time.perf_counter()
    _ = float(stats["final_value"])
    _ = int(stats["total_trades"])
    _ = stats.strategy
    profile.stats_read_seconds = time.perf_counter() - start

    return [
        StageTiming("data_generate", data_seconds),
        StageTiming("init_runner_feed", init_seconds),
        StageTiming("run_loop", run_seconds),
        StageTiming("stats_read", profile.stats_read_seconds),
        StageTiming(
            "total",
            data_seconds + init_seconds + run_seconds + profile.stats_read_seconds,
        ),
    ], stats, profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Lite target_weights portfolio path.")
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--bars", type=int, default=240)
    parser.add_argument("--holdings", type=int, default=50)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    timings, stats, profile = run_benchmark(
        symbols=args.symbols,
        bars=args.bars,
        holdings=args.holdings,
        rebalance_every=args.rebalance_every,
        cash=args.cash,
        seed=args.seed,
    )
    total_data_bars = args.symbols * args.bars
    total = next(t.seconds for t in timings if t.name == "total")

    print("TargetWeights Benchmark")
    print(f"symbols={args.symbols} bars={args.bars} holdings={args.holdings}")
    print(f"rebalance_every={args.rebalance_every} total_data_bars={total_data_bars:,}")
    for timing in timings:
        print(f"{timing.name:>16}: {timing.seconds:.4f}s")
    print("Profile")
    print(f"{'rebalance_count':>16}: {profile.rebalance_count:,}")
    print(f"{'signal_rank':>16}: {profile.signal_rank_seconds:.6f}s")
    print(f"{'target_weights':>16}: {profile.target_weights_seconds:.6f}s")
    print(f"{'order_submit':>16}: {profile.order_submit_seconds:.6f}s")
    print(f"{'stats_read':>16}: {profile.stats_read_seconds:.6f}s")
    print(f"{'profiled_orders':>16}: {profile.order_count:,}")
    print(f"{'bars/s':>16}: {total_data_bars / total:,.0f}")
    print(f"{'final_value':>16}: {float(stats['final_value']):,.2f}")
    print(f"{'trades':>16}: {int(stats['total_trades'])}")
    strategy = stats.strategy
    print(f"{'orders':>16}: {len(strategy.orders):,}")


if __name__ == "__main__":
    main()
