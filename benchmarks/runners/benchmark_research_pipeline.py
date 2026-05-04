from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import tradelearn.lite as tl
import tradelearn.research as research
import tradelearn.research.portfolio as pf
import tradelearn.research.preprocess as pp
from tradelearn.report.artifacts import write_artifact_bundle


@dataclass(frozen=True)
class StageTiming:
    name: str
    seconds: float


@dataclass(frozen=True)
class ResearchPipelineBenchmarkResult:
    total_bars: int
    final_value: float
    return_pct: float
    total_trades: int
    weight_dates: int
    report_path: Path | None
    artifact_count: int
    segments: list[StageTiming]

    @property
    def elapsed(self) -> float:
        return next(segment.seconds for segment in self.segments if segment.name == "total")

    @property
    def bars_per_sec(self) -> float:
        return self.total_bars / self.elapsed if self.elapsed else 0.0


class ResearchTargetWeightStrategy(tl.Strategy):
    """Execute precomputed research weights through Lite target_weights()."""

    lookback = 20
    rebalance_every = 20

    def init(self) -> None:
        self.start_on_bar(self.lookback + 1)

    def next(self) -> None:
        if len(self.data) % self.rebalance_every != 0:
            return
        self.target_weights(self.research_result.weights[0], close_missing=True)


def make_panel(*, symbols: int, bars: int, seed: int) -> pd.DataFrame:
    """Return a synthetic OHLCV MultiIndex(timestamp, symbol) panel."""

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2010-01-01", periods=bars, freq="B", tz="UTC")
    frames: list[pd.DataFrame] = []
    for i in range(symbols):
        symbol = f"S{i:04d}"
        drift = 0.00015 + (i % 7) * 0.00001
        returns = rng.normal(loc=drift, scale=0.012, size=bars)
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = close * (1.0 + rng.normal(0.0, 0.0015, size=bars))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, size=bars))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, size=bars))
        volume = rng.integers(50_000, 500_000, size=bars).astype(float)
        index = pd.MultiIndex.from_product(
            [timestamps, [symbol]],
            names=["timestamp", "symbol"],
        )
        frames.append(
            pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                },
                index=index,
            )
        )
    return pd.concat(frames).sort_index()


def run_benchmark(
    *,
    symbols: int = 50,
    bars: int = 240,
    holdings: int = 10,
    lookback: int = 20,
    rebalance_every: int = 20,
    split_ratio: float = 0.65,
    cash: float = 1_000_000.0,
    commission: float = 0.0003,
    seed: int = 7,
    write_report: bool = True,
    output_dir: str | Path | None = None,
) -> ResearchPipelineBenchmarkResult:
    """Run a light segmented Stage 12 research pipeline benchmark."""

    total_start = time.perf_counter()
    segments: list[StageTiming] = []

    start = time.perf_counter()
    panel = make_panel(symbols=symbols, bars=bars, seed=seed)
    split = _split_timestamp(panel, split_ratio=split_ratio)
    test_bars = research.split_bars(panel, split=split)
    segments.append(StageTiming("panel", time.perf_counter() - start))

    start = time.perf_counter()
    feature_set = research.FeatureSet(
        {
            "alpha": lambda p: p.close.pct_change(lookback)
            / p.close.pct_change().rolling(lookback).std(),
            "size": lambda p: p.close,
        },
        target={"label": lambda p: p.close.shift(-lookback) / p.close - 1.0},
    )
    features = feature_set.fit_transform(panel, include_target=True).dropna()
    train_features, test_features = research.time_split(features, split=split, level="timestamp")
    preprocess = research.Pipeline(
        [
            pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95)),
            pp.Neutralizer(columns=["alpha"], exposures=["size"], method="ols"),
            pp.StandardScaler(columns=["alpha"]),
        ]
    )
    _train_features = preprocess.fit_transform(train_features)
    test_features = preprocess.transform(test_features)
    segments.append(StageTiming("factor", time.perf_counter() - start))

    start = time.perf_counter()
    scores = test_features["alpha"].rename("score")
    allocator = pf.Allocator(
        select=pf.TopK(k=holdings),
        weight=pf.EqualWeight(gross=0.95),
        constrain=pf.Constraints(max_weight=0.5, normalize=True),
    )
    weights = allocator.build(scores)
    with research.ResearchRun("benchmark_research_pipeline") as run:
        research_result = run.finish(
            features=test_features,
            target=test_features.get("label"),
            scores=scores,
            weights=weights,
            artifacts={
                "symbols": [f"S{i:04d}" for i in range(symbols)],
                "split": str(split),
                "lookback": lookback,
            },
        )
    weight_dates = int(weights.index.get_level_values("timestamp").nunique()) if not weights.empty else 0
    segments.append(StageTiming("weights", time.perf_counter() - start))

    start = time.perf_counter()
    backtest = tl.Backtest(
        test_bars,
        ResearchTargetWeightStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=True,
    )
    stats = backtest.run(
        research_result=research_result,
        lookback=lookback,
        rebalance_every=rebalance_every,
    )
    segments.append(StageTiming("backtest", time.perf_counter() - start))

    out_dir = Path(output_dir or "benchmarks/output/research_pipeline")
    report_path: Path | None = None
    start = time.perf_counter()
    if write_report:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = Path(backtest.report(out_dir / "report.html"))
    segments.append(StageTiming("report", time.perf_counter() - start))

    start = time.perf_counter()
    artifact_count = 0
    if write_report:
        artifact_dir = out_dir / "artifacts"
        artifacts = write_artifact_bundle(
            stats.stats,
            artifact_dir,
            strategy=stats.strategy,
            market_data=pd.DataFrame(test_bars),
            log_report=False,
            log_plot=False,
        )
        artifact_count = len(artifacts)
    segments.append(StageTiming("mlflow_artifacts", time.perf_counter() - start))
    segments.append(StageTiming("total", time.perf_counter() - total_start))

    return ResearchPipelineBenchmarkResult(
        total_bars=int(len(panel)),
        final_value=float(stats.summary["final_value"]),
        return_pct=float(stats.summary["return_pct"]),
        total_trades=int(stats.summary.get("total_trades", 0)),
        weight_dates=weight_dates,
        report_path=report_path,
        artifact_count=artifact_count,
        segments=segments,
    )


def _split_timestamp(panel: pd.DataFrame, *, split_ratio: float) -> pd.Timestamp:
    timestamps = pd.DatetimeIndex(panel.index.get_level_values("timestamp").unique()).sort_values()
    position = min(max(int(len(timestamps) * float(split_ratio)), 1), len(timestamps) - 1)
    return pd.Timestamp(timestamps[position])


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.4f}s"


def _print_result(result: ResearchPipelineBenchmarkResult) -> None:
    print("Stage 12 Research Pipeline Benchmark")
    print(f"Total bars: {result.total_bars:,}")
    print(f"Final value: {result.final_value:,.2f}")
    print(f"Return: {result.return_pct:.2f}%")
    print(f"Trades: {result.total_trades:,}")
    print(f"Weight dates: {result.weight_dates:,}")
    print(f"Bars/s: {result.bars_per_sec:,.0f}")
    print()
    print(f"{'Segment':<18} {'Time':>12} {'Share':>9}")
    total = result.elapsed
    for segment in result.segments:
        share = segment.seconds / total * 100.0 if total else 0.0
        print(f"{segment.name:<18} {_format_seconds(segment.seconds):>12} {share:>8.1f}%")
    if result.report_path is not None:
        print(f"\nReport: {result.report_path}")
    print(f"Artifacts: {result.artifact_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=50)
    parser.add_argument("--bars", type=int, default=240)
    parser.add_argument("--holdings", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--split-ratio", type=float, default=0.65)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="benchmarks/output/research_pipeline")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(
        symbols=args.symbols,
        bars=args.bars,
        holdings=args.holdings,
        lookback=args.lookback,
        rebalance_every=args.rebalance_every,
        split_ratio=args.split_ratio,
        cash=args.cash,
        commission=args.commission,
        seed=args.seed,
        write_report=not args.no_write_report,
        output_dir=args.output_dir,
    )
    _print_result(result)


if __name__ == "__main__":
    main()
