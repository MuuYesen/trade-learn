"""Engine end-to-end workflow demo.

Run from the repository root:

    python examples/full_workflow_engine.py

This file mirrors ``examples/full_workflow_lite.py`` with the Engine API:
data provider -> strategy -> stats -> report -> plot -> optional MLflow.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.resources import INLINE

import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider
from tradelearn.engine.analyzers import MLflowAnalyzer

OUTPUT_DIR = Path("examples/output/full_workflow_engine")
SYMBOLS = ("NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG")
START = "2023-01-01"
END = "2024-01-01"
CASH = 100_000.0
COMMISSION = 0.0003
USE_LIVE_DATA = os.getenv("TRADELEARN_DEMO_LIVE_DATA", "0") == "1"
LOG_MLFLOW = os.getenv("TRADELEARN_DEMO_MLFLOW", "0") == "1"

warnings.filterwarnings(
    "ignore",
    message="you are using nologin method, data you access may be limited",
    category=UserWarning,
)


class EngineMomentumPortfolio(bt.IndexEnhanceStrategy):
    """Monthly top-2 momentum portfolio using Engine target weights."""

    params = (
        ("lookback", 20),
        ("top_k", 2),
        ("gross", 0.8),
    )

    def __init__(self) -> None:
        super().__init__()
        self.addminperiod(self.p.lookback + 1)

    def next(self) -> None:
        if len(self) % 20 != 0:
            return

        scores: dict[str, float] = {}
        for data in self.datas:
            previous = data.close[-self.p.lookback]
            if previous and previous == previous:
                scores[data._name] = data.close[0] / previous - 1.0

        selected = [
            ticker
            for ticker, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                : self.p.top_k
            ]
        ]
        if not selected:
            return

        weight = self.p.gross / len(selected)
        self.target_weights({ticker: weight for ticker in selected}, close_missing=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bars = load_bars()
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.setcash(CASH)
    cerebro.setcommission(COMMISSION)
    for symbol in SYMBOLS:
        cerebro.adddata(bars.xs(symbol, level="symbol"), name=symbol)
    cerebro.addstrategy(EngineMomentumPortfolio)

    if LOG_MLFLOW:
        cerebro.addanalyzer(
            MLflowAnalyzer,
            name="mlflow",
            experiment="tradelearn-full-workflow",
            run_name="engine-full-workflow",
            artifact_bundle=True,
            log_report=True,
            log_plot=True,
            artifact_path="engine",
        )

    [strategy] = cerebro.run()
    report_path = cerebro.report(OUTPUT_DIR / "report.html")
    plot_path = write_plot(cerebro.plot(), OUTPUT_DIR / "plot.html")

    print("Engine full workflow")
    print(f"  bars={len(bars)} symbols={len(SYMBOLS)}")
    print(f"  final_value={strategy.stats.summary['final_value']:.2f}")
    print(f"  return_pct={strategy.stats.summary['return_pct']:.2f}")
    print(f"  trades={strategy.stats.summary['total_trades']}")
    print(f"  report={report_path}")
    print(f"  plot={plot_path}")
    if LOG_MLFLOW:
        print(f"  mlflow={strategy.analyzer_results.get('mlflow', {})}")


def load_bars() -> pd.DataFrame:
    """Load panel Bars from TradingView or deterministic local demo data."""

    if USE_LIVE_DATA:
        provider = TradingViewProvider(n_bars=1500)
        return provider.history_ohlc(list(SYMBOLS), start=START, end=END, freq="1d")
    return make_demo_panel(SYMBOLS, periods=180)


def make_demo_panel(symbols: tuple[str, ...], periods: int) -> pd.DataFrame:
    """Create deterministic OHLCV data for offline demos."""

    rng = np.random.default_rng(7)
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    frames: list[pd.DataFrame] = []
    for offset, symbol in enumerate(symbols):
        drift = 0.0008 + offset * 0.0002
        noise = rng.normal(0.0, 0.01, periods)
        close = 100 + offset * 20 + np.cumsum(drift + noise) * 100
        open_ = np.r_[close[0], close[:-1]]
        spread = 1.0 + rng.random(periods)
        frame = pd.DataFrame(
            {
                "timestamp": index,
                "symbol": symbol,
                "open": open_,
                "high": np.maximum(open_, close) + spread,
                "low": np.minimum(open_, close) - spread,
                "close": close,
                "volume": rng.integers(100_000, 300_000, periods),
            }
        )
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.set_index(["timestamp", "symbol"]).sort_index()
    bars.attrs.update(
        market="DEMO",
        freq="1d",
        engine="synthetic",
        source="full_workflow_engine",
        adjust="none",
    )
    return bars


def write_plot(charts: list[object], path: Path) -> Path:
    """Write the first returned chart to HTML."""

    if not charts:
        return path
    path.write_text(file_html(charts[0], INLINE, "Tradelearn Engine Plot"))
    return path


if __name__ == "__main__":
    main()
