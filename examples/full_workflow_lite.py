"""Lite end-to-end workflow demo.

Run from the repository root:

    python examples/full_workflow_lite.py

This file mirrors ``examples/full_workflow_engine.py`` with the Lite API:
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

import tradelearn.lite as tl
from tradelearn.data import TradingViewProvider

OUTPUT_DIR = Path("examples/output/full_workflow_lite")
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


class LiteMomentumPortfolio(tl.Strategy):
    """Monthly top-2 momentum portfolio using Lite target weights."""

    lookback = 20
    top_k = 2
    gross = 0.8

    def init(self) -> None:
        self.start_on_bar(self.lookback + 1)

    def next(self) -> None:
        if len(self.data) % 20 != 0:
            return

        scores: dict[str, float] = {}
        for ticker, feed in self._target_weight_data_map().items():
            close = feed.get_array("close")
            cursor = feed._cursor
            previous = close[cursor - self.lookback]
            if previous and previous == previous:
                scores[ticker] = close[cursor] / previous - 1.0

        selected = [
            ticker
            for ticker, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                : self.top_k
            ]
        ]
        if not selected:
            return

        weight = self.gross / len(selected)
        self.target_weights({ticker: weight for ticker in selected}, close_missing=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bars = load_bars()

    backtest = tl.Backtest(
        bars,
        LiteMomentumPortfolio,
        cash=CASH,
        commission=COMMISSION,
        trade_on_close=True,
    )
    stats = backtest.run()
    report_path = backtest.report(OUTPUT_DIR / "report.html")
    plot_path = write_plot(backtest.plot(), OUTPUT_DIR / "plot.html")

    if LOG_MLFLOW:
        backtest.log_mlflow(
            experiment_name="tradelearn-full-workflow",
            run_name="lite-full-workflow",
            params={
                "symbols": ",".join(SYMBOLS),
                "lookback": LiteMomentumPortfolio.lookback,
                "top_k": LiteMomentumPortfolio.top_k,
                "gross": LiteMomentumPortfolio.gross,
            },
            tags={"mode": "lite"},
            artifact_bundle=True,
            log_report=True,
            log_plot=True,
        )

    print("Lite full workflow")
    print(f"  bars={len(bars)} symbols={len(SYMBOLS)}")
    print(f"  final_value={stats.summary['final_value']:.2f}")
    print(f"  return_pct={stats.summary['return_pct']:.2f}")
    print(f"  trades={stats.summary['total_trades']}")
    print(f"  report={report_path}")
    print(f"  plot={plot_path}")


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
        source="full_workflow_lite",
        adjust="none",
    )
    return bars


def write_plot(charts: list[object], path: Path) -> Path:
    """Write the first returned chart to HTML."""

    if not charts:
        return path
    path.write_text(file_html(charts[0], INLINE, "Tradelearn Lite Plot"))
    return path


if __name__ == "__main__":
    main()
