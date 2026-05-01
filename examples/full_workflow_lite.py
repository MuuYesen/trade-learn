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
        for data in self.datas:
            previous = data.close[-self.lookback]
            if previous and previous == previous:
                scores[data._name] = data.close[0] / previous - 1.0

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

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = TradingViewProvider(n_bars=1500)
    bars = provider.history_ohlc(list(SYMBOLS), start=START, end=END, freq="1d")

    backtest = tl.Backtest(
        bars,
        LiteMomentumPortfolio,
        cash=CASH,
        commission=COMMISSION,
        trade_on_close=True,
    )
    stats = backtest.run()
    report_path = backtest.report(OUTPUT_DIR / "report.html")
    plot_path = OUTPUT_DIR / "plot.html"
    charts = backtest.plot()
    plot_path.write_text(file_html(charts[0], INLINE, "Tradelearn Lite Plot"))

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
