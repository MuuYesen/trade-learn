"""Engine end-to-end workflow demo.

Run from the repository root:

    python examples/full_workflow_engine.py

This file mirrors ``examples/full_workflow_lite.py`` with the Engine API:
data provider -> factor research -> strategy -> stats -> report -> plot ->
optional MLflow.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from bokeh.embed import file_html
from bokeh.resources import INLINE

import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.factor import FactorAnalyzer
from tradelearn.portfolio import select_top

OUTPUT_DIR = Path("examples/output/full_workflow_engine")
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

        selected = select_top(scores, k=self.p.top_k)
        if not selected:
            return

        weight = self.p.gross / len(selected)
        self.target_weights({ticker: weight for ticker in selected}, close_missing=True)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = TradingViewProvider(n_bars=1500)
    bars = provider.history_ohlc(list(SYMBOLS), start=START, end=END, freq="1d")

    close = bars["close"].unstack("symbol")
    momentum = close.pct_change(EngineMomentumPortfolio.params[0][1])
    volatility = close.pct_change().rolling(EngineMomentumPortfolio.params[0][1]).std()
    factor = (momentum / volatility).stack().rename("momentum_quality")
    forward_returns = close.pct_change().shift(-1).stack().rename("forward_return")

    factor_analyzer = FactorAnalyzer(
        factor.dropna(),
        forward_returns=forward_returns.dropna(),
        quantiles=3,
    )
    factor_report_path = factor_analyzer.report(OUTPUT_DIR / "factor_report.html")

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.setcash(CASH)
    cerebro.setcommission(COMMISSION)
    cerebro.adddata(bars)
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
    plot_path = OUTPUT_DIR / "plot.html"
    charts = cerebro.plot()
    plot_path.write_text(file_html(charts[0], INLINE, "Tradelearn Engine Plot"))

    print("Engine full workflow")
    print(f"  bars={len(bars)} symbols={len(SYMBOLS)}")
    print(f"  factor_ic={factor_analyzer.summary()['ic_mean']:.4f}")
    print(f"  final_value={strategy.stats.summary['final_value']:.2f}")
    print(f"  return_pct={strategy.stats.summary['return_pct']:.2f}")
    print(f"  trades={strategy.stats.summary['total_trades']}")
    print(f"  factor_report={factor_report_path}")
    print(f"  report={report_path}")
    print(f"  plot={plot_path}")
    if LOG_MLFLOW:
        print(f"  mlflow={strategy.analyzer_results.get('mlflow', {})}")
