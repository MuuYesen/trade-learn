"""Lite index-enhancement research workflow.

Run from the repository root:

    python examples/research/index_enhance_lite.py

The example keeps research outside the strategy, passes the final
ResearchResult into the strategy, and lets Lite execute target_weights().
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import tradelearn.lite as tl
import tradelearn.research as research
import tradelearn.research.explore as ex
import tradelearn.research.portfolio as pf
import tradelearn.research.preprocess as pp
from tradelearn.data import TradingViewProvider
from tradelearn.research import ResearchRun


class LiteResearchIndexEnhance(tl.Strategy):
    """Execute precomputed research weights through Lite target_weights()."""

    lookback = 20
    rebalance_bars = 20

    def init(self) -> None:
        self.start_on_bar(self.lookback + 1)

    def next(self) -> None:
        if len(self.data) % self.rebalance_bars != 0:
            return
        self.target_weights(self.research_result.weights[0], close_missing=True)


if __name__ == "__main__":
    output_dir = Path("examples/output/research_lite")
    symbols = ("NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG")
    start = "2023-01-01"
    end = "2024-01-01"
    lookback = 20
    cash = 100_000.0
    commission = 0.0003
    log_mlflow = True
    upload_artifacts = True
    mlflow_uri = "http://127.0.0.1:5050"

    warnings.filterwarnings(
        "ignore",
        message="you are using nologin method, data you access may be limited",
        category=UserWarning,
    )
    logging.getLogger("tvDatafeed.main").setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    output_dir.mkdir(parents=True, exist_ok=True)

    provider = TradingViewProvider(n_bars=1500)
    bars = provider.history_ohlc(list(symbols), start=start, end=end, freq="1d")

    close = bars["close"].unstack("symbol")
    returns = close.pct_change()
    alpha = close.pct_change(lookback) / returns.rolling(lookback).std()
    features = alpha.stack().rename("alpha").to_frame().dropna()
    features.index.names = ["timestamp", "symbol"]
    exposures = close.stack().rename("size").to_frame().reindex(features.index)
    split_at = features.index.get_level_values("timestamp").max()

    with ResearchRun("lite_index_enhance_research") as run:
        data_profile = ex.profile(bars)
        train_features, test_features = research.time_split(
            features,
            split=split_at,
            level="timestamp",
        )
        train_exposures = exposures.reindex(train_features.index)
        test_exposures = exposures.reindex(test_features.index)
        winsorizer = pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95))
        neutralizer = pp.Neutralizer(columns=["alpha"], method="ols")
        scaler = pp.StandardScaler(columns=["alpha"])
        train_features = winsorizer.fit_transform(train_features)
        train_features = pp.winsorize_mad(train_features, columns=["alpha"], scale=5.0)
        train_features = neutralizer.fit_transform(
            train_features,
            exposures=train_exposures,
        )
        train_features = scaler.fit_transform(train_features)
        test_features = winsorizer.transform(test_features)
        test_features = pp.winsorize_mad(test_features, columns=["alpha"], scale=5.0)
        test_features = neutralizer.transform(
            test_features,
            exposures=test_exposures,
        )
        test_features = scaler.transform(test_features)
        scores = test_features["alpha"].droplevel("timestamp")
        selected = pf.select_top(scores.to_dict(), k=2)
        weights = pf.equal_weight(selected, gross=0.95)
        weights = pf.apply_constraints(weights, max_weight=0.5, normalize=True)
        research_result = run.finish(
            features=test_features,
            scores=scores,
            selected=selected,
            weights=weights,
            artifacts={
                "symbols": list(symbols),
                "lookback": lookback,
                "profile": data_profile.to_dict(),
            },
        )

    backtest = tl.Backtest(
        bars,
        LiteResearchIndexEnhance,
        cash=cash,
        commission=commission,
        trade_on_close=True,
    )
    stats = backtest.run(research_result=research_result, lookback=lookback)
    report_path = backtest.report(output_dir / "report.html")

    if log_mlflow:
        print("Logging Lite run to MLflow...")
        try:
            backtest.log_mlflow(
                experiment_name="tradelearn-research-examples",
                run_name="lite-index-enhance",
                uri=mlflow_uri,
                params=research_result.params,
                tags={"mode": "lite", "workflow": "research"},
                log_mlflow=log_mlflow,
                upload_artifacts=upload_artifacts,
                log_report=upload_artifacts,
                log_plot=False,
            )
            print("MLflow logging finished")
        except Exception as exc:
            print(f"MLflow logging skipped: {type(exc).__name__}: {exc}")

    print("Lite research index enhance")
    print(f"  symbols={','.join(symbols)}")
    print(f"  selected={research_result.selected}")
    print(f"  weights={research_result.weights.to_dict()}")
    print(f"  final_value={stats.summary['final_value']:.2f}")
    print(f"  return_pct={stats.summary['return_pct']:.2f}")
    print(f"  report={report_path}")
