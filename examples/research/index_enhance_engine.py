"""Engine index-enhancement research workflow.

Run from the repository root:

    python examples/research/index_enhance_engine.py

The example keeps research outside the strategy, passes the final
ResearchResult into the strategy, and lets Engine execute target_weights().
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

import tradelearn.engine as bt
import tradelearn.research as research
import tradelearn.research.explore as ex
import tradelearn.research.portfolio as pf
import tradelearn.research.preprocess as pp
from tradelearn.data import TradingViewProvider
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.research import ResearchRun


class EngineResearchIndexEnhance(bt.IndexEnhanceStrategy):
    """Execute precomputed research weights through Engine target_weights()."""

    params = (("lookback", 20), ("rebalance_bars", 20))

    def __init__(self) -> None:
        super().__init__()
        self.addminperiod(self.p.lookback + 1)

    def next(self) -> None:
        if len(self) % self.p.rebalance_bars != 0:
            return
        self.target_weights(self.research_result.weights[0], close_missing=True)


if __name__ == "__main__":
    output_dir = Path("examples/output/research_engine")
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

    with ResearchRun("engine_index_enhance_research") as run:
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
        scores = test_features["alpha"].rename("score")
        weight_parts = []
        for dt, daily_scores in scores.groupby(level="timestamp"):
            daily_scores = daily_scores.droplevel("timestamp").dropna()
            if daily_scores.empty:
                continue
            selected = pf.select_top(daily_scores, k=2)
            daily_weights = pf.equal_weight(selected, gross=0.95)
            daily_weights = pf.apply_constraints(
                daily_weights,
                max_weight=0.5,
                normalize=True,
            )
            daily_weights.index = pd.MultiIndex.from_product(
                [[dt], daily_weights.index],
                names=["timestamp", "symbol"],
            )
            weight_parts.append(daily_weights)
        weights = (
            pd.concat(weight_parts).sort_index().rename("weight")
            if weight_parts
            else pd.Series(
                dtype="float64",
                index=pd.MultiIndex.from_arrays([[], []], names=["timestamp", "symbol"]),
                name="weight",
            )
        )
        research_result = run.finish(
            features=test_features,
            scores=scores,
            selected=None,
            weights=weights,
            artifacts={
                "symbols": list(symbols),
                "lookback": lookback,
                "profile": data_profile.to_dict(),
            },
        )

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.setcash(cash)
    cerebro.setcommission(commission)
    cerebro.adddata(bars)
    cerebro.addstrategy(
        EngineResearchIndexEnhance,
        research_result=research_result,
        lookback=lookback,
        rebalance_bars=20,
    )

    if log_mlflow:
        print("Logging Engine run to MLflow...")
        cerebro.addanalyzer(
            MLflowAnalyzer,
            name="mlflow",
            experiment="tradelearn-research-examples",
            run_name="engine-index-enhance",
            uri=mlflow_uri,
            log_mlflow=log_mlflow,
            upload_artifacts=upload_artifacts,
            log_report=upload_artifacts,
            log_plot=False,
        )

    [strategy] = cerebro.run()
    if log_mlflow:
        print(f"MLflow logging status: {strategy.analyzer_results.get('mlflow', {})}")
    report_path = cerebro.report(output_dir / "report.html")

    print("Engine research index enhance")
    print(f"  symbols={','.join(symbols)}")
    print(
        f"  weight_dates={research_result.weights.index.get_level_values('timestamp').nunique()}"
    )
    print(
        f"  latest_weights={research_result.weights.groupby(level='timestamp').tail(2).to_dict()}"
    )
    print(f"  final_value={strategy.stats.summary['final_value']:.2f}")
    print(f"  return_pct={strategy.stats.summary['return_pct']:.2f}")
    print(f"  report={report_path}")
