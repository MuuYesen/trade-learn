"""Engine index-enhancement workflow with live-style in-strategy inference.

Run from the repository root:

    python examples/research/index_enhance_engine_live.py

The main block stops after fitting the training-time preprocessors and model.
During backtest execution the strategy builds the current cross section from
Engine data feeds, transforms it with fitted preprocessors, predicts scores,
builds weights, and submits Backtrader-style order_target_percent() calls.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

import tradelearn.engine as bt
import tradelearn.research as research
import tradelearn.research.explore as ex
import tradelearn.research.portfolio as pf
import tradelearn.research.preprocess as pp
from tradelearn.data import TradingViewProvider
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.research import ResearchRun


class EngineLiveIndexEnhance(bt.Strategy):
    """Run live-style feature cleaning, prediction, weighting, and execution."""

    params = (
        ("lookback", 20),
        ("history_window", 21),
        ("rebalance_bars", 20),
        ("runtime_pipeline", None),
        ("scorer", None),
        ("allocator", None),
    )

    def __init__(self) -> None:
        super().__init__()
        self.addminperiod(self.p.lookback + 1)

    def next(self) -> None:
        if len(self) % self.p.rebalance_bars != 0:
            return

        features = self.p.runtime_pipeline.transform(self.history_panel(self.p.history_window))
        if features.empty:
            return

        scores = self.p.scorer.predict(features)
        weights = self.p.allocator.build(scores)
        for data in self.datas:
            self.order_target_percent(
                data=data,
                target=weights.get(data._name),
            )


if __name__ == "__main__":
    output_dir = Path("examples/output/research_engine_live")
    symbols = ("NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG")
    start = "2023-01-01"
    end = "2024-01-01"
    split = "2023-09-01"
    lookback = 20
    cash = 100_000.0
    commission = 0.0003
    log_mlflow = True
    upload_artifacts = True

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

    feature_set = research.FeatureSet(
        features={
            "alpha": lambda p: p.close.pct_change(lookback)
            / p.close.pct_change().rolling(lookback).std(),
            "size": lambda p: p.close,
        },
        target={
            "label": lambda p: p.close.shift(-lookback) / p.close - 1.0,
        },
    )
    preprocess = research.Pipeline(
        [
            pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95)),
            pp.Neutralizer(columns=["alpha"], exposures=["size"], method="ols"),
            pp.StandardScaler(columns=["alpha"]),
        ]
    )

    with ResearchRun("index_enhance_live_research") as run:
        data_profile = ex.profile(bars)
        dataset = feature_set.fit_transform(bars, include_target=True).dropna()
        train_features, _ = research.time_split(dataset, split=split, level="timestamp")
        train_features = preprocess.fit_transform(train_features)

        model = GradientBoostingRegressor(
            random_state=7,
            n_estimators=50,
            max_depth=3,
        )
        model.fit(train_features[["alpha"]], train_features["label"])

        research_result = run.finish(
            features=train_features[["alpha", "size"]],
            target=train_features["label"],
            model=model,
            selected_features=("alpha",),
            artifacts={
                "symbols": list(symbols),
                "split": split,
                "lookback": lookback,
                "profile": data_profile.to_dict(),
                "execution": "strategy_inference",
            },
        )

    runtime_pipeline = research.Pipeline([feature_set, preprocess])
    scorer = research.ModelScorer(model, features=("alpha",))
    allocator = pf.Allocator(
        select=pf.TopK(k=2),
        weight=pf.EqualWeight(gross=0.95),
        constrain=pf.Constraints(max_weight=0.5, normalize=True),
    )

    test_bars = research.split_bars(bars, split=split)
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.setcash(cash)
    cerebro.setcommission(commission)
    cerebro.adddata(test_bars)
    cerebro.addstrategy(
        EngineLiveIndexEnhance,
        research_result=research_result,
        lookback=lookback,
        history_window=lookback + 1,
        runtime_pipeline=runtime_pipeline,
        scorer=scorer,
        allocator=allocator,
        rebalance_bars=20,
    )

    if log_mlflow:
        print("Logging Engine live-style run to MLflow...")
        cerebro.addanalyzer(
            MLflowAnalyzer,
            name="mlflow",
            experiment="tradelearn-research-examples",
            run_name="engine-index-enhance-live",
            params={"runtime.mode": "engine", "runtime.inference": "strategy"},
            upload_artifacts=upload_artifacts,
        )

    [strategy] = cerebro.run()
    if log_mlflow:
        print(f"MLflow logging status: {strategy.analyzer_results.get('mlflow', {})}")
    report_path = cerebro.report(output_dir / "report.html")

    print("Engine live-style index enhance")
    print(f"  symbols={','.join(symbols)}")
    print("  model=GradientBoostingRegressor")
    print("  runtime=clean -> predict -> select -> weight -> order_target_percent")
    print(f"  final_value={strategy.stats.summary['final_value']:.2f}")
    print(f"  return_pct={strategy.stats.summary['return_pct']:.2f}")
    print(f"  report={report_path}")
