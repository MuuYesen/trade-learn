"""Lite index-enhancement workflow with live-style in-strategy inference.

Run from the repository root:

    python examples/research/index_enhance_lite_live.py

The main block stops after fitting the training-time preprocessors and model.
During backtest execution the strategy builds the current cross section,
transforms it with fitted preprocessors, predicts scores, builds weights, and
submits target_weights() orders.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

import tradelearn.lite as tl
import tradelearn.research as research
import tradelearn.research.explore as ex
import tradelearn.research.portfolio as pf
import tradelearn.research.preprocess as pp
from tradelearn.data import TradingViewProvider
from tradelearn.research import ResearchRun


class LiteLiveIndexEnhance(tl.Strategy):
    """Run live-style feature cleaning, prediction, weighting, and execution."""

    lookback = 20
    history_window = 21
    rebalance_bars = 20
    runtime_pipeline = None
    scorer = None
    weight_builder = None

    def init(self) -> None:
        self.start_on_bar(self.lookback + 1)

    def next(self) -> None:
        if len(self.data) % self.rebalance_bars != 0:
            return

        features = self.runtime_pipeline.transform(self.history_panel(self.history_window))
        if features.empty:
            return

        scores = self.scorer.predict(features)
        weights = self.weight_builder.build(scores)
        self.target_weights(weights, close_missing=True)


if __name__ == "__main__":
    output_dir = Path("examples/output/research_lite_live")
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
    weight_builder = pf.WeightBuilder(
        select=pf.TopK(k=2),
        weight=pf.EqualWeight(gross=0.95),
        constrain=pf.Constraints(max_weight=0.5, normalize=True),
    )

    test_bars = research.split_bars(bars, split=split)
    backtest = tl.Backtest(
        test_bars,
        LiteLiveIndexEnhance,
        cash=cash,
        commission=commission,
        trade_on_close=True,
    )
    stats = backtest.run(
        research_result=research_result,
        lookback=lookback,
        history_window=lookback + 1,
        runtime_pipeline=runtime_pipeline,
        scorer=scorer,
        weight_builder=weight_builder,
    )
    report_path = backtest.report(output_dir / "report.html")

    if log_mlflow:
        print("Logging Lite live-style run to MLflow...")
        try:
            backtest.log_mlflow(
                experiment_name="tradelearn-research-examples",
                run_name="lite-index-enhance-live",
                params={"runtime.mode": "lite", "runtime.inference": "strategy"},
                upload_artifacts=upload_artifacts,
            )
            print("MLflow logging finished")
        except Exception as exc:
            print(f"MLflow logging skipped: {type(exc).__name__}: {exc}")

    print("Lite live-style index enhance")
    print(f"  symbols={','.join(symbols)}")
    print("  model=GradientBoostingRegressor")
    print("  runtime=clean -> predict -> select -> weight -> target_weights")
    print(f"  final_value={stats.summary['final_value']:.2f}")
    print(f"  return_pct={stats.summary['return_pct']:.2f}")
    print(f"  report={report_path}")
