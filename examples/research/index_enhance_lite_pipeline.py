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

from sklearn.ensemble import GradientBoostingRegressor

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
        {
            "alpha": lambda p: p.close.pct_change(lookback)
            / p.close.pct_change().rolling(lookback).std(),
            "size": lambda p: p.close,
        },
        target={
            "label": lambda p: p.close.shift(-lookback) / p.close - 1.0,
        },
    )
    features = feature_set.fit_transform(bars, include_target=True).dropna()

    with ResearchRun("index_enhance_research") as run:

        train_features, test_features = research.time_split(
            features,
            split=split,
            level="timestamp",
        )
        preprocess = research.Pipeline(
            [
                pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95)),
                pp.Neutralizer(columns=["alpha"], exposures=["size"], method="ols"),
                pp.StandardScaler(columns=["alpha"]),
            ]
        )
        train_features = preprocess.fit_transform(train_features)
        test_features = preprocess.transform(test_features)

        model = GradientBoostingRegressor(
            random_state=7,
            n_estimators=50,
            max_depth=3,
        )
        model.fit(train_features[["alpha"]], train_features["label"])
        scores = research.ModelScorer(model, features=("alpha",), current=False).predict(
            test_features
        )

        weights = pf.WeightBuilder(
            select=pf.TopK(k=2),
            weight=pf.EqualWeight(gross=0.95),
            constrain=pf.Constraints(max_weight=0.5, normalize=True),
        ).build(scores)

        research_result = run.finish(
            features=test_features,
            target=test_features["label"],
            model=model,
            scores=scores,
            weights=weights,
            artifacts={
                "symbols": list(symbols),
                "split": split,
                "lookback": lookback,
                "profile": ex.profile(bars).to_dict(),
            },
        )

    test_bars = research.split_bars(bars, split=split)
    backtest = tl.Backtest(
        test_bars,
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
                params={"runtime.mode": "lite", "runtime.pipeline": True},
                upload_artifacts=upload_artifacts,
            )
            print("MLflow logging finished")
        except Exception as exc:
            print(f"MLflow logging skipped: {type(exc).__name__}: {exc}")

    print("Lite research index enhance")
    print(f"  symbols={','.join(symbols)}")
    print(
        f"  weight_dates={research_result.weights.index.get_level_values('timestamp').nunique()}"
    )
    print(
        f"  latest_weights={research_result.weights.groupby(level='timestamp').tail(2).to_dict()}"
    )
    print(f"  final_value={stats.summary['final_value']:.2f}")
    print(f"  return_pct={stats.summary['return_pct']:.2f}")
    print(f"  report={report_path}")
