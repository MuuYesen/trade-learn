from __future__ import annotations

import pandas as pd

from tradelearn.ml import (
    EqualWeightOptimizer,
    FactorTransformer,
    ModelAdapter,
    RiskPolicy,
    StrategyPipeline,
    TopKSelector,
)
from tradelearn.ml.features import FeatureStore, feature


class LinearModel:
    def __init__(self) -> None:
        self.fitted_shape = None

    def fit(self, X, y):
        self.fitted_shape = (len(X), len(X[0]))
        return self

    def predict(self, X):
        return [row[0] - row[1] for row in X]


def test_strategy_pipeline_fits_scores_and_weights() -> None:
    frame = pd.DataFrame(
        {
            "value": [0.9, 0.2, 0.6],
            "risk": [0.1, 0.3, 0.2],
            "unused": [1.0, 2.0, 3.0],
        },
        index=["A", "B", "C"],
    )
    model = LinearModel()
    pipeline = StrategyPipeline(
        [
            ("features", FactorTransformer(["value", "risk"])),
            ("model", ModelAdapter(model)),
            ("selector", TopKSelector(k=2)),
            ("optimizer", EqualWeightOptimizer(gross=1.0)),
            ("risk", RiskPolicy(max_weight=0.6)),
        ]
    )

    result = pipeline.fit(frame, [1.0, 0.0, 1.0]).predict_weights(frame)

    assert model.fitted_shape == (3, 2)
    assert result.scores.round(10).to_dict() == {"A": 0.8, "B": -0.1, "C": 0.4}
    assert result.weights.to_dict() == {"A": 0.5, "C": 0.5}
    assert result.selected == ["A", "C"]


def test_factor_transformer_accepts_callable_features() -> None:
    frame = pd.DataFrame({"close": [10.0, 11.0], "open": [9.0, 12.0]}, index=["A", "B"])

    def spread(data: pd.DataFrame) -> pd.Series:
        return (data["close"] - data["open"]).rename("spread")

    transformed = FactorTransformer(["close", spread]).fit_transform(frame)

    assert list(transformed.columns) == ["close", "spread"]
    assert transformed.loc["A", "spread"] == 1.0


def test_pipeline_can_use_precomputed_score_column() -> None:
    frame = pd.DataFrame({"score": [0.1, 0.8, 0.5]}, index=["A", "B", "C"])
    pipeline = StrategyPipeline(
        [
            ("model", ModelAdapter(score_column="score")),
            ("selector", TopKSelector(k=1)),
            ("optimizer", EqualWeightOptimizer()),
        ]
    )

    result = pipeline.predict_weights(frame)

    assert result.weights.to_dict() == {"B": 1.0}
    assert result.as_weight_dict() == {"B": 1.0}


def test_topk_selector_ascending_threshold_uses_portfolio_filtering() -> None:
    scores = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20})
    selector = TopKSelector(k=3, ascending=True, threshold=0.20)

    assert selector.select(scores) == ["A", "C"]


def test_risk_policy_normalizes_and_drops_small_weights() -> None:
    policy = RiskPolicy(max_weight=0.4, min_abs_weight=0.05)
    weights = pd.Series({"A": 0.9, "B": 0.2, "C": 0.01})

    adjusted = policy.apply(weights)

    assert adjusted.to_dict() == {"A": 0.4, "B": 0.2}


def test_factor_transformer_can_read_registered_feature_store_feature(tmp_path) -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01", tz="UTC"), "A"),
            (pd.Timestamp("2024-01-01", tz="UTC"), "B"),
        ],
        names=["timestamp", "symbol"],
    )
    frame = pd.DataFrame(
        {
            "open": [10.0, 12.0],
            "high": [12.0, 13.0],
            "low": [9.5, 9.8],
            "close": [11.5, 10.0],
            "volume": [1000.0, 1200.0],
        },
        index=index,
    )
    store = FeatureStore(tmp_path)

    @feature(name="spread", version=1)
    def spread(data: pd.DataFrame) -> pd.Series:
        return data["close"] - data["open"]

    store.register(spread)

    transformed = FactorTransformer(["spread"], feature_store=store).fit_transform(frame)

    assert transformed["spread"].to_dict() == {
        (pd.Timestamp("2024-01-01", tz="UTC"), "A"): 1.5,
        (pd.Timestamp("2024-01-01", tz="UTC"), "B"): -2.0,
    }
    assert store.exists(frame, "spread")


def test_strategy_pipeline_exposes_serializable_step_params(tmp_path) -> None:
    store = FeatureStore(tmp_path)
    pipeline = StrategyPipeline(
        [
            ("features", FactorTransformer(["value"], feature_store=store)),
            ("model", ModelAdapter(score_column="score")),
            ("selector", TopKSelector(k=2, threshold=0.1)),
            ("optimizer", EqualWeightOptimizer(gross=0.8)),
            ("risk", RiskPolicy(max_weight=0.4, min_abs_weight=0.01, normalize=True)),
        ]
    )

    params = pipeline.get_params()

    assert params == {
        "steps": ["features", "model", "selector", "optimizer", "risk"],
        "features": {
            "type": "FactorTransformer",
            "features": ["value"],
            "feature_store": True,
        },
        "model": {
            "type": "ModelAdapter",
            "estimator": None,
            "score_column": "score",
        },
        "selector": {
            "type": "TopKSelector",
            "k": 2,
            "ascending": False,
            "threshold": 0.1,
        },
        "optimizer": {
            "type": "EqualWeightOptimizer",
            "gross": 0.8,
        },
        "risk": {
            "type": "RiskPolicy",
            "max_weight": 0.4,
            "min_abs_weight": 0.01,
            "normalize": True,
        },
    }


def test_strategy_pipeline_exposes_model_feature_importance() -> None:
    class ImportanceModel(LinearModel):
        feature_importances_ = [0.25, 0.75]

    frame = pd.DataFrame({"value": [0.9, 0.2], "quality": [0.1, 0.8]}, index=["A", "B"])
    pipeline = StrategyPipeline(
        [
            ("features", FactorTransformer(["value", "quality"])),
            ("model", ModelAdapter(ImportanceModel())),
        ]
    )

    explanation = pipeline.fit(frame, [1.0, 0.0]).explain()

    assert explanation.to_dict() == {"value": 0.25, "quality": 0.75}
