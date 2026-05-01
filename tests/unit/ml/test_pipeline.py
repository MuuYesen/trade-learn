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


def test_risk_policy_normalizes_and_drops_small_weights() -> None:
    policy = RiskPolicy(max_weight=0.4, min_abs_weight=0.05)
    weights = pd.Series({"A": 0.9, "B": 0.2, "C": 0.01})

    adjusted = policy.apply(weights)

    assert adjusted.to_dict() == {"A": 0.4, "B": 0.2}
