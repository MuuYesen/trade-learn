"""Scikit-learn style strategy research pipeline.

This module stays above the backtest runtime.  It turns research data into
scores and target weights; ``engine`` / ``lite`` strategies decide when to pass
those weights into ``target_weights()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradelearn.portfolio import EqualWeightOptimizer, RiskPolicy, TopKSelector

if TYPE_CHECKING:
    from tradelearn.ml.features import FeatureStore

FeatureInput = str | Callable[[pd.DataFrame], pd.Series | pd.DataFrame | Iterable[float]]


@dataclass(frozen=True)
class PipelineResult:
    """Output of a strategy pipeline prediction."""

    scores: pd.Series
    selected: list[str]
    weights: pd.Series

    def as_weight_dict(self) -> dict[str, float]:
        """Return weights as a plain dict suitable for ``target_weights()``."""
        return {str(symbol): float(weight) for symbol, weight in self.weights.items()}


class FactorTransformer:
    """Select or compute feature columns from a cross-sectional DataFrame."""

    def __init__(
        self,
        features: Sequence[FeatureInput],
        *,
        feature_store: FeatureStore | None = None,
    ) -> None:
        self.features = tuple(features)
        self.feature_store = feature_store
        self.feature_names_: list[str] = []

    def fit(self, data: pd.DataFrame, y: Any | None = None) -> FactorTransformer:
        """Record output feature names."""
        transformed = self.transform(data)
        self.feature_names_ = list(transformed.columns)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing configured features."""
        frames: list[pd.DataFrame] = []
        for feature in self.features:
            if isinstance(feature, str):
                if feature in data:
                    frames.append(data[[feature]].copy())
                elif self.feature_store is not None:
                    frames.append(self.feature_store.compute(data, [feature]))
                else:
                    raise KeyError(f"feature column {feature!r} not found")
                continue
            values = feature(data)
            feature_name = getattr(feature, "__name__", "feature")
            frames.append(_feature_frame(values, data.index, feature_name))
        if not frames:
            return pd.DataFrame(index=data.index)
        return pd.concat(frames, axis=1)

    def fit_transform(self, data: pd.DataFrame, y: Any | None = None) -> pd.DataFrame:
        """Fit and transform in one pass."""
        self.fit(data, y)
        return self.transform(data)

    def get_params(self) -> dict[str, Any]:
        """Return serializable transformer parameters for tracking."""
        return {
            "type": type(self).__name__,
            "features": [_feature_name(feature) for feature in self.features],
            "feature_store": self.feature_store is not None,
        }


class ModelAdapter:
    """Adapter for sklearn-like estimators or precomputed score columns."""

    def __init__(self, estimator: Any | None = None, *, score_column: str | None = None) -> None:
        if estimator is None and score_column is None:
            raise ValueError("ModelAdapter requires estimator or score_column")
        self.estimator = estimator
        self.score_column = score_column

    def fit(self, data: pd.DataFrame, y: Sequence[float] | pd.Series | None = None) -> ModelAdapter:
        """Fit wrapped estimator when possible."""
        if self.estimator is not None and hasattr(self.estimator, "fit") and y is not None:
            self.estimator.fit(_matrix(data), list(y))
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return model scores indexed like input data."""
        if self.score_column is not None:
            if self.score_column not in data:
                raise KeyError(f"score column {self.score_column!r} not found")
            return data[self.score_column].astype(float).rename("score")
        if self.estimator is None or not hasattr(self.estimator, "predict"):
            raise ValueError("estimator must implement predict().")
        values = self.estimator.predict(_matrix(data))
        return pd.Series(values, index=data.index, name="score", dtype="float64")

    def get_params(self) -> dict[str, Any]:
        """Return serializable model parameters for tracking."""
        return {
            "type": type(self).__name__,
            "estimator": None if self.estimator is None else type(self.estimator).__name__,
            "score_column": self.score_column,
        }

    def feature_importance(self, feature_names: Sequence[str] | None = None) -> pd.Series:
        """Return estimator-provided feature importance or coefficients."""
        if self.estimator is None:
            return pd.Series(dtype="float64", name="importance")
        values = None
        if hasattr(self.estimator, "feature_importances_"):
            values = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            values = self.estimator.coef_
        if values is None:
            return pd.Series(dtype="float64", name="importance")
        flat_values = _flatten_importance(values)
        names = list(feature_names or [])
        if len(names) != len(flat_values):
            names = [f"feature_{idx}" for idx in range(len(flat_values))]
        return pd.Series(flat_values, index=names, name="importance", dtype="float64")


class StrategyPipeline:
    """Pipeline that maps research data to scores and target weights."""

    def __init__(self, steps: Sequence[tuple[str, Any]]) -> None:
        self.steps = list(steps)
        self.named_steps = {name: step for name, step in self.steps}

    def fit(
        self,
        data: pd.DataFrame,
        y: Sequence[float] | pd.Series | None = None,
    ) -> StrategyPipeline:
        """Fit transformer and model steps."""
        transformed = data
        for _name, step in self.steps:
            if isinstance(step, (TopKSelector, EqualWeightOptimizer, RiskPolicy)):
                continue
            if isinstance(step, ModelAdapter):
                step.fit(transformed, y)
                continue
            if hasattr(step, "fit_transform"):
                transformed = step.fit_transform(transformed, y)
            else:
                if hasattr(step, "fit"):
                    step.fit(transformed, y)
                if hasattr(step, "transform"):
                    transformed = step.transform(transformed)
        return self

    def predict_scores(self, data: pd.DataFrame) -> pd.Series:
        """Return model scores for the provided cross section."""
        transformed = data
        model: ModelAdapter | None = None
        for _name, step in self.steps:
            if isinstance(step, (TopKSelector, EqualWeightOptimizer, RiskPolicy)):
                continue
            if isinstance(step, ModelAdapter):
                model = step
                break
            if hasattr(step, "transform"):
                transformed = step.transform(transformed)
        if model is None:
            raise ValueError("StrategyPipeline requires a ModelAdapter step")
        return model.predict(transformed)

    def predict_weights(self, data: pd.DataFrame) -> PipelineResult:
        """Return scores, selected symbols, and final target weights."""
        scores = self.predict_scores(data)
        selector = self._step_of_type(TopKSelector)
        optimizer = self._step_of_type(EqualWeightOptimizer)
        if selector is not None:
            selected = selector.select(scores)
        else:
            selected = [str(item) for item in scores.index]
        if optimizer is not None:
            weights = optimizer.optimize(selected, scores)
        else:
            weights = scores.loc[selected]
        for _name, step in self.steps:
            if isinstance(step, RiskPolicy):
                weights = step.apply(weights)
        return PipelineResult(scores=scores, selected=selected, weights=weights)

    def get_params(self) -> dict[str, Any]:
        """Return serializable pipeline parameters for MLflow/report tracking."""
        params: dict[str, Any] = {"steps": [name for name, _step in self.steps]}
        for name, step in self.steps:
            if hasattr(step, "get_params"):
                params[name] = step.get_params()
            else:
                params[name] = {"type": type(step).__name__}
        return params

    def explain(self) -> pd.Series:
        """Return model feature importance when the estimator exposes it."""
        model = self._step_of_type(ModelAdapter)
        if model is None:
            return pd.Series(dtype="float64", name="importance")
        feature_names: list[str] = []
        for _name, step in self.steps:
            if step is model:
                break
            names = getattr(step, "feature_names_", None)
            if names:
                feature_names = [str(name) for name in names]
        return model.feature_importance(feature_names)

    def _step_of_type(self, cls: type[Any]) -> Any | None:
        for _name, step in self.steps:
            if isinstance(step, cls):
                return step
        return None


def _feature_name(feature: FeatureInput) -> str:
    if isinstance(feature, str):
        return feature
    return str(getattr(feature, "feature_name", getattr(feature, "__name__", "feature")))


def _feature_frame(values: Any, index: pd.Index, default_name: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reindex(index).copy()
    if isinstance(values, pd.Series):
        name = values.name or default_name
        return values.reindex(index).rename(str(name)).to_frame()
    return pd.Series(values, index=index, name=default_name).to_frame()


def _matrix(data: pd.DataFrame) -> list[list[float]]:
    return data.astype(float).values.tolist()


def _flatten_importance(values: Any) -> list[float]:
    if hasattr(values, "ravel"):
        return [float(value) for value in values.ravel()]
    if isinstance(values, list | tuple):
        if values and isinstance(values[0], list | tuple):
            return [float(value) for row in values for value in row]
        return [float(value) for value in values]
    return [float(values)]


__all__ = [
    "EqualWeightOptimizer",
    "FactorTransformer",
    "ModelAdapter",
    "PipelineResult",
    "RiskPolicy",
    "StrategyPipeline",
    "TopKSelector",
]
