"""Scikit-learn style strategy research pipeline.

This module stays above the backtest runtime.  It turns research data into
scores and target weights; ``engine`` / ``lite`` strategies decide when to pass
those weights into ``target_weights()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

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

    def __init__(self, features: Sequence[FeatureInput]) -> None:
        self.features = tuple(features)
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
                if feature not in data:
                    raise KeyError(f"feature column {feature!r} not found")
                frames.append(data[[feature]].copy())
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


class TopKSelector:
    """Select the top-k symbols by score."""

    def __init__(
        self,
        k: int,
        *,
        ascending: bool = False,
        threshold: float | None = None,
    ) -> None:
        self.k = int(k)
        self.ascending = bool(ascending)
        self.threshold = threshold

    def select(self, scores: pd.Series) -> list[str]:
        """Return selected symbol labels."""
        ranked = scores.dropna().sort_values(ascending=self.ascending)
        if self.threshold is not None:
            if self.ascending:
                ranked = ranked[ranked <= self.threshold]
            else:
                ranked = ranked[ranked >= self.threshold]
        return [str(symbol) for symbol in ranked.head(self.k).index]


class EqualWeightOptimizer:
    """Build equal weights for selected symbols."""

    def __init__(self, gross: float = 1.0) -> None:
        self.gross = float(gross)

    def optimize(self, selected: Sequence[str], scores: pd.Series | None = None) -> pd.Series:
        """Return equal positive weights for selected symbols."""
        if not selected:
            return pd.Series(dtype="float64")
        weight = self.gross / len(selected)
        return pd.Series({str(symbol): weight for symbol in selected}, dtype="float64")


class RiskPolicy:
    """Post-process portfolio weights with simple risk constraints."""

    def __init__(
        self,
        *,
        max_weight: float | None = None,
        min_abs_weight: float = 0.0,
        normalize: bool = False,
    ) -> None:
        self.max_weight = max_weight
        self.min_abs_weight = float(min_abs_weight)
        self.normalize = bool(normalize)

    def apply(self, weights: pd.Series) -> pd.Series:
        """Apply clipping, pruning, and optional normalization."""
        adjusted = weights.astype(float).copy()
        if self.max_weight is not None:
            cap = float(self.max_weight)
            adjusted = adjusted.clip(lower=-cap, upper=cap)
        if self.min_abs_weight > 0:
            adjusted = adjusted[adjusted.abs() >= self.min_abs_weight]
        if self.normalize and not adjusted.empty:
            gross = adjusted.abs().sum()
            if gross > 0:
                adjusted = adjusted / gross
        return adjusted


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

    def _step_of_type(self, cls: type[Any]) -> Any | None:
        for _name, step in self.steps:
            if isinstance(step, cls):
                return step
        return None


def _feature_frame(values: Any, index: pd.Index, default_name: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reindex(index).copy()
    if isinstance(values, pd.Series):
        name = values.name or default_name
        return values.reindex(index).rename(str(name)).to_frame()
    return pd.Series(values, index=index, name=default_name).to_frame()


def _matrix(data: pd.DataFrame) -> list[list[float]]:
    return data.astype(float).values.tolist()


__all__ = [
    "EqualWeightOptimizer",
    "FactorTransformer",
    "ModelAdapter",
    "PipelineResult",
    "RiskPolicy",
    "StrategyPipeline",
    "TopKSelector",
]
