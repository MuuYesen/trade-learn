"""Shared ML strategy behavior for engine and Lite facades."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from tradelearn.ml.registry import ModelRegistry

FeatureSpec = str | Callable[[Any], float]
TargetSpec = str | Callable[[pd.DataFrame], Sequence[float] | pd.Series]


class MLStrategyMixin:
    """Shared train -> predict -> signal -> order behavior."""

    params = (
        ("threshold", 0.0),
        ("size", 1.0),
        ("training_data", None),
        ("allow_short", False),
        ("model_registry", None),
    )
    model: Any = None
    features: tuple[FeatureSpec, ...] = ()
    target: TargetSpec | None = None
    threshold = 0.0
    size = 1.0
    training_data = None
    allow_short = False
    model_registry = None

    def start(self) -> None:
        if self.model is None:
            raise ValueError("MLStrategy requires a model with predict().")
        if isinstance(self.model, str):
            registry = self._ml_param("model_registry") or ModelRegistry()
            self.model_ = registry.load(self.model)
        else:
            self.model_ = copy.deepcopy(self.model)
        self.predictions: list[float] = []
        training_data = self._ml_param("training_data")
        if training_data is None:
            training_data = self._default_training_data()
        if training_data is not None and self.target is not None and hasattr(self.model_, "fit"):
            X, y = self._training_matrix(pd.DataFrame(training_data))
            if X and y:
                self.model_.fit(X, y)

    def next(self) -> None:
        prediction = self.predict_next()
        self.predictions.append(prediction)
        self.apply_prediction(prediction)

    def predict_next(self) -> float:
        vector = self.feature_vector()
        if not hasattr(self.model_, "predict"):
            raise ValueError("MLStrategy model must implement predict().")
        result = self.model_.predict([vector])
        if isinstance(result, pd.Series):
            return float(result.iloc[0])
        if isinstance(result, Sequence) and not isinstance(result, str | bytes):
            return float(result[0])
        if hasattr(result, "__len__") and not isinstance(result, str | bytes):
            return float(result[0])
        return float(result)

    def feature_vector(self) -> list[float]:
        return [self._feature_value(feature) for feature in self.features]

    def apply_prediction(self, prediction: float) -> None:
        threshold = float(self._ml_param("threshold"))
        size = float(self._ml_param("size"))
        position = self._ml_position()
        if prediction > threshold:
            if position.size <= 0:
                self.buy(size=size)
        elif prediction < -threshold:
            if position.size > 0:
                self.sell(size=position.size)
            elif bool(self._ml_param("allow_short")):
                self.sell(size=size)
        elif position:
            self.close()

    def _ml_param(self, name: str) -> Any:
        params = getattr(self, "p", None)
        if params is not None and hasattr(params, name):
            return getattr(params, name)
        return getattr(self, name)

    def _ml_position(self) -> Any:
        position = getattr(self, "position", None)
        if callable(position):
            return position()
        return position

    def _feature_value(self, feature: FeatureSpec) -> float:
        if isinstance(feature, str):
            line = getattr(self.data, feature, None)
            if line is not None:
                return float(line[0])
            frame = self._default_training_data()
            cursor = getattr(getattr(self.data, "close", None), "_cursor", None)
            if frame is None or cursor is None or feature not in frame:
                raise AttributeError(f"data feed has no feature column {feature!r}")
            return float(frame[feature].iloc[cursor])
        return float(feature(self))

    def _default_training_data(self) -> pd.DataFrame | None:
        frame = getattr(self.data, "_frame", None)
        if frame is not None:
            return frame
        feed = getattr(self.data, "_feed", None)
        return getattr(feed, "_frame", None)

    def _training_matrix(self, frame: pd.DataFrame) -> tuple[list[list[float]], list[float]]:
        features = []
        for feature in self.features:
            if isinstance(feature, str):
                features.append(frame[feature].rename(feature))
            else:
                values = feature(frame)
                features.append(pd.Series(values, index=frame.index, name=feature.__name__))
        X_frame = pd.concat(features, axis=1)
        y_series = self._target_series(frame)
        aligned = pd.concat([X_frame, y_series.rename("__target__")], axis=1).dropna()
        X = aligned.drop(columns=["__target__"]).astype(float).values.tolist()
        y = aligned["__target__"].astype(float).tolist()
        return X, y

    def _target_series(self, frame: pd.DataFrame) -> pd.Series:
        if self.target is None:
            raise ValueError("MLStrategy requires target for training.")
        if isinstance(self.target, str):
            return frame[self.target]
        values = self.target(frame)
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values, index=frame.index)


__all__ = ["FeatureSpec", "MLStrategyMixin", "TargetSpec"]
