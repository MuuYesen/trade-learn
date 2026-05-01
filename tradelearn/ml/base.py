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
        ("size", None),
        ("training_data", None),
        ("allow_short", False),
        ("model_registry", None),
    )
    model: Any = None
    features: tuple[FeatureSpec, ...] = ()
    target: TargetSpec | None = None
    threshold = 0.0
    size = None
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
            frame = pd.DataFrame(training_data)
            if self._has_custom_feature_vector():
                X, y = self._training_matrix_from_feature_vector(frame)
            else:
                X, y = self._training_matrix(frame)
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
        size = self._ml_param("size")
        position = self._ml_position()
        if prediction > threshold:
            if position.size <= 0:
                self._ml_target_position(1.0, size=size)
        elif prediction < -threshold:
            if position.size > 0:
                self._ml_target_position(0.0, size=size)
            elif bool(self._ml_param("allow_short")):
                self._ml_target_position(-1.0, size=size)
        elif position:
            self._ml_target_position(0.0, size=size)

    def _ml_target_position(self, target: float, *, size: Any = None) -> Any:
        if size is None:
            return self.order_target_percent(target=target)
        fixed_size = float(size)
        if target > 0:
            return self.buy(size=fixed_size)
        if target < 0:
            return self.sell(size=fixed_size)
        return self.close()

    def _has_custom_feature_vector(self) -> bool:
        return type(self).feature_vector is not MLStrategyMixin.feature_vector

    def _ml_get_cursor(self) -> int:
        data = self.data
        if hasattr(data, "_cursor"):
            return int(data._cursor)
        close = getattr(data, "close", None)
        if close is not None:
            source = getattr(close, "_cursor_source", None)
            if source is not None:
                return int(source._cursor)
            return int(getattr(close, "_cursor", 0))
        return 0

    def _ml_advance_cursor(self, pos: int) -> None:
        data = self.data
        if hasattr(data, "_advance"):
            data._advance(pos)
        elif hasattr(data, "_cursor"):
            data._cursor = pos

    def _training_matrix_from_feature_vector(
        self, frame: pd.DataFrame
    ) -> tuple[list[list[float]], list[float]]:
        """Build training matrix by replaying cursor and calling feature_vector()."""
        target_series = self._target_series(frame)
        n = len(frame)
        orig_cursor = self._ml_get_cursor()
        X: list[list[float]] = []
        y: list[float] = []
        try:
            for i in range(n):
                self._ml_advance_cursor(i)
                yi = target_series.iloc[i]
                if yi != yi:  # NaN target
                    continue
                try:
                    vec = self.feature_vector()
                except (IndexError, KeyError, AttributeError):
                    continue
                if isinstance(vec, dict):
                    row = list(vec.values())
                else:
                    row = list(vec)
                if any(v != v for v in row):  # NaN feature
                    continue
                X.append([float(v) for v in row])
                y.append(float(yi))
        finally:
            self._ml_advance_cursor(orig_cursor)
        return X, y

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
