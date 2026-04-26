"""ML strategy base class built on the backtest Strategy facade."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from tradelearn.backtest import Strategy

FeatureSpec = str | Callable[[Any], float]
TargetSpec = str | Callable[[pd.DataFrame], Sequence[float] | pd.Series]


class MLStrategy(Strategy):
    """Base class for train -> predict -> signal -> order strategies."""

    params = (
        ("threshold", 0.0),
        ("size", 1.0),
        ("training_data", None),
        ("allow_short", False),
    )
    model: Any = None
    features: tuple[FeatureSpec, ...] = ()
    target: TargetSpec | None = None

    def start(self) -> None:
        if self.model is None:
            raise ValueError("MLStrategy requires a model with predict().")
        self.model_ = copy.deepcopy(self.model)
        self.predictions: list[float] = []
        training_data = self.p.training_data
        if training_data is None:
            training_data = getattr(self.data, "_frame", None)
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
        return float(result)

    def feature_vector(self) -> list[float]:
        return [self._feature_value(feature) for feature in self.features]

    def apply_prediction(self, prediction: float) -> None:
        threshold = float(self.p.threshold)
        size = float(self.p.size)
        if prediction > threshold:
            if self.position.size <= 0:
                self.buy(size=size)
        elif prediction < -threshold:
            if self.position.size > 0:
                self.sell(size=self.position.size)
            elif bool(self.p.allow_short):
                self.sell(size=size)
        elif self.position:
            self.close()

    def _feature_value(self, feature: FeatureSpec) -> float:
        if isinstance(feature, str):
            line = getattr(self.data, feature, None)
            if line is not None:
                return float(line[0])
            frame = getattr(self.data, "_frame", None)
            cursor = getattr(self.data.close, "_cursor", None)
            if frame is None or cursor is None or feature not in frame:
                raise AttributeError(f"data feed has no feature column {feature!r}")
            return float(frame[feature].iloc[cursor])
        return float(feature(self))

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
