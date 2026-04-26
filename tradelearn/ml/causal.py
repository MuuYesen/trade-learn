"""Causal feature selection helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

CausalBackend = Callable[[pd.DataFrame, pd.Series], dict[str, float] | pd.Series]


@dataclass
class CausalSelector:
    """Select candidate causal features with a sklearn-style API."""

    max_features: int | None = None
    min_score: float = 0.0
    backend: CausalBackend | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CausalSelector:
        """Fit selector scores and selected feature names."""
        frame = pd.DataFrame(X)
        target = pd.Series(y, index=frame.index).astype(float)
        scores = self._score(frame, target)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = [name for name, score in ranked if score >= self.min_score]
        if self.max_features is not None:
            selected = selected[: self.max_features]
        self.scores_ = scores
        self.selected_features_ = selected
        return self

    def select(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Return selected feature names after fitting."""
        return self.fit(X, y).selected_features_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing only selected features."""
        if not hasattr(self, "selected_features_"):
            raise ValueError("CausalSelector must be fitted before transform().")
        return pd.DataFrame(X).loc[:, self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and return selected feature columns."""
        return self.fit(X, y).transform(X)

    def _score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if self.backend is not None:
            raw_scores = self.backend(X, y)
            if isinstance(raw_scores, pd.Series):
                return {str(name): float(score) for name, score in raw_scores.items()}
            return {str(name): float(score) for name, score in raw_scores.items()}
        aligned = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        if aligned.empty:
            return {str(column): 0.0 for column in X.columns}
        target = aligned["__target__"].astype(float)
        scores: dict[str, float] = {}
        for column in X.columns:
            feature = aligned[column].astype(float)
            if feature.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
                scores[str(column)] = 0.0
                continue
            score = feature.corr(target)
            scores[str(column)] = 0.0 if pd.isna(score) else abs(float(score))
        return scores
