from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


class SymbolicFeatureGenerator:
    """Generate symbolic features through optional gplearn.

    This class owns the heavy symbolic-transformer configuration and follows
    the sklearn-style ``fit`` / ``transform`` split to avoid train/test leakage.
    """

    def __init__(
        self,
        *,
        n_features: int = 20,
        random_state: int | None = None,
        generations: int = 3,
        population_size: int = 100,
        metric: str = "pearson",
        **kwargs: Any,
    ) -> None:
        self.n_features = int(n_features)
        self.random_state = random_state
        self.generations = int(generations)
        self.population_size = int(population_size)
        self.metric = str(metric)
        self.kwargs = dict(kwargs)
        self.transformer_: Any | None = None
        self.input_columns_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.fill_values_: pd.Series | None = None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str],
        target: str,
    ) -> SymbolicFeatureGenerator:
        """Fit the symbolic transformer on training data."""
        transformer_type = _load_symbolic_transformer()
        frame = pd.DataFrame(data)
        feature_columns = [str(column) for column in columns]
        target_name = str(target)
        clean = frame[feature_columns + [target_name]].dropna()
        if clean.empty:
            raise ValueError("symbolic feature training data is empty after dropping NA")

        transformer = transformer_type(
            n_components=self.n_features,
            generations=self.generations,
            population_size=self.population_size,
            random_state=self.random_state,
            metric=self.metric,
            feature_names=feature_columns,
            **self.kwargs,
        )
        transformer.fit(clean[feature_columns], clean[target_name])
        self.transformer_ = transformer
        self.input_columns_ = feature_columns
        self.feature_names_ = [f"symbolic_{i + 1}" for i in range(self.n_features)]
        self.fill_values_ = clean[feature_columns].median()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the fitted symbolic transformer."""
        if self.transformer_ is None or self.input_columns_ is None:
            raise RuntimeError("fit() must be called before transform()")
        frame = pd.DataFrame(data)
        feature_frame = frame[self.input_columns_]
        fill_values = self.fill_values_
        if fill_values is not None:
            feature_frame = feature_frame.fillna(fill_values)
        values = np.asarray(
            self.transformer_.transform(feature_frame)
        )
        columns_out = self.feature_names_ or [
            f"symbolic_{i + 1}" for i in range(values.shape[1])
        ]
        return pd.DataFrame(values, index=frame.index, columns=columns_out)

    def fit_transform(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str],
        target: str,
    ) -> pd.DataFrame:
        """Fit the symbolic transformer and return generated features."""
        return self.fit(data, columns=columns, target=target).transform(data)

    def get_params(self) -> dict[str, Any]:
        """Return serializable generator parameters for tracking artifacts."""
        params = {
            "type": type(self).__name__,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "generations": self.generations,
            "population_size": self.population_size,
            "metric": self.metric,
        }
        params.update(self.kwargs)
        return params


def _load_symbolic_transformer():
    try:
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise ImportError(
            "SymbolicFeatureGenerator requires optional dependency `gplearn`. "
            "Install it with `pip install gplearn`."
        ) from exc
    return SymbolicTransformer


__all__ = ["SymbolicFeatureGenerator"]
