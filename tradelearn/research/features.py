from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradelearn.core import ContractError, validate_bars


class FeatureSet:
    """Executable feature definitions for research pipelines."""

    def __init__(
        self,
        features: Mapping[str, Any],
        *,
        target: Mapping[str, Any] | None = None,
    ) -> None:
        self.features = dict(features)
        self.target = None if target is None else dict(target)

    def fit(self, data: Any) -> FeatureSet:
        """No-op fit so FeatureSet can be used inside Pipeline."""
        return self

    def transform(self, data: Any, *, include_target: bool = False) -> pd.DataFrame:
        """Build features, excluding target columns by default."""
        return self._transform(data, include_target=include_target)

    def fit_transform(self, data: Any, *, include_target: bool = False) -> pd.DataFrame:
        """Build features while keeping Pipeline semantics."""
        return self.transform(data, include_target=include_target)

    def _transform(self, data: Any, *, include_target: bool) -> pd.DataFrame:
        specs = dict(self.features)
        if include_target and self.target:
            specs.update(self.target)
        return _PanelView(data).to_dataset(specs)

    def get_params(self) -> dict[str, Any]:
        """Return serializable feature builder parameters."""
        return {
            "type": type(self).__name__,
            "features": list(self.features),
            "target": [] if self.target is None else list(self.target),
        }


class _PanelView:
    """Internal wide-table view over MultiIndex(timestamp, symbol) bars."""

    def __init__(self, bars: Any) -> None:
        self.bars = validate_bars(pd.DataFrame(bars).copy())

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name in self.bars.columns:
            return self.field(name)
        raise AttributeError(name)

    def field(self, name: str) -> pd.DataFrame:
        if name not in self.bars.columns:
            raise KeyError(f"bars column not found: {name}")
        wide = self.bars[name].unstack("symbol")
        wide.index.name = "timestamp"
        return wide.sort_index()

    def to_dataset(self, features: Mapping[str, Any]) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for name, spec in features.items():
            value = spec(self) if callable(spec) else spec
            columns[str(name)] = _feature_series(value, name=str(name))
        dataset = pd.concat(columns, axis=1)
        dataset.index.names = ["timestamp", "symbol"]
        return dataset.sort_index()


def _feature_series(value: object, *, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        series = value.stack(future_stack=True)
    else:
        series = pd.Series(value)
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ContractError(
            f"FeatureSet feature {name!r} must produce MultiIndex(timestamp, symbol) values"
        )
    series = series.copy()
    series.index.names = ["timestamp", "symbol"]
    series.name = name
    return series


FeatureBuilder = FeatureSet

__all__ = ["FeatureBuilder", "FeatureSet"]
