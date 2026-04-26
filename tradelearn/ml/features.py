"""Versioned feature computation and parquet-backed reuse."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.data.bars import bars_fingerprint

FeatureFn = Callable[[pd.DataFrame], pd.Series | pd.DataFrame | Iterable[float]]


@dataclass(frozen=True)
class FeatureDefinition:
    """Registered feature metadata."""

    name: str
    version: str
    func: FeatureFn
    factor_type: str
    horizon: int


def feature(
    *,
    name: str,
    version: int | str,
    factor_type: str = "custom",
    horizon: int = 1,
) -> Callable[[FeatureFn], FeatureFn]:
    """Annotate a callable as a versioned FeatureStore feature."""

    def decorator(func: FeatureFn) -> FeatureFn:
        func.feature_name = name
        func.feature_version = str(version)
        func.feature_type = factor_type
        func.feature_horizon = int(horizon)
        return func

    return decorator


class FeatureStore:
    """Compute and reuse versioned factor features for Bars inputs."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, func: FeatureFn) -> FeatureFn:
        """Register one decorated feature callable."""
        definition = _definition_from(func)
        self._features[definition.name] = definition
        return func

    def compute(self, bars: pd.DataFrame, features: Iterable[str | FeatureFn]) -> pd.DataFrame:
        """Return a Factor DataFrame for selected features, using cached values when present."""
        frames: list[pd.DataFrame] = []
        definitions: list[FeatureDefinition] = []
        for item in features:
            definition = self._resolve(item)
            definitions.append(definition)
            frames.append(self._read_or_compute(bars, definition))
        if not frames:
            result = pd.DataFrame(index=bars.index)
        else:
            result = pd.concat(frames, axis=1)
        result.attrs.update(_factor_attrs(definitions))
        return result

    def exists(self, bars: pd.DataFrame, feature_name: str) -> bool:
        """Return whether the selected feature is cached for the Bars fingerprint."""
        definition = self._features[feature_name]
        return self._data_path(bars, definition).exists() and self._meta_path(
            bars,
            definition,
        ).exists()

    def _resolve(self, item: str | FeatureFn) -> FeatureDefinition:
        if isinstance(item, str):
            if item not in self._features:
                raise KeyError(f"unknown feature: {item}")
            return self._features[item]
        definition = _definition_from(item)
        self._features.setdefault(definition.name, definition)
        return definition

    def _read_or_compute(self, bars: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        data_path = self._data_path(bars, definition)
        meta_path = self._meta_path(bars, definition)
        if data_path.exists() and meta_path.exists():
            frame = pd.read_parquet(data_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame.attrs.update(meta["attrs"])
            return frame

        frame = self._compute_one(bars, definition)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_path)
        meta_path.write_text(
            json.dumps(
                {
                    "name": definition.name,
                    "version": definition.version,
                    "bars_fingerprint": bars_fingerprint(bars),
                    "attrs": dict(frame.attrs),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return frame

    def _compute_one(self, bars: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        values = definition.func(bars)
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
            if list(frame.columns) != [definition.name]:
                frame = frame.iloc[:, [0]].rename(columns={frame.columns[0]: definition.name})
        elif isinstance(values, pd.Series):
            frame = values.rename(definition.name).to_frame()
        else:
            frame = pd.Series(values, index=bars.index, name=definition.name).to_frame()
        frame = frame.reindex(bars.index)
        frame.attrs.update(_factor_attrs([definition]))
        return frame

    def _data_path(self, bars: pd.DataFrame, definition: FeatureDefinition) -> Path:
        return self.root / definition.name / definition.version / (
            f"{bars_fingerprint(bars)}.parquet"
        )

    def _meta_path(self, bars: pd.DataFrame, definition: FeatureDefinition) -> Path:
        return self.root / definition.name / definition.version / f"{bars_fingerprint(bars)}.json"


def _definition_from(func: FeatureFn) -> FeatureDefinition:
    name = getattr(func, "feature_name", getattr(func, "__name__", None))
    if not name:
        raise ValueError("feature callable requires a name")
    return FeatureDefinition(
        name=str(name),
        version=str(getattr(func, "feature_version", "1")),
        func=func,
        factor_type=str(getattr(func, "feature_type", "custom")),
        horizon=int(getattr(func, "feature_horizon", 1)),
    )


def _factor_attrs(definitions: list[FeatureDefinition]) -> dict[str, Any]:
    if not definitions:
        return {"factor_type": "custom", "horizon": 1, "version": {}}
    factor_types = {definition.factor_type for definition in definitions}
    horizons = {definition.horizon for definition in definitions}
    return {
        "factor_type": next(iter(factor_types)) if len(factor_types) == 1 else "mixed",
        "horizon": next(iter(horizons)) if len(horizons) == 1 else max(horizons),
        "version": {definition.name: definition.version for definition in definitions},
    }
