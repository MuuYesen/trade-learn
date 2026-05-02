from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.research.run import current_run, tracked


@tracked(category="preprocess")
def fill_missing(
    data: pd.DataFrame | pd.Series,
    *,
    columns: Sequence[str] | None = None,
    method: str = "median",
    value: float | None = None,
    by: str | Sequence[str] | None = None,
) -> pd.DataFrame | pd.Series:
    """Fill missing values for selected columns."""
    frame, restore = _as_frame(data)
    target_columns = _columns(frame, columns)

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for column in target_columns:
            if method == "median":
                fill_value = group[column].median()
            elif method == "mean":
                fill_value = group[column].mean()
            elif method == "zero":
                fill_value = 0.0
            elif method == "constant":
                fill_value = value
            else:
                raise ValueError(f"Unsupported fill method: {method}")
            group[column] = group[column].fillna(fill_value)
        return group

    return restore(_apply_by(frame, by, transform))


@tracked(category="preprocess")
def fill_by_group(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    group: str,
    fallback: str = "median",
) -> pd.DataFrame:
    """Fill missing values by group-level statistics.

    This is the current API equivalent of 1.x ``Miss.replace_nan_indu``.
    """
    frame = pd.DataFrame(data).copy()
    target_columns = _columns(frame, columns)
    if group not in frame.columns:
        raise KeyError(f"group column not found: {group}")
    grouped = frame.groupby(group, dropna=False)[target_columns]
    if fallback == "median":
        group_values = grouped.transform("median")
        fallback_values = frame[target_columns].median(numeric_only=True)
    elif fallback == "mean":
        group_values = grouped.transform("mean")
        fallback_values = frame[target_columns].mean(numeric_only=True)
    else:
        raise ValueError(f"Unsupported fallback: {fallback}")
    frame[target_columns] = frame[target_columns].fillna(group_values)
    frame[target_columns] = frame[target_columns].fillna(fallback_values)
    return frame


@tracked(category="preprocess")
def clip_outliers(
    data: pd.DataFrame | pd.Series,
    *,
    columns: Sequence[str] | None = None,
    limits: tuple[float, float] = (0.01, 0.99),
    by: str | Sequence[str] | None = None,
) -> pd.DataFrame | pd.Series:
    """Alias for quantile winsorization with a clearer preprocessing name."""
    return _winsorize_impl(data, columns=columns, limits=limits, by=by)


@tracked(category="preprocess")
def winsorize_mad(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    scale: float = 5.0,
) -> pd.DataFrame:
    """Clip selected columns by median absolute deviation.

    This is the current API equivalent of 1.x ``Outlier.winorize_med``.
    """
    frame = pd.DataFrame(data).copy()
    for column in _columns(frame, columns):
        values = frame[column].astype("float64")
        median = values.median()
        mad = (values - median).abs().median()
        if pd.isna(mad):
            continue
        lower = median - float(scale) * mad
        upper = median + float(scale) * mad
        frame[column] = values.clip(lower=lower, upper=upper)
    return frame


@tracked(category="preprocess")
def rank(
    data: pd.DataFrame | pd.Series,
    *,
    columns: Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    pct: bool = True,
    ascending: bool = True,
    method: str = "average",
) -> pd.DataFrame | pd.Series:
    """Rank selected columns, optionally by date/symbol group."""
    frame, restore = _as_frame(data)

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for column in _columns(group, columns):
            group[column] = group[column].rank(
                pct=pct,
                ascending=ascending,
                method=method,
            )
        return group

    return restore(_apply_by(frame, by, transform))


@tracked(category="preprocess")
def label_by_quantile(
    data: pd.DataFrame,
    *,
    target: str,
    positive_quantile: float = 0.7,
    negative_quantile: float = 0.3,
    label: str = "label",
    positive: int = 1,
    negative: int = 0,
    drop_target: bool = False,
) -> pd.DataFrame:
    """Create binary labels from target quantiles.

    Rows between the negative and positive thresholds keep missing labels.
    This is the current API equivalent of 1.x ``Label.label_by_percent``.
    """
    frame = pd.DataFrame(data).copy()
    if target not in frame.columns:
        raise KeyError(f"target column not found: {target}")
    values = frame[target]
    upper = values.quantile(float(positive_quantile))
    lower = values.quantile(float(negative_quantile))
    frame[label] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[values >= upper, label] = int(positive)
    frame.loc[values <= lower, label] = int(negative)
    if drop_target:
        frame = frame.drop(columns=[target])
    return frame


class Winsorizer:
    """Train/test-safe quantile clipping transformer."""

    def __init__(
        self,
        *,
        columns: Sequence[str] | None = None,
        limits: tuple[float, float] = (0.01, 0.99),
        by: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = None if columns is None else [str(column) for column in columns]
        self.limits = limits
        self.by = by
        self.columns_: list[str] | None = None
        self.lower_: dict[str, float] | None = None
        self.upper_: dict[str, float] | None = None
        self.group_lower_: dict[tuple[Any, ...], dict[str, float]] | None = None
        self.group_upper_: dict[tuple[Any, ...], dict[str, float]] | None = None

    def fit(self, data: pd.DataFrame | pd.Series) -> Winsorizer:
        """Fit clipping bounds from training data."""
        _record_transformer_step(self)
        frame, _ = _as_frame(data)
        self.columns_ = _columns(frame, self.columns)
        lower_q, upper_q = self.limits
        self.lower_ = {
            column: float(frame[column].quantile(lower_q)) for column in self.columns_
        }
        self.upper_ = {
            column: float(frame[column].quantile(upper_q)) for column in self.columns_
        }
        if self.by is not None:
            self.group_lower_ = {}
            self.group_upper_ = {}
            keys = _group_keys(frame, self.by)
            for key, group in frame.groupby(keys, sort=False, dropna=False):
                normalized = _normalise_group_key(key)
                self.group_lower_[normalized] = {
                    column: float(group[column].quantile(lower_q)) for column in self.columns_
                }
                self.group_upper_[normalized] = {
                    column: float(group[column].quantile(upper_q)) for column in self.columns_
                }
        return self

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Clip data using fitted training bounds."""
        _require_fitted(self.columns_, self.lower_, self.upper_)
        frame, restore = _as_frame(data)
        assert self.columns_ is not None
        assert self.lower_ is not None
        assert self.upper_ is not None

        def transform_group(
            group: pd.DataFrame,
            key: tuple[Any, ...] | None = None,
        ) -> pd.DataFrame:
            group = group.copy()
            lower = self.lower_
            upper = self.upper_
            if key is not None and self.group_lower_ is not None and self.group_upper_ is not None:
                lower = self.group_lower_.get(key, self.lower_)
                upper = self.group_upper_.get(key, self.upper_)
            for column in self.columns_:
                group[column] = group[column].clip(lower=lower[column], upper=upper[column])
            return group

        if self.by is None:
            return restore(transform_group(frame))
        keys = _group_keys(frame, self.by)
        pieces = [
            transform_group(group, _normalise_group_key(key))
            for key, group in frame.groupby(keys, sort=False, dropna=False)
        ]
        return restore(pd.concat(pieces).reindex(frame.index))

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Fit bounds and return transformed data."""
        return self.fit(data).transform(data)

    def get_params(self) -> dict[str, Any]:
        """Return serializable transformer parameters for tracking artifacts."""
        return {
            "type": type(self).__name__,
            "columns": self.columns,
            "limits": self.limits,
            "by": self.by,
        }


class StandardScaler:
    """Train/test-safe z-score transformer."""

    def __init__(
        self,
        *,
        columns: Sequence[str] | None = None,
        ddof: int = 0,
        by: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = None if columns is None else [str(column) for column in columns]
        self.ddof = int(ddof)
        self.by = by
        self.columns_: list[str] | None = None
        self.mean_: dict[str, float] | None = None
        self.scale_: dict[str, float] | None = None
        self.group_mean_: dict[tuple[Any, ...], dict[str, float]] | None = None
        self.group_scale_: dict[tuple[Any, ...], dict[str, float]] | None = None

    def fit(self, data: pd.DataFrame | pd.Series) -> StandardScaler:
        """Fit scaling statistics from training data."""
        _record_transformer_step(self)
        frame, _ = _as_frame(data)
        self.columns_ = _columns(frame, self.columns)
        self.mean_ = {column: float(frame[column].mean()) for column in self.columns_}
        self.scale_ = {
            column: _safe_scale(frame[column].std(ddof=self.ddof)) for column in self.columns_
        }
        if self.by is not None:
            self.group_mean_ = {}
            self.group_scale_ = {}
            keys = _group_keys(frame, self.by)
            for key, group in frame.groupby(keys, sort=False, dropna=False):
                normalized = _normalise_group_key(key)
                self.group_mean_[normalized] = {
                    column: float(group[column].mean()) for column in self.columns_
                }
                self.group_scale_[normalized] = {
                    column: _safe_scale(group[column].std(ddof=self.ddof))
                    for column in self.columns_
                }
        return self

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Scale data using fitted training statistics."""
        _require_fitted(self.columns_, self.mean_, self.scale_)
        frame, restore = _as_frame(data)
        assert self.columns_ is not None
        assert self.mean_ is not None
        assert self.scale_ is not None

        def transform_group(
            group: pd.DataFrame,
            key: tuple[Any, ...] | None = None,
        ) -> pd.DataFrame:
            group = group.copy()
            mean = self.mean_
            scale = self.scale_
            if key is not None and self.group_mean_ is not None and self.group_scale_ is not None:
                mean = self.group_mean_.get(key, self.mean_)
                scale = self.group_scale_.get(key, self.scale_)
            for column in self.columns_:
                group[column] = (group[column] - mean[column]) / scale[column]
            return group

        if self.by is None:
            return restore(transform_group(frame))
        keys = _group_keys(frame, self.by)
        pieces = [
            transform_group(group, _normalise_group_key(key))
            for key, group in frame.groupby(keys, sort=False, dropna=False)
        ]
        return restore(pd.concat(pieces).reindex(frame.index))

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Fit scaling statistics and return transformed data."""
        return self.fit(data).transform(data)

    def get_params(self) -> dict[str, Any]:
        """Return serializable transformer parameters for tracking artifacts."""
        return {
            "type": type(self).__name__,
            "columns": self.columns,
            "ddof": self.ddof,
            "by": self.by,
        }


class Neutralizer:
    """Train/test-safe exposure neutralization transformer."""

    def __init__(
        self,
        *,
        columns: Sequence[str],
        method: str = "ols",
    ) -> None:
        self.columns = [str(column) for column in columns]
        self.method = method
        self.columns_: list[str] | None = None
        self.exposure_columns_: list[str] | None = None
        self.exposure_fill_values_: pd.Series | None = None
        self.coef_: dict[str, np.ndarray] | None = None

    def fit(self, data: pd.DataFrame, *, exposures: pd.DataFrame) -> Neutralizer:
        """Fit exposure coefficients from training data."""
        _record_transformer_step(self)
        if self.method != "ols":
            raise ValueError(f"Unsupported neutralize method: {self.method}")
        frame = pd.DataFrame(data)
        exposure_frame = pd.DataFrame(exposures).reindex(frame.index)
        self.columns_ = list(self.columns)
        self.exposure_columns_ = [str(column) for column in exposure_frame.columns]
        x = exposure_frame.astype("float64")
        self.exposure_fill_values_ = x.median(numeric_only=True)
        x = x.fillna(self.exposure_fill_values_)
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype="float64")])
        self.coef_ = {}
        for column in self.columns_:
            y = frame[column].astype("float64")
            mask = y.notna()
            if not mask.any():
                self.coef_[column] = np.zeros(design.shape[1], dtype="float64")
                continue
            beta, *_ = np.linalg.lstsq(
                design[mask.to_numpy()],
                y[mask].to_numpy(),
                rcond=None,
            )
            self.coef_[column] = beta
        return self

    def transform(self, data: pd.DataFrame, *, exposures: pd.DataFrame) -> pd.DataFrame:
        """Neutralize data using fitted training exposure coefficients."""
        _require_fitted(self.columns_, self.exposure_columns_, self.coef_)
        frame = pd.DataFrame(data).copy()
        assert self.columns_ is not None
        assert self.exposure_columns_ is not None
        assert self.coef_ is not None
        exposure_frame = pd.DataFrame(exposures).reindex(frame.index)
        x = exposure_frame[self.exposure_columns_].astype("float64")
        if self.exposure_fill_values_ is not None:
            x = x.fillna(self.exposure_fill_values_)
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype="float64")])
        for column in self.columns_:
            y = frame[column].astype("float64")
            frame[column] = y - design @ self.coef_[column]
        return frame

    def fit_transform(self, data: pd.DataFrame, *, exposures: pd.DataFrame) -> pd.DataFrame:
        """Fit coefficients and return neutralized data."""
        return self.fit(data, exposures=exposures).transform(data, exposures=exposures)

    def get_params(self) -> dict[str, Any]:
        """Return serializable transformer parameters for tracking artifacts."""
        return {
            "type": type(self).__name__,
            "columns": self.columns,
            "method": self.method,
        }


def _columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return [str(column) for column in frame.select_dtypes("number").columns]
    return [str(column) for column in columns]


def _winsorize_impl(
    data: pd.DataFrame | pd.Series,
    *,
    columns: Sequence[str] | None,
    limits: tuple[float, float],
    by: str | Sequence[str] | None,
) -> pd.DataFrame | pd.Series:
    frame, restore = _as_frame(data)
    lower_q, upper_q = limits

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for column in _columns(group, columns):
            lower = group[column].quantile(lower_q)
            upper = group[column].quantile(upper_q)
            group[column] = group[column].clip(lower=lower, upper=upper)
        return group

    return restore(_apply_by(frame, by, transform))


def _as_frame(
    data: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, Callable[[pd.DataFrame], pd.DataFrame | pd.Series]]:
    if isinstance(data, pd.Series):
        name = data.name or "value"
        frame = data.to_frame(name=name)

        def restore(result: pd.DataFrame) -> pd.Series:
            return result[name].rename(data.name)

        return frame, restore

    frame = pd.DataFrame(data).copy()
    return frame, lambda result: result


def _apply_by(
    frame: pd.DataFrame,
    by: str | Sequence[str] | None,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    if by is None:
        return transform(frame)
    keys = _group_keys(frame, by)
    grouped = frame.groupby(keys, group_keys=False, sort=False, dropna=False)
    return grouped.apply(transform)


def _group_keys(frame: pd.DataFrame, by: str | Sequence[str]) -> pd.Series | list[pd.Series]:
    names = [by] if isinstance(by, str) else list(by)
    keys = [_group_key(frame, name) for name in names]
    return keys[0] if len(keys) == 1 else keys


def _group_key(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    if isinstance(frame.index, pd.MultiIndex) and name in frame.index.names:
        return pd.Series(frame.index.get_level_values(name), index=frame.index, name=name)
    if frame.index.name == name:
        return pd.Series(frame.index, index=frame.index, name=name)
    raise KeyError(f"group key not found: {name}")


def _normalise_group_key(key: Any) -> tuple[Any, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def _safe_scale(value: float) -> float:
    if not value or np.isnan(value):
        return 1.0
    return float(value)


def _require_fitted(*values: Any) -> None:
    if any(value is None for value in values):
        raise RuntimeError("fit() must be called before transform()")


def _record_transformer_step(transformer: Any) -> None:
    run = current_run()
    if run is not None:
        run.record_step(type(transformer).__name__, "preprocess", transformer.get_params())


__all__ = [
    "Neutralizer",
    "StandardScaler",
    "Winsorizer",
    "clip_outliers",
    "fill_by_group",
    "fill_missing",
    "label_by_quantile",
    "rank",
    "winsorize_mad",
]
