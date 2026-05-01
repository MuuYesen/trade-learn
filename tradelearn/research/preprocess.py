from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tradelearn.research.run import tracked


@tracked(category="preprocess")
def fill_missing(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "median",
    value: float | None = None,
) -> pd.DataFrame:
    """Fill missing values for selected columns."""
    frame = pd.DataFrame(data).copy()
    target_columns = _columns(frame, columns)
    for column in target_columns:
        if method == "median":
            fill_value = frame[column].median()
        elif method == "mean":
            fill_value = frame[column].mean()
        elif method == "zero":
            fill_value = 0.0
        elif method == "constant":
            fill_value = value
        else:
            raise ValueError(f"Unsupported fill method: {method}")
        frame[column] = frame[column].fillna(fill_value)
    return frame


@tracked(category="preprocess")
def winsorize(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    limits: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Clip selected columns to quantile limits."""
    frame = pd.DataFrame(data).copy()
    lower_q, upper_q = limits
    for column in _columns(frame, columns):
        lower = frame[column].quantile(lower_q)
        upper = frame[column].quantile(upper_q)
        frame[column] = frame[column].clip(lower=lower, upper=upper)
    return frame


@tracked(category="preprocess")
def standardize(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    ddof: int = 0,
) -> pd.DataFrame:
    """Z-score selected columns."""
    frame = pd.DataFrame(data).copy()
    for column in _columns(frame, columns):
        mean = frame[column].mean()
        std = frame[column].std(ddof=ddof)
        frame[column] = 0.0 if not std or np.isnan(std) else (frame[column] - mean) / std
    return frame


@tracked(category="preprocess")
def neutralize(
    data: pd.DataFrame,
    *,
    exposures: pd.DataFrame,
    columns: Sequence[str],
    method: str = "ols",
) -> pd.DataFrame:
    """Neutralize selected columns against exposure columns using OLS residuals."""
    if method != "ols":
        raise ValueError(f"Unsupported neutralize method: {method}")
    frame = pd.DataFrame(data).copy()
    exposure_frame = pd.DataFrame(exposures).reindex(frame.index)
    x = exposure_frame.astype("float64")
    x = x.fillna(x.median(numeric_only=True))
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype="float64")])
    for column in columns:
        y = frame[column].astype("float64")
        mask = y.notna()
        if not mask.any():
            continue
        beta, *_ = np.linalg.lstsq(design[mask.to_numpy()], y[mask].to_numpy(), rcond=None)
        fitted = design @ beta
        frame[column] = y - fitted
    return frame


def _columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return [str(column) for column in frame.select_dtypes("number").columns]
    return [str(column) for column in columns]


__all__ = ["fill_missing", "neutralize", "standardize", "winsorize"]
