"""Shared helpers for metric calculations."""

from typing import Literal

import pandas as pd

NanPolicy = Literal["drop", "zero", "propagate", "raise"]


def validate_periods(periods: int) -> None:
    """Validate an annualization period count."""
    if periods <= 0:
        raise ValueError("periods must be a positive integer")


def apply_nan_policy(
    values: pd.Series | pd.DataFrame,
    nan_policy: NanPolicy = "drop",
) -> pd.Series | pd.DataFrame:
    """Apply a common NaN policy to pandas inputs."""
    if nan_policy == "drop":
        return values.dropna()
    if nan_policy == "zero":
        return values.fillna(0.0)
    if nan_policy == "propagate":
        return values
    if nan_policy == "raise":
        if values.isna().to_numpy().any():
            raise ValueError("NaN values are not allowed when nan_policy='raise'")
        return values
    raise ValueError("nan_policy must be one of: drop, zero, propagate, raise")
