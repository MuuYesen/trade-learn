"""Shared helpers for metric calculations."""

from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

NanPolicy = Literal["drop", "zero", "propagate", "raise"]


def validate_periods(periods: int) -> None:
    """Validate an annualization period count."""
    if isinstance(periods, (bool, np.bool_)) or not isinstance(periods, Integral) or periods <= 0:
        raise ValueError("periods must be a positive integer")


def apply_nan_policy(
    values: pd.Series | pd.DataFrame,
    nan_policy: NanPolicy = "drop",
) -> pd.Series | pd.DataFrame:
    """Apply a common NaN policy to pandas inputs."""
    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Metric inputs must not contain infinite values")
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
