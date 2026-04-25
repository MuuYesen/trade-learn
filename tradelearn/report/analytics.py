"""Shared report analytics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradelearn import metrics


def monthly_returns_matrix(returns: pd.Series) -> pd.DataFrame:
    """Return monthly returns pivoted by year and month."""
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    frame = monthly.to_frame("return")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    result = frame.pivot(index="year", columns="month", values="return")
    result["year_total"] = (1.0 + result).prod(axis=1) - 1.0
    result.loc["month_avg"] = result.mean(axis=0)
    return result


def rolling_sharpe(returns: pd.Series, window: int, periods: int) -> pd.Series:
    """Return rolling Sharpe ratio using tradelearn.metrics.sharpe."""
    values = [
        np.nan
        if index + 1 < window
        else metrics.sharpe(returns.iloc[index + 1 - window : index + 1], periods)
        for index in range(len(returns))
    ]
    return pd.Series(values, index=returns.index, name="rolling_sharpe")
