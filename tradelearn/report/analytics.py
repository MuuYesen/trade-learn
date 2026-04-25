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


def top_drawdowns(returns: pd.Series, limit: int = 10) -> pd.DataFrame:
    """Return the largest drawdown episodes."""
    drawdown = metrics.drawdown_series(returns)
    episodes: list[dict[str, object]] = []
    in_drawdown = False
    peak = None
    valley = None
    max_drawdown = 0.0
    previous_date = None

    for date, value in drawdown.items():
        if value < 0 and not in_drawdown:
            in_drawdown = True
            peak = previous_date or date
            valley = date
            max_drawdown = float(value)
        elif value < 0 and in_drawdown:
            if value < max_drawdown:
                valley = date
                max_drawdown = float(value)
        elif value >= 0 and in_drawdown:
            episodes.append(
                _drawdown_episode(
                    peak=peak,
                    valley=valley,
                    recovery=date,
                    max_drawdown=max_drawdown,
                )
            )
            in_drawdown = False
        previous_date = date

    if in_drawdown:
        episodes.append(
            _drawdown_episode(
                peak=peak,
                valley=valley,
                recovery=pd.NaT,
                max_drawdown=max_drawdown,
            )
        )
    columns = ["peak", "valley", "recovery", "max_drawdown", "duration"]
    return pd.DataFrame(episodes, columns=columns).sort_values("max_drawdown").head(limit)


def exposure_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """Return daily symbol exposure weights from position values."""
    if positions.empty or not {"date", "symbol", "value"}.issubset(positions.columns):
        return pd.DataFrame()
    exposure = positions.pivot_table(
        index="date",
        columns="symbol",
        values="value",
        aggfunc="sum",
    ).sort_index()
    totals = exposure.abs().sum(axis=1)
    return exposure.div(totals.replace(0, np.nan), axis=0).fillna(0.0)


def exposure_correlation(positions: pd.DataFrame) -> pd.DataFrame:
    """Return a symbol correlation matrix from daily exposure weights."""
    exposure = exposure_weights(positions)
    if exposure.empty:
        return pd.DataFrame()
    return exposure.corr()


def _drawdown_episode(
    *,
    peak: object,
    valley: object,
    recovery: object,
    max_drawdown: float,
) -> dict[str, object]:
    """Return a drawdown episode row."""
    end = valley if pd.isna(recovery) else recovery
    duration = (end - peak).days if hasattr(end - peak, "days") else 0
    return {
        "peak": peak,
        "valley": valley,
        "recovery": recovery,
        "max_drawdown": max_drawdown,
        "duration": duration,
    }
