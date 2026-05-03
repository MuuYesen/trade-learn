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


def annual_returns(returns: pd.Series) -> pd.Series:
    """Return annual compounded returns."""
    return ((1.0 + returns.dropna()).resample("YE").prod() - 1.0).rename("annual_return")


def rolling_returns(returns: pd.Series) -> pd.Series:
    """Return cumulative rolling returns."""
    return ((1.0 + returns.dropna()).cumprod() - 1.0).rename("rolling_returns")


def rolling_volatility(returns: pd.Series, window: int, periods: int) -> pd.Series:
    """Return annualized rolling volatility."""
    return (
        returns.dropna().rolling(window, min_periods=2).std() * periods ** 0.5
    ).rename("rolling_volatility")


def rolling_sharpe(returns: pd.Series, window: int, periods: int) -> pd.Series:
    """Return rolling Sharpe ratio using tradelearn.metrics.sharpe."""
    values = [
        np.nan
        if index + 1 < window
        else metrics.sharpe(returns.iloc[index + 1 - window : index + 1], periods)
        for index in range(len(returns))
    ]
    return pd.Series(values, index=returns.index, name="rolling_sharpe")


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    """Return rolling beta using tradelearn.metrics.beta."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner")
    aligned.columns = ["returns", "benchmark"]
    values = [
        np.nan
        if index + 1 < window
        else metrics.beta(
            aligned["returns"].iloc[index + 1 - window : index + 1],
            aligned["benchmark"].iloc[index + 1 - window : index + 1],
        )
        for index in range(len(aligned))
    ]
    return pd.Series(values, index=aligned.index, name="rolling_beta")


def active_returns(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Return strategy returns minus benchmark returns on aligned dates."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner")
    aligned.columns = ["returns", "benchmark"]
    return (aligned["returns"] - aligned["benchmark"]).rename("active_return")


def active_return(returns: pd.Series, benchmark: pd.Series) -> float:
    """Return compounded strategy excess return over benchmark."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    aligned.columns = ["returns", "benchmark"]
    strategy_return = float((1.0 + aligned["returns"]).prod() - 1.0)
    benchmark_return = float((1.0 + aligned["benchmark"]).prod() - 1.0)
    return strategy_return - benchmark_return


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods: int) -> float:
    """Return annualized active return volatility."""
    active = active_returns(returns, benchmark).dropna()
    if active.empty:
        return np.nan
    return float(active.std(ddof=1) * periods ** 0.5)


def active_weights(
    positions: pd.DataFrame,
    benchmark_weights: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """Return active position weights against optional benchmark weights."""
    exposure = exposure_weights(positions)
    if exposure.empty:
        return pd.DataFrame()
    if benchmark_weights is None:
        return exposure.rename_axis(index="date")
    benchmark = pd.DataFrame(benchmark_weights).copy()
    if isinstance(benchmark_weights, pd.Series):
        benchmark = benchmark_weights.to_frame().T
    benchmark = benchmark.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    columns = sorted(set(exposure.columns) | set(benchmark.columns))
    aligned_exposure = exposure.reindex(columns=columns, fill_value=0.0)
    aligned_benchmark = benchmark.reindex(
        index=aligned_exposure.index,
        columns=columns,
        fill_value=0.0,
    )
    return (aligned_exposure - aligned_benchmark).rename_axis(index="date")


def performance_attribution(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Return high-level benchmark-relative return attribution rows."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    aligned.columns = ["returns", "benchmark"]
    if aligned.empty:
        strategy_return = np.nan
        benchmark_return = np.nan
    else:
        strategy_return = float((1.0 + aligned["returns"]).prod() - 1.0)
        benchmark_return = float((1.0 + aligned["benchmark"]).prod() - 1.0)
    return pd.DataFrame(
        [
            {"component": "strategy_return", "value": strategy_return},
            {"component": "benchmark_return", "value": benchmark_return},
            {"component": "active_return", "value": strategy_return - benchmark_return},
        ]
    )


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


def trade_distribution(trades: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Return trade PnL histogram bins with summary stats in attrs."""
    if trades.empty or "pnl" not in trades:
        result = pd.DataFrame(columns=["left", "right", "count"])
        result.attrs["mean"] = np.nan
        result.attrs["median"] = np.nan
        return result
    pnl = pd.Series(trades["pnl"], dtype="float64")
    bucket_count = min(bins, max(1, len(pnl)))
    histogram, edges = np.histogram(pnl, bins=bucket_count)
    result = pd.DataFrame(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "count": histogram,
        }
    )
    result.attrs["mean"] = float(pnl.mean())
    result.attrs["median"] = float(pnl.median())
    return result


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
    exposure = exposure.apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
