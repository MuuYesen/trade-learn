"""Risk metrics for simple return series."""

import math

import numpy as np
import pandas as pd

from tradelearn.metrics._common import NanPolicy, apply_nan_policy, validate_periods
from tradelearn.metrics.returns import annual_return, excess_returns


def volatility(
    returns: pd.Series,
    periods: int,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized volatility.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    periods : int
        Periods per year.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Sample standard deviation annualized by ``sqrt(periods)``.

    Examples
    --------
    >>> import pandas as pd
    >>> round(volatility(pd.Series([0.01, 0.02, -0.01, 0.0]), periods=252), 4)
    0.2049
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    return float(clean.std(ddof=1) * math.sqrt(periods))


def sharpe(
    returns: pd.Series,
    periods: int,
    rf: float = 0.0,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    periods : int
        Periods per year.
    rf : float, default 0.0
        Annualized risk-free rate.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Examples
    --------
    >>> import pandas as pd
    >>> round(sharpe(pd.Series([0.01, 0.02, -0.01, 0.0]), periods=252), 4)
    6.1482
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    std = clean.std(ddof=1)
    if std == 0:
        return np.nan
    adjusted = excess_returns(clean, rf=rf, periods=periods)
    return float(adjusted.mean() / std * math.sqrt(periods))


def downside_risk(
    returns: pd.Series,
    periods: int,
    required: float = 0.0,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized downside risk.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    periods : int
        Periods per year.
    required : float, default 0.0
        Per-period required return threshold.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Annualized downside deviation.

    Examples
    --------
    >>> import pandas as pd
    >>> round(downside_risk(pd.Series([0.02, -0.01, -0.03, 0.01]), periods=12), 4)
    0.0548
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    downside = np.minimum(clean - required, 0.0)
    return float(math.sqrt(np.mean(np.square(downside))) * math.sqrt(periods))


def sortino(
    returns: pd.Series,
    periods: int,
    rf: float = 0.0,
    required: float = 0.0,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized Sortino ratio.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    periods : int
        Periods per year.
    rf : float, default 0.0
        Annualized risk-free rate.
    required : float, default 0.0
        Per-period downside threshold.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Annualized Sortino ratio.

    Examples
    --------
    >>> import pandas as pd
    >>> round(sortino(pd.Series([0.02, -0.01, -0.03, 0.01]), periods=12), 4)
    -0.5477
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    downside = downside_risk(clean, periods=periods, required=required, nan_policy="propagate")
    if downside == 0:
        return np.nan
    adjusted = excess_returns(clean, rf=rf, periods=periods)
    return float(adjusted.mean() / (downside / math.sqrt(periods)) * math.sqrt(periods))


def drawdown_series(
    returns: pd.Series,
    nan_policy: NanPolicy = "zero",
) -> pd.Series:
    """Compute drawdown at each point in a return series.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "zero"
        How missing returns are handled.

    Returns
    -------
    pandas.Series
        Drawdown series, where zero means no drawdown.

    Examples
    --------
    >>> import pandas as pd
    >>> drawdown_series(pd.Series([0.10, -0.20, 0.05])).round(2).tolist()
    [0.0, -0.2, -0.16]
    """
    clean = apply_nan_policy(returns, nan_policy)
    equity = (1.0 + clean).cumprod()
    running_max = equity.cummax().clip(lower=1.0)
    return equity / running_max - 1.0


def max_drawdown(
    returns: pd.Series,
    nan_policy: NanPolicy = "zero",
) -> float:
    """Maximum drawdown.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "zero"
        How missing returns are handled.

    Returns
    -------
    float
        Most negative drawdown.

    Examples
    --------
    >>> import pandas as pd
    >>> round(max_drawdown(pd.Series([0.10, -0.20, 0.05, -0.10])), 4)
    -0.244
    """
    return float(drawdown_series(returns, nan_policy=nan_policy).min())


def calmar(
    returns: pd.Series,
    periods: int,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Calmar ratio.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    periods : int
        Periods per year.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Annual return divided by absolute maximum drawdown.

    Examples
    --------
    >>> import pandas as pd
    >>> round(calmar(pd.Series([0.10, -0.20, 0.05, -0.10]), periods=12), 4)
    -1.7414
    """
    clean = apply_nan_policy(returns, nan_policy)
    drawdown = abs(max_drawdown(clean, nan_policy="propagate"))
    if drawdown == 0:
        return np.nan
    return float(annual_return(clean, periods=periods, nan_policy="propagate") / drawdown)


def var(
    returns: pd.Series,
    cutoff: float = 0.05,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Historical value at risk percentile.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    cutoff : float, default 0.05
        Lower-tail percentile expressed as a fraction.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Historical lower-tail percentile.

    Examples
    --------
    >>> import pandas as pd
    >>> round(var(pd.Series([-0.10, -0.04, -0.01, 0.02, 0.05, 0.09])), 4)
    -0.085
    """
    clean = apply_nan_policy(returns, nan_policy)
    return float(np.percentile(clean, cutoff * 100.0))


def cvar(
    returns: pd.Series,
    cutoff: float = 0.05,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Conditional value at risk.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    cutoff : float, default 0.05
        Lower-tail percentile expressed as a fraction.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Mean return at or below the historical VaR threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> round(cvar(pd.Series([-0.10, -0.04, -0.01, 0.02, 0.05, 0.09])), 4)
    -0.1
    """
    clean = apply_nan_policy(returns, nan_policy)
    threshold = var(clean, cutoff=cutoff, nan_policy="propagate")
    return float(clean[clean <= threshold].mean())


def beta(
    returns: pd.Series,
    benchmark: pd.Series,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Beta to a benchmark.

    Parameters
    ----------
    returns : pandas.Series
        Asset or strategy returns.
    benchmark : pandas.Series
        Benchmark returns.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    float
        OLS slope using sample covariance and variance.

    Examples
    --------
    >>> import pandas as pd
    >>> round(beta(pd.Series([0.01, 0.02, 0.03]), pd.Series([0.02, 0.03, 0.04])), 4)
    1.0
    """
    aligned = _align_pair(returns, benchmark, nan_policy)
    r = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    variance = b.var(ddof=1)
    if variance == 0:
        return np.nan
    return float(r.cov(b) / variance)


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int,
    rf: float = 0.0,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized alpha to a benchmark.

    Parameters
    ----------
    returns : pandas.Series
        Asset or strategy returns.
    benchmark : pandas.Series
        Benchmark returns.
    periods : int
        Periods per year.
    rf : float, default 0.0
        Annualized risk-free rate.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    float
        Annualized intercept.

    Examples
    --------
    >>> import pandas as pd
    >>> round(alpha(pd.Series([0.01, 0.02, 0.03]), pd.Series([0.02, 0.03, 0.04]), 252), 4)
    -2.52
    """
    validate_periods(periods)
    aligned = _align_pair(returns, benchmark, nan_policy)
    r = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    period_rf = rf / periods
    slope = beta(r, b, nan_policy="propagate")
    return float(((r - period_rf).mean() - slope * (b - period_rf).mean()) * periods)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized information ratio.

    Parameters
    ----------
    returns : pandas.Series
        Asset or strategy returns.
    benchmark : pandas.Series
        Benchmark returns.
    periods : int
        Periods per year.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    float
        Active return divided by tracking error.

    Examples
    --------
    >>> import pandas as pd
    >>> r = pd.Series([0.01, 0.03, 0.02])
    >>> b = pd.Series([0.02, 0.01, 0.02])
    >>> round(information_ratio(r, b, 252), 4)
    3.4641
    """
    validate_periods(periods)
    aligned = _align_pair(returns, benchmark, nan_policy)
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active.std(ddof=1)
    if tracking_error == 0:
        return np.nan
    return float(active.mean() / tracking_error * math.sqrt(periods))


def tail_ratio(
    returns: pd.Series,
    cutoff: float = 0.05,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Right-tail to left-tail ratio.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    cutoff : float, default 0.05
        Tail percentile expressed as a fraction.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Absolute ratio of upper and lower percentiles.

    Examples
    --------
    >>> import pandas as pd
    >>> round(tail_ratio(pd.Series([-0.10, -0.04, -0.01, 0.02, 0.05, 0.09])), 4)
    0.9412
    """
    clean = apply_nan_policy(returns, nan_policy)
    lower = np.percentile(clean, cutoff * 100.0)
    upper = np.percentile(clean, (1.0 - cutoff) * 100.0)
    if lower == 0:
        return np.nan
    return float(abs(upper / lower))


def omega(
    returns: pd.Series,
    threshold: float = 0.0,
    periods: int = 252,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Omega gain/loss ratio.

    Parameters
    ----------
    returns : pandas.Series
        Periodic simple returns.
    threshold : float, default 0.0
        Annualized return threshold.
    periods : int, default 252
        Periods per year.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled.

    Returns
    -------
    float
        Sum of gains divided by absolute sum of losses.

    Examples
    --------
    >>> import pandas as pd
    >>> round(omega(pd.Series([0.03, 0.01, -0.02, -0.01]), threshold=0.12, periods=12), 4)
    0.4
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    adjusted = clean - threshold / periods
    gains = adjusted[adjusted > 0].sum()
    losses = adjusted[adjusted < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / abs(losses))


def _align_pair(
    first: pd.Series,
    second: pd.Series,
    nan_policy: NanPolicy,
) -> pd.DataFrame:
    """Align two series on their shared index."""
    left, right = first.align(second, join="inner")
    frame = pd.concat([left, right], axis=1)
    frame.columns = ["first", "second"]
    return apply_nan_policy(frame, nan_policy)
