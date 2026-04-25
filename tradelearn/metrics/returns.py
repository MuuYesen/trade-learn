"""Return series metrics.

The functions in this module use simple returns, defined as
``r_t = p_t / p_{t-1} - 1``.
"""

import numpy as np
import pandas as pd

from tradelearn.metrics._common import NanPolicy, apply_nan_policy, validate_periods


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple returns from prices.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Price series or wide price frame.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Periodic simple returns with the first missing period removed.

    Examples
    --------
    >>> import pandas as pd
    >>> simple_returns(pd.Series([100.0, 105.0, 102.9])).round(4).tolist()
    [0.05, -0.02]
    """
    return prices.pct_change().iloc[1:]


def cum_returns(
    returns: pd.Series | pd.DataFrame,
    starting_value: float = 0.0,
    nan_policy: NanPolicy = "zero",
) -> pd.Series | pd.DataFrame:
    """Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Periodic simple returns.
    starting_value : float, default 0.0
        If zero, return cumulative return. Otherwise return an equity curve
        starting from this value.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "zero"
        How missing returns are handled before compounding.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Cumulative returns or equity curve.

    Examples
    --------
    >>> import pandas as pd
    >>> cum_returns(pd.Series([0.10, -0.05, 0.02])).round(4).tolist()
    [0.1, 0.045, 0.0659]
    """
    clean = apply_nan_policy(returns, nan_policy)
    compounded = (1.0 + clean).cumprod()
    if starting_value == 0:
        return compounded - 1.0
    return compounded * starting_value


def annual_return(
    returns: pd.Series | pd.DataFrame,
    periods: int,
    nan_policy: NanPolicy = "drop",
) -> float | pd.Series:
    """Compute compound annual growth rate from periodic simple returns.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Periodic simple returns.
    periods : int
        Periods per year. For example, 252 for daily stock returns.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing returns are handled before calculation.

    Returns
    -------
    float or pandas.Series
        Annualized compound return.

    Examples
    --------
    >>> import pandas as pd
    >>> round(annual_return(pd.Series([0.01, 0.02, -0.005, 0.004]), periods=252), 4)
    5.1113
    """
    validate_periods(periods)
    clean = apply_nan_policy(returns, nan_policy)
    if len(clean) == 0:
        return np.nan

    total_return = (1.0 + clean).prod()
    years = len(clean) / periods
    return total_return ** (1.0 / years) - 1.0


def log_to_simple(log_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert log returns to simple returns.

    Parameters
    ----------
    log_returns : pandas.Series or pandas.DataFrame
        Log returns.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Simple returns.

    Examples
    --------
    >>> import math, pandas as pd
    >>> log_to_simple(pd.Series([0.0, math.log(1.05)])).round(4).tolist()
    [0.0, 0.05]
    """
    return np.exp(log_returns) - 1.0


def excess_returns(
    returns: pd.Series | pd.DataFrame,
    rf: float,
    periods: int,
) -> pd.Series | pd.DataFrame:
    """Subtract per-period risk-free return from simple returns.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Periodic simple returns.
    rf : float
        Annualized risk-free rate.
    periods : int
        Periods per year.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Excess returns.

    Examples
    --------
    >>> import pandas as pd
    >>> excess_returns(pd.Series([0.01, 0.02]), rf=0.024, periods=12).round(4).tolist()
    [0.008, 0.018]
    """
    validate_periods(periods)
    return returns - (rf / periods)
