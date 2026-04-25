"""Factor evaluation metrics."""

import math

import numpy as np
import pandas as pd

from tradelearn.metrics._common import NanPolicy, apply_nan_policy, validate_periods


def ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    nan_policy: NanPolicy = "drop",
) -> pd.Series:
    """Compute per-date Pearson information coefficient.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    forward_returns : pandas.Series
        Forward returns indexed by ``(date, symbol)``.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    pandas.Series
        Pearson correlation by date.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
    ...     names=["date", "symbol"],
    ... )
    >>> ic(pd.Series([1, 2, 3], idx), pd.Series([0.1, 0.2, 0.3], idx)).tolist()
    [1.0]
    """
    aligned = _align_factor_and_returns(factor, forward_returns, nan_policy)
    result = aligned.groupby(level=0).apply(lambda frame: frame["factor"].corr(frame["returns"]))
    result.index.name = _date_level_name(factor)
    result.name = "ic"
    return result


def rank_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    nan_policy: NanPolicy = "drop",
) -> pd.Series:
    """Compute per-date Spearman rank information coefficient.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    forward_returns : pandas.Series
        Forward returns indexed by ``(date, symbol)``.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    pandas.Series
        Spearman rank correlation by date.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
    ...     names=["date", "symbol"],
    ... )
    >>> rank_ic(pd.Series([1, 2, 3], idx), pd.Series([0.3, 0.2, 0.1], idx)).tolist()
    [-1.0]
    """
    aligned = _align_factor_and_returns(factor, forward_returns, nan_policy)

    def _spearman(frame: pd.DataFrame) -> float:
        return frame["factor"].rank().corr(frame["returns"].rank())

    result = aligned.groupby(level=0).apply(_spearman)
    result.index.name = _date_level_name(factor)
    result.name = "rank_ic"
    return result


def ic_ir(
    ic_series: pd.Series,
    periods: int,
    nan_policy: NanPolicy = "drop",
) -> float:
    """Annualized information coefficient information ratio.

    Parameters
    ----------
    ic_series : pandas.Series
        Information coefficient values through time.
    periods : int
        Periods per year.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing IC values are handled.

    Returns
    -------
    float
        Annualized IC mean divided by sample standard deviation.

    Examples
    --------
    >>> import pandas as pd
    >>> round(ic_ir(pd.Series([0.10, 0.20, -0.05, 0.15]), periods=12), 4)
    3.2071
    """
    validate_periods(periods)
    clean = apply_nan_policy(ic_series, nan_policy)
    std = clean.std(ddof=1)
    if np.isclose(std, 0.0):
        return np.nan
    return float(clean.mean() / std * math.sqrt(periods))


def factor_returns(
    factor: pd.Series,
    prices: pd.Series,
    quantiles: int = 5,
    nan_policy: NanPolicy = "drop",
) -> pd.DataFrame:
    """Compute quantile forward returns from prices.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    prices : pandas.Series
        Prices indexed by ``(date, symbol)``.
    quantiles : int, default 5
        Number of date-level factor quantiles.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    pandas.DataFrame
        Mean next-period returns by date and factor quantile.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
    ...     names=["date", "symbol"],
    ... )
    >>> out = factor_returns(pd.Series([1, 2], idx[:2]), pd.Series([100, 100, 110, 90], idx), 2)
    >>> out.round(4).to_dict("list")
    {1: [0.1], 2: [-0.1]}
    """
    forward = _forward_returns_from_prices(prices)
    return quantile_returns(factor, forward, quantiles=quantiles, nan_policy=nan_policy)


def quantile_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 5,
    nan_policy: NanPolicy = "drop",
) -> pd.DataFrame:
    """Compute mean forward returns by factor quantile.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    forward_returns : pandas.Series
        Forward returns indexed by ``(date, symbol)``.
    quantiles : int, default 5
        Number of date-level factor quantiles.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing aligned rows are handled.

    Returns
    -------
    pandas.DataFrame
        Mean forward return by date and quantile. Quantiles are 1-based.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01"]), ["A", "B"]],
    ...     names=["date", "symbol"],
    ... )
    >>> out = quantile_returns(pd.Series([1, 2], idx), pd.Series([0.1, -0.1], idx), 2)
    >>> out.to_dict("list")
    {1: [0.1], 2: [-0.1]}
    """
    if quantiles <= 0:
        raise ValueError("quantiles must be a positive integer")
    aligned = _align_factor_and_returns(factor, forward_returns, nan_policy)

    def _assign_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["quantile"] = pd.qcut(
            frame["factor"].rank(method="first"),
            q=min(quantiles, len(frame)),
            labels=False,
        )
        frame["quantile"] = frame["quantile"].astype(int) + 1
        return frame

    assigned = aligned.groupby(level=0, group_keys=False).apply(_assign_quantiles)
    grouped = assigned.groupby([assigned.index.get_level_values(0), "quantile"])["returns"].mean()
    result = grouped.unstack("quantile").sort_index(axis=1)
    result.index.name = _date_level_name(factor)
    result.columns.name = None
    return result


def turnover(
    factor: pd.Series,
    lag: int = 1,
    nan_policy: NanPolicy = "drop",
) -> pd.Series:
    """Compute factor rank turnover.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    lag : int, default 1
        Date lag used for rank autocorrelation.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing rows are handled.

    Returns
    -------
    pandas.Series
        ``1 - autocorrelation`` by date.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
    ...     names=["date", "symbol"],
    ... )
    >>> turnover(pd.Series([1, 2, 2, 1], idx)).tolist()
    [2.0]
    """
    result = 1.0 - autocorrelation(factor, lag=lag, nan_policy=nan_policy)
    result.name = "turnover"
    return result


def autocorrelation(
    factor: pd.Series,
    lag: int = 1,
    nan_policy: NanPolicy = "drop",
) -> pd.Series:
    """Compute lagged factor rank autocorrelation by date.

    Parameters
    ----------
    factor : pandas.Series
        Factor values indexed by ``(date, symbol)``.
    lag : int, default 1
        Number of date observations to lag.
    nan_policy : {"drop", "zero", "propagate", "raise"}, default "drop"
        How missing rows are handled.

    Returns
    -------
    pandas.Series
        Cross-sectional rank autocorrelation by date.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
    ...     names=["date", "symbol"],
    ... )
    >>> autocorrelation(pd.Series([1, 2, 1, 2], idx)).round(4).tolist()
    [1.0]
    """
    if lag <= 0:
        raise ValueError("lag must be a positive integer")
    _validate_multiindex(factor)
    clean = apply_nan_policy(factor.rename("factor").to_frame(), nan_policy)["factor"]
    rank_frame = clean.groupby(level=0).rank(method="average").unstack()
    correlations: list[float] = []
    index = []
    for position in range(lag, len(rank_frame)):
        previous = rank_frame.iloc[position - lag]
        current = rank_frame.iloc[position]
        aligned = pd.concat([previous, current], axis=1).dropna()
        correlations.append(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])))
        index.append(rank_frame.index[position])
    result = pd.Series(correlations, index=pd.Index(index, name=_date_level_name(factor)))
    result.name = "autocorrelation"
    return result


def _forward_returns_from_prices(prices: pd.Series) -> pd.Series:
    """Compute next-period returns per symbol from MultiIndex prices."""
    _validate_multiindex(prices)
    forward = prices.groupby(level=1, group_keys=False).pct_change().groupby(level=1).shift(-1)
    forward.name = "returns"
    return forward


def _align_factor_and_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    nan_policy: NanPolicy,
) -> pd.DataFrame:
    """Align factor values with forward returns on their shared MultiIndex."""
    _validate_multiindex(factor)
    _validate_multiindex(forward_returns)
    left, right = factor.align(forward_returns, join="inner")
    frame = pd.concat([left.rename("factor"), right.rename("returns")], axis=1)
    return apply_nan_policy(frame, nan_policy)


def _validate_multiindex(series: pd.Series) -> None:
    """Validate the factor metric MultiIndex shape."""
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise ValueError("factor metrics require a MultiIndex indexed by (date, symbol)")


def _date_level_name(series: pd.Series) -> str | None:
    """Return the date level name for output indexes."""
    return series.index.names[0]
