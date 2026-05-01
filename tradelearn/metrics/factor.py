"""Factor evaluation metrics."""

import math

import numpy as np
import pandas as pd

from tradelearn.metrics._common import NanPolicy, apply_nan_policy, validate_periods


def ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    nan_policy: NanPolicy = "drop",
    groupby: pd.Series | None = None,
    by_group: bool = False,
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
    if groupby is not None:
        aligned = _attach_group(aligned, groupby)
    if by_group:
        group_keys = [aligned.index.get_level_values(0), aligned["group"]]
        result = (
            aligned[["factor", "returns"]]
            .groupby(group_keys)
            .apply(lambda frame: frame["factor"].corr(frame["returns"]))
            .unstack(level=-1)
        )
        result.index.name = _date_level_name(factor)
        result.columns.name = None
        return result
    result = aligned.groupby(level=0).apply(lambda frame: frame["factor"].corr(frame["returns"]))
    result.index.name = _date_level_name(factor)
    result.name = "ic"
    return result


def rank_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    nan_policy: NanPolicy = "drop",
    groupby: pd.Series | None = None,
    by_group: bool = False,
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

    if groupby is not None:
        aligned = _attach_group(aligned, groupby)
    if by_group:
        group_keys = [aligned.index.get_level_values(0), aligned["group"]]
        result = (
            aligned[["factor", "returns"]]
            .groupby(group_keys)
            .apply(_spearman)
            .unstack(level=-1)
        )
        result.index.name = _date_level_name(factor)
        result.columns.name = None
        return result
    result = aligned.groupby(level=0).apply(_spearman)
    result.index.name = _date_level_name(factor)
    result.name = "rank_ic"
    return result


def clean_factor_and_forward_returns(
    factors: pd.DataFrame,
    *,
    factor: str,
    prices: pd.Series | pd.DataFrame,
    periods: tuple[int, ...] = (1,),
    quantiles: int = 5,
    groupby: str | pd.Series | None = None,
    nan_policy: NanPolicy = "drop",
) -> pd.DataFrame:
    """Return an Alphalens-style clean factor frame.

    ``factors`` is a factor table indexed by ``(date, symbol)``. ``factor``
    names the column to analyze, and ``prices`` supplies aligned close prices.
    Output is indexed by ``(date, symbol)`` and contains ``factor``,
    ``forward_return_N`` columns, ``factor_quantile``, and optional ``group``.
    """

    if quantiles <= 0:
        raise ValueError("quantiles must be a positive integer")
    factor_values = _factor_column(factors, factor)
    price_values = _price_series(prices)
    _validate_multiindex(factor_values)
    _validate_multiindex(price_values)
    if any(period <= 0 for period in periods):
        raise ValueError("periods must contain positive integers")

    frame = factor_values.rename("factor").to_frame()
    for period in periods:
        forward = _forward_returns_from_prices(price_values, periods=period)
        frame[f"forward_return_{period}"] = forward
    frame = apply_nan_policy(frame, nan_policy)
    frame["factor_quantile"] = _factor_quantiles(frame["factor"], quantiles)
    if groupby is not None:
        frame = _attach_group(frame, _group_series(factors, groupby))
    return frame.dropna(subset=["factor_quantile"]).astype({"factor_quantile": "int64"})


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
    groupby: pd.Series | None = None,
    group_neutral: bool = False,
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
    if groupby is not None:
        aligned = _attach_group(aligned, groupby)
    if group_neutral:
        if "group" not in aligned.columns:
            raise ValueError("group_neutral=True requires groupby")
        aligned = aligned.copy()
        aligned["returns"] = aligned["returns"] - aligned.groupby(
            [aligned.index.get_level_values(0), "group"]
        )["returns"].transform("mean")

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


def mean_monthly_ic(ic_series: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Return mean IC by year and month."""

    if isinstance(ic_series, pd.DataFrame):
        source = ic_series.mean(axis=1)
    else:
        source = ic_series
    clean = source.dropna()
    index = pd.to_datetime(clean.index)
    grouped = clean.groupby([index.year, index.month]).mean()
    result = grouped.unstack(fill_value=np.nan)
    result.index.name = "year"
    result.columns.name = "month"
    return result


def event_returns(
    prices: pd.Series,
    events: pd.MultiIndex,
    before: int = 5,
    after: int = 5,
) -> pd.DataFrame:
    """Return average event-window returns around ``(date, symbol)`` events."""

    _validate_multiindex(prices)
    if before < 0 or after < 0:
        raise ValueError("before and after must be non-negative")
    if not isinstance(events, pd.MultiIndex) or events.nlevels < 2:
        raise ValueError("events must be a MultiIndex indexed by (date, symbol)")
    price_frame = prices.unstack()
    windows: list[pd.Series] = []
    offsets = list(range(-before, after + 1))
    for event_date, symbol, *_ in events:
        symbol = str(symbol)
        if symbol not in price_frame.columns:
            continue
        dates = price_frame.index
        try:
            loc = dates.get_loc(pd.Timestamp(event_date))
        except KeyError:
            continue
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            continue
        start = loc - before
        end = loc + after
        if start < 0 or end >= len(dates):
            continue
        window = price_frame[symbol].iloc[start : end + 1]
        base = price_frame[symbol].iloc[loc]
        if pd.isna(base) or base == 0:
            continue
        row = (window / base - 1.0).reset_index(drop=True)
        row.index = offsets
        windows.append(row)
    if not windows:
        return pd.DataFrame(index=pd.Index(offsets, name="offset"), columns=["mean", "count"])
    data = pd.DataFrame(windows)
    return pd.DataFrame(
        {"mean": data.mean(axis=0), "count": data.count(axis=0)},
        index=pd.Index(offsets, name="offset"),
    )


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


def _forward_returns_from_prices(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Compute next-period returns per symbol from MultiIndex prices."""
    _validate_multiindex(prices)
    forward = (
        prices.groupby(level=1, group_keys=False)
        .pct_change(periods=periods)
        .groupby(level=1)
        .shift(-periods)
    )
    forward.name = "returns"
    return forward


def _factor_column(factors: pd.DataFrame, factor: str) -> pd.Series:
    """Return the selected factor column from a factor table."""
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame indexed by (date, symbol)")
    if factor not in factors:
        raise ValueError(f"factors must contain factor column {factor!r}")
    return pd.Series(factors[factor], index=factors.index, name=factor).sort_index()


def _price_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a close-price series from a price input."""
    if isinstance(prices, pd.Series):
        return prices.rename("prices").sort_index()
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas Series or DataFrame")
    if "close" in prices:
        return pd.Series(prices["close"], index=prices.index, name="prices").sort_index()
    if len(prices.columns) == 1:
        column = prices.columns[0]
        return pd.Series(prices[column], index=prices.index, name="prices").sort_index()
    raise ValueError("prices DataFrame must contain a 'close' column or exactly one column")


def _group_series(factors: pd.DataFrame, groupby: str | pd.Series) -> pd.Series:
    """Return group labels aligned with the factor table."""
    if isinstance(groupby, str):
        if groupby not in factors:
            raise ValueError(f"factors must contain groupby column {groupby!r}")
        return pd.Series(factors[groupby], index=factors.index, name="group").sort_index()
    return pd.Series(groupby, name="group").sort_index()


def _factor_quantiles(factor: pd.Series, quantiles: int) -> pd.Series:
    """Return date-level factor quantiles."""

    def _assign(frame: pd.Series) -> pd.Series:
        labels = pd.qcut(
            frame.rank(method="first"),
            q=min(quantiles, len(frame)),
            labels=False,
        )
        return labels.astype(float) + 1

    result = factor.groupby(level=0, group_keys=False).apply(_assign)
    result.name = "factor_quantile"
    return result


def _attach_group(frame: pd.DataFrame, groupby: pd.Series) -> pd.DataFrame:
    """Attach an aligned group column to a factor frame."""

    group = groupby.rename("group")
    aligned = frame.join(group, how="left")
    return aligned.dropna(subset=["group"])


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
