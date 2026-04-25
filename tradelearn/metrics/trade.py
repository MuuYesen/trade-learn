"""Trade-level performance metrics."""

import numpy as np
import pandas as pd


def win_rate(trades: pd.Series | pd.DataFrame) -> float:
    """Compute the fraction of profitable trades.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    float
        Positive trade count divided by total trade count.

    Examples
    --------
    >>> import pandas as pd
    >>> win_rate(pd.Series([100.0, -50.0, 0.0, 25.0]))
    0.5
    """
    pnl = _pnl_series(trades)
    if len(pnl) == 0:
        return np.nan
    return float((pnl > 0).sum() / len(pnl))


def profit_factor(trades: pd.Series | pd.DataFrame) -> float:
    """Compute gross profit divided by absolute gross loss.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    float
        Sum of wins divided by the absolute sum of losses.

    Examples
    --------
    >>> import pandas as pd
    >>> round(profit_factor(pd.Series([100.0, -50.0, -25.0, 25.0])), 4)
    1.6667
    """
    pnl = _pnl_series(trades)
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return np.nan
    return float(pnl[pnl > 0].sum() / abs(losses))


def avg_win(trades: pd.Series | pd.DataFrame) -> float:
    """Compute average profitable trade PnL.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    float
        Mean PnL for positive trades.

    Examples
    --------
    >>> import pandas as pd
    >>> avg_win(pd.Series([100.0, -50.0, 0.0, 25.0]))
    62.5
    """
    wins = _pnl_series(trades)
    wins = wins[wins > 0]
    if len(wins) == 0:
        return np.nan
    return float(wins.mean())


def avg_loss(trades: pd.Series | pd.DataFrame) -> float:
    """Compute average losing trade PnL.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    float
        Mean PnL for negative trades.

    Examples
    --------
    >>> import pandas as pd
    >>> avg_loss(pd.Series([100.0, -50.0, 0.0, -25.0]))
    -37.5
    """
    losses = _pnl_series(trades)
    losses = losses[losses < 0]
    if len(losses) == 0:
        return np.nan
    return float(losses.mean())


def max_consecutive_wins(trades: pd.Series | pd.DataFrame) -> int:
    """Compute the longest run of profitable trades.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    int
        Longest consecutive positive-PnL streak.

    Examples
    --------
    >>> import pandas as pd
    >>> max_consecutive_wins(pd.Series([100.0, 25.0, -10.0, 50.0]))
    2
    """
    return _max_streak(_pnl_series(trades), positive=True)


def max_consecutive_losses(trades: pd.Series | pd.DataFrame) -> int:
    """Compute the longest run of losing trades.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    int
        Longest consecutive negative-PnL streak.

    Examples
    --------
    >>> import pandas as pd
    >>> max_consecutive_losses(pd.Series([100.0, -25.0, -10.0, 50.0]))
    2
    """
    return _max_streak(_pnl_series(trades), positive=False)


def expectancy(trades: pd.Series | pd.DataFrame) -> float:
    """Compute expected PnL per trade.

    Parameters
    ----------
    trades : pandas.Series or pandas.DataFrame
        Trade PnL values, or a frame containing a ``pnl`` column.

    Returns
    -------
    float
        ``win_rate * avg_win - loss_rate * abs(avg_loss)``.

    Examples
    --------
    >>> import pandas as pd
    >>> expectancy(pd.Series([100.0, -50.0, -25.0, 25.0]))
    12.5
    """
    pnl = _pnl_series(trades)
    if len(pnl) == 0:
        return np.nan

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_fraction = len(wins) / len(pnl)
    loss_fraction = len(losses) / len(pnl)
    average_win = wins.mean() if len(wins) else 0.0
    average_loss = losses.mean() if len(losses) else 0.0
    return float(win_fraction * average_win - loss_fraction * abs(average_loss))


def _pnl_series(trades: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize supported trade inputs to a float PnL series."""
    if isinstance(trades, pd.DataFrame):
        if "pnl" not in trades:
            raise ValueError("trades DataFrame must contain a 'pnl' column")
        pnl = trades["pnl"]
    else:
        pnl = trades
    return pd.Series(pnl, dtype="float64").dropna()


def _max_streak(pnl: pd.Series, positive: bool) -> int:
    """Return the longest positive or negative streak."""
    best = 0
    current = 0
    for value in pnl:
        hit = value > 0 if positive else value < 0
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
