"""Tests for trade-level metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from tradelearn.metrics.trade import (
    avg_loss,
    avg_win,
    expectancy,
    max_consecutive_losses,
    max_consecutive_wins,
    profit_factor,
    win_rate,
)


def test_win_rate_counts_positive_pnl_trades() -> None:
    """win_rate is positive trades divided by total trades."""
    trades = pd.Series([100.0, -50.0, 0.0, 25.0])

    result = win_rate(trades)

    assert math.isclose(result, 0.5)


def test_profit_factor_divides_wins_by_absolute_losses() -> None:
    """profit_factor uses gross profit over gross loss."""
    trades = pd.Series([100.0, -50.0, -25.0, 25.0])

    result = profit_factor(trades)

    assert math.isclose(result, 125.0 / 75.0)


def test_avg_win_and_avg_loss_ignore_flat_trades() -> None:
    """Average win and loss exclude zero pnl trades."""
    trades = pd.Series([100.0, -50.0, 0.0, 25.0, -25.0])

    assert math.isclose(avg_win(trades), 62.5)
    assert math.isclose(avg_loss(trades), -37.5)


def test_consecutive_streaks_track_positive_and_negative_runs() -> None:
    """Consecutive metrics track the longest positive or negative run."""
    trades = pd.Series([100.0, 25.0, -10.0, -5.0, -1.0, 0.0, 50.0])

    assert max_consecutive_wins(trades) == 2
    assert max_consecutive_losses(trades) == 3


def test_expectancy_uses_win_rate_average_win_and_average_loss() -> None:
    """expectancy follows the documented win/loss expectation formula."""
    trades = pd.Series([100.0, -50.0, -25.0, 25.0])

    result = expectancy(trades)

    expected = 0.5 * 62.5 - 0.5 * 37.5
    assert math.isclose(result, expected)


def test_trade_metrics_accept_dataframe_with_pnl_column() -> None:
    """Trade metrics accept backtest-like frames with a pnl column."""
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 25.0]})

    assert math.isclose(win_rate(trades), 2 / 3)
    assert math.isclose(avg_loss(trades), -50.0)


def test_trade_metrics_return_nan_for_empty_or_missing_sides() -> None:
    """Undefined ratios return NaN instead of raising."""
    empty = pd.Series([], dtype=float)
    only_wins = pd.Series([10.0, 20.0])

    assert np.isnan(win_rate(empty))
    assert np.isnan(avg_win(empty))
    assert np.isnan(avg_loss(only_wins))
    assert np.isnan(expectancy(empty))
    assert np.isnan(profit_factor(only_wins))


def test_dataframe_trade_metrics_require_pnl_column() -> None:
    """DataFrame inputs must expose a pnl column."""
    with pytest.raises(ValueError, match="pnl"):
        win_rate(pd.DataFrame({"profit": [1.0]}))
