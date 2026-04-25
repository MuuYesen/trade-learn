"""Tests for return metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from tradelearn.metrics.returns import (
    annual_return,
    cum_returns,
    excess_returns,
    log_to_simple,
    simple_returns,
)


def test_simple_returns_drops_first_period_and_preserves_index() -> None:
    """simple_returns computes percentage changes from prices."""
    prices = pd.Series([100.0, 105.0, 102.9], index=pd.date_range("2024-01-01", periods=3))

    result = simple_returns(prices)

    expected = pd.Series([0.05, -0.02], index=prices.index[1:])
    pd.testing.assert_series_equal(result, expected)


def test_cum_returns_compounds_simple_returns() -> None:
    """cum_returns compounds periodic simple returns."""
    returns = pd.Series([0.10, -0.05, 0.02], index=pd.RangeIndex(3))

    result = cum_returns(returns)

    expected = pd.Series([0.10, 0.045, 0.0659], index=returns.index)
    pd.testing.assert_series_equal(result, expected)


def test_cum_returns_with_starting_value_returns_equity_curve() -> None:
    """starting_value switches cumulative returns to an equity curve."""
    returns = pd.Series([0.10, -0.05, 0.02], index=pd.RangeIndex(3))

    result = cum_returns(returns, starting_value=100.0)

    expected = pd.Series([110.0, 104.5, 106.59], index=returns.index)
    pd.testing.assert_series_equal(result, expected)


def test_annual_return_uses_explicit_periods() -> None:
    """annual_return uses the documented CAGR formula with explicit periods."""
    returns = pd.Series([0.01, 0.02, -0.005, 0.004])

    result = annual_return(returns, periods=252)

    total_return = np.prod(1.0 + returns.to_numpy())
    expected = total_return ** (252 / len(returns)) - 1.0
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_annual_return_rejects_missing_or_invalid_periods() -> None:
    """periods is required and must be positive."""
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError, match="periods"):
        annual_return(returns, periods=0)


def test_log_to_simple_converts_log_returns() -> None:
    """log_to_simple converts log returns to simple returns."""
    log_returns = pd.Series([0.0, math.log(1.05), math.log(0.98)])

    result = log_to_simple(log_returns)

    expected = pd.Series([0.0, 0.05, -0.02])
    pd.testing.assert_series_equal(result, expected)


def test_excess_returns_subtracts_per_period_risk_free_rate() -> None:
    """excess_returns subtracts annual rf divided by periods."""
    returns = pd.Series([0.01, 0.02, -0.005])

    result = excess_returns(returns, rf=0.024, periods=12)

    expected = pd.Series([0.008, 0.018, -0.007])
    pd.testing.assert_series_equal(result, expected)
