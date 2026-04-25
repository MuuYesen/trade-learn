"""Tests for risk metrics."""

import math

import numpy as np
import pandas as pd

from tradelearn.metrics.risk import (
    alpha,
    beta,
    calmar,
    cvar,
    downside_risk,
    drawdown_series,
    information_ratio,
    max_drawdown,
    omega,
    sharpe,
    sortino,
    tail_ratio,
    var,
    volatility,
)


def test_volatility_annualizes_sample_standard_deviation() -> None:
    """volatility uses sample standard deviation and sqrt(periods)."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.00])

    result = volatility(returns, periods=252)

    expected = returns.std(ddof=1) * math.sqrt(252)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_sharpe_uses_excess_return_mean_over_sample_std() -> None:
    """sharpe subtracts per-period risk-free return."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.00])

    result = sharpe(returns, periods=252, rf=0.0252)

    excess = returns - 0.0252 / 252
    expected = excess.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_downside_risk_and_sortino_use_required_return_threshold() -> None:
    """sortino delegates denominator semantics to downside_risk."""
    returns = pd.Series([0.02, -0.01, -0.03, 0.01])

    downside = downside_risk(returns, periods=12, required=0.0)
    ratio = sortino(returns, periods=12, rf=0.0, required=0.0)

    expected_downside = math.sqrt((np.minimum(returns, 0.0) ** 2).mean()) * math.sqrt(12)
    expected_sortino = returns.mean() / (expected_downside / math.sqrt(12)) * math.sqrt(12)
    assert math.isclose(downside, expected_downside, rel_tol=1e-12)
    assert math.isclose(ratio, expected_sortino, rel_tol=1e-12)


def test_drawdown_series_max_drawdown_and_calmar() -> None:
    """drawdown metrics use cumulative return peaks."""
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])

    drawdowns = drawdown_series(returns)

    expected = pd.Series([0.0, -0.20, -0.16, -0.244])
    pd.testing.assert_series_equal(drawdowns.round(3), expected)
    assert math.isclose(max_drawdown(returns), -0.244, rel_tol=1e-12)
    assert math.isclose(calmar(returns, periods=12), annual_return_for_test(returns, 12) / 0.244)


def test_var_cvar_and_tail_ratio_use_historical_percentiles() -> None:
    """tail metrics are percentile based."""
    returns = pd.Series([-0.10, -0.04, -0.01, 0.02, 0.05, 0.09])

    assert math.isclose(var(returns, cutoff=0.05), np.percentile(returns, 5))
    expected_cvar = returns[returns <= np.percentile(returns, 5)].mean()
    assert math.isclose(cvar(returns, cutoff=0.05), expected_cvar)
    expected_tail = abs(np.percentile(returns, 95) / np.percentile(returns, 5))
    assert math.isclose(tail_ratio(returns, cutoff=0.05), expected_tail)


def test_beta_alpha_and_information_ratio_align_on_inner_index() -> None:
    """benchmark metrics align inputs on common dates."""
    idx = pd.date_range("2024-01-01", periods=4)
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    benchmark = pd.Series([0.015, 0.025, 0.035, 0.045], index=idx.shift(1, freq="D"))

    result_beta = beta(returns, benchmark)
    result_alpha = alpha(returns, benchmark, periods=252, rf=0.0)
    result_ir = information_ratio(returns, benchmark, periods=252)

    aligned_returns, aligned_benchmark = returns.align(benchmark, join="inner")
    expected_beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
    expected_alpha = (aligned_returns.mean() - expected_beta * aligned_benchmark.mean()) * 252
    active = aligned_returns - aligned_benchmark
    expected_ir = active.mean() / active.std(ddof=1) * math.sqrt(252)
    assert math.isclose(result_beta, expected_beta, rel_tol=1e-12)
    assert math.isclose(result_alpha, expected_alpha, rel_tol=1e-12)
    assert math.isclose(result_ir, expected_ir, rel_tol=1e-12)


def test_omega_uses_gain_loss_ratio_above_threshold() -> None:
    """omega sums gains and losses around a per-period threshold."""
    returns = pd.Series([0.03, 0.01, -0.02, -0.01])

    result = omega(returns, threshold=0.12, periods=12)

    per_period_threshold = 0.12 / 12
    adjusted = returns - per_period_threshold
    expected = adjusted[adjusted > 0].sum() / abs(adjusted[adjusted < 0].sum())
    assert math.isclose(result, expected, rel_tol=1e-12)


def annual_return_for_test(returns: pd.Series, periods: int) -> float:
    """Small local copy of the CAGR formula for calmar expectations."""
    return (1.0 + returns).prod() ** (periods / len(returns)) - 1.0
