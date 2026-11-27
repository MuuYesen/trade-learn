"""Known answers and data contracts for Sharpe inference."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from tradelearn import metrics


def test_sharpe_iid_uses_observation_frequency():
    unit_variance = pd.Series(np.arange(1460, dtype=float))
    centered = (unit_variance - unit_variance.mean()) / unit_variance.std()
    returns = centered + 1 / sqrt(365)
    assert metrics.sharpe(returns, 365) == pytest.approx(1)
    assert metrics.sharpe_se(returns, 365) == pytest.approx(0.500342348551)


def test_sharpe_hac_uses_the_lagged_influence_covariance():
    # A symmetric paired-sign pattern has mean zero and positive lag-1 covariance.
    returns = pd.Series([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    # Sample-normalized influence has population variance 7/8 and lag-1
    # covariance (3/8)*(7/8). Bartlett lag-1 weight is 1/2.
    expected = sqrt((7 / 8) * (1 + 3 / 8) / 8)
    assert metrics.sharpe_se(returns, 1, method="hac", bandwidth=1) == pytest.approx(expected)
    assert metrics.sharpe_se(returns, 1, method="hac", bandwidth=0) < expected


def test_paired_bootstrap_keeps_the_same_sample_on_both_sides():
    dates = pd.date_range("2020-01-01", periods=100)
    returns = pd.Series(np.sin(np.arange(100)) / 100, index=dates)
    result = metrics.paired_sharpe_diff(returns, 2 * returns, 365, block_length=7, draws=128)
    assert result.estimate == pytest.approx(0)
    assert result.standard_error == pytest.approx(0)
    assert (result.lower, result.upper) == pytest.approx((0, 0))
    assert result.observations == 100
    assert result.valid_draws == 128


def test_paired_bootstrap_estimates_right_minus_left_and_is_reproducible():
    left = pd.Series(np.sin(np.arange(100)) / 100)
    right = left + .002
    run = metrics.paired_sharpe_diff
    args = dict(block_length=5, draws=200, seed=42)
    result = run(left, right, 252, **args)
    expected = metrics.sharpe(right, 252) - metrics.sharpe(left, 252)
    assert result.estimate == pytest.approx(expected)
    assert result.lower < expected < result.upper
    assert result == run(left, right, 252, **args)


def test_time_series_inference_requires_ordered_aligned_finite_series():
    returns = pd.Series([0.01, -0.02, 0.03, -0.04], index=[1, 2, 3, 4])
    for other in (returns.iloc[::-1], returns.iloc[1:], returns.rename(index={4: 3})):
        with pytest.raises(ValueError):
            metrics.paired_sharpe_diff(returns, other, 365, block_length=2)
    with pytest.raises(ValueError):
        metrics.sharpe_se(returns * np.nan, 365)
    with pytest.raises(ValueError):
        metrics.sharpe_se(returns.iloc[::-1], 365, method="hac")
    with pytest.raises(ValueError):
        metrics.sharpe_se(returns * 0, 365)


def test_psr_dsr_and_minimum_length_known_answers():
    returns = pd.Series([-.01, .03, -.01, .03, -.01, .03, -.01, .03, -.01, .03])
    ratio = returns.mean() / returns.std()
    shape = sqrt(1 + (1 - 1) * ratio**2 / 4)
    normal = NormalDist()
    psr = normal.cdf(ratio * sqrt(len(returns) - 1) / shape)
    assert metrics.probabilistic_sharpe(returns) == pytest.approx(psr)
    assert metrics.minimum_track_record_length(returns) == 14
    assert metrics.minimum_track_record_length(-returns) == np.inf
    assert metrics.deflated_sharpe(returns, n_trials=5, variance_trials=.02) < psr
    with pytest.raises(ValueError, match="at least two"):
        metrics.deflated_sharpe(returns, n_trials=1, variance_trials=.02)


def test_mean_hac_for_chronological_ic_or_excess_return_series():
    scores = pd.Series([-1., -1., 1., 1., 1., 1., -1., -1.])
    assert metrics.hac_mean_se(scores, bandwidth=1) == pytest.approx(sqrt((1 + 3 / 8) / 8))
