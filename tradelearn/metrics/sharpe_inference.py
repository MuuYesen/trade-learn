"""Inference for Sharpe ratios using explicitly timed excess returns."""

import math
from dataclasses import dataclass
from numbers import Integral
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from tradelearn.metrics._common import apply_nan_policy, validate_periods

_NORMAL = NormalDist()


@dataclass(frozen=True)
class PairedSharpeDifference:
    """Right minus left annualized Sharpe and its paired bootstrap uncertainty."""

    estimate: float
    standard_error: float
    lower: float
    upper: float
    observations: int
    valid_draws: int


def _values(returns: pd.Series, *, ordered: bool = False) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if ordered and (
        isinstance(returns.index, pd.MultiIndex)
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or returns.index.hasnans
    ):
        raise ValueError("returns must have a unique, complete and increasing index")
    values = apply_nan_policy(returns, "raise").to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two returns are required")
    deviation = values.std(ddof=1)
    if deviation == 0 or not math.isfinite(deviation):
        raise ValueError("returns must have finite, nonzero sample volatility")
    return values


def sharpe_se(
    returns: pd.Series,
    periods: int,
    *,
    method: Literal["iid", "hac"] = "iid",
    bandwidth: int | None = None,
) -> float:
    """SE of the square-root annualized Sharpe of periodic excess returns.

    IID uses the normal-return approximation. HAC uses Bartlett weights on
    the joint mean and variance influence function. Neither variant rescales
    the point estimate for autocorrelation. Supply regular periods for HAC.
    """
    validate_periods(periods)
    if method not in ("iid", "hac"):
        raise ValueError("method must be 'iid' or 'hac'")
    values = _values(returns, ordered=method == "hac")
    n = len(values)
    mean = values.mean()
    if method == "iid":
        if bandwidth is not None:
            raise ValueError("bandwidth applies only to HAC")
        per_period = mean / values.std(ddof=1)
        return math.sqrt(periods * (1 + per_period**2 / 2) / n)

    sigma = values.std(ddof=0)
    standardized = (values - mean) / sigma
    # Convert the population-variance influence to the sample-variance ratio.
    influence = standardized - (mean / (2 * sigma)) * (standardized**2 - 1)
    influence *= math.sqrt((n - 1) / n)
    return math.sqrt(periods) * _hac_mean_se(influence, bandwidth)


def hac_mean_se(values: pd.Series, *, bandwidth: int | None = None) -> float:
    """Bartlett HAC standard error of a mean, in the input series units.

    Supply a finite and regular chronological series, such as one IC per
    evaluation period. Use regression HAC for jointly estimated coefficients.
    """
    return _hac_mean_se(_values(values, ordered=True), bandwidth)


def _hac_mean_se(values: np.ndarray, bandwidth: int | None) -> float:
    n = len(values)
    if bandwidth is None:
        bandwidth = min(n - 1, math.floor(4 * (n / 100) ** (2 / 9)))
    if (
        isinstance(bandwidth, (bool, np.bool_))
        or not isinstance(bandwidth, Integral)
        or not 0 <= bandwidth < n
    ):
        raise ValueError("bandwidth must be an integer between zero and observations minus one")
    centered = values - values.mean()
    long_run_variance = np.dot(centered, centered) / n
    for lag in range(1, bandwidth + 1):
        weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * weight * np.dot(centered[lag:], centered[:-lag]) / n
    return math.sqrt(max(0, long_run_variance) / n)


def paired_sharpe_diff(
    left: pd.Series,
    right: pd.Series,
    periods: int,
    *,
    block_length: int,
    draws: int = 4000,
    seed: int | None = 0,
    confidence: float = 0.95,
) -> PairedSharpeDifference:
    """Sharpe(right) minus Sharpe(left) with a circular block percentile interval.

    Inputs are regular, strictly aligned excess-return series. Both series
    use the same rows in every draw. Zero-volatility draws have no Sharpe
    estimate and their exclusion is visible in valid_draws.
    """
    validate_periods(periods)
    left_values = _values(left, ordered=True)
    right_values = _values(right, ordered=True)
    if not left.index.equals(right.index):
        raise ValueError("both series must have the same index")
    values = np.column_stack((left_values, right_values))
    n = len(values)
    if (
        isinstance(block_length, (bool, np.bool_))
        or not isinstance(block_length, Integral)
        or not 1 <= block_length < n
    ):
        raise ValueError("block_length must be an integer between one and observations minus one")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    blocks = (n + block_length - 1) // block_length
    offsets = np.arange(block_length)
    scale = math.sqrt(periods)
    batch_size = max(1, min(512, 200_000 // n))
    samples = []
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        starts = rng.integers(0, n, size=(size, blocks, 1))
        indexes = ((starts + offsets) % n).reshape(size, -1)[:, :n]
        sample = values[indexes]
        volatility = sample.std(axis=1, ddof=1)
        valid = (volatility > 0).all(axis=1) & np.isfinite(volatility).all(axis=1)
        ratios = sample[valid].mean(axis=1) / volatility[valid]
        samples.append(scale * (ratios[:, 1] - ratios[:, 0]))
    differences = np.concatenate(samples)
    if len(differences) < 2:
        raise ValueError("fewer than two bootstrap draws had valid volatility in both series")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(differences, [alpha, 1 - alpha])
    ratios = values.mean(axis=0) / values.std(axis=0, ddof=1)
    return PairedSharpeDifference(
        estimate=float((ratios[1] - ratios[0]) * scale),
        standard_error=float(differences.std(ddof=1)),
        lower=float(lower),
        upper=float(upper),
        observations=n,
        valid_draws=len(differences),
    )


def _psr_inputs(returns: pd.Series) -> tuple[int, float, float]:
    values = _values(returns)
    n = len(values)
    if n < 4:
        raise ValueError("at least four returns are required to estimate the first four moments")
    estimate = float(values.mean() / values.std(ddof=1))
    z = (values - values.mean()) / values.std(ddof=0)
    skew = np.mean(z**3)
    kurtosis = np.mean(z**4)
    variance = 1 - skew * estimate + (kurtosis - 1) * estimate**2 / 4
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("invalid Sharpe estimator variance")
    return n, estimate, float(variance)


def probabilistic_sharpe(returns: pd.Series, *, benchmark: float = 0.0) -> float:
    """PSR against a prespecified per-period Sharpe for independent observations."""
    if not math.isfinite(benchmark):
        raise ValueError("benchmark must be finite")
    n, estimate, variance = _psr_inputs(returns)
    return _NORMAL.cdf((estimate - benchmark) * math.sqrt((n - 1) / variance))


def deflated_sharpe(returns: pd.Series, *, n_trials: int, variance_trials: float) -> float:
    """PSR against the expected null maximum of independent search trials.

    variance_trials is the cross-trial variance of per-period Sharpe
    estimates on comparable returns. For a single prespecified trial,
    use probabilistic_sharpe against the research benchmark.
    """
    if isinstance(n_trials, (bool, np.bool_)) or not isinstance(n_trials, Integral) or n_trials < 2:
        raise ValueError("n_trials must be an integer of at least two")
    if not math.isfinite(variance_trials) or variance_trials < 0:
        raise ValueError("variance_trials must be finite and nonnegative")
    gamma = 0.5772156649015329
    maximum = -(
        (1 - gamma) * _NORMAL.inv_cdf(1 / n_trials)
        + gamma * _NORMAL.inv_cdf(math.exp(-1) / n_trials)
    )
    return probabilistic_sharpe(returns, benchmark=math.sqrt(variance_trials) * maximum)


def minimum_track_record_length(
    returns: pd.Series, *, benchmark: float = 0.0, confidence: float = 0.95
) -> int | float:
    """Required periods at the observed PSR inputs, or infinity below the benchmark."""
    if not math.isfinite(benchmark):
        raise ValueError("benchmark must be finite")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be greater than one half and less than one")
    _, estimate, variance = _psr_inputs(returns)
    if estimate <= benchmark:
        return math.inf
    return math.ceil(1 + variance * (_NORMAL.inv_cdf(confidence) / (estimate - benchmark))**2)
