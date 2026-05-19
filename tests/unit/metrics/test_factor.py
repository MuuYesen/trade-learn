"""Tests for factor metrics."""

import math

import pandas as pd
import pytest

from tradelearn.metrics.factor import (
    autocorrelation,
    factor_returns,
    ic,
    ic_ir,
    quantile_returns,
    quantile_turnover,
    rank_ic,
    turnover,
)


def test_ic_computes_pearson_correlation_by_date() -> None:
    """ic returns per-date Pearson correlation across symbols."""
    factor, forward = _factor_and_forward_returns()

    result = ic(factor, forward)

    expected = pd.Series([1.0, -1.0], index=pd.Index(_dates(), name="date"), name="ic")
    pd.testing.assert_series_equal(result, expected)


def test_rank_ic_computes_spearman_correlation_by_date() -> None:
    """rank_ic returns per-date Spearman rank correlation."""
    factor, forward = _factor_and_forward_returns()

    result = rank_ic(factor, forward)

    expected = pd.Series([1.0, -1.0], index=pd.Index(_dates(), name="date"), name="rank_ic")
    pd.testing.assert_series_equal(result, expected)


def test_ic_ir_annualizes_mean_ic_by_sample_std() -> None:
    """ic_ir annualizes IC mean over sample standard deviation."""
    values = pd.Series([0.10, 0.20, -0.05, 0.15])

    result = ic_ir(values, periods=12)

    expected = values.mean() / values.std(ddof=1) * math.sqrt(12)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_quantile_returns_groups_forward_returns_by_factor_quantile() -> None:
    """quantile_returns averages forward returns inside date-level quantiles."""
    factor, forward = _factor_and_forward_returns()

    result = quantile_returns(factor, forward, quantiles=2)

    expected = pd.DataFrame(
        {1: [-0.005, 0.03], 2: [0.04, -0.03]},
        index=pd.Index(_dates(), name="date"),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_factor_returns_computes_forward_returns_from_prices() -> None:
    """factor_returns derives next-period simple returns from price data."""
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
        ]
    )
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-02", "AAA", 110.0),
            ("2024-01-02", "BBB", 90.0),
            ("2024-01-03", "AAA", 99.0),
            ("2024-01-03", "BBB", 99.0),
        ]
    )

    result = factor_returns(factor, prices, quantiles=2)

    expected = pd.DataFrame(
        {1: [0.10, -0.10], 2: [-0.10, 0.10]},
        index=pd.Index(pd.to_datetime(["2024-01-01", "2024-01-02"]), name="date"),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_autocorrelation_and_turnover_track_rank_changes() -> None:
    """turnover is one minus factor rank autocorrelation."""
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-01", "CCC", 3.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
            ("2024-01-02", "CCC", 3.0),
            ("2024-01-03", "AAA", 3.0),
            ("2024-01-03", "BBB", 2.0),
            ("2024-01-03", "CCC", 1.0),
        ]
    )

    corr = autocorrelation(factor, lag=1)
    turns = turnover(factor)

    expected_corr = pd.Series(
        [1.0, -1.0],
        index=pd.Index(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="date"),
        name="autocorrelation",
    )
    expected_turnover = pd.Series(
        [0.0, 2.0],
        index=expected_corr.index,
        name="turnover",
    )
    pd.testing.assert_series_equal(corr, expected_corr)
    pd.testing.assert_series_equal(turns, expected_turnover)


def test_quantile_turnover_tracks_new_names_in_selected_quantile() -> None:
    """quantile_turnover returns the share of names newly entering a quantile."""
    index = pd.MultiIndex.from_tuples(
        [
            ("2024-01-01", "AAA"),
            ("2024-01-01", "BBB"),
            ("2024-01-01", "CCC"),
            ("2024-01-02", "AAA"),
            ("2024-01-02", "BBB"),
            ("2024-01-02", "CCC"),
            ("2024-01-03", "AAA"),
            ("2024-01-03", "BBB"),
            ("2024-01-03", "CCC"),
        ],
        names=["date", "symbol"],
    )
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(index.get_level_values(0)), index.get_level_values(1)],
        names=index.names,
    )
    quantiles = pd.Series([1, 2, 2, 1, 1, 2, 2, 1, 2], index=index)

    result = quantile_turnover(quantiles, quantile=2)

    expected = pd.Series(
        [0.0, 0.5],
        index=pd.Index(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="date"),
        name=2,
    )
    pd.testing.assert_series_equal(result, expected)


def test_factor_metrics_return_nan_or_raise_for_undefined_inputs() -> None:
    """Factor metrics handle undefined ratios and invalid arguments."""
    constant_ic = pd.Series([0.1, 0.1, 0.1])
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
        ]
    )

    assert math.isnan(ic_ir(constant_ic, periods=12))
    with pytest.raises(ValueError, match="quantiles"):
        quantile_returns(factor, factor, quantiles=0)
    with pytest.raises(ValueError, match="lag"):
        autocorrelation(factor, lag=0)
    with pytest.raises(ValueError, match="period"):
        quantile_turnover(factor, quantile=1, period=0)
    with pytest.raises(ValueError, match="MultiIndex"):
        autocorrelation(pd.Series([1.0, 2.0]))


def _factor_and_forward_returns() -> tuple[pd.Series, pd.Series]:
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-01", "CCC", 3.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
            ("2024-01-02", "CCC", 3.0),
        ]
    )
    forward = _series(
        [
            ("2024-01-01", "AAA", -0.02),
            ("2024-01-01", "BBB", 0.01),
            ("2024-01-01", "CCC", 0.04),
            ("2024-01-02", "AAA", 0.05),
            ("2024-01-02", "BBB", 0.01),
            ("2024-01-02", "CCC", -0.03),
        ]
    )
    return factor, forward


def _dates() -> pd.DatetimeIndex:
    return pd.to_datetime(["2024-01-01", "2024-01-02"])


def _series(rows: list[tuple[str, str, float]]) -> pd.Series:
    dates = pd.to_datetime([row[0] for row in rows])
    symbols = [row[1] for row in rows]
    values = [row[2] for row in rows]
    index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    return pd.Series(values, index=index, dtype="float64")
