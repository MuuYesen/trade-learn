"""Tests for the factor analysis facade."""

import math

import pandas as pd

from tradelearn.factor import FactorAnalyzer
from tradelearn.metrics import factor as factor_metrics


def test_factor_analyzer_delegates_core_metrics() -> None:
    """FactorAnalyzer exposes alphalens-style factor diagnostics."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, periods=12, quantiles=2)

    pd.testing.assert_series_equal(analyzer.ic(), factor_metrics.ic(factor, forward))
    pd.testing.assert_series_equal(analyzer.rank_ic(), factor_metrics.rank_ic(factor, forward))
    pd.testing.assert_frame_equal(
        analyzer.quantile_returns(),
        factor_metrics.quantile_returns(factor, forward, quantiles=2),
    )
    pd.testing.assert_series_equal(analyzer.turnover(), factor_metrics.turnover(factor))
    pd.testing.assert_series_equal(
        analyzer.autocorrelation(),
        factor_metrics.autocorrelation(factor),
    )
    assert math.isclose(
        analyzer.ic_ir(),
        factor_metrics.ic_ir(factor_metrics.ic(factor, forward), periods=12),
        rel_tol=1e-12,
    )


def test_factor_analyzer_computes_returns_from_prices() -> None:
    """FactorAnalyzer can derive forward returns from aligned prices."""
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
    analyzer = FactorAnalyzer(factor, prices=prices, quantiles=2)

    pd.testing.assert_frame_equal(
        analyzer.factor_returns(),
        factor_metrics.factor_returns(factor, prices, quantiles=2),
    )


def test_factor_analyzer_summary_contains_stable_keys() -> None:
    """summary returns scalar diagnostics useful for reports."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, periods=12, quantiles=2)

    summary = analyzer.summary()

    assert set(summary) == {
        "ic_mean",
        "ic_std",
        "ic_ir",
        "rank_ic_mean",
        "turnover_mean",
        "autocorrelation_mean",
    }
    assert math.isclose(summary["ic_mean"], analyzer.ic().mean(), rel_tol=1e-12)
    assert math.isclose(summary["rank_ic_mean"], analyzer.rank_ic().mean(), rel_tol=1e-12)


def test_factor_analyzer_quantile_stats_summarizes_groups() -> None:
    """quantile_stats summarizes grouped forward returns for reports."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    stats = analyzer.quantile_stats()
    quantile_returns = analyzer.quantile_returns()

    assert list(stats.columns) == ["mean", "std", "count", "cumulative_return"]
    assert list(stats.index) == [1, 2]
    assert stats.loc[1, "mean"] == quantile_returns[1].mean()
    assert stats.loc[2, "mean"] == quantile_returns[2].mean()
    assert stats.loc[1, "count"] == quantile_returns[1].count()
    assert stats.loc[2, "cumulative_return"] == (1.0 + quantile_returns[2]).prod() - 1.0


def test_factor_analyzer_quantile_decay_returns_rolling_group_means() -> None:
    """quantile_decay returns rolling mean returns by quantile."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    decay = analyzer.quantile_decay(window=2)
    quantile_returns = analyzer.quantile_returns()

    pd.testing.assert_frame_equal(decay, quantile_returns.rolling(2, min_periods=1).mean())


def test_factor_analyzer_quantile_spread_returns_top_minus_bottom() -> None:
    """quantile_spread returns top-minus-bottom quantile returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    spread = analyzer.quantile_spread()
    quantile_returns = analyzer.quantile_returns()
    expected = quantile_returns[2] - quantile_returns[1]
    expected.name = "quantile_spread"

    pd.testing.assert_series_equal(spread, expected)
    assert spread.name == "quantile_spread"


def test_factor_analyzer_requires_returns_or_prices_for_return_metrics() -> None:
    """Return-based methods fail clearly when no returns source is configured."""
    factor, _ = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor)

    try:
        analyzer.ic()
    except ValueError as exc:
        assert "forward_returns or prices" in str(exc)
    else:
        raise AssertionError("ic should require forward returns or prices")


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


def _series(rows: list[tuple[str, str, float]]) -> pd.Series:
    dates = pd.to_datetime([row[0] for row in rows])
    symbols = [row[1] for row in rows]
    values = [row[2] for row in rows]
    index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    return pd.Series(values, index=index, dtype="float64")
