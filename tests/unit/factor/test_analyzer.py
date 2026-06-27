"""Tests for the factor analysis facade."""

import math

import pandas as pd
import pytest

from tradelearn.factor import (
    FactorAnalyzer,
    MultiFactorAnalyzer,
    MultiPeriodFactorAnalyzer,
    clean_factor_and_forward_returns,
)
from tradelearn.metrics import factor as factor_metrics


def test_factor_analyzer_delegates_core_metrics() -> None:
    """FactorAnalyzer exposes alphalens-style factor diagnostics."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, periods=12, quantiles=2)

    pd.testing.assert_series_equal(analyzer.ic(), factor_metrics.ic(factor, forward))
    pd.testing.assert_series_equal(
        analyzer.factor_information_coefficient(),
        factor_metrics.rank_ic(factor, forward).rename("factor_information_coefficient"),
    )
    pd.testing.assert_frame_equal(
        analyzer.mean_return_by_quantile(),
        factor_metrics.quantile_returns(factor, forward, quantiles=2),
    )
    turnover = analyzer.quantile_turnover(quantile=2)
    assert turnover.name == 2
    assert not turnover.empty
    pd.testing.assert_series_equal(
        analyzer.factor_rank_autocorrelation(),
        factor_metrics.autocorrelation(factor).rename("factor_rank_autocorrelation"),
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
        analyzer.mean_return_by_quantile(),
        factor_metrics.factor_returns(factor, prices, quantiles=2),
    )
    assert list(analyzer.factor_returns().columns) == ["1D"]


def test_factor_analyzer_price_derived_ic_uses_symbol_forward_returns() -> None:
    """Price-derived IC aligns each symbol to its own next-period return."""
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
    expected_forward = _series(
        [
            ("2024-01-01", "AAA", 0.10),
            ("2024-01-01", "BBB", -0.10),
            ("2024-01-02", "AAA", -0.10),
            ("2024-01-02", "BBB", 0.10),
        ]
    )
    analyzer = FactorAnalyzer(factor, prices=prices)

    pd.testing.assert_series_equal(analyzer.ic(), factor_metrics.ic(factor, expected_forward))


def test_factor_clean_data_accepts_factor_frame_and_prices() -> None:
    """clean_factor_and_forward_returns uses factor table plus prices as the public path."""
    factors = _factor_price_frame()[["value_score"]]
    prices = _factor_price_frame()["close"]

    clean = clean_factor_and_forward_returns(
        factors,
        factor="value_score",
        prices=prices,
        periods=(1, 2),
        quantiles=2,
    )
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1, 2), quantiles=2)

    assert isinstance(analyzer, MultiPeriodFactorAnalyzer)
    assert set(analyzer.keys()) == {1, 2}
    assert list(analyzer.summary().index) == [1, 2]
    assert list(analyzer.ic().columns) == [1, 2]
    assert isinstance(analyzer[1], FactorAnalyzer)
    assert analyzer[1].forward_period == 1
    assert analyzer[2].forward_period == 2
    expected = _series(
        [
            ("2024-01-01", "AAA", 0.10),
            ("2024-01-01", "BBB", -0.10),
        ]
    ).rename("forward_returns")
    pd.testing.assert_series_equal(analyzer[1].forward_returns.dropna(), expected)


def test_factor_clean_data_accepts_multiple_factors() -> None:
    """Factor analysis can compare multiple factor columns from one factor table."""
    factors = _factor_price_frame()[["value_score"]].copy()
    factors["reverse_score"] = -factors["value_score"]
    prices = _factor_price_frame()["close"]

    clean = clean_factor_and_forward_returns(
        factors,
        factor=("value_score", "reverse_score"),
        prices=prices,
        periods=(1, 2),
        quantiles=2,
    )
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1, 2), quantiles=2)

    assert isinstance(analyzer, MultiFactorAnalyzer)
    assert set(analyzer.keys()) == {"value_score", "reverse_score"}
    assert isinstance(analyzer["value_score"], MultiPeriodFactorAnalyzer)
    assert list(analyzer.summary().index.names) == ["factor", "period"]
    assert ("value_score", 1) in analyzer.summary().index
    assert ("reverse_score", 2) in analyzer.summary().index


def test_factor_analyzer_builds_period_analyzers_from_clean_factor_data() -> None:
    """FactorAnalyzer can expose separate 1D/5D/10D analyzers from clean factor data."""
    factor, _ = _factor_and_forward_returns()
    clean = pd.DataFrame(
        {
            "factor": factor,
            "forward_return_1": [0.01, 0.02, 0.03, 0.02, 0.01, -0.01],
            "forward_return_5": [0.03, 0.04, 0.05, 0.01, 0.02, 0.03],
            "forward_return_10": [0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            "factor_quantile": [1, 1, 2, 1, 2, 2],
        }
    )

    analyzers = FactorAnalyzer.from_clean_factor_data(clean, periods=(1, 5, 10), quantiles=2)

    assert isinstance(analyzers, MultiPeriodFactorAnalyzer)
    assert set(analyzers.keys()) == {1, 5, 10}
    assert analyzers[1].forward_returns is not None
    pd.testing.assert_series_equal(
        analyzers[5].forward_returns,
        clean["forward_return_5"].rename("forward_returns"),
    )
    assert analyzers[10].quantiles == 2


def test_factor_analyzer_multi_period_summary_prefixes_metrics() -> None:
    """multi_period_summary returns one row per prediction horizon."""
    factor, _ = _factor_and_forward_returns()
    clean = pd.DataFrame(
        {
            "factor": factor,
            "forward_return_1": [0.01, 0.02, 0.03, 0.02, 0.01, -0.01],
            "forward_return_5": [0.03, 0.04, 0.05, 0.01, 0.02, 0.03],
            "factor_quantile": [1, 1, 2, 1, 2, 2],
        }
    )

    summary = FactorAnalyzer.multi_period_summary(clean, periods=(1, 5), quantiles=2)

    assert list(summary.index) == [1, 5]
    assert {
        "ic_mean",
        "rank_ic_mean",
        "mean_returns_spread_mean",
    }.issubset(summary.columns)


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
        "rank_ic_std",
        "rank_ic_ir",
        "mean_returns_spread_mean",
        "mean_returns_spread_annualized",
        "mean_returns_spread_cumulative_return",
        "monotonicity",
        "quantile_turnover_mean",
        "factor_rank_autocorrelation_mean",
        "observations",
        "ic_dates",
    }
    assert math.isclose(summary["ic_mean"], analyzer.ic().mean(), rel_tol=1e-12)
    assert math.isclose(summary["rank_ic_mean"], analyzer.factor_information_coefficient().mean())
    assert math.isclose(
        summary["rank_ic_std"],
        analyzer.factor_information_coefficient().std(ddof=1),
        rel_tol=1e-12,
    )
    assert math.isclose(
        summary["rank_ic_ir"],
        factor_metrics.ic_ir(analyzer.factor_information_coefficient(), periods=12),
        rel_tol=1e-12,
    )
    assert math.isclose(
        summary["mean_returns_spread_mean"],
        analyzer.compute_mean_returns_spread()[0].mean(),
        rel_tol=1e-12,
    )
    assert math.isclose(
        summary["mean_returns_spread_annualized"],
        analyzer.compute_mean_returns_spread()[0].mean() * 12,
        rel_tol=1e-12,
    )
    assert math.isclose(
        summary["monotonicity"],
        analyzer.monotonicity()["spearman_rho"],
        rel_tol=1e-12,
    )
    assert summary["observations"] == len(factor)
    assert summary["ic_dates"] == analyzer.factor_information_coefficient().count()


def test_factor_analyzer_evaluate_pandas_matches_existing_metrics() -> None:
    """evaluate(engine='pandas') exposes the existing factor diagnostics together."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, periods=12, quantiles=2)

    result = analyzer.evaluate(engine="pandas")

    assert result.engine == "pandas"
    pd.testing.assert_series_equal(result.ic, analyzer.ic().rename("ic"))
    pd.testing.assert_series_equal(
        result.rank_ic,
        analyzer.factor_information_coefficient().rename("rank_ic"),
    )
    pd.testing.assert_frame_equal(
        result.mean_return_by_quantile,
        analyzer.mean_return_by_quantile(),
    )
    pd.testing.assert_frame_equal(result.quantile_counts, analyzer.quantile_counts())
    pd.testing.assert_series_equal(
        result.mean_returns_spread,
        analyzer.compute_mean_returns_spread()[0],
    )
    assert result.summary()["observations"] == len(factor)


def test_factor_analyzer_evaluate_rejects_unknown_engine() -> None:
    """evaluate validates engine selection explicitly."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    with pytest.raises(ValueError, match="engine"):
        analyzer.evaluate(engine="unknown")


def test_factor_analyzer_evaluate_rust_matches_pandas_backend() -> None:
    """Rust factor analytics backend matches the pandas reference output."""
    pytest.importorskip("tradelearn._rust")
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    try:
        actual = analyzer.evaluate(engine="rust")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    expected = analyzer.evaluate(engine="pandas")

    assert actual.engine == "rust"
    pd.testing.assert_series_equal(actual.ic, expected.ic)
    pd.testing.assert_series_equal(actual.rank_ic, expected.rank_ic)
    pd.testing.assert_frame_equal(actual.mean_return_by_quantile, expected.mean_return_by_quantile)
    pd.testing.assert_frame_equal(
        actual.quantile_counts.astype("int64"),
        expected.quantile_counts.astype("int64"),
    )
    pd.testing.assert_series_equal(actual.mean_returns_spread, expected.mean_returns_spread)


def test_factor_analyzer_evaluate_rust_uses_clean_factor_quantiles() -> None:
    """Rust evaluation honors canonical quantile labels from clean data."""
    pytest.importorskip("tradelearn._rust")
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01"]), list("ABCDE")],
        names=["date", "symbol"],
    )
    clean = pd.DataFrame(
        {
            "factor": [1.0, 1.0, 1.0, 1.0, 1.0],
            "forward_return_5": [0.10, 0.20, 0.30, 0.40, 0.50],
            "factor_quantile": [5, 4, 3, 2, 1],
        },
        index=index,
    )
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(5,), quantiles=5)[5]

    actual = analyzer.evaluate(engine="rust")
    expected = analyzer.evaluate(engine="pandas")

    pd.testing.assert_frame_equal(actual.mean_return_by_quantile, expected.mean_return_by_quantile)
    pd.testing.assert_frame_equal(
        actual.quantile_counts.astype("int64"),
        expected.quantile_counts.astype("int64"),
    )
    pd.testing.assert_series_equal(actual.mean_returns_spread, expected.mean_returns_spread)


def test_factor_analyzer_quantile_stats_summarizes_groups() -> None:
    """quantile_stats summarizes grouped forward returns for reports."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    stats = analyzer.quantile_stats()
    quantile_returns = analyzer.mean_return_by_quantile()

    assert list(stats.columns) == ["mean", "std", "count", "cumulative_return"]
    assert list(stats.index) == [1, 2]
    assert stats.loc[1, "mean"] == quantile_returns[1].mean()
    assert stats.loc[2, "mean"] == quantile_returns[2].mean()
    assert stats.loc[1, "count"] == quantile_returns[1].count()
    assert stats.loc[2, "cumulative_return"] == (1.0 + quantile_returns[2]).prod() - 1.0


def test_factor_analyzer_quantile_counts_returns_daily_group_sizes() -> None:
    """quantile_counts returns per-date sample counts by quantile."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    counts = analyzer.quantile_counts()

    assert list(counts.columns) == [1, 2]
    assert counts.index.equals(analyzer.mean_return_by_quantile().index)
    assert counts.loc[pd.Timestamp("2024-01-01"), 1] == 2
    assert counts.loc[pd.Timestamp("2024-01-01"), 2] == 1
    assert counts.loc[pd.Timestamp("2024-01-02"), 1] == 2
    assert counts.loc[pd.Timestamp("2024-01-02"), 2] == 1


def test_factor_analyzer_quantile_counts_uses_valid_forward_return_sample() -> None:
    """quantile_counts ignores factor rows without usable forward returns."""
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-01", "CCC", 3.0),
        ]
    )
    forward = _series(
        [
            ("2024-01-01", "AAA", 0.01),
            ("2024-01-01", "BBB", 0.02),
        ]
    )
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=3)

    counts = analyzer.quantile_counts()

    assert list(counts.columns) == [1, 2]
    assert counts.loc[pd.Timestamp("2024-01-01"), 1] == 1
    assert counts.loc[pd.Timestamp("2024-01-01"), 2] == 1


def test_factor_analyzer_quantile_decay_returns_rolling_group_means() -> None:
    """quantile_decay returns rolling mean returns by quantile."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    decay = analyzer.quantile_decay(window=2)
    quantile_returns = analyzer.mean_return_by_quantile()

    pd.testing.assert_frame_equal(decay, quantile_returns.rolling(2, min_periods=1).mean())


def test_factor_analyzer_quantile_cumulative_returns_compounds_group_returns() -> None:
    """quantile_cumulative_returns compounds grouped forward returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    cumulative = analyzer.quantile_cumulative_returns()
    expected = (1.0 + analyzer.mean_return_by_quantile()).cumprod() - 1.0

    pd.testing.assert_frame_equal(cumulative, expected)


def test_factor_analyzer_quantile_cumulative_returns_skips_overlapping_periods() -> None:
    """quantile_cumulative_returns compounds non-overlapping forward returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(
        factor,
        forward_returns=forward,
        forward_period=2,
        quantiles=2,
    )

    cumulative = analyzer.quantile_cumulative_returns()
    expected = (1.0 + analyzer.mean_return_by_quantile().iloc[::2]).cumprod() - 1.0

    pd.testing.assert_frame_equal(cumulative, expected)


def test_factor_analyzer_quantile_spread_returns_top_minus_bottom() -> None:
    """quantile_spread returns top-minus-bottom quantile returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    spread = analyzer.compute_mean_returns_spread()[0]
    quantile_returns = analyzer.mean_return_by_quantile()
    expected = quantile_returns[2] - quantile_returns[1]
    expected.name = "mean_returns_spread"

    pd.testing.assert_series_equal(spread, expected)
    assert spread.name == "mean_returns_spread"


def test_factor_analyzer_long_short_returns_exposes_sides_and_spread() -> None:
    """long_short_returns returns long, short, and spread factor returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    returns = analyzer.long_short_returns()
    quantile_returns = analyzer.mean_return_by_quantile()

    assert list(returns.columns) == ["long", "short", "spread"]
    pd.testing.assert_series_equal(returns["long"], quantile_returns[2], check_names=False)
    pd.testing.assert_series_equal(returns["short"], -quantile_returns[1], check_names=False)
    pd.testing.assert_series_equal(
        returns["spread"],
        analyzer.compute_mean_returns_spread()[0],
        check_names=False,
    )


def test_factor_analyzer_long_short_cumulative_returns_compounds_portfolios() -> None:
    """long_short_cumulative_returns compounds long, short, and spread returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    cumulative = analyzer.long_short_cumulative_returns()
    expected = (1.0 + analyzer.long_short_returns()).cumprod() - 1.0

    pd.testing.assert_frame_equal(cumulative, expected)


def test_factor_analyzer_summary_compounds_spread_non_overlapping_periods() -> None:
    """summary cumulative spread uses non-overlapping forward returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(
        factor,
        forward_returns=forward,
        forward_period=2,
        quantiles=2,
    )

    spread = analyzer.compute_mean_returns_spread()[0]
    expected = float((1.0 + spread.iloc[::2]).cumprod().iloc[-1] - 1.0)

    assert math.isclose(
        analyzer.summary()["mean_returns_spread_cumulative_return"],
        expected,
        rel_tol=1e-12,
    )


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


def _factor_price_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "AAA"),
            (pd.Timestamp("2024-01-01"), "BBB"),
            (pd.Timestamp("2024-01-02"), "AAA"),
            (pd.Timestamp("2024-01-02"), "BBB"),
            (pd.Timestamp("2024-01-03"), "AAA"),
            (pd.Timestamp("2024-01-03"), "BBB"),
        ],
        names=["date", "symbol"],
    )
    return pd.DataFrame(
        {
            "value_score": [1.0, 2.0, 1.0, 2.0, 1.5, 2.5],
            "close": [100.0, 100.0, 110.0, 90.0, 99.0, 99.0],
        },
        index=index,
    )


def test_factor_analyzer_monotonicity_detects_ordered_quantiles() -> None:
    """monotonicity() returns spearman_rho=1 when quantiles are perfectly ordered."""
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-01", "CCC", 3.0),
        ]
    )
    forward = _series(
        [
            ("2024-01-01", "AAA", 0.01),
            ("2024-01-01", "BBB", 0.02),
            ("2024-01-01", "CCC", 0.03),
        ]
    )
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=3)

    result = analyzer.monotonicity()

    assert "spearman_rho" in result
    assert "is_monotone" in result
    assert isinstance(result["spearman_rho"], float)
    assert isinstance(result["is_monotone"], bool)


def test_factor_analyzer_plot_returns_bokeh_layout() -> None:
    """plot() returns a Bokeh layout when data is available."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    layout = analyzer.plot()

    assert layout is not None


def test_factor_analyzer_plot_includes_alphalens_style_sections() -> None:
    """plot() includes returns, IC, turnover, and distribution diagnostics."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    layout = analyzer.plot()
    titles = _collect_titles(layout)

    assert "Factor Mean Return by Quantile" in titles
    assert "Factor Quantile Returns Violin" in titles
    assert "Factor Quantile Spread" in titles
    assert "Factor Events Distribution" in titles
    assert "Factor IC Histogram" in titles
    assert "Factor IC QQ" in titles
    assert "Factor Quantile Counts" in titles


def test_factor_analyzer_html_writes_file(tmp_path) -> None:
    """html() writes an Alphalens-style standalone report."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)
    output = tmp_path / "factor_report.html"

    result = analyzer.html(str(output))

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
    content = output.read_text()
    assert "<html" in content.lower()
    assert "TradeLearn Factor Analysis" in content
    assert "Factor Summary" in content
    assert "Quantile Statistics" in content
    assert "Information Coefficient" in content
    assert "Factor Mean Return by Quantile" in content
    assert "IC Histogram" in content


def test_factor_analyzer_report_dispatches_html(tmp_path) -> None:
    """report() is the user-facing alias for factor HTML reports."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)
    output = tmp_path / "factor_report.html"

    result = analyzer.report(str(output))

    assert result == output
    assert output.exists()
    assert "<html" in output.read_text().lower()


def test_multi_period_factor_analyzer_report_dispatches_single_period_html(tmp_path) -> None:
    """Single-period clean-data analysis can write the factor report directly."""
    factor, _ = _factor_and_forward_returns()
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-01", "CCC", 100.0),
            ("2024-01-02", "AAA", 101.0),
            ("2024-01-02", "BBB", 99.0),
            ("2024-01-02", "CCC", 102.0),
            ("2024-01-03", "AAA", 102.0),
            ("2024-01-03", "BBB", 98.0),
            ("2024-01-03", "CCC", 101.0),
        ]
    )
    clean = clean_factor_and_forward_returns(
        factor.rename("value").to_frame(),
        factor="value",
        prices=prices,
        periods=(1,),
        quantiles=2,
    )
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1,), quantiles=2)
    output = tmp_path / "factor_report.html"

    result = analyzer.report(str(output))

    assert result == output
    assert output.exists()
    assert "<html" in output.read_text().lower()


def test_multi_period_factor_analyzer_report_defaults_to_all_periods(tmp_path) -> None:
    """Multi-period clean-data analysis writes one report with all horizons by default."""
    factor, _ = _factor_and_forward_returns()
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-01", "CCC", 100.0),
            ("2024-01-02", "AAA", 101.0),
            ("2024-01-02", "BBB", 99.0),
            ("2024-01-02", "CCC", 102.0),
            ("2024-01-03", "AAA", 102.0),
            ("2024-01-03", "BBB", 98.0),
            ("2024-01-03", "CCC", 101.0),
            ("2024-01-04", "AAA", 103.0),
            ("2024-01-04", "BBB", 97.0),
            ("2024-01-04", "CCC", 100.0),
        ]
    )
    clean = clean_factor_and_forward_returns(
        factor.rename("value").to_frame(),
        factor="value",
        prices=prices,
        periods=(1, 2),
        quantiles=2,
    )
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1, 2), quantiles=2)
    output = tmp_path / "factor_multi_period.html"

    result = analyzer.report(str(output))

    assert result == output
    content = output.read_text()
    assert "Multi-Period Factor Analysis" in content
    assert "Period Summary" in content
    assert "1-bar Forward Return" in content
    assert "2-bar Forward Return" in content


def test_clean_factor_and_forward_returns_builds_alphalens_style_frame() -> None:
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-02", "AAA", 3.0),
            ("2024-01-02", "BBB", 4.0),
        ]
    )
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-02", "AAA", 110.0),
            ("2024-01-02", "BBB", 90.0),
            ("2024-01-03", "AAA", 121.0),
            ("2024-01-03", "BBB", 99.0),
        ]
    )
    groups = pd.Series(
        ["tech", "finance", "tech", "finance"],
        index=factor.index,
        name="group",
    )

    clean = clean_factor_and_forward_returns(
        factor.rename("value").to_frame(),
        factor="value",
        prices=prices,
        periods=(1,),
        quantiles=2,
        groupby=groups,
    )

    assert list(clean.columns) == ["factor", "forward_return_1", "factor_quantile", "group"]
    assert math.isclose(
        clean.loc[(pd.Timestamp("2024-01-01"), "AAA"), "forward_return_1"],
        0.10,
        rel_tol=1e-12,
    )
    assert clean.loc[(pd.Timestamp("2024-01-01"), "AAA"), "factor_quantile"] == 1
    assert clean.loc[(pd.Timestamp("2024-01-01"), "BBB"), "factor_quantile"] == 2
    assert clean.loc[(pd.Timestamp("2024-01-01"), "BBB"), "group"] == "finance"


def test_clean_factor_and_forward_returns_drops_tied_bins_like_alphalens() -> None:
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 1.0),
            ("2024-01-01", "CCC", 1.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
            ("2024-01-02", "CCC", 3.0),
        ]
    )
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-01", "CCC", 100.0),
            ("2024-01-02", "AAA", 101.0),
            ("2024-01-02", "BBB", 102.0),
            ("2024-01-02", "CCC", 103.0),
            ("2024-01-03", "AAA", 102.0),
            ("2024-01-03", "BBB", 104.0),
            ("2024-01-03", "CCC", 106.0),
        ]
    )
    clean = clean_factor_and_forward_returns(
        factor.rename("value").to_frame(),
        factor="value",
        prices=prices,
        periods=(1,),
        quantiles=3,
    )

    assert len(clean) == 3
    assert pd.Timestamp("2024-01-01") not in clean.index.get_level_values(0)
    assert set(clean["factor_quantile"]) == {1, 2, 3}


def test_factor_analyzer_from_clean_data_uses_existing_factor_quantiles() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("2024-01-01", "AAA"),
            ("2024-01-01", "BBB"),
            ("2024-01-01", "CCC"),
        ],
        names=["date", "symbol"],
    )
    clean = pd.DataFrame(
        {
            "factor": [1.0, 1.0, 1.0],
            "forward_return_1": [0.01, 0.02, 0.03],
            "factor_quantile": [1, 1, 3],
        },
        index=index,
    )

    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1,), quantiles=3)[1]

    quantile_returns = analyzer.mean_return_by_quantile()
    assert quantile_returns.loc["2024-01-01", 1] == 0.015
    assert quantile_returns.loc["2024-01-01", 3] == 0.03
    assert 2 not in quantile_returns.columns


def test_factor_analyzer_group_ic_and_group_neutral_returns() -> None:
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-01-01", "CCC", 1.0),
            ("2024-01-01", "DDD", 2.0),
            ("2024-01-02", "AAA", 1.0),
            ("2024-01-02", "BBB", 2.0),
            ("2024-01-02", "CCC", 1.0),
            ("2024-01-02", "DDD", 2.0),
        ]
    )
    forward = _series(
        [
            ("2024-01-01", "AAA", 0.01),
            ("2024-01-01", "BBB", 0.03),
            ("2024-01-01", "CCC", -0.02),
            ("2024-01-01", "DDD", 0.00),
            ("2024-01-02", "AAA", 0.02),
            ("2024-01-02", "BBB", 0.04),
            ("2024-01-02", "CCC", -0.01),
            ("2024-01-02", "DDD", 0.01),
        ]
    )
    groups = pd.Series(
        ["tech", "tech", "finance", "finance"] * 2,
        index=factor.index,
        name="group",
    )
    analyzer = FactorAnalyzer(factor, forward_returns=forward, groups=groups, quantiles=2)

    by_group = analyzer.ic(by_group=True)
    neutral = analyzer.mean_return_by_quantile(group_neutral=True)

    assert set(by_group.columns) == {"finance", "tech"}
    assert list(neutral.columns) == [1, 2]
    assert abs(float(neutral.mean().mean())) < 0.03


def test_factor_analyzer_monthly_ic_heatmap_and_event_returns() -> None:
    factor = _series(
        [
            ("2024-01-01", "AAA", 1.0),
            ("2024-01-01", "BBB", 2.0),
            ("2024-02-01", "AAA", 1.0),
            ("2024-02-01", "BBB", 2.0),
        ]
    )
    forward = _series(
        [
            ("2024-01-01", "AAA", 0.01),
            ("2024-01-01", "BBB", 0.02),
            ("2024-02-01", "AAA", 0.03),
            ("2024-02-01", "BBB", 0.01),
        ]
    )
    prices = _series(
        [
            ("2024-01-01", "AAA", 100.0),
            ("2024-01-02", "AAA", 101.0),
            ("2024-01-03", "AAA", 103.02),
            ("2024-01-01", "BBB", 100.0),
            ("2024-01-02", "BBB", 99.0),
            ("2024-01-03", "BBB", 98.01),
        ]
    )
    events = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "AAA"), (pd.Timestamp("2024-01-02"), "BBB")],
        names=["date", "symbol"],
    )
    analyzer = FactorAnalyzer(factor, forward_returns=forward)

    heatmap = analyzer.monthly_ic_heatmap()
    event = FactorAnalyzer.event_returns(prices, events, before=1, after=1)

    assert 2024 in heatmap.index
    assert {1, 2}.issubset(set(heatmap.columns))
    assert list(event.index) == [-1, 0, 1]
    assert event.loc[0, "mean"] == 0.0


def _series(rows: list[tuple[str, str, float]]) -> pd.Series:
    dates = pd.to_datetime([row[0] for row in rows])
    symbols = [row[1] for row in rows]
    values = [row[2] for row in rows]
    index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    return pd.Series(values, index=index, dtype="float64")


def _collect_titles(layout) -> set[str]:
    titles: set[str] = set()
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        title = getattr(getattr(item, "title", None), "text", "")
        if title:
            titles.add(title)
        titles.update(_collect_titles(item))
    return titles
