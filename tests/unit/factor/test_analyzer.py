"""Tests for the factor analysis facade."""

import math

import pandas as pd

from tradelearn.factor import FactorAnalyzer, clean_factor_and_forward_returns
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
        "quantile_spread_mean",
        "quantile_spread_cumulative_return",
        "turnover_mean",
        "autocorrelation_mean",
    }
    assert math.isclose(summary["ic_mean"], analyzer.ic().mean(), rel_tol=1e-12)
    assert math.isclose(summary["rank_ic_mean"], analyzer.rank_ic().mean(), rel_tol=1e-12)
    assert math.isclose(
        summary["quantile_spread_mean"],
        analyzer.quantile_spread().mean(),
        rel_tol=1e-12,
    )


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


def test_factor_analyzer_quantile_counts_returns_daily_group_sizes() -> None:
    """quantile_counts returns per-date sample counts by quantile."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    counts = analyzer.quantile_counts()

    assert list(counts.columns) == [1, 2]
    assert counts.index.equals(analyzer.quantile_returns().index)
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
    quantile_returns = analyzer.quantile_returns()

    pd.testing.assert_frame_equal(decay, quantile_returns.rolling(2, min_periods=1).mean())


def test_factor_analyzer_quantile_cumulative_returns_compounds_group_returns() -> None:
    """quantile_cumulative_returns compounds grouped forward returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    cumulative = analyzer.quantile_cumulative_returns()
    expected = (1.0 + analyzer.quantile_returns()).cumprod() - 1.0

    pd.testing.assert_frame_equal(cumulative, expected)


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


def test_factor_analyzer_long_short_returns_exposes_sides_and_spread() -> None:
    """long_short_returns returns long, short, and spread factor returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    returns = analyzer.long_short_returns()
    quantile_returns = analyzer.quantile_returns()

    assert list(returns.columns) == ["long", "short", "spread"]
    pd.testing.assert_series_equal(returns["long"], quantile_returns[2], check_names=False)
    pd.testing.assert_series_equal(returns["short"], quantile_returns[1], check_names=False)
    pd.testing.assert_series_equal(
        returns["spread"],
        analyzer.quantile_spread(),
        check_names=False,
    )


def test_factor_analyzer_long_short_cumulative_returns_compounds_portfolios() -> None:
    """long_short_cumulative_returns compounds long, short, and spread returns."""
    factor, forward = _factor_and_forward_returns()
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)

    cumulative = analyzer.long_short_cumulative_returns()
    expected = (1.0 + analyzer.long_short_returns()).cumprod() - 1.0

    pd.testing.assert_frame_equal(cumulative, expected)


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

    assert "Mean Return by Quantile" in titles
    assert "Quantile Spread" in titles
    assert "IC Histogram" in titles
    assert "IC QQ" in titles
    assert "Quantile Counts" in titles


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
    assert "Tradelearn Factor Analysis" in content
    assert "Factor Summary" in content
    assert "Quantile Statistics" in content
    assert "Information Coefficient" in content
    assert "Mean Return by Quantile" in content
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
        factor,
        prices,
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
    neutral = analyzer.quantile_returns(group_neutral=True)

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
