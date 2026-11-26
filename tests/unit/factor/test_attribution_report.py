"""PerformanceAttribution must render like every other analysis facade.

FactorAnalyzer exposes plot()/to_analysis()/html()/report(); attribution only
returned DataFrames, so there was no interface to show. These tests pin the
same contract: real computed numbers, no invented series, and an empty input
degrades to an empty report instead of raising.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradelearn.factor import PerformanceAttribution


def _attribution() -> PerformanceAttribution:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    returns = pd.Series([0.0100, -0.0050, 0.0075], index=dates, name="returns")
    positions = pd.DataFrame({"AAA": [0.6, 0.2, 0.4], "BBB": [0.4, 0.8, 0.6]}, index=dates)
    factor_returns = pd.DataFrame(
        {"value": [0.010, -0.020, 0.005], "momentum": [0.020, 0.010, -0.010]}, index=dates
    )
    factor_loadings = pd.DataFrame(
        {
            "value": [1.0, -0.5, 0.5, 0.0, 0.8, -0.2],
            "momentum": [0.2, 1.0, 0.4, 0.8, 0.1, 0.9],
        },
        index=pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"]),
    )
    return PerformanceAttribution(
        returns=returns,
        positions=positions,
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
    )


def test_to_analysis_carries_the_computed_tables_and_metrics() -> None:
    analysis = _attribution().to_analysis()

    assert analysis["title"] == "TradeLearn Performance Attribution"
    assert {"Attribution Summary", "Factor Exposures", "Return Attribution"} <= set(
        analysis["tables"]
    )
    summary, exposure_summary = _attribution().summary()
    for name, value in summary.items():
        assert analysis["metrics"][name]["value"] == pytest.approx(value)
    assert analysis["metadata"]["factors"] == list(exposure_summary.index)
    assert analysis["metadata"]["observations"] == 3
    assert analysis["figures"] == {}


def test_plot_returns_a_bokeh_layout_of_attribution_charts() -> None:
    layout = _attribution().plot()

    assert layout.children, "attribution must render at least one chart"
    titles = [
        child.title.text
        for child in layout.children
        if getattr(child, "title", None) is not None
    ]
    assert "Factor Exposures" in titles
    assert "Cumulative Attribution" in titles


def test_report_writes_html_with_the_real_numbers(tmp_path: Path) -> None:
    path = _attribution().report(str(tmp_path / "attribution.html"))

    assert Path(path).exists()
    text = Path(path).read_text(errors="ignore")
    assert "Performance Attribution" in text
    assert "specific_returns" in text


def test_figures_are_only_computed_when_requested() -> None:
    analysis = _attribution().to_analysis(include_figures=True)
    assert set(analysis["figures"]) == {"Attribution charts"}


def test_empty_attribution_degrades_instead_of_raising(tmp_path: Path) -> None:
    empty = PerformanceAttribution(
        returns=pd.Series(dtype="float64"),
        positions=pd.DataFrame(),
        factor_returns=pd.DataFrame(),
        factor_loadings=pd.DataFrame(),
    )

    analysis = empty.to_analysis()
    assert analysis["metadata"]["observations"] == 0
    assert empty.plot().children == []
    assert Path(empty.report(str(tmp_path / "empty.html"))).exists()
