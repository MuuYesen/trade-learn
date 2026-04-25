"""Tests for HTML report export."""

from types import SimpleNamespace

import pandas as pd

from tradelearn import metrics
from tradelearn.report import Reporter


def test_reporter_html_writes_single_file_tear_sheet(tmp_path) -> None:
    """Reporter.html writes a shareable HTML tear sheet."""
    path = tmp_path / "report.html"
    stats = _stats()

    result = Reporter(stats, periods=252).html(path)

    assert result == path
    html = path.read_text()
    assert "<!doctype html>" in html.lower()
    assert "Summary Stats" in html
    assert "demo-strategy" in html
    assert "run-001" in html
    assert "Generated" in html
    assert "Equity Curve" in html
    assert "Drawdown" in html
    assert "Top 10 Drawdowns" in html
    assert "Monthly Returns Heatmap" in html
    assert "Rolling Sharpe" in html
    assert "Trade Distribution" in html
    assert "Tradelearn" in html
    assert "Bokeh" in html
    assert "annual_return" in html


def test_reporter_html_accepts_mapping_stats(tmp_path) -> None:
    """Reporter.html accepts dict-shaped stats."""
    path = tmp_path / "mapping-report.html"

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "summary": {"strategy_name": "demo"},
            "config": {"strategy": "demo"},
        }
    ).html(path)

    assert path.exists()


def _stats() -> SimpleNamespace:
    returns = _returns()
    return SimpleNamespace(
        returns=returns,
        equity=metrics.cum_returns(returns, starting_value=100_000.0),
        trades=_trades(),
        positions=pd.DataFrame(),
        orders=pd.DataFrame(),
        summary={"strategy_name": "demo-strategy"},
        analyzers={},
        config={"strategy": "demo", "run_id": "run-001"},
    )


def _returns() -> pd.Series:
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="returns",
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame({"pnl": [100.0, -50.0, 25.0, -10.0]})
