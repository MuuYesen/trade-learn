"""Tests for pyfolio-compatible report entrypoints."""

from pathlib import Path

import pandas as pd

from tradelearn.report import pyfolio
from tradelearn.report.pyfolio import create_full_tear_sheet


def test_create_full_tear_sheet_writes_html_report(tmp_path: Path) -> None:
    """pyfolio facade writes a Tradelearn HTML tear sheet."""
    output = tmp_path / "pyfolio-report.html"

    result = create_full_tear_sheet(
        _returns(),
        positions=_positions(),
        transactions=_transactions(),
        benchmark_rets=_benchmark(),
        output=output,
    )

    html = output.read_text()
    assert result == output
    assert "Summary Stats" in html
    assert "Equity Curve" in html
    assert "Annual Returns" in html
    assert "Monthly Returns Distribution" in html
    assert "Return Quantiles" in html
    assert "Benchmark" in html
    assert (tmp_path / "stats.json").exists()
    assert (tmp_path / "trades.parquet").exists()


def test_create_returns_tear_sheet_is_report_alias(tmp_path: Path) -> None:
    """returns tear sheet uses the same report writer with a pyfolio name."""
    output = tmp_path / "returns.html"

    result = pyfolio.create_returns_tear_sheet(_returns(), output=output)

    assert result == output
    assert "Rolling Returns" in output.read_text()


def test_pyfolio_report_alias_writes_default_html(tmp_path: Path) -> None:
    """report() is a compact pyfolio-compatible alias."""
    output = tmp_path / "report-without-suffix"

    result = pyfolio.report(_returns(), output=output)

    assert result == tmp_path / "report-without-suffix.html"
    assert result.exists()


def _returns() -> pd.Series:
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="returns",
    )


def _benchmark() -> pd.Series:
    return pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="benchmark",
    )


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [60_000.0, 61_000.0, 62_000.0, 61_500.0, 63_000.0],
            "cash": [40_000.0, 39_000.0, 38_000.0, 38_500.0, 37_000.0],
        },
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [100.0, -100.0],
            "price": [10.0, 11.0],
            "symbol": ["AAA", "AAA"],
            "pnl": [0.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-05"], utc=True),
    )
