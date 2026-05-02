"""Tests for external returns report entrypoints."""

from pathlib import Path

import pandas as pd

from tradelearn.report import Reporter


def test_reporter_from_returns_writes_external_returns_report(tmp_path: Path) -> None:
    """Reporter.from_returns writes the external returns report."""
    output = tmp_path / "external.html"

    result = Reporter.from_returns(
        returns=_returns(),
        positions=_positions(),
        transactions=_transactions(),
        benchmark=_benchmark(),
    ).report(output)

    html = output.read_text()
    assert result == output
    assert "Summary Stats" in html
    assert "Equity Curve" in html
    assert "Benchmark" in html
    assert (tmp_path / "stats.json").exists()
    assert not list(tmp_path.glob("*.parquet"))


def test_reporter_from_returns_adds_default_html_suffix(tmp_path: Path) -> None:
    """Reporter.from_returns uses Reporter.report suffix handling."""
    output = tmp_path / "external-without-suffix"

    result = Reporter.from_returns(returns=_returns()).report(output)

    assert result == tmp_path / "external-without-suffix.html"
    assert result.exists()


def test_reporter_from_returns_accepts_value_named_position_column(tmp_path: Path) -> None:
    """External positions accept wide positions when a symbol is named value."""
    output = tmp_path / "value-symbol.html"
    positions = _positions().rename(columns={"AAA": "value"})

    Reporter.from_returns(returns=_returns(), positions=positions).report(output)

    assert output.exists()


def test_reporter_from_returns_ignores_non_numeric_position_columns(tmp_path: Path) -> None:
    """External positions ignore metadata columns in wide positions."""
    output = tmp_path / "metadata-positions.html"
    positions = _positions()
    positions["sector"] = "technology"

    Reporter.from_returns(returns=_returns(), positions=positions).report(output)

    assert output.exists()


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
