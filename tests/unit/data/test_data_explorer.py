"""Tests for raw Bars exploratory analysis."""

from __future__ import annotations

import pandas as pd

from tradelearn.data import DataExplorer, explore, normalize_bars


def test_data_explorer_summarizes_bars_schema_and_missing_values() -> None:
    """DataExplorer exposes compact raw Bars profile tables."""
    bars = _bars()
    bars.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "BBB"), "volume"] = pd.NA

    explorer = explore(bars)

    assert isinstance(explorer, DataExplorer)
    assert explorer.summary()["rows"] == 6
    assert explorer.summary()["symbols"] == 2
    assert explorer.summary()["start"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert explorer.summary()["end"] == pd.Timestamp("2024-01-03", tz="UTC")
    assert explorer.summary()["freq"] == "1d"
    assert explorer.summary()["rows_by_symbol"] == {"AAA": 3, "BBB": 3}

    schema = explorer.schema()
    assert schema.loc["volume", "missing"] == 1
    assert schema.loc["volume", "missing_pct"] == 1 / 6
    assert schema.loc["close", "non_missing"] == 6

    missing = explorer.missing()
    assert missing.loc["volume", "missing"] == 1
    assert "mean" in explorer.describe().index


def test_data_explorer_reports_ohlcv_quality_issues() -> None:
    """OHLCV quality checks catch market-data-specific data errors."""
    bars = _bars()
    bars.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "AAA"), "high"] = 9.0
    bars.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "AAA"), "low"] = 10.0
    bars.loc[(pd.Timestamp("2024-01-03", tz="UTC"), "AAA"), "close"] = 20.0
    bars.loc[(pd.Timestamp("2024-01-03", tz="UTC"), "BBB"), "volume"] = -1.0

    quality = explore(bars).ohlcv_quality()

    assert quality.loc["high_below_low", "count"] == 1
    assert quality.loc["close_outside_range", "count"] == 1
    assert quality.loc["negative_volume", "count"] == 1
    assert quality.loc["duplicate_index", "count"] == 0


def test_data_explorer_returns_outliers_and_correlation() -> None:
    """DataExplorer derives returns, outliers, and multi-symbol correlations."""
    bars = _bars()
    bars.loc[(pd.Timestamp("2024-01-03", tz="UTC"), "AAA"), "close"] = 30.0
    explorer = DataExplorer(bars)

    returns = explorer.returns()
    outliers = explorer.outliers(zscore=1.0)
    correlation = explorer.correlation()

    assert returns.index.names == ["timestamp", "symbol"]
    assert returns.name == "return"
    assert ("AAA" in outliers["symbol"].astype(str).tolist())
    assert correlation.index.tolist() == ["AAA", "BBB"]
    assert correlation.columns.tolist() == ["AAA", "BBB"]


def test_data_explorer_report_writes_html(tmp_path) -> None:
    """report() writes a standalone HTML profile for raw data."""
    path = tmp_path / "data.html"

    result = explore(_bars()).report(path)

    assert result == path
    html = path.read_text(encoding="utf-8")
    assert "Data Exploration Report" in html
    assert "OHLCV Quality" in html
    assert "AAA" in html


def _bars() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "open": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "high": [11.0, 12.0, 13.0, 21.0, 20.0, 19.0],
            "low": [9.0, 10.0, 11.0, 19.0, 18.0, 17.0],
            "close": [10.5, 11.5, 12.5, 20.5, 19.5, 18.5],
            "volume": [100.0, 120.0, 130.0, 200.0, 180.0, 160.0],
        }
    )
    return normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")
