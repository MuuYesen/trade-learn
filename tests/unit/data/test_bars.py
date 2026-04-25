"""Tests for Bars normalization and cache."""

from pathlib import Path

import pandas as pd
import pytest

from tradelearn.core import ContractError
from tradelearn.data import BarsCache, bars_fingerprint, normalize_bars


def test_normalize_bars_builds_utc_multiindex_and_metadata() -> None:
    """Raw OHLCV rows become contract-valid Bars."""
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 09:30", "2024-01-02 09:30"],
            "symbol": ["AAA", "AAA"],
            "open": [10, 11],
            "high": [12, 12],
            "low": [9, 10],
            "close": [11, 11.5],
            "volume": [1000, 1200],
        }
    )

    bars = normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")

    assert bars.index.names == ["timestamp", "symbol"]
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    assert bars.attrs == {
        "market": "US",
        "freq": "1d",
        "adjust": "pre",
        "engine": "test",
        "source": "fixture",
    }
    assert bars["open"].dtype == "float64"


def test_normalize_bars_applies_pre_adjustment() -> None:
    """pre adjustment scales OHLC prices by adj_factor."""
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "symbol": ["AAA"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [1000.0],
            "adj_factor": [0.5],
        }
    )

    bars = normalize_bars(raw, market="CN", freq="1d", engine="test", source="fixture")

    row = bars.iloc[0]
    assert row["open"] == 5.0
    assert row["high"] == 6.0
    assert row["low"] == 4.5
    assert row["close"] == 5.5
    assert row["adj_factor"] == 0.5


def test_normalize_bars_rejects_duplicate_symbol_timestamps() -> None:
    """Each symbol timestamp must be unique."""
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAA", "AAA"],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "volume": [100.0, 100.0],
        }
    )

    with pytest.raises(ContractError, match="duplicate"):
        normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")


def test_bars_cache_round_trips_with_fingerprint(tmp_path: Path) -> None:
    """BarsCache stores parquet data and a stable metadata fingerprint."""
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAA", "AAA"],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100.0, 120.0],
        }
    )
    bars = normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")
    cache = BarsCache(tmp_path)

    written = cache.write("test", "AAA", "1d", bars)
    loaded = cache.read("test", "AAA", "1d")

    assert written.data_path.exists()
    assert written.meta_path.exists()
    pd.testing.assert_frame_equal(loaded, bars)
    assert loaded.attrs == bars.attrs
    assert cache.fingerprint("test", "AAA", "1d") == bars_fingerprint(bars)
