"""Tests for Bars normalization and cache."""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from tradelearn.core import ContractError
from tradelearn.data import (
    BarsCache,
    CacheExpiredError,
    CacheMissError,
)
from tradelearn.data.bars import bars_fingerprint, normalize_bars
from tradelearn.research import FeatureSet


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


def test_feature_set_exposes_wide_fields_and_dataset() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [10.0, 20.0, 11.0, 22.0],
            "high": [11.0, 21.0, 12.0, 23.0],
            "low": [9.0, 19.0, 10.0, 17.0],
            "close": [10.0, 20.0, 12.0, 18.0],
            "volume": [100.0, 200.0, 120.0, 180.0],
        }
    )
    bars = normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")

    features = FeatureSet(
        {
            "ret_1d": lambda p: p.close.pct_change(),
        },
        target={"label": lambda p: p.close.shift(-1) / p.close - 1},
    )
    dataset = features.fit_transform(bars, include_target=True)

    assert not hasattr(features, "dataset")
    assert dataset.index.names == ["timestamp", "symbol"]
    assert dataset.columns.tolist() == ["ret_1d", "label"]
    assert dataset.loc[(pd.Timestamp("2024-01-02", tz="UTC"), "AAA"), "ret_1d"] == pytest.approx(0.2)
    assert dataset.loc[(pd.Timestamp("2024-01-01", tz="UTC"), "BBB"), "label"] == pytest.approx(-0.1)


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


def test_bars_cache_reports_ttl_freshness(tmp_path: Path) -> None:
    """BarsCache freshness compares metadata age against the configured TTL."""
    bars = _sample_bars()
    cache = BarsCache(tmp_path, ttl=timedelta(days=1))

    cache.write("test", "AAA", "1d", bars, now=pd.Timestamp("2024-01-01T00:00:00Z"))

    assert cache.exists("test", "AAA", "1d")
    assert cache.is_fresh("test", "AAA", "1d", now=pd.Timestamp("2024-01-01T12:00:00Z"))
    assert not cache.is_fresh(
        "test",
        "AAA",
        "1d",
        now=pd.Timestamp("2024-01-03T00:00:00Z"),
    )


def test_bars_cache_rejects_stale_entries_unless_offline(tmp_path: Path) -> None:
    """Online reads reject stale cache entries while offline reads reuse them."""
    bars = _sample_bars()
    cache = BarsCache(tmp_path, ttl=timedelta(hours=1))
    cache.write("test", "AAA", "1d", bars, now=pd.Timestamp("2024-01-02T00:00:00Z"))

    stale_now = pd.Timestamp("2024-01-02T02:00:00Z")

    with pytest.raises(CacheExpiredError, match="stale"):
        cache.read("test", "AAA", "1d", now=stale_now)

    loaded = BarsCache(tmp_path, ttl=timedelta(hours=1), offline=True).read(
        "test",
        "AAA",
        "1d",
        now=stale_now,
    )

    expected = bars.copy()
    expected.attrs["cache_stale"] = True
    pd.testing.assert_frame_equal(loaded, expected)
    assert loaded.attrs["cache_stale"] is True


def test_bars_cache_offline_mode_still_requires_cached_files(tmp_path: Path) -> None:
    """Offline mode does not hide missing cache entries."""
    cache = BarsCache(tmp_path, offline=True)

    with pytest.raises(CacheMissError, match="missing"):
        cache.read("test", "AAA", "1d")


def _sample_bars() -> pd.DataFrame:
    """Build a small deterministic Bars frame for cache tests."""
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
    return normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture")
