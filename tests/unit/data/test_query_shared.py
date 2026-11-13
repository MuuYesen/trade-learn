import pandas as pd
import pytest
from tradelearn.data.providers import _filter_dates
from tradelearn.data import DuckDBBarsBackend, open_dataset, BarsCache
from tradelearn.core.errors import ContractError


def panel():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), ["A", "B"]],
        names=["timestamp", "symbol"],
    )
    return pd.DataFrame(
        {
            name: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            for name in ("open", "high", "low", "close", "volume")
        },
        index=index,
    )


def test_aware_provider_filter_and_exact_local_query_share_utc_normalization():
    frame = panel().reset_index()
    begin = pd.Timestamp("2026-01-01 09:30", tz="Asia/Tokyo")
    end = pd.Timestamp("2026-01-01 10:00", tz="Asia/Tokyo")
    selected = _filter_dates(frame, start=begin, end=end)
    assert selected.close.tolist() == [3.0, 4.0]
    pd.testing.assert_frame_equal(frame, panel().reset_index())


def test_dataset_cache_and_duckdb_filter_projection_match_without_network(tmp_path):
    bars = panel()
    path = tmp_path / "bars.parquet"
    bars.reset_index().to_parquet(path, index=False)
    cache = BarsCache(tmp_path / "cache")
    cache.write("fixture", "A", "1h", bars)
    db = DuckDBBarsBackend()
    db.write(bars)
    options = dict(
        symbols=["A"],
        start="2026-01-01 00:30Z",
        end="2026-01-01 02:00Z",
        columns=["close", "close"],
    )
    with open_dataset(path) as ds:
        a = ds.read(**options)
        b = db.read(**options)
        c = cache.read("fixture", "A", "1h", **options)
        assert a.index.tolist() == b.index.tolist() == c.index.tolist()
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        pd.testing.assert_frame_equal(a.reset_index(drop=True), c.reset_index(drop=True))
        for query in (ds, db):
            with pytest.raises(ValueError):
                query.read(columns=["close; DROP TABLE bars"])
            with pytest.raises(ValueError):
                query.read(symbols=[])


def test_duckdb_validates_full_bars_before_field_projection():
    db = DuckDBBarsBackend()
    db.write(panel())
    db.conn.execute("UPDATE bars SET high=0")
    with pytest.raises(ContractError, match="OHLC"):
        db.read(columns=["close"])
