import pandas as pd
import pytest

from tradelearn import data
from tradelearn.core.errors import ContractError


def test_file_dataset_query_and_profile(tmp_path):
    pytest.importorskip("duckdb")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "symbol": ["A", "B", "A"],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    path = tmp_path / "bars.csv"
    frame.to_csv(path, index=False)
    with data.open_dataset(path) as dataset:
        bars = dataset.read(symbols=["A"], start="2026-01-02")
        assert len(bars) == 1
        assert bars.close.iloc[0] == 3
        assert bars.attrs["source_files"] == [str(path)]
        assert dataset.profile(symbols=["B"]).summary()["rows"] == 1
        assert dataset.read(columns=["close"]).columns.tolist() == ["close"]
        assert dataset.read(start="2030-01-01").empty
        with pytest.raises(ValueError):
            dataset.read(columns=["close; DROP TABLE bars"])
    frame = pd.concat([frame, frame.iloc[:1]])
    frame.to_csv(path, index=False)
    with data.open_dataset(path) as dataset:
        with pytest.raises(ContractError, match="duplicate"):
            dataset.read()


def test_dataset_is_public():
    assert callable(data.open_dataset)


def test_cache_uses_same_filtered_file_query(tmp_path):
    from tradelearn.core.contracts import validate_bars
    from tradelearn.data import BarsCache

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4, tz="UTC"), ["A"]], names=["timestamp", "symbol"]
    )
    bars = pd.DataFrame(
        {field: [1.0, 2.0, 3.0, 4.0] for field in ["open", "high", "low", "close", "volume"]},
        index=index,
    )
    cache = BarsCache(tmp_path)
    cache.write("fixture", "A", "1d", validate_bars(bars))
    result = cache.read("fixture", "A", "1d", start="2026-01-03", columns=["close"])
    assert result.close.tolist() == [3.0, 4.0]
    assert "query" in result.attrs


def test_multiple_files_pushdown_matches_pandas_and_rejects_schema_change(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, tz="UTC"),
            "symbol": ["A"] * 6,
            **{name: range(1, 7) for name in ["open", "high", "low", "close", "volume"]},
        }
    )
    for i, part in enumerate([frame.iloc[:3], frame.iloc[3:]]):
        part.to_parquet(tmp_path / f"{i}.parquet", index=False)
    with data.open_dataset(tmp_path / "*.parquet") as dataset:
        result = dataset.read(start="2026-01-03", end="2026-01-05")
        assert result.close.tolist() == [3, 4, 5]
        batches = list(dataset.iter_batches(batch_size=2))
        assert [len(batch) for batch in batches] == [2, 2, 2]
        pd.testing.assert_frame_equal(pd.concat(batches), dataset.read())
    frame.iloc[:1].drop(columns="volume").to_parquet(tmp_path / "bad.parquet", index=False)
    with data.open_dataset(tmp_path / "*.parquet") as dataset:
        with pytest.raises(ValueError, match="schema"):
            dataset.read()


def test_dataset_rejects_missing_timestamp(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00Z", None]),
            "symbol": ["A", "A"],
            **{name: [1.0, 2.0] for name in ["open", "high", "low", "close", "volume"]},
        }
    )
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    with data.open_dataset(path) as dataset:
        with pytest.raises(ValueError, match="timestamp"):
            dataset.read()
