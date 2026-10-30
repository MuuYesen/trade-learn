from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.data import DuckDBBarsBackend
from tradelearn.data.bars import normalize_bars


class _FakeResult:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def fetchdf(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeConnection:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = pd.DataFrame() if frame is None else frame
        self.calls: list[tuple[str, list[object]]] = []
        self.registered: dict[str, pd.DataFrame] = {}
        self.unregistered: list[str] = []

    def execute(self, sql: str, params: list[object] | None = None) -> _FakeResult:
        self.calls.append((sql, [] if params is None else list(params)))
        return _FakeResult(self.frame)

    def register(self, name: str, frame: pd.DataFrame) -> None:
        self.registered[name] = frame.copy()

    def unregister(self, name: str) -> None:
        self.unregistered.append(name)


def _bars() -> pd.DataFrame:
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


def test_duckdb_backend_writes_contract_bars_with_schema() -> None:
    conn = _FakeConnection()
    backend = DuckDBBarsBackend(conn=conn)

    backend.write(_bars(), mode="replace")

    statements = [sql for sql, _params in conn.calls]
    assert any("CREATE TABLE IF NOT EXISTS bars" in sql for sql in statements)
    assert any("DELETE FROM bars" in sql for sql in statements)
    assert any("INSERT INTO bars" in sql for sql in statements)
    assert "bars_input" in conn.registered
    assert conn.unregistered == ["bars_input"]


def test_duckdb_backend_reads_filtered_column_subset() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "symbol": ["AAA", "AAA"],
            "close": [10.5, 11.5],
        }
    )
    conn = _FakeConnection(frame)
    backend = DuckDBBarsBackend(conn=conn)

    result = backend.read(
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-31",
        columns=["close"],
        market="US",
        freq="1d",
    )

    sql, params = conn.calls[-1]
    assert "SELECT timestamp, symbol, close" in sql
    assert "symbol IN (?)" in sql
    assert "timestamp >= ?" in sql
    assert "timestamp <= ?" in sql
    assert params == [
        "AAA",
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-31T00:00:00Z"),
    ]
    assert result.index.names == ["timestamp", "symbol"]
    assert result.attrs["engine"] == "duckdb"
    assert result.attrs["market"] == "US"
    assert result.attrs["freq"] == "1d"
    assert result["close"].tolist() == [10.5, 11.5]


def test_duckdb_backend_requires_duckdb_when_connection_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):
        if name == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="pip install trade-learn\\[duckdb\\]"):
        DuckDBBarsBackend("stage11.duckdb")


def test_duckdb_backend_reads_cross_section_by_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
            "symbol": ["AAA", "BBB"],
            "close": [11.5, 21.5],
            "volume": [120.0, 220.0],
        }
    )
    conn = _FakeConnection(frame)
    backend = DuckDBBarsBackend(conn=conn)

    result = backend.read_cross_section(
        "2024-01-02",
        symbols=["BBB", "AAA"],
        columns=["close", "volume"],
    )

    sql, params = conn.calls[-1]
    assert "timestamp >= ?" in sql
    assert "timestamp <= ?" in sql
    assert params[-2:] == [
        pd.Timestamp("2024-01-02T00:00:00Z"),
        pd.Timestamp("2024-01-02T00:00:00Z"),
    ]
    assert result.index.name == "symbol"
    assert result.loc["AAA", "close"] == 11.5
    assert result.loc["BBB", "volume"] == 220.0
    assert result.attrs["asof"] == pd.Timestamp("2024-01-02T00:00:00Z")


def test_duckdb_backend_read_panel_aliases_bars_contract_read() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "symbol": ["AAA", "BBB"],
            "close": [10.5, 21.5],
        }
    )
    conn = _FakeConnection(frame)
    backend = DuckDBBarsBackend(conn=conn)

    result = backend.read_panel(
        symbols=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-01-31",
        columns=["close"],
    )

    assert result.index.names == ["timestamp", "symbol"]
    assert result["close"].tolist() == [10.5, 21.5]
