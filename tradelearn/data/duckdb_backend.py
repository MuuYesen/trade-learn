"""Optional DuckDB backend for Bars contract storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.core.contracts import validate_bars
from tradelearn.data.bars import REQUIRED_COLUMNS


class DuckDBBarsBackend:
    """Columnar Bars storage backed by DuckDB.

    DuckDB is optional. Tests and custom integrations may inject a connection
    object with ``execute/register/unregister`` methods to avoid importing the
    package.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        conn: Any | None = None,
        table: str = "bars",
    ) -> None:
        if conn is None:
            try:
                import duckdb
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "DuckDBBarsBackend requires duckdb; install with "
                    "`pip install trade-learn[all]` or `pip install duckdb`."
                ) from exc
            conn = duckdb.connect(str(path or ":memory:"))
        self.conn = conn
        self.table = _table_identifier(table)

    def ensure_schema(self) -> None:
        """Create the Bars table if it does not exist."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                vwap DOUBLE,
                amount DOUBLE,
                adj_factor DOUBLE,
                PRIMARY KEY (timestamp, symbol)
            )
            """
        )

    def write(self, bars: pd.DataFrame, *, mode: str = "append") -> None:
        """Write contract-valid Bars rows into DuckDB."""
        validate_bars(bars)
        if mode not in {"append", "replace"}:
            raise ValueError("mode must be 'append' or 'replace'")
        self.ensure_schema()
        frame = bars.reset_index()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for column in ("vwap", "amount", "adj_factor"):
            if column not in frame:
                frame[column] = None
        columns = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "amount",
            "adj_factor",
        ]
        frame = frame[columns]
        self.conn.register("bars_input", frame)
        try:
            if mode == "replace":
                self.conn.execute(f"DELETE FROM {self.table}")
            self.conn.execute(
                f"""
                INSERT INTO {self.table} ({", ".join(columns)})
                SELECT {", ".join(columns)}
                FROM bars_input
                """
            )
        finally:
            self.conn.unregister("bars_input")

    def read(
        self,
        *,
        symbols: str | list[str] | tuple[str, ...] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
        market: str | None = None,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Read Bars rows with optional symbol, date, and column filters."""
        self.ensure_schema()
        selected_columns = _selected_columns(columns)
        where, params = _filters(symbols=symbols, start=start, end=end)
        sql = (
            f"SELECT {', '.join(['timestamp', 'symbol', *selected_columns])} "
            f"FROM {self.table}{where} ORDER BY timestamp, symbol"
        )
        frame = self.conn.execute(sql, params).fetchdf()
        if frame.empty:
            result = pd.DataFrame(columns=list(selected_columns))
            result.index = pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex([], tz="UTC", name="timestamp"),
                    pd.Index([], name="symbol"),
                ],
                names=["timestamp", "symbol"],
            )
        else:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["symbol"] = frame["symbol"].astype(str)
            result = frame.set_index(["timestamp", "symbol"]).sort_index()
        result.attrs.update(
            {
                "engine": "duckdb",
                "source": "duckdb",
            }
        )
        if market is not None:
            result.attrs["market"] = market
        if freq is not None:
            result.attrs["freq"] = freq
        return result

    def read_cross_section(
        self,
        timestamp: str | pd.Timestamp,
        *,
        symbols: str | list[str] | tuple[str, ...] | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
        market: str | None = None,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Read one timestamp as a symbol-indexed cross section."""
        asof = _utc_timestamp(timestamp)
        frame = self.read(
            symbols=symbols,
            start=asof,
            end=asof,
            columns=columns,
            market=market,
            freq=freq,
        )
        if frame.empty:
            result = pd.DataFrame(columns=frame.columns)
            result.index = pd.Index([], name="symbol")
        else:
            result = frame.reset_index(level="timestamp", drop=True).sort_index()
            result.index.name = "symbol"
        result.attrs.update(frame.attrs)
        result.attrs["asof"] = asof
        return result

    def read_panel(
        self,
        *,
        symbols: str | list[str] | tuple[str, ...] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
        market: str | None = None,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Read a timestamp/symbol indexed Bars panel."""
        return self.read(
            symbols=symbols,
            start=start,
            end=end,
            columns=columns,
            market=market,
            freq=freq,
        )


def _selected_columns(columns: list[str] | tuple[str, ...] | None) -> list[str]:
    if columns is None:
        return list(REQUIRED_COLUMNS)
    allowed = set(REQUIRED_COLUMNS) | {"vwap", "amount", "adj_factor"}
    selected = list(dict.fromkeys(str(column) for column in columns))
    unknown = [column for column in selected if column not in allowed]
    if unknown:
        raise ValueError(f"unsupported Bars columns: {unknown}")
    return selected


def _filters(
    *,
    symbols: str | list[str] | tuple[str, ...] | None,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if symbols is not None:
        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        if not symbol_list:
            raise ValueError("symbols must not be empty")
        clauses.append(f"symbol IN ({', '.join(['?'] * len(symbol_list))})")
        params.extend(str(symbol) for symbol in symbol_list)
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(_utc_timestamp(start))
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(_utc_timestamp(end))
    return ("" if not clauses else " WHERE " + " AND ".join(clauses)), params


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _table_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"invalid DuckDB table name: {value!r}")
    return value
