"""On-demand queries over CSV/Parquet files, without a duplicate database."""

from __future__ import annotations

import glob
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.core.contracts import validate_bars
from tradelearn.data.duckdb_backend import _filters, _selected_columns
from tradelearn.data.explorer import DataExplorer


class Dataset:
    """A local file set queried by DuckDB with explicit UTC timestamps.

    CSVs must include timestamp (with timezone), symbol and OHLCV columns.
    Parquet uses the same columns. Files are never rewritten or repaired.
    """

    def __init__(self, paths: str | Path | Sequence[str | Path]) -> None:
        patterns = [paths] if isinstance(paths, (str, Path)) else list(paths)
        files = sorted(
            {
                str(Path(path).resolve())
                for pattern in patterns
                for path in glob.glob(str(pattern), recursive=True)
            }
        )
        if not files:
            raise FileNotFoundError(f"no dataset files matched: {patterns}")
        suffixes = {Path(path).suffix.lower() for path in files}
        if not suffixes <= {".csv", ".parquet"} or len(suffixes) != 1:
            raise ValueError("dataset requires homogeneous CSV or Parquet files")
        import duckdb

        self.files = files
        self.conn = duckdb.connect(":memory:")
        self.kind = "read_csv_auto" if suffixes == {".csv"} else "read_parquet"

    def read(
        self,
        *,
        symbols: Any = None,
        start: Any = None,
        end: Any = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read inclusive UTC boundaries and requested fields with SQL pushdown.

        Full OHLCV is validated before projection. Filtering executes at source;
        no cache is silently reused after a file changes.
        """
        sql, params, selected = self._query(symbols, start, end, columns)
        frame = self.conn.execute(sql, params).fetchdf()
        return self._result(frame, selected, symbols, start, end)

    def _query(self, symbols, start, end, columns):
        selected = _selected_columns(columns)
        where, params = _filters(symbols=symbols, start=start, end=end)
        types = None
        for path in self.files:
            schema = self.conn.execute(f"DESCRIBE SELECT * FROM {self.kind}(?)", [path]).fetchdf()
            current = dict(zip(schema.column_name, schema.column_type, strict=True))
            if types is not None and current != types:
                raise ValueError("dataset schema differs between files")
            types = current
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume", *selected}
        missing = required - set(types)
        if missing:
            raise ValueError(f"dataset missing columns: {sorted(missing)}")
        timestamp_type = types["timestamp"]
        if timestamp_type not in {"TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ_NS", "TIMESTAMPTZ"}:
            raise ValueError("dataset timestamp must carry an explicit timezone")
        # DuckDB 2 preserves nanosecond Parquet timezone types; cast filter
        # parameters to source precision instead of truncating the source.
        for operator in (">=", "<="):
            where = where.replace(
                f"timestamp {operator} ?", f"timestamp {operator} CAST(? AS {timestamp_type})"
            )
        queried = list(dict.fromkeys(["open", "high", "low", "close", "volume", *selected]))
        projection = ", ".join(["timestamp", "symbol", *queried])
        sql = (
            f"SELECT {projection} FROM {self.kind}(?, union_by_name=false)"
            f"{where} ORDER BY timestamp, symbol"
        )
        return sql, [self.files, *params], selected

    def _result(self, frame, selected, symbols, start, end):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if frame.timestamp.isna().any():
            raise ValueError("dataset timestamp must be present")
        if frame.symbol.isna().any() or (frame.symbol.astype(str).str.len() == 0).any():
            raise ValueError("dataset symbol must be present")
        frame["symbol"] = frame.symbol.astype(str)
        bars = frame.set_index(["timestamp", "symbol"])
        validate_bars(bars)
        result = bars.loc[:, selected].copy()
        result.attrs.update(
            engine="duckdb",
            source="files",
            source_files=self.files.copy(),
            query={"symbols": symbols, "start": str(start), "end": str(end), "columns": selected},
            source_versions=[
                {
                    "path": path,
                    "size": Path(path).stat().st_size,
                    "mtime_ns": Path(path).stat().st_mtime_ns,
                }
                for path in self.files
            ],
        )
        return result

    def iter_batches(
        self,
        *,
        batch_size: int = 65536,
        symbols: Any = None,
        start: Any = None,
        end: Any = None,
        columns: Sequence[str] | None = None,
    ):
        """Stream ordered validated frames; reject duplicates across batches.

        Consume/close the iterator before another query on this connection.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sql, params, selected = self._query(symbols, start, end, columns)
        reader = self.conn.sql(sql, params=params).to_arrow_reader(batch_size)
        previous = None
        try:
            for batch in reader:
                result = self._result(batch.to_pandas(), selected, symbols, start, end)
                if len(result):
                    if previous == result.index[0]:
                        raise ValueError("duplicate timestamp/symbol across batches")
                    previous = result.index[-1]
                yield result
        finally:
            reader.close()

    def profile(self, **query: Any) -> DataExplorer:
        """Return the existing explorer over the explicitly selected data."""
        return DataExplorer(self.read(**query))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Dataset:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_dataset(paths: str | Path | Sequence[str | Path]) -> Dataset:
    """Open explicit local CSV/Parquet paths or glob patterns."""
    return Dataset(paths)


__all__ = ["Dataset", "open_dataset"]
