"""Raw market data exploratory analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.data.bars import REQUIRED_COLUMNS


class DataExplorer:
    """Explore raw Bars-like market data before research or backtesting."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Create an explorer from a Bars contract frame or raw DataFrame."""
        self.data = _as_bars_like(data)

    def summary(self) -> dict[str, Any]:
        """Return compact dataset-level diagnostics."""
        frame = self.data
        timestamps = _timestamps(frame)
        symbols = _symbols(frame)
        result: dict[str, Any] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "symbols": int(symbols.nunique()) if symbols is not None else 0,
            "start": timestamps.min() if timestamps is not None and len(timestamps) else pd.NaT,
            "end": timestamps.max() if timestamps is not None and len(timestamps) else pd.NaT,
            "freq": frame.attrs.get("freq") or _infer_frequency(timestamps, symbols),
            "rows_by_symbol": {},
        }
        if symbols is not None:
            counts = pd.Series(symbols, index=frame.index).astype(str).value_counts().sort_index()
            result["rows_by_symbol"] = {str(key): int(value) for key, value in counts.items()}
        return result

    def schema(self) -> pd.DataFrame:
        """Return dtype, missingness, and cardinality by column."""
        frame = self.data
        rows = max(len(frame), 1)
        result = pd.DataFrame(
            {
                "dtype": frame.dtypes.astype(str),
                "non_missing": frame.notna().sum(),
                "missing": frame.isna().sum(),
                "missing_pct": frame.isna().sum() / rows,
                "unique": frame.nunique(dropna=True),
            }
        )
        result.index.name = "column"
        return result

    def missing(self) -> pd.DataFrame:
        """Return columns sorted by missing value count."""
        result = self.schema()[["missing", "missing_pct", "non_missing"]].copy()
        return result.sort_values(["missing", "missing_pct"], ascending=False)

    def describe(self) -> pd.DataFrame:
        """Return numeric descriptive statistics including skew and kurtosis."""
        numeric = self.data.select_dtypes(include="number")
        if numeric.empty:
            return pd.DataFrame()
        described = numeric.describe()
        described.loc["skew"] = numeric.skew(numeric_only=True)
        described.loc["kurt"] = numeric.kurt(numeric_only=True)
        return described

    def ohlcv_quality(self) -> pd.DataFrame:
        """Return market-data-specific quality checks."""
        frame = self.data
        checks = {
            "missing_required_columns": len(
                [column for column in REQUIRED_COLUMNS if column not in frame.columns]
            ),
            "duplicate_index": int(frame.index.duplicated().sum()),
            "high_below_low": _count_mask(frame, frame.get("high") < frame.get("low")),
            "close_outside_range": _count_mask(frame, _close_outside_valid_range(frame)),
            "non_positive_price": _non_positive_prices(frame),
            "negative_volume": _count_mask(frame, frame.get("volume") < 0),
            "unsorted_index": int(not frame.index.is_monotonic_increasing),
        }
        result = pd.DataFrame.from_dict(checks, orient="index", columns=["count"])
        result.index.name = "check"
        return result

    def gaps(self) -> pd.DataFrame:
        """Return observed timestamp gaps by symbol."""
        timestamps = _timestamps(self.data)
        symbols = _symbols(self.data)
        if timestamps is None or symbols is None:
            return pd.DataFrame(columns=["symbol", "start", "end", "gap"])
        frame = pd.DataFrame({"timestamp": timestamps, "symbol": symbols.astype(str)}).sort_values(
            ["symbol", "timestamp"]
        )
        frame["previous"] = frame.groupby("symbol")["timestamp"].shift()
        frame["gap"] = frame["timestamp"] - frame["previous"]
        result = frame.dropna(subset=["previous"])[["symbol", "previous", "timestamp", "gap"]]
        return (
            result.rename(columns={"previous": "start", "timestamp": "end"})
            .reset_index(drop=True)
        )

    def returns(self, column: str = "close") -> pd.Series:
        """Return per-symbol percentage returns."""
        if column not in self.data.columns:
            raise KeyError(f"missing price column: {column}")
        symbols = _symbols(self.data)
        if symbols is None:
            result = self.data[column].pct_change()
        else:
            result = self.data[column].groupby(symbols).pct_change()
        result = result.rename("return")
        return result

    def outliers(self, *, column: str = "close", zscore: float = 4.0) -> pd.DataFrame:
        """Return return outliers by absolute z-score."""
        returns = self.returns(column).replace([np.inf, -np.inf], np.nan)
        std = returns.std(skipna=True)
        if pd.isna(std) or std == 0:
            return pd.DataFrame(columns=["timestamp", "symbol", "return", "zscore"])
        zvalues = (returns - returns.mean(skipna=True)) / std
        mask = zvalues.abs() >= zscore
        timestamps = _timestamps(self.data)
        symbols = _symbols(self.data)
        result = pd.DataFrame(
            {
                "timestamp": timestamps if timestamps is not None else self.data.index,
                "symbol": symbols.astype(str) if symbols is not None else "",
                "return": returns,
                "zscore": zvalues,
            }
        )
        return result.loc[mask.fillna(False)].reset_index(drop=True)

    def correlation(self, column: str = "close") -> pd.DataFrame:
        """Return multi-symbol return correlation."""
        returns = self.returns(column)
        timestamps = _timestamps(self.data)
        symbols = _symbols(self.data)
        if timestamps is None or symbols is None:
            numeric = self.data.select_dtypes(include="number")
            return numeric.corr()
        panel = pd.DataFrame(
            {"timestamp": timestamps, "symbol": symbols.astype(str), "return": returns}
        ).pivot(index="timestamp", columns="symbol", values="return")
        return panel.corr()

    def report(self, path: str | Path) -> Path:
        """Write a standalone HTML data exploration report."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_html_report(self), encoding="utf-8")
        return output

def _as_bars_like(data: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(data).copy()
    if isinstance(frame.index, pd.MultiIndex) and {"timestamp", "symbol"}.issubset(
        set(frame.index.names)
    ):
        return frame.sort_index()
    if {"timestamp", "symbol"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = frame["symbol"].astype(str)
        return frame.set_index(["timestamp", "symbol"]).sort_index()
    return frame


def _timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex | pd.Series | None:
    if isinstance(frame.index, pd.MultiIndex) and "timestamp" in frame.index.names:
        return pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], utc=True)
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    return None


def _symbols(frame: pd.DataFrame) -> pd.Index | pd.Series | None:
    if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        return pd.Index(frame.index.get_level_values("symbol"), name="symbol")
    if "symbol" in frame.columns:
        return frame["symbol"].astype(str)
    return None


def _infer_frequency(
    timestamps: pd.DatetimeIndex | pd.Series | None,
    symbols: pd.Index | pd.Series | None,
) -> str | None:
    if timestamps is None or len(timestamps) < 3:
        return None
    if symbols is not None:
        frame = pd.DataFrame({"timestamp": timestamps, "symbol": symbols.astype(str)})
        first_symbol = str(frame["symbol"].iloc[0])
        values = frame.loc[frame["symbol"] == first_symbol, "timestamp"]
    else:
        values = pd.Series(timestamps)
    inferred = pd.infer_freq(pd.DatetimeIndex(values.sort_values().drop_duplicates()))
    return {"D": "1d", "W": "1w", "h": "1h", "H": "1h"}.get(str(inferred), inferred)


def _count_mask(frame: pd.DataFrame, mask: Any) -> int:
    if mask is None or isinstance(mask, bool):
        return 0
    try:
        return int(pd.Series(mask, index=frame.index).fillna(False).sum())
    except ValueError:
        return 0


def _non_positive_prices(frame: pd.DataFrame) -> int:
    price_columns = [column for column in ("open", "high", "low", "close") if column in frame]
    if not price_columns:
        return 0
    return int((frame[price_columns] <= 0).any(axis=1).sum())


def _close_outside_valid_range(frame: pd.DataFrame) -> pd.Series | bool:
    if not {"high", "low", "close"}.issubset(frame.columns):
        return False
    valid_range = frame["high"] >= frame["low"]
    outside = (frame["close"] < frame["low"]) | (frame["close"] > frame["high"])
    return valid_range & outside


def _html_report(explorer: DataExplorer) -> str:
    summary = explorer.summary()
    body = [
        "<h1>Data Exploration Report</h1>",
        "<h2>Overview</h2>",
        _mapping_table(summary),
        "<h2>Schema</h2>",
        explorer.schema().to_html(),
        "<h2>Missing Values</h2>",
        explorer.missing().to_html(),
        "<h2>Descriptive Statistics</h2>",
        explorer.describe().to_html(),
        "<h2>OHLCV Quality</h2>",
        explorer.ohlcv_quality().to_html(),
        "<h2>Return Correlation</h2>",
        explorer.correlation().to_html(),
    ]
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Data Exploration Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:32px;color:#1f2933}"
        "table{border-collapse:collapse;margin:12px 0 24px;width:100%}"
        "th,td{border:1px solid #d8dee4;padding:6px 8px;text-align:right}"
        "th{background:#f6f8fa}td:first-child,th:first-child{text-align:left}"
        "h1,h2{color:#102a43}"
        "</style></head><body>"
        + "\n".join(body)
        + "</body></html>"
    )


def _mapping_table(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        if isinstance(value, dict):
            value = ", ".join(
                f"{inner_key}: {inner_value}" for inner_key, inner_value in value.items()
            )
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


__all__ = ["DataExplorer"]
