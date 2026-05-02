"""Bars contract normalization."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd

from tradelearn.core import ContractError, validate_bars

Market = Literal["CN", "US", "HK", "CRYPTO"]
Frequency = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
Adjustment = Literal["pre", "post", "none"]

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


class MarketPanel:
    """Wide-table view over MultiIndex(timestamp, symbol) market bars."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = validate_bars(pd.DataFrame(bars).copy())

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name in self.bars.columns:
            return self.field(name)
        raise AttributeError(name)

    def field(self, name: str) -> pd.DataFrame:
        """Return one bars column as timestamp x symbol wide data."""

        if name not in self.bars.columns:
            raise KeyError(f"bars column not found: {name}")
        wide = self.bars[name].unstack("symbol")
        wide.index.name = "timestamp"
        return wide.sort_index()

    def to_dataset(self, features: dict[str, object]) -> pd.DataFrame:
        """Build a MultiIndex(timestamp, symbol) research dataset.

        Feature callables receive this ``MarketPanel`` and must return either a
        timestamp x symbol DataFrame or a compatible Series.
        """

        columns: dict[str, pd.Series] = {}
        for name, spec in features.items():
            value = spec(self) if callable(spec) else spec
            columns[str(name)] = _feature_series(value, name=str(name))
        dataset = pd.concat(columns, axis=1)
        dataset.index.names = ["timestamp", "symbol"]
        return dataset.sort_index()


def normalize_bars(
    raw: pd.DataFrame,
    *,
    market: Market,
    freq: Frequency,
    engine: str,
    source: str,
    adjust: Adjustment = "pre",
) -> pd.DataFrame:
    """Normalize raw OHLCV rows into the Bars contract.

    Parameters
    ----------
    raw : pandas.DataFrame
        DataFrame with timestamp, symbol, OHLCV columns.
    market : {"CN", "US", "HK", "CRYPTO"}
        Market identifier.
    freq : {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        Bar frequency.
    engine : str
        Data provider identifier.
    source : str
        Source URL or provider-specific marker.
    adjust : {"pre", "post", "none"}, default "pre"
        Adjustment mode. When ``pre`` and ``adj_factor`` exists, OHLC prices are
        scaled by ``adj_factor``.

    Returns
    -------
    pandas.DataFrame
        Contract-valid Bars DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({
    ...     "timestamp": ["2024-01-01"], "symbol": ["AAA"],
    ...     "open": [1], "high": [2], "low": [1], "close": [2], "volume": [100],
    ... })
    >>> normalize_bars(raw, market="US", freq="1d", engine="test", source="fixture").attrs["adjust"]
    'pre'
    """
    bars = raw.copy()
    _require_columns(bars, ("timestamp", "symbol", *REQUIRED_COLUMNS))

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["symbol"] = bars["symbol"].astype(str)
    for column in REQUIRED_COLUMNS + tuple(_optional_numeric_columns(bars)):
        bars[column] = bars[column].astype("float64")

    if adjust == "pre" and "adj_factor" in bars.columns:
        for column in PRICE_COLUMNS:
            bars[column] = bars[column] * bars["adj_factor"]

    bars = bars.set_index(["timestamp", "symbol"]).sort_index()
    bars.index.names = ["timestamp", "symbol"]
    if bars.index.has_duplicates:
        raise ContractError("Bars index contains duplicate timestamp/symbol rows")

    bars.attrs.update(
        {
            "market": market,
            "freq": freq,
            "adjust": adjust,
            "engine": engine,
            "source": source,
        }
    )
    return validate_bars(bars)


def bars_fingerprint(bars: pd.DataFrame) -> str:
    """Return a deterministic fingerprint for Bars values and metadata."""
    validate_bars(bars)
    payload = {
        "attrs": dict(sorted(bars.attrs.items())),
        "index": [
            [str(timestamp), symbol]
            for timestamp, symbol in bars.index.to_flat_index()
        ],
        "columns": list(bars.columns),
        "values": bars.astype("float64", errors="ignore").to_json(
            orient="split",
            date_format="iso",
            double_precision=12,
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise if required raw columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"Bars missing required columns: {missing}")


def _optional_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return optional numeric Bars columns present in a frame."""
    return [column for column in ("vwap", "amount", "adj_factor") if column in frame.columns]


def _feature_series(value: object, *, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        series = value.stack(future_stack=True)
    else:
        series = pd.Series(value)
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ContractError(
            f"MarketPanel feature {name!r} must produce MultiIndex(timestamp, symbol) values"
        )
    series = series.copy()
    series.index.names = ["timestamp", "symbol"]
    series.name = name
    return series
