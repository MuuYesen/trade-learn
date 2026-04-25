"""Market data provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pandas as pd

from tradelearn.data.bars import Frequency, normalize_bars

TDX_FREQ_CATEGORY: dict[str, int] = {
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 4,
    "1w": 5,
    "1m": 8,
}


class DataProvider(Protocol):
    """Protocol for OHLCV market data providers."""

    def history_ohlc(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        freq: Frequency = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars."""
        ...


class PytdxProvider:
    """pytdx2-backed A-share OHLCV provider."""

    def __init__(
        self,
        *,
        host: str = "119.147.212.81",
        port: int = 7709,
        timeout: float = 5.0,
        count: int = 800,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        """Create a provider with optional pytdx2 client injection."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.count = count
        self._client_factory = client_factory

    def history_ohlc(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        freq: Frequency = "1d",
    ) -> pd.DataFrame:
        """Fetch A-share OHLCV data and return contract-valid Bars."""
        if freq not in TDX_FREQ_CATEGORY:
            raise ValueError(f"Unsupported pytdx2 frequency: {freq}")

        market, code = infer_tdx_market(symbol)
        client = self._make_client()
        connected = client.connect(self.host, self.port, time_out=self.timeout)
        if not connected:
            raise ConnectionError(f"Unable to connect to pytdx2 host {self.host}:{self.port}")

        try:
            rows = client.get_security_bars(
                TDX_FREQ_CATEGORY[freq],
                market,
                code,
                0,
                self.count,
            )
            raw = client.to_df(rows)
        finally:
            client.disconnect()

        raw = _normalize_pytdx_columns(raw, code)
        raw = _filter_dates(raw, start=start, end=end)
        return normalize_bars(
            raw,
            market="CN",
            freq=freq,
            engine="pytdx2",
            source=f"{self.host}:{self.port}",
            adjust="none",
        )

    def _make_client(self) -> object:
        """Create the underlying pytdx2 client."""
        if self._client_factory is not None:
            return self._client_factory()

        from pytdx2.hq import TdxHq_API

        return TdxHq_API()


def infer_tdx_market(symbol: str) -> tuple[int, str]:
    """Infer TDX market id and bare code from an A-share symbol."""
    normalized = symbol.strip().upper()
    if normalized.startswith("SH."):
        return 1, normalized[3:]
    if normalized.startswith("SZ."):
        return 0, normalized[3:]
    if normalized.startswith(("5", "6", "9")):
        return 1, normalized
    return 0, normalized


def _normalize_pytdx_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map pytdx2 row columns to raw Bars input columns."""
    frame = raw.copy()
    if "datetime" in frame.columns:
        frame = frame.rename(columns={"datetime": "timestamp"})
    elif "date" in frame.columns:
        frame = frame.rename(columns={"date": "timestamp"})
    if "vol" in frame.columns:
        frame = frame.rename(columns={"vol": "volume"})
    frame["symbol"] = symbol
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    optional = [column for column in ("amount", "vwap") if column in frame.columns]
    return frame[columns + optional]


def _filter_dates(
    raw: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Filter raw rows by inclusive start/end timestamps."""
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        frame = frame[frame["timestamp"] <= end_ts]
    return frame
