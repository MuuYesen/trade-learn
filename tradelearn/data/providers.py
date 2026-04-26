"""Market data provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pandas as pd

from tradelearn.data.bars import Frequency, normalize_bars

OPEN_TDX_PERIOD: dict[str, str] = {
    "5m": "MIN_5",
    "15m": "MIN_15",
    "30m": "MIN_30",
    "1h": "MIN_60",
    "1d": "DAILY",
    "1w": "WEEKLY",
    "1m": "MONTHLY",
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


class OpenTdxProvider:
    """opentdx-backed A-share OHLCV provider."""

    def __init__(
        self,
        *,
        host: str = "119.147.212.81",
        port: int = 7709,
        timeout: float = 5.0,
        count: int = 800,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        """Create a provider with optional opentdx client injection."""
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
        if freq not in OPEN_TDX_PERIOD:
            raise ValueError(f"Unsupported opentdx frequency: {freq}")

        market, code = infer_tdx_market(symbol)
        client = self._make_client()
        rows = client.stock_kline(
            market=self._market_value(market),
            code=code,
            period=self._period_value(freq),
            start=0,
            count=self.count,
        )
        if not rows:
            raise ConnectionError(f"opentdx returned no rows for {symbol}")
        raw = pd.DataFrame(rows)

        raw = _normalize_opentdx_columns(raw, code)
        raw = _filter_dates(raw, start=start, end=end)
        return normalize_bars(
            raw,
            market="CN",
            freq=freq,
            engine="opentdx",
            source=f"{self.host}:{self.port}",
            adjust="none",
        )

    def _make_client(self) -> object:
        """Create the underlying opentdx client."""
        if self._client_factory is not None:
            return self._client_factory()

        from opentdx.tdxClient import TdxClient

        return TdxClient()

    @staticmethod
    def _market_value(market: int) -> object:
        """Return an opentdx market enum when the library is available."""
        try:
            from opentdx.tdxClient import MARKET
        except ModuleNotFoundError:
            return market
        return MARKET.SH if market == 1 else MARKET.SZ

    @staticmethod
    def _period_value(freq: Frequency) -> object:
        """Return an opentdx period enum when the library is available."""
        try:
            from opentdx.tdxClient import PERIOD
        except ModuleNotFoundError:
            period_values = {
                "MIN_5": 0,
                "MIN_15": 1,
                "MIN_30": 2,
                "MIN_60": 3,
                "DAILY": 4,
                "WEEKLY": 5,
                "MONTHLY": 6,
            }
            return period_values[OPEN_TDX_PERIOD[freq]]
        return getattr(PERIOD, OPEN_TDX_PERIOD[freq])


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


def _normalize_opentdx_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map opentdx row columns to raw Bars input columns."""
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
