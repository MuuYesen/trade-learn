"""Market data provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from tradelearn.data.bars import Frequency, normalize_bars

TDX_PERIOD: dict[str, str] = {
    "5m": "MIN_5",
    "15m": "MIN_15",
    "30m": "MIN_30",
    "1h": "MIN_60",
    "1d": "DAILY",
    "1w": "WEEKLY",
    "1m": "MONTHLY",
}

TRADINGVIEW_INTERVAL: dict[str, str] = {
    "1m": "in_1_minute",
    "3m": "in_3_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "45m": "in_45_minute",
    "1h": "in_1_hour",
    "2h": "in_2_hour",
    "3h": "in_3_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1w": "in_weekly",
    "1M": "in_monthly",
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


class TdxProvider:
    """TDX-backed A-share OHLCV provider."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int = 7709,
        timeout: float = 5.0,
        count: int = 800,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        """Create a provider with optional TDX client injection."""
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
        if freq not in TDX_PERIOD:
            raise ValueError(f"Unsupported TDX frequency: {freq}")

        market, code = infer_tdx_market(symbol)
        client = self._make_client()
        self._prepare_client(client)
        rows = client.stock_kline(
            market=self._market_value(market),
            code=code,
            period=self._period_value(freq),
            start=0,
            count=self.count,
        )
        if not rows:
            raise ConnectionError(f"TDX returned no rows for {symbol}")
        raw = pd.DataFrame(rows)

        raw = _normalize_tdx_columns(raw, code)
        raw = _filter_dates(raw, start=start, end=end)
        return normalize_bars(
            raw,
            market="CN",
            freq=freq,
            engine="tdx",
            source=self._source_label(client),
            adjust="none",
        )

    def _make_client(self) -> object:
        """Create the underlying TDX client."""
        if self._client_factory is not None:
            return self._client_factory()

        from opentdx.tdxClient import TdxClient

        return TdxClient()

    def _prepare_client(self, client: object) -> None:
        """Connect TDX quotation client following the official default flow."""
        quotation_client = getattr(client, "quotation_client", None)
        connect = getattr(quotation_client, "connect", None)
        if quotation_client is None or connect is None:
            return

        connection = (
            connect(time_out=self.timeout)
            if self.host is None
            else connect(ip=self.host, port=self.port, time_out=self.timeout)
        )
        if connection is None:
            self._raise_connection_error(quotation_client)
        login = getattr(connection, "login", None)
        login_ok = login() if callable(login) else True
        if not getattr(quotation_client, "connected", False):
            self._raise_connection_error(quotation_client)
        if login_ok is False:
            self._raise_connection_error(quotation_client, reason="login failed")

    def _raise_connection_error(
        self,
        quotation_client: object,
        *,
        reason: str = "connection not established",
    ) -> None:
        """Raise a provider error with endpoint diagnostics."""
        selected_ip = getattr(quotation_client, "ip", None)
        selected_port = getattr(quotation_client, "port", None)
        configured = f"{self.host}:{self.port}" if self.host is not None else "auto"
        selected = (
            f"; selected={selected_ip}:{selected_port}"
            if selected_ip is not None or selected_port is not None
            else ""
        )
        raise ConnectionError(f"TDX {reason} for {configured}{selected}")

    @staticmethod
    def _source_label(client: object) -> str:
        """Return the connected TDX endpoint label when available."""
        quotation_client: Any = getattr(client, "quotation_client", None)
        ip = getattr(quotation_client, "ip", None)
        port = getattr(quotation_client, "port", None)
        return f"{ip}:{port}" if ip is not None and port is not None else "tdx:auto"

    @staticmethod
    def _market_value(market: int) -> object:
        """Return a TDX market enum when the library is available."""
        try:
            from opentdx.tdxClient import MARKET
        except ModuleNotFoundError:
            return market
        return MARKET.SH if market == 1 else MARKET.SZ

    @staticmethod
    def _period_value(freq: Frequency) -> object:
        """Return a TDX period enum when the library is available."""
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
            return period_values[TDX_PERIOD[freq]]
        return getattr(PERIOD, TDX_PERIOD[freq])


class TradingViewProvider:
    """tvdatafeed-backed OHLCV provider."""

    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
        n_bars: int = 5000,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        self.username = username
        self.password = password
        self.n_bars = int(n_bars)
        self._client_factory = client_factory

    def history_ohlc(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        freq: Frequency = "1d",
        exchange: str | None = None,
    ) -> pd.DataFrame:
        """Fetch TradingView OHLCV data and return contract-valid Bars."""
        if freq not in TRADINGVIEW_INTERVAL:
            raise ValueError(f"Unsupported TradingView frequency: {freq}")

        exchange_name, tv_symbol = _split_tradingview_symbol(symbol, exchange=exchange)
        client = self._make_client()
        rows = client.get_hist(
            symbol=tv_symbol,
            exchange=exchange_name,
            interval=self._interval_value(freq),
            n_bars=self.n_bars,
        )
        if rows is None or len(rows) == 0:
            raise ConnectionError(f"TradingView returned no rows for {symbol}")
        raw = _normalize_tradingview_columns(pd.DataFrame(rows), exchange_name, tv_symbol)
        raw = _filter_dates(raw, start=start, end=end)
        return normalize_bars(
            raw,
            market="GLOBAL",
            freq=freq,
            engine="tradingview",
            source="tvdatafeed",
            adjust="none",
        )

    def _make_client(self) -> object:
        if self._client_factory is not None:
            return self._client_factory()
        from tvDatafeed import TvDatafeed

        return TvDatafeed(username=self.username, password=self.password)

    @staticmethod
    def _interval_value(freq: Frequency) -> object:
        interval_name = TRADINGVIEW_INTERVAL[freq]
        try:
            from tvDatafeed import Interval
        except ModuleNotFoundError:
            return interval_name
        return getattr(Interval, interval_name)


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


def _split_tradingview_symbol(symbol: str, *, exchange: str | None = None) -> tuple[str, str]:
    normalized = symbol.strip()
    if exchange is not None:
        return exchange.strip().upper(), normalized.upper()
    if ":" in normalized:
        exchange_name, tv_symbol = normalized.split(":", 1)
        return exchange_name.strip().upper(), tv_symbol.strip().upper()
    raise ValueError("TradingView symbols require an exchange, e.g. 'NASDAQ:AAPL'")


def _normalize_tdx_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map TDX row columns to raw Bars input columns."""
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


def _normalize_tradingview_columns(
    raw: pd.DataFrame,
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    """Map tvdatafeed rows to raw Bars input columns."""
    frame = raw.copy()
    if "datetime" not in frame.columns:
        frame = frame.reset_index(names="datetime")
    frame = frame.rename(columns={"datetime": "timestamp"})
    frame["symbol"] = f"{exchange}:{symbol}"
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"TradingView data missing column(s): {missing}")
    return frame[columns]


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
