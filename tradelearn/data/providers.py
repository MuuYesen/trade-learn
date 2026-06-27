"""Market data provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import logging
import os
import sys
import time
from contextlib import contextmanager

import pandas as pd
from tradelearn.utils.console import smart_tqdm as tqdm

from tradelearn.core import get_logger
from tradelearn.data.bars import Frequency, normalize_bars

LOGGER = get_logger("data.providers")

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


@dataclass(frozen=True)
class TdxSymbol:
    """Resolved TDX request symbol and canonical output symbol."""

    exchange: str
    market: int
    code: str
    canonical: str


class DataProvider(Protocol):
    """Protocol for OHLCV market data providers."""

    def history_ohlc(
        self,
        symbol: str | list[str] | tuple[str, ...],
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
        symbol: str | list[str] | tuple[str, ...],
        *,
        start: str | None = None,
        end: str | None = None,
        freq: Frequency = "1d",
    ) -> pd.DataFrame:
        """Fetch A-share OHLCV data and return contract-valid Bars."""
        if freq not in TDX_PERIOD:
            raise ValueError(f"Unsupported TDX frequency: {freq}")
        if _is_symbol_collection(symbol):
            LOGGER.info(
                "TDX history_ohlc started symbols=%s start=%s end=%s freq=%s",
                len(symbol),
                start,
                end,
                freq,
            )
            results = []
            for item in tqdm(symbol, desc="TdxProvider.history_ohlc", leave=True, unit="symbol"):
                results.append(self.history_ohlc(item, start=start, end=end, freq=freq))
            return _combine_bars(results)

        tdx_symbol = resolve_tdx_symbol(symbol)
        LOGGER.info(
            "TDX history_ohlc started symbol=%s request=%s:%s start=%s end=%s freq=%s",
            symbol,
            tdx_symbol.exchange,
            tdx_symbol.code,
            start,
            end,
            freq,
        )
        client = self._make_client()
        self._prepare_client(client)
        rows = client.stock_kline(
            market=self._market_value(tdx_symbol.market),
            code=tdx_symbol.code,
            period=self._period_value(freq),
            start=0,
            count=self.count,
        )
        if not rows:
            raise ConnectionError(f"TDX returned no rows for {symbol}")
        raw = pd.DataFrame(rows)

        raw = _normalize_tdx_columns(raw, tdx_symbol.canonical)
        raw = _filter_dates(raw, start=start, end=end)
        bars = normalize_bars(
            raw,
            market="CN",
            freq=freq,
            engine="tdx",
            source=self._source_label(client),
            adjust="none",
        )
        LOGGER.info(
            "TDX history_ohlc finished symbol=%s rows=%s columns=%s index=%s",
            tdx_symbol.canonical,
            len(bars),
            len(bars.columns),
            type(bars.index).__name__,
        )
        return bars

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
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.username = username
        self.password = password
        self.n_bars = int(n_bars)
        self._client_factory = client_factory
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        # Quiet the verbose tvDatafeed logger
        logging.getLogger("tvDatafeed").setLevel(logging.ERROR)

    @contextmanager
    def _suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def history_ohlc(
        self,
        symbol: str | list[str] | tuple[str, ...],
        *,
        start: str | None = None,
        end: str | None = None,
        freq: Frequency = "1d",
        exchange: str | None = None,
    ) -> pd.DataFrame:
        """Fetch TradingView OHLCV data and return contract-valid Bars."""
        if freq not in TRADINGVIEW_INTERVAL:
            raise ValueError(f"Unsupported TradingView frequency: {freq}")
        if _is_symbol_collection(symbol):
            LOGGER.info(
                "TradingView history_ohlc started symbols=%s start=%s end=%s freq=%s",
                len(symbol),
                start,
                end,
                freq,
            )
            results = []
            for item in tqdm(symbol, desc="TradingViewProvider.history_ohlc", leave=True, unit="symbol"):
                results.append(self.history_ohlc(
                    item,
                    start=start,
                    end=end,
                    freq=freq,
                    exchange=exchange,
                ))
            return _combine_bars(results)

        exchange_name, tv_symbol = _split_tradingview_symbol(symbol, exchange=exchange)
        LOGGER.info(
            "TradingView history_ohlc started symbol=%s request=%s:%s start=%s end=%s freq=%s",
            symbol,
            exchange_name,
            tv_symbol,
            start,
            end,
            freq,
        )
        rows = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            client = self._make_client()
            try:
                rows = client.get_hist(
                    symbol=tv_symbol,
                    exchange=exchange_name,
                    interval=self._interval_value(freq),
                    n_bars=self.n_bars,
                )
            except Exception as exc:
                last_error = exc
                rows = None
            if rows is not None and len(rows) > 0:
                break
            if attempt < self.max_retries:
                LOGGER.warning(
                    "TradingView history_ohlc retrying symbol=%s attempt=%s/%s",
                    symbol,
                    attempt + 1,
                    self.max_retries,
                )
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay)
        if rows is None or len(rows) == 0:
            if last_error is not None:
                raise ConnectionError(f"TradingView returned no rows for {symbol}") from last_error
            raise ConnectionError(f"TradingView returned no rows for {symbol}")
        raw = _normalize_tradingview_columns(pd.DataFrame(rows), exchange_name, tv_symbol)
        raw = _filter_dates(raw, start=start, end=end)
        bars = normalize_bars(
            raw,
            market="GLOBAL",
            freq=freq,
            engine="tradingview",
            source="tvdatafeed",
            adjust="none",
        )
        LOGGER.info(
            "TradingView history_ohlc finished symbol=%s rows=%s columns=%s index=%s",
            f"{exchange_name}:{tv_symbol}",
            len(bars),
            len(bars.columns),
            type(bars.index).__name__,
        )
        return bars

    def _make_client(self) -> object:
        if self._client_factory is not None:
            return self._client_factory()

        from tvDatafeed import TvDatafeed
        with self._suppress_stdout():
            return TvDatafeed(username=self.username, password=self.password)

    @staticmethod
    def _interval_value(freq: Frequency) -> object:
        from tvDatafeed import Interval
        return getattr(Interval, TRADINGVIEW_INTERVAL[freq])


def infer_tdx_market(symbol: str) -> tuple[int, str]:
    """Infer TDX market id and bare code from an A-share symbol."""
    tdx_symbol = resolve_tdx_symbol(symbol)
    return tdx_symbol.market, tdx_symbol.code


def resolve_tdx_symbol(symbol: str) -> TdxSymbol:
    """Resolve a user TDX symbol into request fields and canonical symbol."""
    normalized = symbol.strip().upper()
    exchange, code = _split_tdx_symbol(normalized)
    if exchange is None:
        exchange = _infer_tdx_exchange(code)

    market = 1 if exchange == "SH" else 0
    return TdxSymbol(
        exchange=exchange,
        market=market,
        code=code,
        canonical=f"{exchange}:{code}",
    )


def _split_tdx_symbol(symbol: str) -> tuple[str | None, str]:
    """Split explicit SH/SZ prefixes from a TDX symbol."""
    for separator in (":", "."):
        if separator in symbol:
            exchange, code = symbol.split(separator, 1)
            exchange = exchange.strip().upper()
            code = code.strip().upper()
            if exchange not in {"SH", "SZ"}:
                raise ValueError("TDX symbols only support SH or SZ exchange prefixes")
            return exchange, _normalize_tdx_code(code)
    return None, _normalize_tdx_code(symbol)


def _normalize_tdx_code(code: str) -> str:
    """Validate and normalize a six-digit TDX code."""
    if len(code) != 6 or not code.isdigit():
        raise ValueError("TDX symbols must use a six-digit code")
    return code


def _infer_tdx_exchange(code: str) -> str:
    """Infer TDX exchange from conservative exchange code ranges."""
    if code.startswith(("600", "601", "603", "605", "688", "900")):
        return "SH"
    if code.startswith(
        (
            "500",
            "510",
            "511",
            "512",
            "513",
            "515",
            "516",
            "518",
            "588",
        )
    ):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "SZ"
    if code.startswith(
        (
            "159",
            "160",
            "161",
            "162",
            "163",
            "164",
            "165",
            "166",
            "167",
            "168",
            "169",
        )
    ):
        return "SZ"
    raise ValueError(
        f"Ambiguous TDX symbol: {code}; use an explicit exchange prefix like SH:{code} or SZ:{code}"
    )


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


def _is_symbol_collection(symbol: object) -> bool:
    return isinstance(symbol, list | tuple)


def _combine_bars(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("symbol list must not be empty")
    combined = pd.concat(frames).sort_index()
    combined.attrs.update(frames[0].attrs)
    return combined


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
