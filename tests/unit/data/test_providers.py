"""Tests for market data providers."""

from __future__ import annotations

import pytest

from tradelearn.data.providers import (
    DataProvider,
    TdxProvider,
    TdxSymbol,
    TradingViewProvider,
    infer_tdx_market,
    resolve_tdx_symbol,
)


def enum_value(value: object) -> object:
    """Return enum value for assertions when the TDX dependency is installed."""
    return getattr(value, "value", value)


def enum_name(value: object) -> object:
    """Return enum name for assertions when optional providers are installed."""
    return getattr(value, "name", value)


class FakeTdxClient:
    """Small TDX-compatible fake client."""

    def __init__(self) -> None:
        """Initialize fake call state."""
        self.calls: list[tuple[object, str, object, int, int]] = []

    def stock_kline(
        self,
        *,
        market: object,
        code: str,
        period: object,
        start: int,
        count: int,
    ) -> list[dict[str, object]]:
        """Return TDX-style kline dictionaries."""
        self.calls.append((market, code, period, start, count))
        return [
            {
                "datetime": "2024-01-01 15:00",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "vol": 100.0,
                "amount": 1000.0,
            },
            {
                "datetime": "2024-01-02 15:00",
                "open": 11.0,
                "high": 12.0,
                "low": 10.0,
                "close": 11.5,
                "vol": 120.0,
                "amount": 1320.0,
            },
        ]


class EmptyTdxClient:
    """Fake client that simulates unavailable TDX data."""

    def stock_kline(self, **_: object) -> list[dict[str, object]]:
        """Return no rows."""
        return []


class FakeQuotationClient:
    """Small fake quotation connection object."""

    def __init__(self, *, connected: bool) -> None:
        """Initialize fake connection state."""
        self.connected = connected
        self.ip = None
        self.port = None
        self.calls: list[tuple[str | None, int, float]] = []

    def connect(
        self,
        ip: str | None = None,
        port: int = 7709,
        time_out: float = 5.0,
    ) -> object:
        """Record explicit connection parameters."""
        self.calls.append((ip, port, time_out))
        self.ip = ip
        self.port = port
        return self

    def login(self) -> object:
        """Return the fake connection object."""
        return self


class ConnectableTdxClient(FakeTdxClient):
    """Fake client with a TDX-style quotation connection."""

    def __init__(self, *, connected: bool = True) -> None:
        """Initialize fake data and connection state."""
        super().__init__()
        self.quotation_client = FakeQuotationClient(connected=connected)


class FakeTvDatafeedClient:
    """Small tvdatafeed-compatible fake client."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object, int]] = []

    def get_hist(
        self,
        *,
        symbol: str,
        exchange: str,
        interval: object,
        n_bars: int,
    ):
        self.calls.append((symbol, exchange, interval, n_bars))
        import pandas as pd

        return pd.DataFrame(
            {
                "symbol": ["NASDAQ:AAPL", "NASDAQ:AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )


class EmptyTvDatafeedClient:
    def get_hist(self, **_: object):
        return None


def test_tdx_provider_fetches_and_normalizes_daily_bars() -> None:
    """TdxProvider converts TDX rows into Bars."""
    client = FakeTdxClient()
    provider: DataProvider = TdxProvider(client_factory=lambda: client)

    bars = provider.history_ohlc("600519", start="2024-01-02", end="2024-01-02", freq="1d")

    calls = [
        (enum_value(market), code, enum_value(period), start, count)
        for market, code, period, start, count in client.calls
    ]
    assert calls == [
        (1, "600519", 4, 0, 800)
    ]
    assert bars.index.names == ["timestamp", "symbol"]
    assert list(bars.index.get_level_values("symbol")) == ["SH:600519"]
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    assert bars.iloc[0]["volume"] == 120.0
    assert bars.attrs["market"] == "CN"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "tdx"


def test_tdx_provider_fetches_symbol_list_as_combined_bars() -> None:
    """TdxProvider accepts a list of symbols and combines normalized Bars."""
    client = FakeTdxClient()
    provider = TdxProvider(client_factory=lambda: client)

    bars = provider.history_ohlc(["600519", "000001"], freq="1d")

    calls = [
        (enum_value(market), code, enum_value(period), start, count)
        for market, code, period, start, count in client.calls
    ]
    assert calls == [
        (1, "600519", 4, 0, 800),
        (0, "000001", 4, 0, 800),
    ]
    assert bars.index.names == ["timestamp", "symbol"]
    assert sorted(set(bars.index.get_level_values("symbol"))) == [
        "SH:600519",
        "SZ:000001",
    ]
    assert bars.attrs["market"] == "CN"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "tdx"


def test_tdx_provider_maps_symbol_market_and_frequency() -> None:
    """Provider maps common A-share symbols and supported frequencies."""
    client = FakeTdxClient()
    provider = TdxProvider(client_factory=lambda: client)

    provider.history_ohlc("000001", freq="1h")

    market, code, period, start, count = client.calls[0]
    assert (enum_value(market), code, enum_value(period), start, count) == (
        0,
        "000001",
        3,
        0,
        800,
    )


def test_tdx_provider_connects_with_configured_host() -> None:
    """Provider uses configured host/port before fetching live rows."""
    client = ConnectableTdxClient()
    provider = TdxProvider(
        host="1.2.3.4",
        port=7710,
        timeout=2.5,
        client_factory=lambda: client,
    )

    provider.history_ohlc("000001")

    assert client.quotation_client.calls == [("1.2.3.4", 7710, 2.5)]


def test_tdx_provider_reports_unestablished_connection() -> None:
    """Provider reports host diagnostics when TDX never connects."""
    client = ConnectableTdxClient(connected=False)
    provider = TdxProvider(
        host="1.2.3.4",
        port=7710,
        timeout=2.5,
        client_factory=lambda: client,
    )

    with pytest.raises(ConnectionError, match="connection not established.*1.2.3.4:7710"):
        provider.history_ohlc("000001")


def test_tdx_provider_uses_official_auto_server_by_default() -> None:
    """Provider follows TDX's default auto server selection."""
    client = ConnectableTdxClient()
    provider = TdxProvider(client_factory=lambda: client)

    provider.history_ohlc("000001")

    assert client.quotation_client.calls == [(None, 7709, 5.0)]


def test_tdx_provider_rejects_unsupported_frequency() -> None:
    """Unsupported frequencies fail before network access."""
    provider = TdxProvider(client_factory=FakeTdxClient)

    with pytest.raises(ValueError, match="Unsupported"):
        provider.history_ohlc("000001", freq="4h")


def test_tdx_provider_reports_empty_live_response() -> None:
    """Empty live responses fail with a concrete provider message."""
    provider = TdxProvider(client_factory=EmptyTdxClient)

    with pytest.raises(ConnectionError, match="returned no rows"):
        provider.history_ohlc("000001")


def test_resolve_tdx_symbol_supports_explicit_exchange_prefix() -> None:
    """Symbol prefixes can explicitly select the TDX market."""
    assert resolve_tdx_symbol("SH.600519") == TdxSymbol(
        exchange="SH",
        market=1,
        code="600519",
        canonical="SH:600519",
    )
    assert resolve_tdx_symbol("SZ:000001") == TdxSymbol(
        exchange="SZ",
        market=0,
        code="000001",
        canonical="SZ:000001",
    )


def test_resolve_tdx_symbol_maps_supported_bare_code_ranges() -> None:
    """Known exchange code ranges map to canonical TDX symbols."""
    assert resolve_tdx_symbol("600519").canonical == "SH:600519"
    assert resolve_tdx_symbol("688001").canonical == "SH:688001"
    assert resolve_tdx_symbol("510300").canonical == "SH:510300"
    assert resolve_tdx_symbol("588000").canonical == "SH:588000"
    assert resolve_tdx_symbol("900901").canonical == "SH:900901"
    assert resolve_tdx_symbol("000001").canonical == "SZ:000001"
    assert resolve_tdx_symbol("300750").canonical == "SZ:300750"
    assert resolve_tdx_symbol("159919").canonical == "SZ:159919"
    assert resolve_tdx_symbol("200012").canonical == "SZ:200012"


def test_resolve_tdx_symbol_rejects_ambiguous_bare_codes() -> None:
    """Ambiguous bare codes fail instead of defaulting to Shenzhen."""
    with pytest.raises(ValueError, match="Ambiguous TDX symbol"):
        resolve_tdx_symbol("123456")


def test_infer_tdx_market_preserves_legacy_tuple_api() -> None:
    """Legacy helper returns the request market and code."""
    assert infer_tdx_market("SH.600519") == (1, "600519")
    assert infer_tdx_market("SZ:000001") == (0, "000001")


def test_tradingview_provider_fetches_and_normalizes_bars() -> None:
    """TradingViewProvider converts tvdatafeed rows into Bars."""

    client = FakeTvDatafeedClient()
    provider: DataProvider = TradingViewProvider(client_factory=lambda: client, n_bars=100)

    bars = provider.history_ohlc(
        "NASDAQ:AAPL",
        start="2024-01-02",
        end="2024-01-02",
        freq="1d",
    )

    calls = [
        (symbol, exchange, enum_name(interval), n_bars)
        for symbol, exchange, interval, n_bars in client.calls
    ]
    assert calls == [("AAPL", "NASDAQ", "in_daily", 100)]
    assert bars.index.names == ["timestamp", "symbol"]
    assert list(bars.index.get_level_values("symbol")) == ["NASDAQ:AAPL"]
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    assert bars.iloc[0]["close"] == 102.0
    assert bars.attrs["market"] == "GLOBAL"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "tradingview"


def test_tradingview_provider_fetches_symbol_list_as_combined_bars() -> None:
    """TradingViewProvider accepts a list of namespaced symbols."""
    client = FakeTvDatafeedClient()
    provider = TradingViewProvider(client_factory=lambda: client, n_bars=50)

    bars = provider.history_ohlc(["NASDAQ:AAPL", "NYSE:IBM"], freq="1d")

    calls = [
        (symbol, exchange, enum_name(interval), n_bars)
        for symbol, exchange, interval, n_bars in client.calls
    ]
    assert calls == [
        ("AAPL", "NASDAQ", "in_daily", 50),
        ("IBM", "NYSE", "in_daily", 50),
    ]
    assert bars.index.names == ["timestamp", "symbol"]
    assert sorted(set(bars.index.get_level_values("symbol"))) == [
        "NASDAQ:AAPL",
        "NYSE:IBM",
    ]
    assert bars.attrs["market"] == "GLOBAL"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "tradingview"


def test_tradingview_provider_accepts_exchange_keyword() -> None:
    """Symbols can be split into symbol plus exchange arguments."""

    client = FakeTvDatafeedClient()
    provider = TradingViewProvider(client_factory=lambda: client)

    provider.history_ohlc("AAPL", exchange="NASDAQ", freq="1h")

    calls = [
        (symbol, exchange, enum_name(interval), n_bars)
        for symbol, exchange, interval, n_bars in client.calls
    ]
    assert calls == [("AAPL", "NASDAQ", "in_1_hour", 5000)]


def test_tradingview_provider_accepts_market_specific_symbols() -> None:
    """TradingView preserves explicit exchange namespaces for non-stock symbols."""
    client = FakeTvDatafeedClient()
    provider = TradingViewProvider(client_factory=lambda: client)

    provider.history_ohlc("BINANCE:BTCUSDT", freq="1d")
    provider.history_ohlc("CME_MINI:ES1!", freq="1d")
    provider.history_ohlc("FX:EURUSD", freq="1d")

    assert [call[:2] for call in client.calls] == [
        ("BTCUSDT", "BINANCE"),
        ("ES1!", "CME_MINI"),
        ("EURUSD", "FX"),
    ]


def test_tradingview_provider_rejects_unsupported_frequency() -> None:
    provider = TradingViewProvider(client_factory=FakeTvDatafeedClient)

    with pytest.raises(ValueError, match="Unsupported"):
        provider.history_ohlc("NASDAQ:AAPL", freq="10m")


def test_tradingview_provider_reports_empty_response() -> None:
    provider = TradingViewProvider(client_factory=EmptyTvDatafeedClient)

    with pytest.raises(ConnectionError, match="returned no rows"):
        provider.history_ohlc("NASDAQ:AAPL")
