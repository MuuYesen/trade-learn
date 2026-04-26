"""Tests for market data providers."""

from __future__ import annotations

import pytest

from tradelearn.data.providers import DataProvider, OpenTdxProvider, infer_tdx_market


def enum_value(value: object) -> object:
    """Return enum value for assertions when opentdx is installed."""
    return getattr(value, "value", value)


class FakeTdxClient:
    """Small opentdx-compatible fake client."""

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
        """Return opentdx-style kline dictionaries."""
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
    """Fake client that simulates unavailable opentdx data."""

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
    """Fake client with an opentdx-style quotation connection."""

    def __init__(self, *, connected: bool = True) -> None:
        """Initialize fake data and connection state."""
        super().__init__()
        self.quotation_client = FakeQuotationClient(connected=connected)


def test_opentdx_provider_fetches_and_normalizes_daily_bars() -> None:
    """OpenTdxProvider converts opentdx rows into Bars."""
    client = FakeTdxClient()
    provider: DataProvider = OpenTdxProvider(client_factory=lambda: client)

    bars = provider.history_ohlc("600519", start="2024-01-02", end="2024-01-02", freq="1d")

    calls = [
        (enum_value(market), code, enum_value(period), start, count)
        for market, code, period, start, count in client.calls
    ]
    assert calls == [
        (1, "600519", 4, 0, 800)
    ]
    assert bars.index.names == ["timestamp", "symbol"]
    assert list(bars.index.get_level_values("symbol")) == ["600519"]
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    assert bars.iloc[0]["volume"] == 120.0
    assert bars.attrs["market"] == "CN"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "opentdx"


def test_opentdx_provider_maps_symbol_market_and_frequency() -> None:
    """Provider maps common A-share symbols and supported frequencies."""
    client = FakeTdxClient()
    provider = OpenTdxProvider(client_factory=lambda: client)

    provider.history_ohlc("000001", freq="1h")

    market, code, period, start, count = client.calls[0]
    assert (enum_value(market), code, enum_value(period), start, count) == (
        0,
        "000001",
        3,
        0,
        800,
    )


def test_opentdx_provider_connects_with_configured_host() -> None:
    """Provider uses configured host/port before fetching live rows."""
    client = ConnectableTdxClient()
    provider = OpenTdxProvider(
        host="1.2.3.4",
        port=7710,
        timeout=2.5,
        client_factory=lambda: client,
    )

    provider.history_ohlc("000001")

    assert client.quotation_client.calls == [("1.2.3.4", 7710, 2.5)]


def test_opentdx_provider_reports_unestablished_connection() -> None:
    """Provider reports host diagnostics when opentdx never connects."""
    client = ConnectableTdxClient(connected=False)
    provider = OpenTdxProvider(
        host="1.2.3.4",
        port=7710,
        timeout=2.5,
        client_factory=lambda: client,
    )

    with pytest.raises(ConnectionError, match="connection not established.*1.2.3.4:7710"):
        provider.history_ohlc("000001")


def test_opentdx_provider_uses_official_auto_server_by_default() -> None:
    """Provider follows opentdx's default auto server selection."""
    client = ConnectableTdxClient()
    provider = OpenTdxProvider(client_factory=lambda: client)

    provider.history_ohlc("000001")

    assert client.quotation_client.calls == [(None, 7709, 5.0)]


def test_opentdx_provider_rejects_unsupported_frequency() -> None:
    """Unsupported frequencies fail before network access."""
    provider = OpenTdxProvider(client_factory=FakeTdxClient)

    with pytest.raises(ValueError, match="Unsupported"):
        provider.history_ohlc("000001", freq="4h")


def test_opentdx_provider_reports_empty_live_response() -> None:
    """Empty live responses fail with a concrete provider message."""
    provider = OpenTdxProvider(client_factory=EmptyTdxClient)

    with pytest.raises(ConnectionError, match="returned no rows"):
        provider.history_ohlc("000001")


def test_infer_tdx_market_supports_explicit_exchange_prefix() -> None:
    """Symbol prefixes can explicitly select the TDX market."""
    assert infer_tdx_market("SH.600519") == (1, "600519")
    assert infer_tdx_market("SZ.000001") == (0, "000001")


def test_infer_tdx_market_maps_common_index_and_etf_symbols() -> None:
    """Common index and ETF codes map to their documented TDX markets."""
    assert infer_tdx_market("000001") == (0, "000001")
    assert infer_tdx_market("600519") == (1, "600519")
    assert infer_tdx_market("SH.000300") == (1, "000300")
    assert infer_tdx_market("510300") == (1, "510300")
    assert infer_tdx_market("159919") == (0, "159919")
