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
