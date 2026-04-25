"""Tests for market data providers."""

from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.data.providers import DataProvider, PytdxProvider, infer_tdx_market


class FakeTdxClient:
    """Small pytdx2-compatible fake client."""

    def __init__(self) -> None:
        """Initialize fake call state."""
        self.calls: list[tuple[int, int, str, int, int]] = []
        self.connected = False
        self.disconnected = False

    def connect(self, host: str, port: int, time_out: float) -> bool:
        """Record connection parameters."""
        self.connected = True
        self.connection = (host, port, time_out)
        return True

    def disconnect(self) -> None:
        """Record disconnection."""
        self.disconnected = True

    def get_security_bars(
        self,
        category: int,
        market: int,
        code: str,
        start: int,
        count: int,
    ) -> list[dict[str, object]]:
        """Return pytdx-style kline dictionaries."""
        self.calls.append((category, market, code, start, count))
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

    def to_df(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        """Convert rows to DataFrame like pytdx2."""
        return pd.DataFrame(rows)


def test_pytdx_provider_fetches_and_normalizes_daily_bars() -> None:
    """PytdxProvider converts pytdx2 rows into Bars."""
    client = FakeTdxClient()
    provider: DataProvider = PytdxProvider(client_factory=lambda: client)

    bars = provider.history_ohlc("600519", start="2024-01-02", end="2024-01-02", freq="1d")

    assert client.connected is True
    assert client.disconnected is True
    assert client.calls == [(4, 1, "600519", 0, 800)]
    assert bars.index.names == ["timestamp", "symbol"]
    assert list(bars.index.get_level_values("symbol")) == ["600519"]
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    assert bars.iloc[0]["volume"] == 120.0
    assert bars.attrs["market"] == "CN"
    assert bars.attrs["freq"] == "1d"
    assert bars.attrs["engine"] == "pytdx2"


def test_pytdx_provider_maps_symbol_market_and_frequency() -> None:
    """Provider maps common A-share symbols and supported frequencies."""
    client = FakeTdxClient()
    provider = PytdxProvider(client_factory=lambda: client)

    provider.history_ohlc("000001", freq="1h")

    assert client.calls[0] == (3, 0, "000001", 0, 800)


def test_pytdx_provider_rejects_unsupported_frequency() -> None:
    """Unsupported frequencies fail before network access."""
    provider = PytdxProvider(client_factory=FakeTdxClient)

    with pytest.raises(ValueError, match="Unsupported"):
        provider.history_ohlc("000001", freq="4h")


def test_infer_tdx_market_supports_explicit_exchange_prefix() -> None:
    """Symbol prefixes can explicitly select the TDX market."""
    assert infer_tdx_market("SH.600519") == (1, "600519")
    assert infer_tdx_market("SZ.000001") == (0, "000001")
