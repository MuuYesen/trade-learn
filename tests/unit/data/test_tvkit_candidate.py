"""Fixed actual tvkit candidate; no market, credential or browser access."""

import asyncio

import pytest

from tradelearn.data._tvkit import TvkitResearchClient

OHLCV = pytest.importorskip("tvkit.api.chart.ohlcv").OHLCV
OHLCVBar = pytest.importorskip("tvkit.api.chart.models.ohlcv").OHLCVBar


def test_fixed_upstream_cannot_preserve_existing_raw_prices_without_io(monkeypatch):
    entered = []

    async def forbidden(self):
        entered.append(1)
        raise AssertionError("no authentication for invalid adjustment")

    monkeypatch.setattr(OHLCV, "__aenter__", forbidden)
    with pytest.raises(ValueError, match="none"):
        TvkitResearchClient().history("NASDAQ:AAPL", adjustment="none")
    assert not entered


def test_actual_client_auth_is_instance_explicit_not_environment_or_browser(monkeypatch):
    monkeypatch.setenv("TVKIT_BROWSER", "chrome")
    monkeypatch.setenv("TVKIT_AUTH_TOKEN", "forbidden-environment-token")
    seen = []

    async def fetch(self, symbol, interval, **kwargs):
        seen.append((self, self._auth_manager.auth_token, symbol, interval, kwargs))
        return [OHLCVBar.from_array([1704067200, 1, 2, 1, 2, 3])]

    monkeypatch.setattr(OHLCV, "get_historical_ohlcv", fetch)
    a = TvkitResearchClient()
    b = TvkitResearchClient(auth_token="explicit-instance-token")
    rows = a.history("NASDAQ:AAPL", adjustment="splits", count=3)
    b.history("NASDAQ:MSFT", adjustment="dividends", start="2024-01-01", end="2024-01-02")
    assert isinstance(rows[0], OHLCVBar)
    assert [x[1] for x in seen] == ["unauthorized_user_token", "explicit-instance-token"]
    assert seen[0][4]["bars_count"] == 3 and seen[1][4]["start"] == "2024-01-01"
    assert "bars_count" not in seen[1][4]
    assert all(x[0]._auth_manager is None and x[0]._max_attempts == 1 for x in seen)


def test_notebook_async_timeout_cancel_and_no_background_task_leak(monkeypatch):
    closed = []
    original = OHLCV.__aexit__

    async def exit(self, *args):
        closed.append(args[0])
        await original(self, *args)

    async def fetch(self, *args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(OHLCV, "__aexit__", exit)
    monkeypatch.setattr(OHLCV, "get_historical_ohlcv", fetch)

    async def notebook():
        client = TvkitResearchClient(timeout=0.01)
        with pytest.raises(RuntimeError, match="history_async"):
            client.history("NASDAQ:AAPL", adjustment="splits")
        with pytest.raises(TimeoutError):
            await client.history_async("NASDAQ:AAPL", adjustment="splits")
        task = asyncio.create_task(
            TvkitResearchClient().history_async("NASDAQ:AAPL", adjustment="splits")
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(notebook())
    assert len(closed) == 2


def test_candidate_gaps_are_real_upstream_api_not_a_claim_of_full_migration():
    # Existing Charting consumes these; do not retire its implementation merely
    # because count/range OHLCV is available in the new dependency.
    assert not hasattr(OHLCV, "resolve_symbol")
    assert not hasattr(OHLCV, "search_symbols")
    assert OHLCVBar.from_array([1704067200, 1, 2, 1, 2]).volume == 0
    from tvkit.api.chart.models.adjustment import Adjustment

    assert {x.value for x in Adjustment} == {"splits", "dividends"}


def test_actual_upstream_range_dispatch_utc_dst_and_date_end(monkeypatch):
    seen = []

    async def fetch(self, symbol, interval, *, start, end, adjustment):
        seen.append((start.isoformat(), end.isoformat(), adjustment.value))
        return [OHLCVBar.from_array([1710046800, 1, 2, 1, 2, 3])]

    monkeypatch.setattr(OHLCV, "_fetch_single_range", fetch)
    client = TvkitResearchClient()
    client.history(
        "NASDAQ:AAPL", adjustment="splits", start="2024-03-10T00:00:00-05:00", end="2024-03-10"
    )
    assert seen == [("2024-03-10T05:00:00+00:00", "2024-03-10T23:59:59+00:00", "splits")]


def test_entitlement_refusal_is_distinct_and_never_retried(monkeypatch):
    from tvkit.api.chart.exceptions import EntitlementError

    calls = []

    async def fetch(self, *args, **kwargs):
        calls.append(1)
        raise EntitlementError("permission denied", symbol="NASDAQ:AAPL", interval="1D")

    monkeypatch.setattr(OHLCV, "get_historical_ohlcv", fetch)
    with pytest.raises(PermissionError, match="entitlement"):
        TvkitResearchClient().history("NASDAQ:AAPL", adjustment="splits")
    assert calls == [1]
