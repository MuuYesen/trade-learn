"""TradingView websocket protocol and provider integration."""

import json

import pytest

from tradelearn.data import providers


def frame(value):
    payload = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


class Socket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent, self.timeouts = [], []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return next(self.messages)

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def close(self, **kwargs):
        self.closed = True


def transport(socket, **kwargs):
    from tradelearn.data.tradingview_transport import TradingViewTransport

    return TradingViewTransport(connection_factory=lambda *a, **k: socket, **kwargs)


def updates():
    return {
        "m": "timescale_update",
        "p": [
            "cs",
            {
                "s1": {
                    "s": [
                        {"v": [200, 2, 4, 1, 3, 20]},
                        {"v": [100, 1, 3, 0, 2, 10]},
                        {"v": [100, 1, 3, 0, 2.5, 11]},
                    ]
                }
            },
        ],
    }


def commands(socket):
    return [json.loads(s.split("~m~", 2)[2]) for s in socket.sent if not s.endswith("~h~42")]


def test_bounded_history_sends_cutoff_handles_fragmented_frames_and_heartbeat():
    data = frame(updates())
    ws = Socket(
        [
            frame("~h~42") + data[:20],
            data[20:] + frame({"m": "series_completed", "p": ["cs", "s1"]}),
        ]
    )
    result = transport(ws).history_window("NASDAQ:AAPL", "1D", count=2, to=200)
    assert [row["time"] for row in result] == [100, 200]
    assert result[0]["close"] == 2.5
    create = next(c for c in commands(ws) if c["m"] == "create_series")
    assert create["p"][4:] == ["1D", ["bar_count", 200, 2]]
    assert frame("~h~42") in ws.sent
    assert ws.closed
    assert all(0 < timeout <= 15 for timeout in ws.timeouts)


def test_forward_replay_uses_negative_bar_count_and_excludes_reference():
    ws = Socket([frame(updates()), frame({"m": "series_completed", "p": ["cs", "s1"]})])
    result = transport(ws).history_window("NASDAQ:AAPL", "1", count=2, to=100, forward=True)
    create = next(c for c in commands(ws) if c["m"] == "create_series")
    assert create["p"][-1] == ["bar_count", 101, -2]
    assert [row["time"] for row in result] == [200]


def test_resolve_preserves_upstream_metadata():
    info = {
        "name": "AAPL",
        "ticker": "AAPL",
        "pricescale": 100,
        "session": "0930-1600",
        "timezone": "America/New_York",
    }
    ws = Socket([frame({"m": "symbol_resolved", "p": ["cs", "symbol_1", info]})])
    assert transport(ws).resolve_symbol("NASDAQ:AAPL") == {"series_id": "symbol_1", **info}
    assert ws.closed


def test_upstream_error_is_safe_and_closes_socket():
    ws = Socket(
        [frame({"m": "symbol_error", "p": ["cs", "symbol_1", "sensitive upstream details"]})]
    )
    with pytest.raises(ConnectionError, match="TradingView request failed") as caught:
        transport(ws).resolve_symbol("NASDAQ:INVALID")
    assert "sensitive" not in str(caught.value)
    assert ws.closed


def test_timeout_closes_socket():
    ws = Socket([])

    def fail():
        raise TimeoutError("private endpoint")

    ws.recv = fail
    with pytest.raises(TimeoutError, match="TradingView request timed out"):
        transport(ws).history_window("NASDAQ:AAPL", "D")
    assert ws.closed


@pytest.mark.parametrize(
    "kwargs", [{"count": 5001}, {"count": 0}, {"forward": True}, {"to": float("nan")}]
)
def test_invalid_window_rejected_before_network(kwargs):
    with pytest.raises(ValueError):
        transport(Socket([])).history_window("NASDAQ:AAPL", "D", **kwargs)


def test_search_returns_node_compatible_metadata(monkeypatch):
    def request(url, **kwargs):
        assert "symbol_search" in url
        return [
            {"exchange": "NASDAQ Global", "symbol": "AAPL", "description": "Apple", "type": "stock"}
        ]

    client = transport(Socket([]))
    monkeypatch.setattr(client, "_http_json", request)
    assert client.search_symbols("Apple") == [
        {
            "id": "NASDAQ:AAPL",
            "exchange": "NASDAQ",
            "fullExchange": "NASDAQ Global",
            "symbol": "AAPL",
            "description": "Apple",
            "type": "stock",
        }
    ]


def test_provider_end_is_sent_to_transport(monkeypatch):
    from tradelearn.data.tradingview_transport import TradingViewTransport

    calls = []

    def history(self, symbol, resolution, **kwargs):
        calls.append((symbol, resolution, kwargs))
        return [{"time": 1704153600, "open": 1, "high": 3, "low": 0, "close": 2, "volume": 10}]

    monkeypatch.setattr(TradingViewTransport, "history_window", history)
    bars = providers.TradingViewProvider(n_bars=20).history_ohlc("NASDAQ:AAPL", end="2024-01-02")
    assert calls == [("NASDAQ:AAPL", "1D", {"count": 20, "to": 1704239999, "adjustment": "none"})]
    assert len(bars) == 1
    assert bars.attrs["source"] == "tradelearn:tradingview"


def test_provider_exposes_chart_methods():
    class Client:
        def history_window(self, *args, **kwargs):
            return [args, kwargs]

        def resolve_symbol(self, symbol, adjustment="splits"):
            return {"name": symbol}

        def search_symbols(self, query):
            return [{"symbol": query}]

    provider = providers.TradingViewProvider(client_factory=Client)
    assert provider.history_window("NASDAQ:AAPL", "1D", count=4, to=123, forward=True) == [
        ("NASDAQ:AAPL", "1D"),
        {"count": 4, "to": 123, "forward": True, "adjustment": "splits"},
    ]
    assert provider.resolve_symbol("NASDAQ:AAPL") == {"name": "NASDAQ:AAPL"}
    assert provider.search_symbols("AAPL") == [{"symbol": "AAPL"}]


def test_missing_websocket_dependency_has_safe_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "websocket", None)
    with pytest.raises(
        ConnectionError, match="TradingView websocket-client dependency is required"
    ):
        transport(Socket([])).resolve_symbol("NASDAQ:AAPL")


def test_heartbeats_cannot_extend_overall_deadline(monkeypatch):
    from tradelearn.data import tradingview_transport as module

    tick = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(tick))
    ws = Socket([frame("~h~42")] * 30)
    with pytest.raises(TimeoutError, match="TradingView request timed out"):
        transport(ws).history_window("NASDAQ:AAPL", "D")
    assert ws.closed


def test_credentials_are_exchanged_for_auth_token(monkeypatch):
    ws = Socket([frame({"m": "symbol_resolved", "p": ["cs", "symbol_1", {"name": "AAPL"}]})])
    client = transport(ws, username="example-user", password="secret")

    def request(url, **kwargs):
        assert url.endswith("/accounts/signin/")
        assert kwargs["data"] == b"username=example-user&password=secret&remember=on"
        return {"user": {"auth_token": "test-token"}}

    monkeypatch.setattr(client, "_http_json", request)
    client.resolve_symbol("NASDAQ:AAPL")
    assert commands(ws)[0] == {"m": "set_auth_token", "p": ["test-token"]}
    assert "secret" not in str(ws.sent)


def test_http_hard_deadline_interrupts_blocked_body(monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    stopped, closed = threading.Event(), threading.Event()

    class Response:
        def __init__(self):
            self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))

        def shutdown(self, how):
            stopped.set()

        def read(self, amount):
            stopped.wait(1)
            return b"[]"

        def close(self):
            closed.set()

    client = transport(Socket([]), timeout=0.05)
    monkeypatch.setattr(client, "_open_http", lambda *args: Response(), raising=False)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        client._http_json("https://example.invalid/", deadline=start + 0.05)
    elapsed = time.monotonic() - start
    assert elapsed < 0.2
    assert stopped.wait(0.2)
    assert closed.wait(0.2)


def test_http_hard_deadline_includes_connect_and_bounds_workers(monkeypatch):
    import threading
    import time

    from tradelearn.data import tradingview_transport as module

    entered, release = threading.Event(), threading.Event()

    def blocked_open(*args):
        entered.set()
        release.wait(1)
        raise TimeoutError()

    client = transport(Socket([]))
    monkeypatch.setattr(client, "_open_http", blocked_open, raising=False)
    monkeypatch.setattr(module, "_HTTP_WORKERS", threading.BoundedSemaphore(1), raising=False)
    try:
        for _ in range(2):
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                client._http_json("https://example.invalid/", deadline=start + 0.05)
            assert time.monotonic() - start < 0.2
        assert entered.is_set()
    finally:
        release.set()


def test_auth_cache_single_flight_expiry_and_credential_isolation(monkeypatch):
    import concurrent.futures
    import threading
    import time

    from tradelearn.data import tradingview_transport as module

    monkeypatch.setattr(module, "_TOKEN_CACHE", {}, raising=False)
    monkeypatch.setattr(module, "_TOKEN_TTL", 0.1, raising=False)
    calls, lock = [], threading.Lock()

    def request(self, url, **kwargs):
        with lock:
            calls.append(kwargs["data"])
        time.sleep(0.02)
        return {"user": {"auth_token": "token-" + str(len(calls))}}

    monkeypatch.setattr(module.TradingViewTransport, "_http_json", request)

    def token(password="one"):
        return transport(Socket([]), username="cached-user", password=password)._token(
            time.monotonic() + 1
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        assert list(pool.map(lambda _: token(), range(5))) == ["token-1"] * 5
    assert len(calls) == 1
    assert token("changed") == "token-2"
    assert len(calls) == 2
    time.sleep(0.11)
    assert token() == "token-3"


def test_failed_login_is_not_cached(monkeypatch):
    import time

    from tradelearn.data import tradingview_transport as module

    monkeypatch.setattr(module, "_TOKEN_CACHE", {}, raising=False)
    responses = iter([{}, {"user": {"auth_token": "recovered"}}])
    client = transport(Socket([]), username="retry-user", password="password")
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: next(responses))
    with pytest.raises(ConnectionError):
        client._token(time.monotonic() + 1)
    assert client._token(time.monotonic() + 1) == "recovered"


def test_http_cleanup_failure_does_not_leak_worker_capacity(monkeypatch):
    import threading
    import time

    from tradelearn.data import tradingview_transport as module

    class Response:
        def read(self, amount):
            return b"[]"

        def close(self):
            raise OSError("already closed")

    client = transport(Socket([]))
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, "_HTTP_WORKERS", slots)
    monkeypatch.setattr(client, "_open_http", lambda *args: Response())
    try:
        client._http_json("https://example.invalid/", deadline=time.monotonic() + 0.05)
    except TimeoutError:
        pass
    assert slots.acquire(timeout=0.1)
    slots.release()


@pytest.mark.parametrize(
    "method", ["resolve", "history", "get_hist", "provider_resolve", "provider_history"]
)
@pytest.mark.parametrize("mode", [None, "none", "splits", "DIVIDENDS"])
def test_adjustment_is_sent_in_actual_resolve_packet(method, mode):
    messages = (
        [frame({"m": "symbol_resolved", "p": ["cs", "symbol_1", {"name": "AAPL"}]})]
        if method.endswith("resolve")
        else [frame(updates()), frame({"m": "series_completed", "p": ["cs", "s1"]})]
    )
    ws = Socket(messages)
    client = transport(ws)
    kwargs = {} if mode is None else {"adjustment": mode}
    if method.startswith("provider_"):
        injected = client
        client = providers.TradingViewProvider(client_factory=lambda: injected)
        method = method.removeprefix("provider_")
    if method == "resolve":
        client.resolve_symbol("NASDAQ:AAPL", **kwargs)
    elif method == "history":
        client.history_window("NASDAQ:AAPL", "1D", **kwargs)
    else:
        client.get_hist("AAPL", "NASDAQ", **kwargs)
    packet = next(c for c in commands(ws) if c["m"] == "resolve_symbol")
    assert json.loads(packet["p"][2][1:])["adjustment"] == (mode or "splits").lower()


@pytest.mark.parametrize("mode", ["pre", "post", "", None, 1])
def test_invalid_adjustment_never_opens_network(mode):
    ws = Socket([])
    for client in (
        transport(ws),
        providers.TradingViewProvider(client_factory=lambda: transport(ws)),
    ):
        with pytest.raises(ValueError, match="adjustment"):
            client.resolve_symbol("NASDAQ:AAPL", adjustment=mode)
        with pytest.raises(ValueError, match="adjustment"):
            client.history_window("NASDAQ:AAPL", "1D", adjustment=mode)
    with pytest.raises(ValueError, match="adjustment"):
        transport(ws).get_hist("AAPL", "NASDAQ", adjustment=mode)
    assert not ws.sent


@pytest.mark.parametrize("tail", [[], [None], [float("nan")], [float("inf")]])
def test_unknown_volume_stays_null_and_json_safe(tail):
    packet = {
        "m": "timescale_update",
        "p": [
            "cs",
            {
                "s1": {
                    "s": [
                        {"v": [100, 1, 3, 0, 2, *tail]},
                    ]
                }
            },
        ],
    }
    ws = Socket([frame(packet), frame({"m": "series_completed", "p": ["cs", "s1"]})])
    result = transport(ws).history_window("NASDAQ:AAPL", "1D", count=1, to=100)
    assert result[0]["volume"] is None
    json.dumps(result, allow_nan=False)


def test_observed_zero_volume_is_preserved():
    packet = {
        "m": "timescale_update",
        "p": [
            "cs",
            {
                "s1": {
                    "s": [
                        {"v": [100, 1, 3, 0, 2, 0]},
                    ]
                }
            },
        ],
    }
    ws = Socket([frame(packet), frame({"m": "series_completed", "p": ["cs", "s1"]})])
    assert transport(ws).history_window("NASDAQ:AAPL", "1D", count=1, to=100)[0]["volume"] == 0


def test_research_bars_adjustment_metadata_matches_transport_request(monkeypatch):
    from tradelearn.data.tradingview_transport import TradingViewTransport

    adjustments = []

    def history(self, symbol, resolution, **kwargs):
        adjustments.append(kwargs["adjustment"])
        return [{"time": 1704153600, "open": 1, "high": 3, "low": 0, "close": 2, "volume": 10}]

    monkeypatch.setattr(TradingViewTransport, "history_window", history)
    bars = providers.TradingViewProvider().history_ohlc("NASDAQ:AAPL", end="2024-01-02")
    assert adjustments == [bars.attrs["adjust"]] == ["none"]
