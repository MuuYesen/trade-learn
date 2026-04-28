from __future__ import annotations

import json

import httpx
import pytest

from tradelearn.brokers import QMTBroker, QMTBrokerError


def _broker_with_transport(handler, **kwargs) -> QMTBroker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return QMTBroker(client=client, **kwargs)


def test_qmt_broker_places_paper_order_with_context() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/orders":
            payload = json.loads(request.read().decode())
            assert payload["mode"] == "paper"
            assert payload["account_id"] == "acct-1"
            assert payload["symbol"] == "000001.SZ"
            return httpx.Response(200, json={"order_id": "qmt-1"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    broker = _broker_with_transport(handler, account_id="acct-1")
    broker.connect()

    order_id = broker.place(
        {
            "symbol": "000001.SZ",
            "side": "buy",
            "type": "limit",
            "size": 100,
            "limit": 10.0,
        }
    )

    assert broker.is_connected() is True
    assert order_id == "qmt-1"
    assert [request.url.path for request in requests] == ["/health", "/orders"]


def test_qmt_broker_requires_live_confirmation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    broker = _broker_with_transport(handler, mode="live")
    broker.connect()

    with pytest.raises(QMTBrokerError, match="confirm_live"):
        broker.place({"symbol": "000001.SZ", "side": "buy", "size": 100})


def test_qmt_broker_enforces_max_order_value_before_http_order() -> None:
    order_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal order_requests
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/orders":
            order_requests += 1
        return httpx.Response(200, json={"ok": True})

    broker = _broker_with_transport(handler, max_order_value=999.0)
    broker.connect()

    with pytest.raises(QMTBrokerError, match="max_order_value"):
        broker.place({"symbol": "000001.SZ", "side": "buy", "size": 100, "limit": 10.0})

    assert order_requests == 0


def test_qmt_broker_queries_account_positions_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/account":
            return httpx.Response(200, json={"cash": 1000.0})
        if request.url.path == "/positions":
            return httpx.Response(200, json={"positions": [{"symbol": "000001.SZ"}]})
        if request.url.path == "/orders/qmt-1":
            return httpx.Response(200, json={"status": "filled"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    broker = _broker_with_transport(handler)
    broker.connect()

    assert broker.account() == {"cash": 1000.0}
    assert broker.positions() == [{"symbol": "000001.SZ"}]
    assert broker.order_status("qmt-1") == {"status": "filled"}
