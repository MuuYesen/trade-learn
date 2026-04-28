from __future__ import annotations

from tradelearn.backtest.core.brokers.rust import RustBroker
from tradelearn.core import BrokerEvent, BrokerEventPump


def test_broker_event_pump_dispatches_polled_proxy_events() -> None:
    events = [
        BrokerEvent("fill", order_id=1, payload={"order_id": 1, "price": 10.0}),
        {"kind": "cancel", "order_id": 2},
        {"kind": "reject", "order_id": 3, "reason": "risk limit"},
    ]
    fills: list[object] = []
    cancels: list[object] = []
    rejects: list[tuple[object, str]] = []

    pump = BrokerEventPump(lambda: events)
    pump.on_fill(fills.append)
    pump.on_cancel(cancels.append)
    pump.on_reject(lambda order_id, reason: rejects.append((order_id, reason)))

    assert pump.poll_once() == 3
    assert fills == [{"order_id": 1, "price": 10.0}]
    assert cancels == [2]
    assert rejects == [(3, "risk limit")]


def test_rust_broker_exposes_proxy_event_pump_for_external_polling() -> None:
    broker = RustBroker(match_mode="exact")
    broker._proxy_events = [
        BrokerEvent("fill", order_id=1, payload={"order_id": 1}),
        {"kind": "reject", "order_id": 2, "reason": "risk"},
    ]
    fills: list[object] = []
    rejects: list[tuple[object, str]] = []

    pump = broker.event_pump()
    pump.on_fill(fills.append)
    pump.on_reject(lambda order_id, reason: rejects.append((order_id, reason)))

    assert pump.poll_once() == 2
    assert broker.drain_proxy_events() == []
    assert fills == [{"order_id": 1}]
    assert rejects == [(2, "risk")]
