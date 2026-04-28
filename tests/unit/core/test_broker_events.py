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


def test_broker_event_pump_dispatches_status_partial_and_replay_events() -> None:
    events = [
        {
            "kind": "status",
            "order_id": 1,
            "status": "accepted",
            "replay": True,
        },
        {
            "kind": "partial",
            "order_id": 1,
            "payload": {"size": 10, "filled": 4},
        },
    ]
    statuses: list[tuple[object, str, bool]] = []
    partials: list[object] = []

    pump = BrokerEventPump(lambda: events)
    pump.on_status(lambda order_id, status, replay: statuses.append((order_id, status, replay)))
    pump.on_partial(partials.append)

    assert pump.poll_once() == 2
    assert statuses == [(1, "accepted", True)]
    assert partials == [{"size": 10, "filled": 4}]


def test_broker_event_carries_live_risk_and_confirmation_fields() -> None:
    event = BrokerEvent(
        "status",
        order_id=9,
        status="pending_confirmation",
        requires_confirmation=True,
        max_notional=10000.0,
        risk_tags=("daily-limit",),
    )

    assert event.requires_confirmation is True
    assert event.max_notional == 10000.0
    assert event.risk_tags == ("daily-limit",)
