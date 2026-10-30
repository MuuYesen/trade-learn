from types import SimpleNamespace

import pytest

from tradelearn.backtest.strategy import Strategy
from tradelearn.core.strategy_events import (
    ObserverSpec,
    add_observer,
    assemble_observers,
    emit_domain,
    invoke_callback,
    observation_failures,
    remove_observer,
)


def test_filtered_immutable_snapshots_and_remove():
    s = Strategy()
    seen = []
    token = add_observer(
        s,
        seen.append,
        events={"business.notify_order"},
        context={"env": "backtest", "run_id": "synthetic"},
    )
    order = SimpleNamespace(
        ref="a", status=1, size=2, data=SimpleNamespace(_name="A"), executed=SimpleNamespace(size=0)
    )
    invoke_callback(s, "notify_order", order)
    assert len(seen) == 1
    assert seen[0].name == "business.notify_order"
    assert seen[0].context["env"] == "backtest"
    with pytest.raises(TypeError):
        seen[0].details["args"][0]["size"] = 9
    order.size = 4
    assert seen[0].details["args"][0]["size"] == 2
    remove_observer(s, token)
    invoke_callback(s, "notify_order", order)
    assert len(seen) == 1


def test_observer_failure_does_not_mask_business_or_skip_other_observers():
    class S(Strategy):
        def next(self):
            raise LookupError("business")

    s = S()
    seen = []

    def bad(e):
        raise OSError("durable failure")

    def handler(e):
        raise RuntimeError("bad handler")

    add_observer(s, bad, error_handler=handler)
    add_observer(s, seen.append)
    with pytest.raises(LookupError, match="business"):
        invoke_callback(s, "next")
    assert len(seen) == 1 and seen[0].phase == "error"
    failures = observation_failures(s)
    assert failures.total >= 2
    assert {f.error_type for f in failures.retained} == {"OSError", "RuntimeError"}


def test_no_super_external_sdk_factory_assembles_once_and_overridden_notify_runs_once():
    events = []
    constructed = []

    def factory(s):
        constructed.append(s)
        return ObserverSpec(events.append, context={"env": "sim", "run_id": "one"})

    class SDKBase(Strategy):
        strategy_observer_factories = (factory,)

    class User(SDKBase):
        def __init__(self):
            self.calls = 0

        def notify_order(self, order):
            self.calls += 1

    s = User()
    assemble_observers(s)
    assemble_observers(s)
    invoke_callback(s, "notify_order", SimpleNamespace(status=3))
    assert len(constructed) == 1 and s.calls == 1 and len(events) == 1


def test_observer_cannot_submit_or_cancel_via_public_strategy():
    s = Strategy()
    s.data = object()
    s.datas = [s.data]

    class Broker:
        def _submit(self, *a, **kw):
            raise AssertionError("must not reach broker")

        def cancel(self, *a):
            raise AssertionError("must not reach broker")

    s.broker = Broker()

    def callback(e):
        with pytest.raises(RuntimeError, match="observer"):
            s.buy(size=1)
        with pytest.raises(RuntimeError, match="observer"):
            s.cancel(object())

    add_observer(s, callback)
    emit_domain(s, "business.custom", "after", {"values": [1, 2]})
    assert not s._pending_size


def test_snapshot_and_controlflow_observer_errors_cannot_mask_business():
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def notify_order(self, order):
            self.calls += 1
            raise LookupError("main")

    class Opaque:
        @property
        def __dict__(self):
            raise RuntimeError("snapshot")

    s = S()
    add_observer(s, lambda e: None, phases=("before", "error"))
    with pytest.raises(LookupError, match="main"):
        invoke_callback(s, "notify_order", Opaque())
    assert s.calls == 1 and observation_failures(s).total >= 1
    other = S()

    def bad(e):
        raise SystemExit("observer exit")

    add_observer(other, bad)
    with pytest.raises(LookupError, match="main"):
        invoke_callback(other, "notify_order", SimpleNamespace(status=0))
    assert observation_failures(other).retained[0].error_type == "SystemExit"


def test_observer_direct_overridden_business_method_is_guarded():
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def next(self):
            self.calls += 1

    s = S()
    add_observer(s, lambda event: s.next(), events={"business.next"})
    invoke_callback(s, "next")
    assert s.calls == 1
    assert observation_failures(s).retained[0].error_type == "RuntimeError"


def test_async_observer_is_not_silently_discarded():
    async def async_observer(event):
        pass

    s = Strategy()
    with pytest.raises(TypeError, match="synchronous"):
        add_observer(s, async_observer)
    add_observer(s, lambda event: async_observer(event))
    invoke_callback(s, "next")
    assert observation_failures(s).retained[0].error_type == "TypeError"


def test_real_order_info_and_slots_snapshot_preserve_existing_identity():
    from tradelearn.backtest.models import Order

    events = []
    s = Strategy()
    add_observer(s, events.append)
    s.runtime_context = SimpleNamespace(symbol="A", ts_event_ns=12, sequence=3, event=None)
    order = Order(
        ref=7, data=None, ordtype=Order.Buy, size=2, info={"client_oid": "already-assigned"}
    )
    invoke_callback(s, "notify_order", order)
    snapshot = events[-1].details["args"][0]
    assert snapshot["info"]["client_oid"] == "already-assigned"
    assert events[-1].details["runtime_context"]["ts_event_ns"] == 12
    assert "event_id" not in events[-1].details

    class Slots:
        __slots__ = ("ts_event", "price")

        def __init__(self):
            self.ts_event = 10
            self.price = 2

    invoke_callback(s, "notify_order", Slots())
    assert events[-1].details["args"][0]["ts_event"] == 10
    assert events[-1].details["args"][0]["price"] == 2


def test_native_quote_snapshot_uses_existing_source_fields_without_sdk_import():
    pytest.importorskip("nautilus_trader")
    from nautilus_trader.model import InstrumentId, Price, Quantity, QuoteTick

    quote = QuoteTick(
        InstrumentId.from_str("EURUSD.SIM"),
        Price.from_str("1.10000"),
        Price.from_str("1.10010"),
        Quantity.from_int(2),
        Quantity.from_int(3),
        12,
        13,
    )
    s = Strategy()
    events = []
    add_observer(s, events.append)
    s.runtime_context = SimpleNamespace(symbol="A", ts_event_ns=12, sequence=1, event=quote)
    invoke_callback(s, "notify_order", None)
    snapshot = events[0].details["runtime_context"]["event"]
    assert snapshot["ts_event"] == 12
    assert snapshot["bid_price"]["raw"] == quote.bid_price.raw
    assert snapshot["bid_price"]["precision"] == quote.bid_price.precision
    assert snapshot["ask_size"]["raw"] == quote.ask_size.raw


def test_no_observer_has_no_snapshot_cost_and_observer_cannot_reemit(monkeypatch):
    import tradelearn.core.strategy_events as domain

    s = Strategy()
    monkeypatch.setattr(
        domain, "_freeze", lambda value, *a, **k: (_ for _ in ()).throw(AssertionError("snapshot"))
    )
    invoke_callback(s, "next")
    monkeypatch.undo()
    seen = []

    def bad(event):
        seen.append(event)
        emit_domain(s, "business.next", "after")

    add_observer(s, bad, events={"business.next"})
    invoke_callback(s, "next")
    assert len(seen) == 1
    assert observation_failures(s).retained[0].error_type == "RuntimeError"
    with pytest.raises(ValueError):
        emit_domain(s, "execution.filled", "after")


def test_observer_cannot_mutate_broker_or_advance_historical_data():
    import pandas as pd

    from tradelearn.backtest.broker import RustBroker
    from tradelearn.backtest.data import DataContainer

    s = Strategy()
    broker = RustBroker(cash=100)
    data = DataContainer(pd.DataFrame({"Open": [1.], "High": [1.], "Low": [1.],
                                       "Close": [1.], "Volume": [1.]}))
    before_cursor = data._cursor
    attempts = [lambda: broker.setcash(0), lambda: broker.setcommission(0.9),
                lambda: data._advance(0)]

    def observer(event):
        for attempt in attempts:
            with pytest.raises(RuntimeError, match="observer"):
                attempt()

    add_observer(s, observer)
    invoke_callback(s, "next")
    assert broker.getcash() == 100
    assert broker.commission_ratio == 0
    assert data._cursor == before_cursor
    assert observation_failures(s).total == 0


def test_diagnostic_warning_failure_cannot_mask_primary(monkeypatch):
    import tradelearn.core.strategy_events as events

    primary = LookupError("business")

    class S(Strategy):
        def next(self):
            raise primary

    def bad(event):
        raise OSError("observer")

    def bad_warning(*args, **kwargs):
        raise SystemExit("warning hook")

    s = S()
    add_observer(s, bad)
    monkeypatch.setattr(events.warnings, "warn", bad_warning)
    with pytest.raises(LookupError) as caught:
        invoke_callback(s, "next")
    assert caught.value is primary
    assert observation_failures(s).total == 1
