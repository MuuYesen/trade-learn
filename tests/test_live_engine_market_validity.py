"""BT Market ignores valid; this must not become an unqualified native GTD order."""

import pytest

pytest.importorskip('nautilus_trader')
from tradelearn import engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.broker import NautilusBroker
from tests.test_live_order_mapping import native_host
from tests.unit.engine.test_bt_validity_actual import CASES, observations
import backtrader as official


@pytest.mark.parametrize('valid', CASES)
@pytest.mark.parametrize('tif', [None, 'ioc'])
def test_plain_engine_market_valid_keeps_bt_public_value_and_native_tif(valid, tif):
    native = native_host()
    sent = []

    class Gateway:
        def lookup_order(self, cid):
            return None
        def validate(self, request, **controls):
            native._make_order(request, controls)
        def submit(self, request, **controls):
            sent.append(native._make_order(request, controls)[0])

    broker = NautilusBroker(Gateway())

    class Business(bt.Strategy):
        def next(self):
            self.order = self.buy(size=1, valid=valid, **({} if tif is None else {'tif': tif}))

    runtime = StrategyRuntime(Business, feeds={'EURUSD': 'bar'}, broker=broker)
    try:
        runtime.start()
        runtime.step_market('EURUSD', dict(ts='2026-01-01T09:30:00Z', open=10,
                                          high=10, low=10, close=10, volume=1))
        assert len(sent) == 1 and sent[0].order_type.name == 'MARKET'
        assert sent[0].time_in_force.name == ('GTC' if tif is None else 'IOC')
        assert not hasattr(sent[0], 'expire_time_ns')
        expected = observations(official, 'Market', valid)[0]
        if expected is None:
            assert runtime.strategy.order.valid is None
        else:
            assert runtime.strategy.order.valid == pytest.approx(expected, abs=1e-10, rel=0)
    finally:
        runtime.cleanup()


@pytest.mark.parametrize('valid', [None, 0])
def test_native_limit_none_and_numeric_zero_are_unbounded_without_gtd(valid):
    from types import SimpleNamespace
    native = native_host()
    sent = []
    broker = NautilusBroker(SimpleNamespace(
        lookup_order=lambda cid: None,
        validate=lambda request, **controls: native._make_order(request, controls),
        submit=lambda request, **controls: sent.append(native._make_order(request, controls)[0]),
    ))
    class Business(bt.Strategy):
        def next(self):
            self.order = self.buy(size=1, price=9, exectype=bt.Order.Limit, valid=valid)
    runtime = StrategyRuntime(Business, feeds={'EURUSD': 'bar'}, broker=broker)
    try:
        runtime.start()
        runtime.step_market('EURUSD', dict(ts='2026-01-01T09:30:00Z', open=10,
                                          high=10, low=10, close=10, volume=1))
        assert len(sent) == 1 and sent[0].time_in_force.name == 'GTC'
        assert sent[0].expire_time_ns is None
        assert runtime.strategy.order.valid == valid
    finally:
        runtime.cleanup()
