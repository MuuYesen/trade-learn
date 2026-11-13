"""Closed Store must not leave an unsendable NA actor in cyclic GC garbage."""

import gc
import weakref
from threading import Thread

import pytest

pytest.importorskip('nautilus_trader')
from tests.unit.engine.test_nautilus_store import store, engine_for, FakeNode


@pytest.mark.parametrize('logging', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_closed_native_store_actor_is_released_on_its_owner_before_cyclic_gc(tmp_path, logging, failure):
    def closed_scope():
        value = store(tmp_path)
        engine = engine_for(value)
        captured = []
        class Node(FakeNode):
            def add_strategy(self, actor):
                captured.append(weakref.ref(actor))
                super().add_strategy(actor)
        primary = ValueError("native fixture failure")
        value._build_node = lambda: Node([], error=primary if failure else None)
        if failure:
            with pytest.raises(ValueError) as caught:
                value.run(engine, monitor=True) if logging else engine.run()
            assert caught.value is primary
        else:
            value.run(engine, monitor=True) if logging else engine.run()
        return captured[0], primary
    enabled = gc.isenabled()
    try:
        gc.collect()
        gc.disable()
        actor, primary = closed_scope()
        retained = actor() is not None
        assert not retained, 'closed native actor still depends on cyclic GC; worker GC is unsafe'
        worker=Thread(target=gc.collect)
        worker.start();worker.join()
    finally:
        # Never intentionally destroy an unsendable actor on a worker in this red test.
        gc.collect()
        if enabled:
            gc.enable()


@pytest.mark.parametrize('logging', [False, True])
def test_real_native_node_dispose_releases_actor_even_with_business_result_retained(tmp_path, logging):
    from types import SimpleNamespace
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.model import Bar, BarType, Money, Currency, OmsType, AccountType
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tradelearn import engine as bt
    from tests.unit.backtest._engine_config import feed as as_feed

    def closed_scope():
        instrument = TestInstrumentProvider.btcusdt_binance()
        bars = BarType.from_str(str(instrument.id)+'-1-MINUTE-LAST-EXTERNAL')
        native = BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
        native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.CASH,
                         [Money(10000,Currency.from_str('USDT'))],
                         fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
        native.add_instrument(instrument)
        native.add_data([Bar(bars,*[instrument.make_price(10)]*4,instrument.make_qty(1),
                             60000000000,60000000000)])
        value=store(tmp_path)
        engine=bt.Cerebro(stdstats=False)
        engine.setbroker(value.getbroker())
        engine.adddata(as_feed(value.getdata(bars)))
        engine.addstrategy(bt.Strategy)
        captured=[]
        class Node:
            def add_strategy(self, actor):
                captured.append(weakref.ref(actor))
                native.add_strategy(actor)
            def handle(self):
                return SimpleNamespace(stop=lambda:None)
            async def run_async(self):
                native.run()
            def dispose(self):
                native.dispose()
        value._build_node=lambda:Node()
        result=value.run(engine,monitor=True) if logging else engine.run()
        return captured[0],result
    enabled=gc.isenabled()
    try:
        gc.collect();gc.disable()
        actor,result=closed_scope()
        retained=actor() is not None
        assert result and not retained, 'closed real native actor is still retained by result/Store cycles'
        worker=Thread(target=gc.collect)
        worker.start();worker.join()
        with pytest.raises(RuntimeError,match='closed'):
            result[0].position
    finally:
        gc.collect()
        if enabled:gc.enable()


@pytest.mark.parametrize('logging', [False, True])
def test_durable_worker_failure_keeps_original_error_but_releases_native_owner(tmp_path, logging):
    import asyncio
    from types import SimpleNamespace
    from tradelearn.core.broker_contracts import OrderRequest
    from tradelearn import engine as bt

    def closed_scope():
        value=store(tmp_path)
        engine=engine_for(value)
        captured=[]
        primary=ValueError('synthetic durable prepare failure')
        request=OrderRequest('BTCUSDT.BINANCE','buy',1,client_oid='OWNER-FAIL-1')
        def failed_prepare(*args):
            raise primary
        class Node(FakeNode):
            def add_strategy(self, actor):
                captured.append(weakref.ref(actor))
                super().add_strategy(actor)
            async def run_async(self):
                # Only lifecycle/IO failure is under test, no native command is sent.
                self.actor.frontend=SimpleNamespace(failed=False,strategy=bt.Strategy(),
                                                     cleanup=lambda error:None)
                self.actor.frontend_broker=value.getbroker()
                self.actor.intent_journal.prepare=failed_prepare
                task=self.actor._schedule(self.actor._admit_and_submit(
                    request,SimpleNamespace(is_reduce_only=False),{}))
                try:
                    await task
                finally:
                    task=None
        value._build_node=lambda:Node([])
        with pytest.raises(ValueError) as caught:
            value.run(engine,monitor=True) if logging else engine.run()
        assert caught.value is primary
        assert request.client_oid in value.getbroker()._unresolved
        return captured[0],primary
    enabled=gc.isenabled()
    try:
        gc.collect();gc.disable()
        actor,primary=closed_scope()
        retained=actor() is not None
        assert not retained, 'retained error traceback still owns the closed native actor'
        assert str(primary)=='synthetic durable prepare failure'
        import traceback
        assert any(frame.name=='_admit_and_submit' for frame in traceback.extract_tb(primary.__traceback__))
        worker=Thread(target=gc.collect);worker.start();worker.join()
    finally:
        gc.collect()
        if enabled:gc.enable()
