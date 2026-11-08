"""Store assembly is offline; native callbacks remain owned by ExecutionHost."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip('nautilus_trader')
from nautilus_trader.adapters.binance import (
    BinanceDataClientConfig, BinanceDataClientFactory, BinanceEnvironment,
    BinanceExecutionClientConfig, BinanceExecutionClientFactory,
)
from nautilus_trader.common import Environment, LoggerConfig
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import AccountId, StrategyId, TraderId
from nautilus_trader.config import StrategyConfig
from tradelearn import engine as bt


def store(tmp_path, account='BINANCE-001', **kwargs):
    config = LiveNodeConfig(
        trader_id=TraderId('TL-001'), environment=Environment.SANDBOX,
        logging=LoggerConfig(bypass_logging=True),
        data_clients={'BINANCE': BinanceDataClientConfig(environment=BinanceEnvironment.TESTNET)},
        exec_clients={'BINANCE': BinanceExecutionClientConfig(
            account_id=AccountId(account), environment=BinanceEnvironment.TESTNET)},
    )
    return bt.stores.NautilusStore(
        config, data_factories={'BINANCE': BinanceDataClientFactory()},
        exec_factories={'BINANCE': BinanceExecutionClientFactory()},
        state_dir=tmp_path, **kwargs)


def engine_for(value, strategy=bt.Strategy, **parameters):
    engine = bt.Cerebro()
    engine.setbroker(value.getbroker())
    engine.adddata(value.getdata('BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL'))
    engine.addstrategy(strategy, **parameters)
    return engine


def test_construction_is_inert_and_account_instances_are_isolated(tmp_path):
    a, b = store(tmp_path), store(tmp_path, 'BINANCE-002')
    assert a is not b
    assert a.getbroker() is a.getbroker()
    assert a.getbroker() is not b.getbroker()
    assert a.account_id == 'BINANCE-001'
    assert a.state_path != b.state_path
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match='not started'):
        a.getbroker().getcash()
    assert a.get_notifications() == []


def test_unknown_parameters_and_duplicate_instrument_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        store(tmp_path, unknown=True)
    a = store(tmp_path)
    a.getdata('BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL')
    with pytest.raises(ValueError, match='instrument'):
        a.getdata('BTCUSDT.BINANCE-5-MINUTE-LAST-EXTERNAL')


@pytest.mark.parametrize('conflict', ['foreign_broker', 'foreign_feed', 'poller', 'paper', 'two_strategies', 'monitor'])
def test_conflicting_drivers_rejected_before_node_build(tmp_path, monkeypatch, conflict):
    a = store(tmp_path)
    engine = engine_for(a)
    if conflict == 'foreign_broker':
        engine.setbroker(store(tmp_path, 'BINANCE-002').getbroker())
    elif conflict == 'foreign_feed':
        engine.adddata(SimpleNamespace(_name='other', is_live=True))
    elif conflict == 'poller':
        engine.kwargs['live_poller'] = lambda: ()
    elif conflict == 'paper':
        engine.mode = 'paper'
    elif conflict == 'two_strategies':
        engine.addstrategy(bt.Strategy)
    monkeypatch.setattr(a, '_build_node', lambda: pytest.fail('conflict reached node build'))
    with pytest.raises((ValueError, NotImplementedError)):
        engine.run(**({'monitor': {'endpoint': 'https://example.invalid', 'token': 'test'}} if conflict == 'monitor' else {}))


class FakeNode:
    def __init__(self, events, error=None):
        self.events, self.error = events, error
        self.actor = None

    def add_strategy(self, actor):
        self.actor = actor
        if hasattr(actor, 'prepare_monitor'):
            # This fake does not register native handles. The real registration
            # path is exercised by the official BacktestEngine replay below.
            actor.prepare_monitor = lambda: None
        self.events.append('add')

    def handle(self):
        return SimpleNamespace(stop=lambda: self.events.append('stop'))

    async def run_async(self):
        self.events.append('run')
        if self.error:
            raise self.error

    def dispose(self):
        self.events.append('dispose')


def test_engine_uses_native_driver_and_always_drains(tmp_path, monkeypatch):
    a = store(tmp_path)
    engine = engine_for(a)
    events = []
    node = FakeNode(events)
    monkeypatch.setattr(a, '_build_node', lambda: node)
    monkeypatch.setattr(engine, '_run_event_mode', lambda: pytest.fail('second runtime'))
    assert engine.run() == []  # Fake node deliberately emits no business start.
    assert events == ['add', 'run', 'dispose']
    # The real host's journal must have released its process lock.
    from tradelearn_runtime.intents import IntentJournal
    journal = IntentJournal(a.state_path / 'intents.sqlite', identity=a.intent_identity)
    journal.close()
    assert [item[0] for item in a.get_notifications()] == ['STARTING', 'STOPPED']
    with pytest.raises(RuntimeError, match='single use'):
        engine.run()


def test_run_failure_preserves_primary_error_and_releases_journal(tmp_path, monkeypatch):
    a = store(tmp_path)
    engine = engine_for(a)
    events = []
    primary = RuntimeError('native run failed')
    node = FakeNode(events, primary)
    monkeypatch.setattr(a, '_build_node', lambda: node)
    with pytest.raises(RuntimeError, match='native run failed') as caught:
        engine.run()
    assert caught.value is primary
    assert events[-1] == 'dispose'
    assert a.get_notifications()[-1][0] == 'FAILED'
    from tradelearn_runtime.intents import IntentJournal
    journal = IntentJournal(a.state_path / 'intents.sqlite', identity=a.intent_identity)
    journal.close()


def test_sync_run_rejects_existing_event_loop_without_side_effects(tmp_path):
    a = store(tmp_path)
    async def call():
        with pytest.raises(RuntimeError, match='event loop'):
            engine_for(a).run()
    asyncio.run(call())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('logged', [False, True])
def test_official_bar_replay_uses_one_runtime_and_existing_params(tmp_path, monkeypatch, logged):
    if logged:
        pytest.importorskip('tradelearn._rust')
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.model import AccountType, OmsType, Money, Currency, Bar, BarType, Price, Quantity
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tradelearn_runtime.broker import NautilusBroker
    instrument = TestInstrumentProvider.btcusdt_binance()
    bar_type = BarType.from_str('BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL')
    native = BacktestEngine(BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    from nautilus_trader.execution import FixedFeeModel
    native.add_venue(instrument.id.venue, OmsType.NETTING, AccountType.CASH,
                     [Money(10000, Currency.from_str('USDT'))],
                     fee_model=FixedFeeModel(Money(0, Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    native.add_data([Bar(bar_type, Price.from_str('10.00'), Price.from_str('10.00'),
                         Price.from_str('10.00'), Price.from_str('10.00'), Quantity.from_str('1.000000'),
                         i * 60_000_000_000, i * 60_000_000_000) for i in range(1, 4)])
    calls = []
    stopped = []
    class Business(bt.TradeLogStrategy if logged else bt.Strategy):
        params = (('period', 2),)
        def __init__(self, label):
            calls.append(('init', label, self.p.period, self.data._name))
        def start(self):
            calls.append(('start',))
        def next(self):
            calls.append(('next', self.data.close[0]))
            if len(self.data) == 2:
                engine.runstop()
        def stop(self):
            calls.append(('stop',))
    a = store(tmp_path)
    engine = bt.Cerebro()
    engine.setbroker(a.getbroker())
    engine.adddata(a.getdata(bar_type))
    engine.addstrategy(Business, 'hello', period=3)
    class ReplayNode:
        def add_strategy(self, actor):
            self.actor = actor
            native.add_strategy(actor)
        def handle(self):
            return SimpleNamespace(stop=lambda: stopped.append(True))
        async def run_async(self):
            native.run()
        def dispose(self):
            native.dispose()
    monkeypatch.setattr(a, '_build_node', ReplayNode)
    [business] = engine.run()
    assert business.broker is a.getbroker()
    assert isinstance(business.broker, NautilusBroker)
    assert business._runtime is a._host.frontend
    assert calls == [('init', 'hello', 3, 'BTCUSDT.BINANCE'), ('start',),
                     ('next', 10.0), ('next', 10.0), ('next', 10.0), ('stop',)]
    assert stopped == [True]
    assert [item[0] for item in a.get_notifications()] == ['STARTING', 'RUNNING', 'STOPPED']


def test_tradelog_strategy_no_super_has_generated_local_context(tmp_path, monkeypatch):
    # Recorder is part of the installed wheel. Source-tree tests may supply the
    # matching prebuilt extension on the package path; no Rust rebuild required.
    pytest.importorskip('tradelearn._rust')
    class Business(bt.TradeLogStrategy):
        def __init__(self):
            ...
        def next(self):
            ...
    a = store(tmp_path)
    engine = engine_for(a, Business)
    events = []
    node = FakeNode(events)
    async def run():
        # Constructing the real Runtime exercises the unchanged param metaclass
        # and observer assembly; no broker or network calls are made.
        node.actor.frontend = bt.StrategyRuntime(
            node.actor._business_cls, feeds={'BTCUSDT.BINANCE': 'bar'},
            broker=a.getbroker(), parameters=node.actor._parameters)
        node.actor.frontend.start()
    node.run_async = run
    monkeypatch.setattr(a, '_build_node', lambda: node)
    [business] = engine.run()
    assert isinstance(business, Business)
    assert Business.monitor_config is None
    assert business.monitor_config.context['run_id'] == a._host.logger.run_id
    assert business.monitor_config.context['environment'] == 'paper'
    assert a._host.logger.sender is None


def test_drain_waits_for_all_workers_and_closes_on_failure(tmp_path):
    from tradelearn_runtime.execution import ExecutionHost
    from tradelearn_runtime.intents import IntentJournal
    async def run():
        journal = IntentJournal(tmp_path / 'intents.sqlite', identity='test')
        finished = []
        async def bad():
            raise RuntimeError('worker failed')
        async def slow():
            await asyncio.sleep(0.02)
            finished.append(True)
        actor = SimpleNamespace(_tasks={asyncio.create_task(bad()), asyncio.create_task(slow())},
                                _worker_error=None, intent_journal=journal)
        with pytest.raises(RuntimeError, match='worker failed'):
            await ExecutionHost.drain(actor)
        assert finished == [True]
        reopened = IntentJournal(tmp_path / 'intents.sqlite', identity='test')
        reopened.close()
    asyncio.run(run())


@pytest.mark.parametrize('phase', ['build', 'add', 'run'])
def test_logged_host_cleanup_at_each_node_failure(tmp_path, monkeypatch, phase):
    pytest.importorskip('tradelearn._rust')
    a = store(tmp_path)
    engine = engine_for(a, bt.TradeLogStrategy)
    node = FakeNode([])
    def fail(*args):
        raise RuntimeError(phase + ' failed')
    if phase == 'build':
        monkeypatch.setattr(a, '_build_node', fail)
    else:
        monkeypatch.setattr(a, '_build_node', lambda: node)
        if phase == 'add':
            node.add_strategy = fail
        else:
            node.error = RuntimeError('run failed')
    with pytest.raises(RuntimeError, match=phase + ' failed'):
        engine.run()
    assert a._host.logger.sender is None
    # Reopening both identities must succeed; cleanup released both writers.
    from tradelearn_runtime.intents import IntentJournal
    journal = IntentJournal(a.state_path / 'intents.sqlite', identity=a.intent_identity)
    journal.close()
    from leafquant_monitor.logger import TradeLogger
    logger = TradeLogger(a._host.logger.settings)
    logger.close()


def test_host_constructor_failure_closes_acquired_intent_lock(tmp_path, monkeypatch):
    import tradelearn_runtime.execution as execution
    from tradelearn_runtime.intents import IntentJournal
    acquired = []
    def fail(self, *args, **kwargs):
        acquired.append(self)
        raise RuntimeError('native construction failed')
    monkeypatch.setattr(IntentJournal, 'unfinished', fail)
    with pytest.raises(RuntimeError, match='native construction failed'):
        execution.create_execution_host(StrategyConfig(strategy_id=StrategyId('TL-001')),
            business_cls=bt.Strategy, feeds={'A': 'bar'}, instruments={},
            intent_path=tmp_path / 'intents.sqlite', intent_identity='test', account_id='BINANCE-001')
    assert acquired[0]._closed
    journal = IntentJournal(tmp_path / 'intents.sqlite', identity='test')
    journal.close()


def test_durable_worker_failure_requests_native_stop(tmp_path, monkeypatch):
    a = store(tmp_path)
    engine = engine_for(a)
    node = FakeNode([])
    stopped = asyncio.Event()
    node.handle = lambda: SimpleNamespace(stop=stopped.set)
    async def bad():
        raise RuntimeError('worker failed')
    async def run():
        node.actor._schedule(bad())
        await asyncio.wait_for(stopped.wait(), timeout=0.2)
    node.run_async = run
    monkeypatch.setattr(a, '_build_node', lambda: node)
    with pytest.raises(RuntimeError, match='worker failed'):
        engine.run()


def test_documented_official_adapter_configuration_constructs_offline(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BINANCE_TESTNET_API_KEY', 'synthetic-test-key')
    monkeypatch.setenv('BINANCE_TESTNET_API_SECRET', 'synthetic-test-secret')
    guide = Path(__file__).resolve().parents[3] / 'docs/guides/live.md'
    code = guide.read_text().split('```python\n', 1)[1].split('```', 1)[0]
    assert code.rstrip().endswith('engine.run()')
    namespace = {}
    exec(code.rsplit('engine.run()', 1)[0], namespace)
    value = namespace['store']
    assert value.account_id == 'BINANCE-001'
    assert namespace['engine'].broker is value.getbroker()
    assert not (tmp_path / '.tradelearn').exists()
