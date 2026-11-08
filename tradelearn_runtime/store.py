"""Inert, per-account NA assembly; execution stays in the existing native host.

The Store model is inspired by Backtrader's public API. No Backtrader source,
metaclasses or singleton implementation are used here.
"""
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import asyncio
import json

from .execution import create_execution_host  # validates optional dependency/version
from .broker import NautilusBroker


class _Disconnected:
    def __getattr__(self, name):
        raise RuntimeError('NautilusStore is not started')


@dataclass(eq=False)
class _Subscription:
    _store: object
    bar_type: object
    _name: str
    is_live: bool = True


class NautilusStore:
    """Assemble one official LiveNodeConfig, account and business strategy.

    Construction/getdata/getbroker never build or connect a node. Native adapter
    parameters remain owned by NA; unknown Store/feed arguments raise TypeError.
    Each Store is single use. A new instance reopens the same durable identity.
    """

    def __init__(self, config, *, data_factories, exec_factories,
                 strategy_config=None, state_dir='.tradelearn/nautilus',
                 account_provider=None):
        from nautilus_trader.live import LiveNodeConfig
        from nautilus_trader.config import StrategyConfig
        from nautilus_trader.model import StrategyId
        if not isinstance(config, LiveNodeConfig):
            raise TypeError('official LiveNodeConfig required')
        if len(config.exec_clients) != 1:
            raise ValueError('one native execution account per Store required')
        execution = next(iter(config.exec_clients.values()))
        account_id = getattr(execution, 'account_id', None)
        if account_id is None:
            raise ValueError('native execution config must expose account_id')
        if not config.data_clients:
            raise ValueError('native data client configuration required')
        if set(data_factories) != set(config.data_clients) or set(exec_factories) != set(config.exec_clients):
            raise ValueError('native factory keys must match client configuration keys')
        from importlib.resources import files
        inventory = json.loads(files('tradelearn_runtime').joinpath('official_adapters.json').read_text())
        for kind, factories in (('data_factory', data_factories), ('execution_factory', exec_factories)):
            allowed = {entry['package'] for entry in inventory['adapters'] if entry[kind]}
            for factory in factories.values():
                module = type(factory).__module__.split('.')
                if len(module) < 3 or module[:2] != ['nautilus_trader', 'adapters'] or module[2] not in allowed:
                    raise ValueError('NautilusStore requires an official adapter factory from the pinned NA version')
        if strategy_config is None:
            strategy_config = StrategyConfig(strategy_id=StrategyId('TL-001'))
        if not isinstance(strategy_config, StrategyConfig):
            raise TypeError('official StrategyConfig required')
        if account_provider is not None and not callable(account_provider):
            raise TypeError('account_provider must be callable')
        self.config = config
        self.strategy_config = strategy_config
        self.account_id = str(account_id)
        self.data_factories = dict(data_factories)
        self.exec_factories = dict(exec_factories)
        self.account_provider = account_provider
        self.intent_identity = json.dumps([str(config.trader_id), self.account_id,
                                           str(strategy_config.strategy_id)], separators=(',', ':'))
        self.state_path = Path(state_dir).resolve() / sha256(self.intent_identity.encode()).hexdigest()
        self._broker = NautilusBroker(_Disconnected())
        self._broker._store = self
        self._feeds = []
        self._notifications = deque()
        self._used = False
        self._handle = self._host = self._engine = None
        self._remote_monitor = None

    def getbroker(self):
        return self._broker

    def getdata(self, bar_type, *, name=None):
        from nautilus_trader.model import BarType
        if self._used:
            raise RuntimeError('cannot add subscriptions after Store start')
        if isinstance(bar_type, str):
            bar_type = BarType.from_str(bar_type)
        if not isinstance(bar_type, BarType):
            raise TypeError('official BarType or its string representation required')
        if any(feed.bar_type.instrument_id == bar_type.instrument_id for feed in self._feeds):
            raise ValueError('one bar subscription per instrument supported by the current bridge')
        name = str(bar_type.instrument_id) if name is None else name
        if not isinstance(name, str) or not name or any(feed._name == name for feed in self._feeds):
            raise ValueError('unique nonempty feed name required')
        feed = _Subscription(self, bar_type, name)
        self._feeds.append(feed)
        return feed

    def get_notifications(self):
        result = []
        while self._notifications:
            result.append(self._notifications.popleft())
        return result

    def stop(self):
        """Request native shutdown; run() owns the awaited drain and disposal."""
        if self._handle is not None:
            self._handle.stop()

    def _validate(self, engine, run_kwargs):
        if self._used:
            raise RuntimeError('NautilusStore is single use; construct a new Store to restart')
        if run_kwargs:
            raise ValueError(f'unsupported native run parameters: {sorted(run_kwargs)}')
        if engine.broker is not self._broker or not engine.datas:
            raise ValueError('set this Store broker and add its feeds before run')
        if any(data not in self._feeds for data in engine.datas):
            raise ValueError('native Store cannot mix foreign feeds or polling drivers')
        names = [data._name for data in engine.datas]
        if len(set(names)) != len(names) or len({id(data) for data in engine.datas}) != len(engine.datas):
            raise ValueError('duplicate native feed registration')
        if engine.mode == 'paper' or set(engine.kwargs) - {'buffer_capacity'}:
            raise ValueError('native Store cannot combine paper/polling/runtime options')
        if len(engine.strats) != 1 or engine._dooptimize or engine.signals:
            raise ValueError('native Store requires exactly one addstrategy registration')
        if engine.analyzers or engine.observers or engine.writers or engine.timers or engine.calendars:
            raise NotImplementedError('native Store does not yet drive Cerebro analyzers/observers/writers/timers/calendars')
        if engine.trade_on_close:
            raise ValueError('native execution cannot use backtest trade_on_close')

    def _build_node(self):
        from nautilus_trader.live import LiveNode
        return LiveNode.build('TradeLearn', self.config,
                              data_factories=self.data_factories, exec_factories=self.exec_factories)

    def _prepare_runtime(self, runtime):
        from tradelearn.core.strategy_events import ObserverSpec
        sizer, options = self._engine._sizer_spec
        runtime.strategy.setsizer(sizer(**options))
        def failed(event):
            self.stop()
        runtime.add_observer(failed, phases=('error',))
        def started(event):
            self._notifications.append(('RUNNING', (), {}))
        return ObserverSpec(observer=started, events=frozenset({'lifecycle.start'}), phases=('after',))

    def _make_host(self, engine):
        strategy_cls, args, parameters = engine.strats[0]
        parameters = dict(parameters)
        monitor = None
        sender_factory = None
        from tradelearn.engine import TradeLogStrategy
        if self._remote_monitor is not None and not issubclass(strategy_cls, TradeLogStrategy):
            raise ValueError('remote monitor requires engine.TradeLogStrategy')
        if issubclass(strategy_cls, TradeLogStrategy):
            if 'monitor_config' in parameters or getattr(strategy_cls, 'monitor_config', None) is not None:
                raise ValueError('Store owns TradeLogStrategy context; custom monitor_config is unsupported')
            from leafquant_monitor import MonitorSettings
            from leafquant_monitor.tradelearn import TradeLogConfig
            monitor = MonitorSettings(
                source=self.account_id.split('-', 1)[0], strategy_id=str(self.strategy_config.strategy_id),
                account_id=self.account_id, state_dir=str(self.state_path / 'logs'),
                framework_strategy_id=str(self.strategy_config.strategy_id), framework_account_id=self.account_id)
            if self._remote_monitor is not None:
                monitor, sender_factory = self._remote_monitor.bind(monitor)
            # Provider runs after host construction; the native host owns the one
            # recorder/run_id. The business observer borrows it and starts no Sender.
            log_config = TradeLogConfig(
                context=dict(environment='live',
                             run_id='pending', source=monitor.source, strategy_id=monitor.strategy_id,
                             account_id=monitor.account_id, state_dir=monitor.state_dir),
                logger_provider=lambda strategy: self._host.logger)
            # Bind a per-run class attribute, not a synthetic user __init__ kwarg.
            # This keeps no-super callbacks and the existing TL params mechanism.
            strategy_cls = type(strategy_cls.__name__, (strategy_cls,),
                                {'monitor_config': log_config, '__module__': strategy_cls.__module__})
        host = create_execution_host(
            self.strategy_config, business_cls=strategy_cls, strategy_args=args,
            parameters=parameters, feeds={data._name: 'bar' for data in engine.datas},
            instruments={data._name: data.bar_type.instrument_id for data in engine.datas},
            subscriptions=[data.bar_type for data in engine.datas],
            account_id=self.account_id, account_provider=self.account_provider,
            intent_path=self.state_path / 'intents.sqlite', intent_identity=self.intent_identity,
            frontend_broker=self._broker, buffer_capacity=engine.kwargs.get('buffer_capacity', 512),
            runtime_observer_factory=self._prepare_runtime, monitor=monitor, sender_factory=sender_factory,
            failure_callback=self.stop)
        if monitor is not None:
            context = log_config.context
            context['run_id'] = host.logger.run_id
            context['environment'] = 'paper' if str(self.config.environment).lower() == 'sandbox' else 'live'
        return host

    def _run_cerebro(self, engine, run_kwargs):
        run_kwargs = dict(run_kwargs)
        remote = run_kwargs.pop('monitor', None)
        self._validate(engine, run_kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Cerebro.run requires a thread without a running event loop')
        if remote is not None:
            from leafquant_monitor.bootstrap import MonitorConnection
            self._remote_monitor = MonitorConnection.from_options(remote)
        self._used = True
        self._engine = engine
        engine._active_store_driver = self
        self._notifications.append(('STARTING', (), {}))
        primary = None
        try:
            result = asyncio.run(self._run_native(engine))
        except BaseException as error:
            primary = error
            self._notifications.append(('FAILED', (), {}))
            raise
        finally:
            self._handle = None
            engine._active_store_driver = None
            if self._remote_monitor is not None:
                try:
                    self._remote_monitor.close()
                except BaseException as cleanup:
                    if primary is None:
                        self._notifications.append(('FAILED', (), {}))
                        raise
                    from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                    _safe_cleanup_diagnostic(primary, cleanup)
        self._notifications.append(('STOPPED', (), {}))
        engine._last_results = result
        return result

    async def _run_native(self, engine):
        from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
        node = None
        primary = None
        try:
            self._host = self._make_host(engine)
            node = self._build_node()
            self._handle = node.handle()  # rc7 moves the wrapper during run_async.
            node.add_strategy(self._host)
            prepare = getattr(self._host, 'prepare_monitor', None)
            if prepare is not None:
                prepare()
            await node.run_async()
            if self._host.frontend is not None and self._host.frontend.failed:
                raise RuntimeError('native business Runtime failed; reconciliation required')
            return [self._host.frontend.strategy] if self._host.frontend is not None else []
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup_errors = []
            if self._host is not None:
                runtime = self._host.frontend
                if runtime is not None:
                    try:
                        runtime.cleanup(primary)
                    except BaseException as error:
                        cleanup_errors.append(error)
                try:
                    await self._host.drain()
                except BaseException as error:
                    cleanup_errors.append(error)
            if node is not None:
                try:
                    node.dispose()
                except BaseException as error:
                    cleanup_errors.append(error)
            if self._host is not None and hasattr(self._host, 'logger'):
                try:
                    await asyncio.to_thread(self._host.logger.finish_delivery)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                error = primary or cleanup_errors.pop(0)
                for cleanup in cleanup_errors:
                    _safe_cleanup_diagnostic(error, cleanup)
                if primary is None:
                    raise error
