"""Unified, inert, per-account NA assembly; execution stays in the existing native host.

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

    def __post_init__(self):
        from tradelearn.backtest.parameters import Params
        from ._data_period import bar_period
        metadata = bar_period(self.bar_type)
        self.p = self.params = Params(metadata)
        self._timeframe = self.p.timeframe
        self._compression = self.p.compression


class NautilusStore:
    """One native Store for official factories and explicitly selected installed adapters.

    Construction/getdata/getbroker never build or connect a node. Native adapter
    parameters remain owned by NA; unknown Store/feed arguments raise TypeError.
    Each Store is single use. A new instance reopens the same durable identity.
    """

    _tradelearn_store_adapter_api = 2
    _tradelearn_store_interface = 1

    @classmethod
    def register_adapter(cls, name, factory):
        """Explicitly trust an inert config -> versioned assembly factory."""
        from tradelearn_runtime._adapters import register
        register(name, factory)

    def __init__(self, config=None, *, adapter=None, data_factories=None,
                 exec_factories=None, strategy_config=None,
                 state_dir=None, account_provider=None, buffer_capacity=512, **adapter_parameters):
        if type(buffer_capacity) is not int or buffer_capacity < 2:
            raise ValueError("buffer_capacity must be an integer at least two")
        self.buffer_capacity = buffer_capacity
        if adapter_parameters and adapter is None:
            raise TypeError('unknown native Store parameters; select an adapter for simple configuration')
        if adapter_parameters and config is not None:
            raise ValueError("simple adapter parameters and native config are mutually exclusive")
        self._close_adapter = None
        self._feed_compiler = None
        self._driver = None
        self._feed_validator = None
        account_id = None
        independent = adapter is not None
        if independent:
            if (data_factories is not None or exec_factories is not None
                    or strategy_config is not None or state_dir is not None
                    or account_provider is not None):
                raise ValueError('independent adapter config owns native assembly parameters')
            from tradelearn_runtime._adapters import resolve
            binding = resolve(adapter)(config, **adapter_parameters)
            if not isinstance(binding, dict) or binding.get('api_version') != 2:
                raise TypeError('adapter assembly API 2 mapping required')
            if 'driver' in binding:
                self._configure_driver(binding)
                return
            allowed = {'api_version', 'config', 'data_factories', 'exec_factories',
                       'account_id', 'strategy_config', 'state_dir', 'account_provider',
                       'validate_feed', 'compile_feed', 'close_adapter'}
            if set(binding) - allowed:
                raise ValueError('unknown independent adapter assembly fields')
            config = binding.get('config')
            data_factories, exec_factories = binding.get('data_factories'), binding.get('exec_factories')
            account_id = binding.get('account_id')
            strategy_config = binding.get('strategy_config')
            state_dir = binding.get('state_dir')
            account_provider = binding.get('account_provider')
            self._close_adapter = binding.get('close_adapter')
            if self._close_adapter is not None and not callable(self._close_adapter):
                raise TypeError('adapter cleanup must be callable')
            self._feed_compiler = binding.get('compile_feed')
            if self._feed_compiler is not None and not callable(self._feed_compiler):
                raise TypeError('adapter feed compiler must be callable')
            self._feed_validator = binding.get('validate_feed')
            if self._feed_validator is not None and not callable(self._feed_validator):
                raise TypeError('adapter feed validator must be callable')
        self._configure_native(config, data_factories=data_factories,
            exec_factories=exec_factories, strategy_config=strategy_config,
            state_dir=state_dir or '.tradelearn/nautilus', account_provider=account_provider,
            account_id=account_id, independent=independent)

    def _configure_native(self, config, *, data_factories, exec_factories,
                          strategy_config, state_dir, account_provider,
                          account_id, independent):
        from nautilus_trader.live import LiveNodeConfig
        from nautilus_trader.config import StrategyConfig
        from nautilus_trader.model import StrategyId
        if not isinstance(config, LiveNodeConfig):
            raise TypeError('official LiveNodeConfig required')
        if len(config.exec_clients) != 1:
            raise ValueError('one native execution account per Store required')
        execution = next(iter(config.exec_clients.values()))
        configured_account = getattr(execution, 'account_id', None)
        if configured_account is not None:
            if account_id is not None and str(account_id) != str(configured_account):
                raise ValueError('adapter account differs from native execution config')
            account_id = configured_account
        if account_id is None:
            raise ValueError('native execution config or adapter must supply account_id')
        from nautilus_trader.model import AccountId
        account_id = AccountId(str(account_id))
        if not config.data_clients:
            raise ValueError('native data client configuration required')
        if set(data_factories) != set(config.data_clients) or set(exec_factories) != set(config.exec_clients):
            raise ValueError('native factory keys must match client configuration keys')
        if not independent:
            self._validate_official_factories(data_factories, exec_factories)
        else:
            self._validate_factory_types(data_factories, exec_factories)
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

    @staticmethod
    def _validate_factory_types(data_factories, exec_factories):
        from nautilus_trader.live.clients import DataClientFactory, ExecutionClientFactory
        for factories, expected in ((data_factories, DataClientFactory),
                                    (exec_factories, ExecutionClientFactory)):
            if any(not isinstance(factory, expected) for factory in factories.values()):
                raise TypeError('installed adapter must supply native factory instances')

    @staticmethod
    def _validate_official_factories(data_factories, exec_factories):
        from importlib.resources import files
        inventory = json.loads(files('tradelearn_runtime').joinpath('official_adapters.json').read_text())
        for kind, factories in (('data_factory', data_factories), ('execution_factory', exec_factories)):
            allowed = {entry['package'] for entry in inventory['adapters'] if entry[kind]}
            for factory in factories.values():
                module_name = type(factory).__module__
                module = module_name.split('.')
                if len(module) < 3 or module[:2] != ['nautilus_trader', 'adapters'] or module[2] not in allowed:
                    raise ValueError('official adapter factory or explicitly selected installed adapter required')
                import importlib
                public_module = importlib.import_module(module_name)
                if getattr(public_module, type(factory).__name__, None) is not type(factory):
                    raise ValueError('official adapter factory type identity mismatch')

    def _configure_driver(self, binding):
        allowed = {'api_version', 'driver', 'account_id', 'trader_id', 'strategy_id', 'state_dir'}
        if set(binding) != allowed:
            raise ValueError('exact driver identity and assembly fields required')
        driver = binding['driver']
        required = ('getdata', 'validate', 'run', 'runtime')
        if not all(callable(getattr(driver, name, None)) for name in required):
            raise TypeError('incomplete native driver contract')
        if not isinstance(getattr(driver, 'broker', None), NautilusBroker):
            raise TypeError('driver must use the existing NautilusBroker')
        from nautilus_trader.model import AccountId, TraderId, StrategyId
        if any(binding[k] is None or not str(binding[k]) for k in ('account_id', 'trader_id', 'strategy_id')):
            raise ValueError('explicit driver account/trader/strategy identity required')
        self.account_id = str(AccountId(str(binding['account_id'])))
        identity = [str(TraderId(str(binding['trader_id']))), self.account_id,
                    str(StrategyId(str(binding['strategy_id'])))]
        self.intent_identity = json.dumps(identity, separators=(',', ':'))
        self.state_path = Path(binding['state_dir']).resolve() / sha256(self.intent_identity.encode()).hexdigest()
        self._driver = driver
        self._broker = driver.broker
        self._broker._store = self
        self._feeds = []
        self._notifications = deque()
        self._used = False
        self._handle = self._host = self._engine = None
        self._remote_monitor = self._monitor_binding = None

    @property
    def strategy_spec(self):
        return getattr(self,'_compiled_strategy_spec',None) or self._engine.strats[0]

    @property
    def remote_monitor_requested(self):
        return self._remote_monitor is not None

    def resolve_monitor(self):
        if self._driver is None:
            raise RuntimeError('native host binds its monitoring context')
        if self._monitor_binding is None:
            if self._remote_monitor is None:
                raise ValueError('explicit SQS monitor settings required')
            settings = getattr(self._driver, 'monitor_settings', None)
            if not callable(settings):
                raise ValueError('driver has no monitoring context provider')
            self._monitor_binding = self._remote_monitor.bind(settings())
        return self._monitor_binding

    def bind_native(self, node, host):
        if self._host is not None:
            raise RuntimeError('Store already owns a native host')
        self._host = host
        self._handle = node.handle()

    def prepare_runtime(self, runtime):
        return self._prepare_runtime(runtime)

    def getbroker(self):
        return self._broker

    def getdata(self, bar_type=None, *, name=None, **parameters):
        from nautilus_trader.model import BarType
        if self._used:
            raise RuntimeError('cannot add subscriptions after Store start')
        if self._driver is not None:
            if bar_type is not None:
                raise TypeError('this driver requires its declared keyword-only feed parameters')
            if name is not None:
                parameters['name'] = name
            feed = self._driver.getdata(self, **parameters)
            if getattr(feed, '_store', None) is not self or not getattr(feed, 'is_live', False):
                raise ValueError('driver returned a foreign or non-live feed')
            if feed not in self._feeds:
                self._feeds.append(feed)
            return feed
        if parameters:
            if bar_type is not None:
                raise ValueError('simple and native feed parameters are mutually exclusive')
            if self._feed_compiler is None:
                raise TypeError('this adapter requires native BarType configuration')
            bar_type = self._feed_compiler(**parameters)
        if isinstance(bar_type, str):
            bar_type = BarType.from_str(bar_type)
        if not isinstance(bar_type, BarType):
            raise TypeError('official BarType or its string representation required')
        if self._feed_validator is not None:
            self._feed_validator(bar_type)
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
        if engine._dooptimize:
            raise ValueError('native Store cannot optimize strategies')
        from .components import strategy_spec,validate_components
        self._compiled_strategy_spec = strategy_spec(engine)
        validate_components(engine,account_provider=getattr(self,'account_provider',None))
        if engine.timers or engine.calendars:
            raise NotImplementedError('TL timer/calendar scheduling is not implemented by the native Store')
        if engine.trade_on_close:
            raise ValueError('native execution cannot use backtest trade_on_close')
        if self._driver is not None:
            self._driver.validate(self, engine)

    def _build_node(self):
        from nautilus_trader.live import LiveNode
        return LiveNode.build('TradeLearn', self.config,
                              data_factories=self.data_factories, exec_factories=self.exec_factories)

    def _prepare_runtime(self, runtime):
        from tradelearn.core.strategy_events import ObserverSpec
        from .components import native_sizer
        runtime.strategy.setsizer(native_sizer(self._engine))
        runtime.strategy.cerebro = self._engine
        provider = getattr(self,'account_provider',None)
        if self._engine.analyzers or self._engine.observers or (self._engine.stdstats and provider is not None):
            from .components import LiveComponents
            runtime._cerebro_components = LiveComponents(self._engine,runtime,account_provider=provider)
        def failed(event):
            self.stop()
        runtime.add_observer(failed, phases=('error',))
        def started(event):
            self._notifications.append(('RUNNING', (), {}))
        return ObserverSpec(observer=started, events=frozenset({'lifecycle.start'}), phases=('after',))

    def _make_host(self, engine):
        strategy_cls, args, parameters = self.strategy_spec
        parameters = dict(parameters)
        monitor = None
        sender_factory = None
        from tradelearn.strategy_logging.tradelearn import TradeLogStrategy
        if self._remote_monitor is not None and not issubclass(strategy_cls, TradeLogStrategy):
            raise ValueError('remote monitor requires TradeLogStrategy')
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
            frontend_broker=self._broker, buffer_capacity=self.buffer_capacity,
            runtime_observer_factory=self._prepare_runtime, monitor=monitor, sender_factory=sender_factory,
            failure_callback=self.stop)
        # Static factory capability is assembly metadata, never Store event forwarding.
        factory = next(iter(self.exec_factories.values()))
        groups = getattr(factory, '_tradelearn_order_groups', None)
        if groups is not None and type(groups) is not bool:
            raise TypeError('adapter linked-order capability must be boolean')
        host._adapter_supports_groups = groups
        if monitor is not None:
            context = log_config.context
            context['run_id'] = host.logger.run_id
            context['environment'] = 'paper' if str(self.config.environment).lower() == 'sandbox' else 'live'
        return host

    def run(self, engine, *, monitor=None):
        """Explicit integration options outside the strict Cerebro parameter set."""
        return self._run_cerebro(engine, {} if monitor is None else {'monitor':monitor})

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
            if self._driver is None:
                result = asyncio.run(self._run_native(engine))
            else:
                self._driver.run(self)
                runtime = self._driver.runtime(self._host)
                if runtime is not None and runtime.failed:
                    raise RuntimeError('business Runtime failed; reconciliation required')
                result = [runtime.strategy] if runtime is not None else []
        except BaseException as error:
            primary = error
            self._notifications.append(('FAILED', (), {}))
            raise
        finally:
            self._handle = None
            engine._active_store_driver = None
            errors = []
            if self._driver is not None:
                logger = getattr(self._host, 'logger', None)
                if logger is not None:
                    try:
                        logger.finish_delivery()
                    except BaseException as cleanup:
                        errors.append(cleanup)
            if self._remote_monitor is not None:
                try:
                    self._remote_monitor.close()
                except BaseException as cleanup:
                    errors.append(cleanup)
            if errors:
                from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                error = primary or errors.pop(0)
                for cleanup in errors:
                    _safe_cleanup_diagnostic(error, cleanup)
                if primary is None:
                    self._notifications.append(('FAILED', (), {}))
                    raise error
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
                components = getattr(runtime,'_cerebro_components',None)
                if components is not None:
                    try:components.finish()
                    except BaseException as error:cleanup_errors.append(error)
                try:
                    await self._host.drain()
                except BaseException as error:
                    cleanup_errors.append(error)
            if node is not None:
                try:
                    node.dispose()
                except BaseException as error:
                    cleanup_errors.append(error)
            if self._close_adapter is not None:
                try:
                    await self._close_adapter()
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
