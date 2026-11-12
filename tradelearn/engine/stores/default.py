"""Per-instance connection assembly for installed TradeLearn adapters.

Factories are registered, never sessions or Store instances. No adapter is loaded
until explicitly selected, and no execution events pass through this module.
"""
import asyncio
from collections import deque
from importlib.metadata import entry_points
import re

_factories = {}


def register_adapter(name, factory):
    """Register a config -> binding factory, without connecting an account.

    Bindings implement broker, getdata(store, **kwargs), validate(store, engine),
    run(store), runtime(host), and monitor_settings() when monitoring is requested.
    run owns its existing native loop and drains/disposes it before returning.
    """
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
        raise ValueError('adapter name must be a lowercase identifier')
    if not callable(factory):
        raise TypeError('adapter factory must be callable')
    if name in _factories:
        if _factories[name] is factory:
            return
        raise ValueError(f'adapter {name!r} already registered')
    _factories[name] = factory


def _factory(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
        raise ValueError('adapter name must be a lowercase identifier')
    if name in _factories:
        return _factories[name]
    matches = list(entry_points(group='tradelearn.store_adapters', name=name))
    if len(matches) > 1:
        raise ValueError(f'ambiguous installed adapter {name!r}')
    if not matches:
        raise ValueError(f'adapter {name!r} is not installed or registered')
    factory = matches[0].load()
    register_adapter(name, factory)
    return factory


class DefaultStore:
    """Assemble one installed TL adapter; construction is inert and single use."""
    def __init__(self, *, adapter, config):
        self._adapter = _factory(adapter)(config)
        self._broker = self._adapter.broker
        self._broker._store = self
        self._host = self._handle = self._engine = None
        self._used = False
        self._notifications = deque()
        self._remote_monitor = None
        self._monitor_binding = None

    def getbroker(self):
        return self._broker

    def getdata(self, **kwargs):
        if self._used:
            raise RuntimeError('cannot add subscriptions after Store start')
        return self._adapter.getdata(self, **kwargs)

    @property
    def remote_monitor_requested(self):
        return self._remote_monitor is not None

    def resolve_monitor(self):
        if self._monitor_binding is None:
            if self._remote_monitor is None:
                raise ValueError('explicit SQS monitor settings required')
            self._monitor_binding = self._remote_monitor.bind(self._adapter.monitor_settings())
        return self._monitor_binding

    def get_notifications(self):
        result = list(self._notifications)
        self._notifications.clear()
        return result

    def stop(self):
        if self._handle is not None:
            self._handle.stop()

    @property
    def strategy_spec(self):
        return self._engine.strats[0]

    def bind_native(self, node, host):
        if self._host is not None:
            raise RuntimeError('Store already owns a native host')
        self._host = host
        self._handle = node.handle()

    def prepare_runtime(self, runtime):
        from tradelearn.core.strategy_events import ObserverSpec
        sizer, options = self._engine._sizer_spec
        runtime.strategy.setsizer(sizer(**options))
        runtime.add_observer(lambda event: self.stop(), phases=('error',))
        return ObserverSpec(observer=lambda event: self._notifications.append(('RUNNING', (), {})),
                            events=frozenset({'lifecycle.start'}), phases=('after',))

    def _validate(self, engine, run_kwargs):
        if self._used:
            raise RuntimeError('DefaultStore is single use')
        if run_kwargs or engine.kwargs or engine.mode == 'paper':
            raise ValueError('DefaultStore refuses polling/paper or unknown runtime parameters')
        if engine.broker is not self._broker or not engine.datas:
            raise ValueError('set this Store broker and feeds before run')
        if any(getattr(data, '_store', None) is not self for data in engine.datas):
            raise ValueError('DefaultStore cannot mix foreign feeds')
        if len({id(data) for data in engine.datas}) != len(engine.datas):
            raise ValueError('duplicate feed registration')
        if len({data._name for data in engine.datas}) != len(engine.datas):
            raise ValueError('duplicate feed name')
        if len(engine.strats) != 1 or engine._dooptimize or engine.signals:
            raise ValueError('DefaultStore requires exactly one business strategy')
        if (engine.trade_on_close or engine.analyzers or engine.observers
                or engine.writers or engine.timers or engine.calendars):
            raise ValueError('unsupported Cerebro components for DefaultStore')
        self._adapter.validate(self, engine)

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
            self._adapter.run(self)
            runtime = self._adapter.runtime(self._host)
            if runtime is not None and runtime.failed:
                raise RuntimeError('business Runtime failed; reconciliation required')
            result = [runtime.strategy] if runtime is not None else []
            engine._last_results = result
        except BaseException as error:
            primary = error
            self._notifications.append(('FAILED', (), {}))
            raise
        finally:
            engine._active_store_driver = None
            self._handle = None
            errors = []
            logger = getattr(self._host, 'logger', None)
            if logger is not None:
                try:
                    logger.finish_delivery()
                except BaseException as error:
                    errors.append(error)
            if self._remote_monitor is not None:
                try:
                    self._remote_monitor.close()
                except BaseException as error:
                    errors.append(error)
            if errors:
                from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                error = primary or errors.pop(0)
                for cleanup in errors:
                    _safe_cleanup_diagnostic(error, cleanup)
                if primary is None:
                    self._notifications.append(('FAILED', (), {}))
                    raise error
        self._notifications.append(('STOPPED', (), {}))
        return result
