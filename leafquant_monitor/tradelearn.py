"""Optional TradeLearn 0.2.9/0.3.0 observation seam; no execution authority or loop."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import inspect
import logging
from types import MappingProxyType

import tradelearn
from tradelearn.core.strategy_events import OBSERVER_API_VERSION, ObserverSpec
from tradelearn.engine import Strategy

from .logger import MonitorSettings, TradeLogger


@dataclass(frozen=True)
class TradeLogConfig:
    """Exactly one of settings (owned), logger or logger_provider (borrowed).

    Context explicitly binds environment, experiment run_id, source, strategy_id,
    account_id and state_dir. It does not replace the Recorder's durable run_id.
    A provider is synchronous, receives the bound strategy and returns TradeLogger.
    Recorder injection is for the settings-owned path only.
    """
    context: Mapping
    settings: MonitorSettings | None = None
    logger: TradeLogger | None = None
    logger_provider: object = None
    recorder: object = None
    error_handler: object = None


def _plain(value):
    # TL MappingProxyType must remain structured JSON, never a repr string.
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('unsupported TL domain value for monitor wire')


def _sync_callable(value, name):
    if not callable(value) or inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(
            inspect.getattr_static(type(value), '__call__', None)):
        raise TypeError(f'{name} must be synchronous callable')


def _context(config):
    if not isinstance(config.context, Mapping):
        raise ValueError('explicit monitor run context required')
    context = _plain(config.context)
    required = ('run_id', 'source', 'strategy_id', 'account_id', 'state_dir')
    if context.get('environment') not in ('backtest', 'paper', 'live') or any(
            not isinstance(context.get(key), str) or not context[key] or context[key].strip() != context[key]
            for key in required):
        raise ValueError('explicit environment/run/identity/state context required')
    return context


def _binding(context, settings):
    if not isinstance(settings, MonitorSettings):
        raise TypeError('MonitorSettings required')
    for key in ('source', 'strategy_id', 'account_id', 'state_dir'):
        if context[key] != getattr(settings, key):
            raise ValueError(f'monitor context {key} differs from logger binding')


def _intent_details(details):
    # Runtime snapshots include the feed and the returned order projection.
    # Neither is another execution fact: keep only the request and correlation
    # identifiers, never tick OHLC placeholders, fills, fees or position values.
    result = {key: _plain(details[key]) for key in
              ('side', 'size', 'price', 'exectype', 'controls') if key in details}
    order = details.get('order')
    if isinstance(order, Mapping):
        identifiers = {key: _plain(order[key]) for key in
                       ('ref', 'client_oid', 'order_id') if key in order}
        info = order.get('info')
        if isinstance(info, Mapping):
            for key in ('client_oid', 'order_id'):
                if key in info and key not in identifiers:
                    identifiers[key] = _plain(info[key])
        if identifiers:
            result['order'] = identifiers
    return result


class _Monitor:
    def __init__(self, logger, context, owned, error_handler):
        self.logger = logger
        self.context = context
        self.owned = owned
        self.closed = False
        self.error_handler = error_handler
        from .tl_timing import TLBoundaryTiming
        self.timing = TLBoundaryTiming(logger)

    def failed(self, failure):
        logging.getLogger(__name__).error('TL monitor observation failed category=%s', failure.error_type)
        if self.error_handler is not None:
            return self.error_handler(failure)

    def _finish(self):
        if not self.owned or self.closed:
            return
        self.closed = True
        # Always attempt close even if flush failed. Neither error controls trading.
        failure = None
        try:
            result = self.logger.flush()
            if result.get('error'):
                raise RuntimeError('monitor flush reported persistence failure')
        except Exception as exc:
            failure = exc
        try:
            result = self.logger.close()
            if result.get('error'):
                raise RuntimeError('monitor close reported persistence failure')
        except Exception as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure

    def observe(self, event):
        runtime = event.details.get('runtime_context') or {}
        source_ts = runtime.get('ts_event_ns')
        fields = dict(event=event.name, phase=event.phase, run_context=_plain(self.context))
        if runtime.get('symbol') is not None:
            fields['symbol'] = runtime['symbol']
        if event.phase == 'error':
            fields['exception_type'] = event.error_type
            self.logger.record('exception', fields, source_ts_ns=source_ts)
            return
        if event.name.startswith('intent.'):
            fields.update(context_type='tradelearn_intent', details=_intent_details(event.details))
            self.logger.record('order_context', fields, source_ts_ns=source_ts)
            return
        fields['state'] = event.name.removeprefix('lifecycle.')
        if event.name != 'lifecycle.closed':
            self.logger.record('lifecycle', fields, source_ts_ns=source_ts)
            return
        primary = None
        try:
            self.logger.record('lifecycle', fields, source_ts_ns=source_ts)
        except Exception as exc:
            primary = exc
        try:
            self._finish()
        except Exception:
            if primary is None:
                raise
        if primary is not None:
            raise primary


def _assemble(strategy):
    if tradelearn.__version__ not in {'0.2.9', '0.3.0', '0.3.1'} or OBSERVER_API_VERSION != 1:
        raise RuntimeError('TradeLearn 0.2.9, 0.3.0 or 0.3.1 with observer API v1 required')
    config = getattr(strategy, 'monitor_config', None)
    if not isinstance(config, TradeLogConfig):
        raise ValueError('explicit TradeLogConfig required before Runtime assembly')
    if sum(value is not None for value in (config.settings, config.logger, config.logger_provider)) != 1:
        raise ValueError('provide exactly one settings, logger or logger_provider')
    if config.settings is None and config.recorder is not None:
        raise ValueError('recorder injection requires owned settings')
    context = _context(config)
    if config.error_handler is not None:
        _sync_callable(config.error_handler, 'error_handler')
    owned = config.settings is not None
    if owned:
        _binding(context, config.settings)  # Validate before acquiring a writer.
        logger = TradeLogger(config.settings, recorder=config.recorder)
    elif config.logger_provider is not None:
        _sync_callable(config.logger_provider, 'logger_provider')
        logger = config.logger_provider(strategy)
        if inspect.isawaitable(logger):
            if inspect.iscoroutine(logger):
                logger.close()
            raise TypeError('logger_provider returned awaitable')
    else:
        logger = config.logger
    if not isinstance(logger, TradeLogger):
        raise TypeError('logger or provider must supply existing TradeLogger')
    _binding(context, logger.settings)
    # Frozen copy: later mutation of caller context cannot change a run binding.
    monitor = _Monitor(logger, MappingProxyType(context), owned, config.error_handler)
    strategy._leafquant_tl_monitor = monitor
    strategy.monitor_logger = logger
    return ObserverSpec(observer=monitor.observe, events=frozenset({
        'lifecycle.setup', 'lifecycle.init', 'lifecycle.start', 'lifecycle.stop', 'lifecycle.closed',
        'intent.submit_order', 'intent.cancel_order'}), phases=('after',),
        context=context, error_handler=monitor.failed)


def _errors(strategy):
    monitor = strategy._leafquant_tl_monitor
    # Error-only subscription observes every callback failure without snapshotting
    # successful next/market callbacks or logging tick data.
    return ObserverSpec(observer=monitor.observe, phases=('error',), context=monitor.context,
                        error_handler=monitor.failed)


def _timing_callbacks(strategy):
    monitor = strategy._leafquant_tl_monitor
    return ObserverSpec(observer=monitor.timing.observe, events=frozenset({
        'business.next', 'business.prenext', 'business.notify_order'}),
        phases=('before',), context=monitor.context, error_handler=monitor.failed)


def _timing_submissions(strategy):
    monitor = strategy._leafquant_tl_monitor
    return ObserverSpec(observer=monitor.timing.observe, events=frozenset({'intent.submit_order'}),
        phases=('before','after','error'), context=monitor.context, error_handler=monitor.failed)


class TradeLogStrategy(Strategy):
    """Business callbacks belong entirely to TradeLearn, including no-super users."""
    monitor_config = None
    strategy_observer_factories = (_assemble, _errors, _timing_callbacks, _timing_submissions)
