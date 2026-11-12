"""Official Nautilus strategy hooks; observation failures never set trading policy.

Override ``on_strategy_*`` hooks, never the framework callback entries. This
adapter owns no subscriptions and makes no account or position calculations.
"""
from __future__ import annotations

from contextvars import ContextVar
import asyncio
from datetime import timedelta
import logging
import os
import time
import uuid

from nautilus_trader.trading import Strategy

from .logger import MonitorSettings, TradeLogger
from .snapshots import public_fields, public_value


_CALLBACKS = frozenset(name for name in dir(Strategy) if name.startswith('on_'))
_CONCRETE_EVENTS = frozenset(name for name in _CALLBACKS
                            if name.startswith(('on_order_', 'on_position_'))
                            and name not in ('on_order_event', 'on_position_event'))
# Legacy grid callbacks are absent from official v2. Silently accepting them
# would look migrated while dropping every quote/book delivery.
_RESERVED = _CALLBACKS | {'on_order_cancelled', 'on_quote_tick', 'on_order_book_deltas',
                          'submit_order', 'submit_order_list'}


from tradelearn.core.observation_clock import CLOCK_DOMAIN, monotonic_ns


class TradeLogStrategy(Strategy):
    monitor_clock_domain = CLOCK_DOMAIN

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in _RESERVED:
            if name in cls.__dict__ or (hasattr(cls, name) and
                    getattr(cls, name) is not getattr(TradeLogStrategy, name, None)):
                raise TypeError(f'{name} is owned by monitoring; use on_strategy_* hooks')

    def __new__(cls, config, *, monitor=None, recorder=None, sender_factory=None):
        # PyO3's constructor accepts only config; Python-only keywords must not
        # reach the native allocation step.
        return super().__new__(cls, config)

    def __init__(self, config, *, monitor: MonitorSettings | None = None, recorder=None, sender_factory=None):
        if monitor is None:
            path = os.environ.get('LEAFQUANT_MONITOR_CONFIG')
            if not path:
                raise ValueError('explicit monitor settings or LEAFQUANT_MONITOR_CONFIG required')
            monitor = MonitorSettings.from_file(path)
        self._monitor_market_context = ContextVar("leafquant_market_context", default=None)
        self._monitor_market_sequence = 0
        self._monitor_callback_observation = None
        self._monitor_strategy_id = self.strategy_id
        self._monitor_clock = None
        self._monitor_cache = None
        self.logger = TradeLogger(monitor, recorder=recorder, sender_factory=sender_factory)
        self.logger._monotonic_ns = self._event_monotonic_ns
        self.monitor_error = None
        self._monitor_timer = 'leafquant-monitor-' + self.logger.run_id
        self._monitor_running = False
        self._monitor_last_account_ns = None
        self._monitor_last_account_snapshot_ns = None
        self._monitor_account_query_pending = False
        self._monitor_account_query_handle = None
        self._monitor_account_scan_task = None
        self._monitor_account_scan = None
        self._monitor_account_snapshot_requested = False
        self._monitor_market = {}
        self._monitor_expected_market = {}
        self._monitor_bar_sources = {}

    @property
    def id(self):
        # Official 2.0.0rc7 exposes strategy_id; no inherited id property exists.
        return self._monitor_strategy_id

    def _observation_error(self, exc):
        self.monitor_error = type(exc).__name__
        # Error category only; payload, configuration and credentials stay private.
        logging.getLogger(__name__).error('monitor observation failed category=%s', self.monitor_error)

    def _observe(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            self._observation_error(exc)
            return None

    def _business(self, name, *args):
        hook = getattr(self, name, None)
        if hook is None:
            return {} if name == 'on_strategy_save' else None
        measured = name in {'on_strategy_quote','on_strategy_book_deltas','on_strategy_trade','on_strategy_bar'}
        previous = self.logger._active_calculation
        context = None
        start = self._observe(self._event_monotonic_ns) if measured else None
        if type(start) is int and start >= 0:
            context = dict(calculation_id=f'{self.logger.run_id}:calculation:{uuid.uuid4().hex}',
                stage='business_hook',hook=name,start_mono_ns=start,
                clock_domain=self.monitor_clock_domain,trace_ids=[])
            self.logger._active_calculation=context
        outcome='ok'
        try:
            return hook(*args)
        except Exception as exc:
            outcome='exception'
            self._observe(self.logger.record, 'exception',
                          {'hook': name, 'exception_type': type(exc).__name__})
            raise
        finally:
            if context is not None:
                try:
                    end=self._observe(self._event_monotonic_ns)
                    if type(end) is int and end>=context['start_mono_ns'] and (context['trace_ids'] or outcome=='exception'):
                        self._observe(self.logger.record,'calculation',dict(context,end_mono_ns=end,outcome=outcome))
                finally:
                    self.logger._active_calculation=previous

    def prepare_monitor(self):
        """Bind public handles after registration, outside native lifecycle borrows.

        Official rc7's engine-driven callbacks hold a mutable Strategy borrow.
        Engine.run callbacks permit public access. Direct native start() holds
        a mutable borrow, so manual start needs this call after registration.
        """
        self._monitor_clock = self.clock
        self._monitor_cache = self.cache

    def _wall_ns(self):
        return self._monitor_clock.timestamp_ns() if self._monitor_clock is not None else time.time_ns()

    def _start_monitor(self):
        if str(self.id) != (self.logger.settings.framework_strategy_id or self.logger.settings.strategy_id):
            raise ValueError('monitor strategy binding does not match official strategy ID')
        if self._monitor_clock is None:
            self._observe(self.prepare_monitor)
        self._monitor_running = True
        self._monitor_last_account_ns = None
        self._monitor_last_account_snapshot_ns = None
        self._observe(self._set_monitor_timer)
        self._monitor_tick(None)

    def _set_monitor_timer(self):
        if self._monitor_clock is None:
            raise RuntimeError('prepare_monitor required after registration for official engine-driven startup')
        if self._monitor_timer in self._monitor_clock.timer_names():
            self._monitor_clock.cancel_timer(self._monitor_timer)
        self._monitor_clock.set_timer(self._monitor_timer,
            timedelta(seconds=self.logger.settings.heartbeat_seconds), callback=self._monitor_tick)

    def _request_monitor_account(self):
        # Logging owns this configured observation request, never user strategy glue.
        # Defer the public dispatch until the native callback's Strategy borrow ends.
        if self._monitor_account_query_pending:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous/backtest environments keep their existing cache observations.
            return False
        self._monitor_account_query_pending = True
        self._monitor_account_query_handle = loop.call_soon(self._run_monitor_account_query)
        return True

    def _run_monitor_account_query(self):
        self._monitor_account_query_pending = False
        self._monitor_account_query_handle = None
        if not self._monitor_running or self.logger.settings.account_source == 'external':
            return
        try:
            from nautilus_trader.model import AccountId
            settings = self.logger.settings
            self.query_account(AccountId(settings.framework_account_id or settings.account_id))
        except Exception as exc:
            self._observation_error(exc)

    def _queue_monitor_account_scan(self, snapshot_due):
        self._monitor_account_snapshot_requested |= snapshot_due
        if self._monitor_account_scan_task is not None:return
        try:loop=asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous engines/backtests have no live async dispatch to starve.
            observed=self._wall_ns()
            changed=self.logger._sample_account_events(self._monitor_cache,observed)
            if changed or self._monitor_account_snapshot_requested:
                self.logger._record_account_snapshot(self._monitor_cache,observed_ts_ns=observed)
                self._monitor_last_account_snapshot_ns=observed
            self._monitor_account_snapshot_requested=False
            return
        self._monitor_account_scan_task=loop.create_task(self._run_monitor_account_scan())

    async def _run_monitor_account_scan(self):
        scan=None
        try:
            # Capture the view on its framework thread at the actual scan start.
            scan=self.logger._begin_account_event_scan(self._monitor_cache,self._wall_ns())
            self._monitor_account_scan=scan
            while not scan.done:
                deadline=time.monotonic_ns()+2_000_000
                for _ in range(16):
                    if not scan.step():break
                    if time.monotonic_ns()>=deadline:break
                if not scan.done:await asyncio.sleep(0)
            if self._monitor_running and (scan.result or self._monitor_account_snapshot_requested):
                # A snapshot reads the current cache, so it gets its own fresh clock.
                observed=self._wall_ns()
                self.logger._record_account_snapshot(self._monitor_cache,observed_ts_ns=observed)
                self._monitor_last_account_snapshot_ns=observed
            self._monitor_account_snapshot_requested=False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._observation_error(exc)
        finally:
            if scan is not None:self.logger._release_account_event_scan(scan)
            if self._monitor_account_scan is scan:self._monitor_account_scan=None
            if self._monitor_account_scan_task is asyncio.current_task():
                self._monitor_account_scan_task=None

    def _finish_monitor_account_scan(self):
        task=self._monitor_account_scan_task
        scan=self._monitor_account_scan
        self._monitor_account_scan_task=None;self._monitor_account_scan=None
        if task is not None:task.cancel()
        try:
            if scan is not None:
                changed=scan.finish()
                self.logger._release_account_event_scan(scan)
            elif task is not None:
                # The queued task may not have started; fulfill its fact capture.
                changed=self.logger._sample_account_events(self._monitor_cache,self._wall_ns())
            else:return
            if changed or self._monitor_account_snapshot_requested:
                observed=self._wall_ns()
                self.logger._record_account_snapshot(self._monitor_cache,observed_ts_ns=observed)
                self._monitor_last_account_snapshot_ns=observed
        except Exception as exc:self._observation_error(exc)
        finally:self._monitor_account_snapshot_requested=False

    def _stop_monitor(self):
        self._monitor_running = False
        self._finish_monitor_account_scan()
        if self._monitor_account_query_handle is not None:
            self._monitor_account_query_handle.cancel()
            self._monitor_account_query_handle = None
        self._monitor_account_query_pending = False
        try:
            if self._monitor_clock is not None and self._monitor_timer in self._monitor_clock.timer_names():
                self._monitor_clock.cancel_timer(self._monitor_timer)
        except Exception as exc:
            self._observation_error(exc)

    def on_start(self):
        self._start_monitor()
        self._observe(self.logger.record, 'lifecycle', {'state': 'started'})
        return self._business('on_strategy_start')

    def on_resume(self):
        self._start_monitor()
        self._observe(self.logger.record, 'lifecycle', {'state': 'resumed'})
        return self._business('on_strategy_resume')

    def on_stop(self):
        self._stop_monitor()
        self._observe(self.logger.record, 'lifecycle', {'state': 'stopped'})
        return self._business('on_strategy_stop')

    def on_reset(self):
        self._stop_monitor()
        self._observe(self.logger.record, 'lifecycle', {'state': 'reset'})
        return self._business('on_strategy_reset')

    def on_dispose(self):
        self._stop_monitor()
        try:
            self._observe(self.logger.record, 'lifecycle', {'state': 'disposed'})
            return self._business('on_strategy_dispose')
        finally:
            # Recorder close is local durability work. No sender/network operation.
            self._observe(self.logger.close)

    def _monitor_tick(self, event):
        self._observe(self._emit_monitor_tick, event)

    def _emit_monitor_tick(self, event):
        observed = self._wall_ns()
        self._observe(self.logger.record, 'heartbeat',
                      {'role': 'strategy', 'expected_running': self._monitor_running}, observed_ts_ns=observed)
        if (self._monitor_running and self.logger.settings.account_source in ('framework', 'framework_and_external')):
            # Observe raw retained events on the framework thread. Full snapshots
            # keep their configured cadence; no financial calculations run here.
            try:
                if self._monitor_cache is None:
                    raise RuntimeError('framework cache not bound; call prepare_monitor after registration')
                period = self.logger.settings.account_seconds * 1_000_000_000
                dispatched = False
                if (self._monitor_last_account_ns is None or observed - self._monitor_last_account_ns >= period):
                    dispatched = self._request_monitor_account()
                    self._monitor_last_account_ns = observed
                last = self._monitor_last_account_snapshot_ns
                due = last is None or observed - last >= period
                self._queue_monitor_account_scan(due and (last is None or not dispatched))
            except Exception as exc:
                self._observation_error(exc)
        self._observe(self._market_summary, observed)

    def expect_market(self, instrument_id, *, session_active, subscription_expected):
        if type(session_active) is not bool or type(subscription_expected) is not bool:
            raise ValueError('explicit boolean market expectations required')
        key = str(instrument_id)
        if key not in self._monitor_expected_market and len(self._monitor_expected_market) >= self.logger.settings.capacity:
            raise ValueError('market expectation capacity exhausted')
        self._monitor_expected_market[key] = (session_active, subscription_expected)

    def _cache_market(self, kind, item):
        key = str(item.instrument_id)
        if key not in self._monitor_market and len(self._monitor_market) >= self.logger.settings.capacity:
            raise ValueError('market observation capacity exhausted')
        observed, monotonic = self._wall_ns(), self._event_monotonic_ns()
        prices = {}
        if kind == 'quote':
            prices = dict(bid_price=str(item.bid_price), ask_price=str(item.ask_price))
        self._monitor_market[key] = dict(kind=kind, source_ts_ns=getattr(item, 'ts_event', None),
            observed_ts_ns=observed, observed_mono_ns=monotonic, **prices)

    def _market_summary(self, observed):
        for key in sorted(self._monitor_market.keys() | self._monitor_expected_market.keys()):
            row = dict(self._monitor_market.get(key, {}))
            session, subscribed = self._monitor_expected_market.get(key, (None, None))
            fields = dict(row, instrument_id=key, session_active=session,
                subscription_expected=subscribed, last_callback_ts_ns=row.get('observed_ts_ns'))
            self.logger.record('market_health', fields, source_ts_ns=row.get('source_ts_ns'),
                               observed_ts_ns=observed)

    def _market_origin(self, kind, item):
        # First SDK callback boundary, before cache/serialization/business work.
        # These are raw points, not socket receipt or calculated latency.
        mono, wall = self._event_monotonic_ns(), self._wall_ns()
        self._monitor_market_sequence += 1
        instrument = item.bar_type.instrument_id if kind == 'bar' else item.instrument_id
        return dict(callback_mono_ns=mono, callback_utc_ns=wall, kind=kind,
                    sequence=self._monitor_market_sequence, instrument_id=str(instrument),
                    source_ts_ns=getattr(item, 'ts_event', None),
                    data_init_ts_ns=getattr(item, 'ts_init', None))

    def _market_callback(self, kind, item, origin):
        # Context-local and stack-scoped: never retain the latest quote or an
        # entire native market object, and never leak it into another task/thread.
        scope = dict(origin=origin, active=True)
        token = self._monitor_market_context.set(scope)
        try:
            if kind == 'bar':
                self._observe(self._cache_bar, item)
            else:
                self._observe(self._cache_market, kind, item)
            return self._business('on_strategy_' + kind, item)
        finally:
            scope['active'] = False
            scope['origin'] = None
            self._monitor_market_context.reset(token)

    def on_quote(self, quote):
        origin = self._observe(self._market_origin, 'quote', quote)
        return self._market_callback('quote', quote, origin)

    def on_book_deltas(self, deltas):
        origin = self._observe(self._market_origin, 'book_deltas', deltas)
        return self._market_callback('book_deltas', deltas, origin)

    def on_trade(self, trade):
        origin = self._observe(self._market_origin, 'trade', trade)
        return self._market_callback('trade', trade, origin)

    def on_bar(self, bar):
        origin = self._observe(self._market_origin, 'bar', bar)
        return self._market_callback('bar', bar, origin)

    def register_bar_source(self, bar_type, *, closed_on_callback, timestamp_basis, source_reference,
                            price_currency=None, quantity_unit=None):
        if (type(closed_on_callback) is not bool or timestamp_basis not in {'open','close','unknown'}
                or not isinstance(source_reference,str) or not source_reference):
            raise ValueError('explicit verified bar source contract required')
        key=str(bar_type)
        if key not in self._monitor_bar_sources and len(self._monitor_bar_sources)>=self.logger.settings.capacity:
            raise ValueError('bar source capacity exhausted')
        profile=dict(closed_on_callback=closed_on_callback,timestamp_basis=timestamp_basis,
            source_reference=source_reference,price_currency=price_currency,quantity_unit=quantity_unit)
        if key in self._monitor_bar_sources and self._monitor_bar_sources[key]!=profile:
            raise ValueError('bar source cannot be rebound during a run')
        self._monitor_bar_sources[key]=profile

    def _cache_bar(self, bar):
        from types import SimpleNamespace
        self._cache_market('bar', SimpleNamespace(instrument_id=bar.bar_type.instrument_id,ts_event=bar.ts_event))
        profile=self._monitor_bar_sources.get(str(bar.bar_type))
        internal=bar.bar_type.is_internally_aggregated()
        time_bar=bar.bar_type.spec.is_time_aggregated()
        if internal:
            closed=True;status='official_internal_aggregator_completed'
            source='NautilusTrader2.0.0rc7/aggregation-completed-handler'
            # A time bar's build/initialization can be delayed and ts_event may
            # be the open. Completion does not prove a numeric close boundary.
            close_ns=bar.ts_event if not time_bar else None
            if time_bar and profile is not None and profile['timestamp_basis']=='close':
                close_ns=bar.ts_event
                source=profile['source_reference']
        elif profile is not None:
            closed=profile['closed_on_callback'];status='registered_source_contract'
            source=profile['source_reference']
            close_ns=bar.ts_event if closed and profile['timestamp_basis']=='close' else None
        else:
            closed=None;status='not_provided_by_external_bar';source=None;close_ns=None
        metadata=self._instrument_metadata(SimpleNamespace(instrument_id=bar.bar_type.instrument_id))
        fields=dict(public_fields(bar,('open','high','low','close','volume','ts_event','ts_init')),
            instrument_id=str(bar.bar_type.instrument_id),bar_type=str(bar.bar_type),
            aggregation_source=public_value(bar.bar_type.aggregation_source),closed=closed,
            closure_status=status,closure_source=source,close_ts_ns=close_ns,interval_complete=None,
            source_revision=None,revision_source=None,instrument_metadata=metadata,
            price_currency=profile['price_currency'] if profile else (metadata or {}).get('quote_currency'),
            quantity_unit=profile['quantity_unit'] if profile else None)
        self.logger.record('bar_observation',fields,source_ts_ns=bar.ts_event)

    def _request(self, order, anchor=None):
        if anchor is False:
            # An entry clock failed. Keep the request fact without claiming a
            # later clock as the missing SDK entry boundary.
            cid = str(order.client_order_id)
            self._observe(self.logger.record, 'submit_request',
                dict(public_fields(order, ('client_order_id', 'instrument_id', 'side', 'quantity', 'price')),
                     phase='local_request', trace_id=self.logger._order_traces.get(cid),
                     clock_domain=self.monitor_clock_domain))
            return cid
        if anchor is None:
            anchor = (self._event_monotonic_ns(), self._wall_ns())
        mono, wall = anchor
        cid = str(order.client_order_id)
        scope = self._monitor_market_context.get()
        origin = scope['origin'] if scope is not None and scope['active'] else None
        if origin is not None and origin['instrument_id'] != str(order.instrument_id):
            origin = None
        # Every invocation is a request, never evidence of venue submission.
        self._observe(self.logger.record, 'submit_request',
            dict(public_fields(order, ('client_order_id', 'instrument_id', 'side', 'quantity', 'price')),
                 phase='local_request', trace_id=self.logger._order_traces.get(cid),
                 clock_domain=self.monitor_clock_domain, timing_origin='SDK_SUBMIT_CALL_V1',
                 submit_mono_ns=mono, submit_utc_ns=wall, causal_market=origin),
            observed_ts_ns=wall, observed_mono_ns=mono)
        return cid

    def _before_native_submission(self,order,position_id=None,client_id=None,params=None):
        """Executor preflight; exceptions block dispatch and are never observer-isolated."""

    def _record_submit_dispatch(self, cid, public_anchor):
        fields = {'client_order_id': cid, 'phase': 'local_dispatch'}
        if public_anchor is not None:
            mono, wall = public_anchor
            fields.update(nautilus_api_entry_mono_ns=mono, nautilus_api_entry_utc_ns=wall,
                          nautilus_api_clock_domain=self.monitor_clock_domain,
                          nautilus_api_origin='NAUTILUS_PUBLIC_SUBMIT_V1')
        self.logger.record('submit_dispatch', fields)

    def submit_order(self, order, position_id=None, client_id=None, params=None):
        self._before_native_submission(order,position_id,client_id,params)
        anchor = self._observe(lambda: (self._event_monotonic_ns(), self._wall_ns()))
        cid = self._observe(self._request, order, anchor if anchor is not None else False)
        public_anchor = self._observe(lambda: (self._event_monotonic_ns(), self._wall_ns()))
        try:
            result = super().submit_order(order, position_id=position_id, client_id=client_id, params=params)
        except Exception as exc:
            self._observe(self.logger.record, 'exception', {'operation': 'submit_order', 'exception_type': type(exc).__name__,
                                                          'client_order_id': cid})
            raise
        self._observe(self._record_submit_dispatch, cid, public_anchor)
        return result

    def _request_list(self, order_list, anchor=None):
        # Official rc7 OrderFactory.bracket returns list[OrderAny], and
        # Strategy.submit_order_list consumes that sequence (not model.OrderList).
        cids = [self._observe(self._request, order, anchor) for order in order_list]
        return [cid for cid in cids if cid is not None]

    def submit_order_list(self, order_list, position_id=None, client_id=None, params=None):
        for order in order_list:self._before_native_submission(order,position_id,client_id,params)
        anchor = self._observe(lambda: (self._event_monotonic_ns(), self._wall_ns()))
        cids = self._observe(self._request_list, order_list, anchor if anchor is not None else False) or []
        public_anchor = self._observe(lambda: (self._event_monotonic_ns(), self._wall_ns()))
        try:
            result = super().submit_order_list(order_list, position_id=position_id, client_id=client_id, params=params)
        except Exception as exc:
            self._observe(self.logger.record, 'exception', {'operation': 'submit_order_list', 'exception_type': type(exc).__name__,
                                                          'client_order_ids': cids})
            raise
        for cid in cids:
            self._observe(self._record_submit_dispatch, cid, public_anchor)
        return result

    def _instrument_metadata(self, event):
        instrument_id = getattr(event, 'instrument_id', None)
        if self._monitor_cache is None or instrument_id is None:
            return None
        instrument = self._monitor_cache.instrument(instrument_id)
        if instrument is None:
            return None
        # These are public instrument facts, not inferred trading units. The
        # server still validates its registered source/instrument profile.
        return dict(public_fields(instrument, ('id', 'base_currency', 'quote_currency', 'settlement_currency',
            'price_precision', 'size_precision', 'price_increment', 'size_increment',
            'multiplier', 'is_inverse', 'asset_class', 'instrument_class')),
            public_type=type(instrument).__name__)

    def _event_monotonic_ns(self):
        return monotonic_ns()

    def _capture_event(self, callback, event):
        # Capture the first public callback boundary before metadata/log work.
        monotonic, observed = self._event_monotonic_ns(), self._wall_ns()
        self._monitor_callback_observation = dict(observed_ts_ns=observed, observed_mono_ns=monotonic)
        metadata = self._observe(self._instrument_metadata, event)
        self._observe(self.logger.record_event, callback[3:], event,
                      observed_ts_ns=observed, observed_mono_ns=monotonic,
                      extra={'instrument_metadata': metadata, 'clock_domain': self.monitor_clock_domain,
                             'timing_origin': 'SDK_SUBMIT_CALL_V1'})


def _callback_entry(name):
    def entry(self, *args):
        previous = self._monitor_callback_observation
        market_token = self._monitor_market_context.set(None)
        if name in _CONCRETE_EVENTS:
            self._observe(self._capture_event, name, args[0])
        hook = 'on_strategy_fill' if name == 'on_order_filled' else 'on_strategy_' + name[3:]
        try:
            return self._business(hook, *args)
        finally:
            self._monitor_market_context.reset(market_token)
            if name in _CONCRETE_EVENTS:
                self._monitor_callback_observation = previous
    entry.__name__ = name
    entry.__doc__ = 'Owned framework entry; business hook is on_strategy_' + name[3:]
    return entry


# Include every public concrete event in the installed official release. Aggregate
# callbacks dispatch business hooks only, so concrete + aggregate never double log.
for _name in sorted(_CALLBACKS):
    if _name not in TradeLogStrategy.__dict__:
        setattr(TradeLogStrategy, _name, _callback_entry(_name))
