"""Official rc7 Strategy facade: owns the live frontend on its callback thread."""
from decimal import Decimal
import math
import asyncio
from dataclasses import asdict

import sys

if not ((3, 13) <= sys.version_info[:2] < (3, 15)):
    raise RuntimeError('Nautilus execution bridge requires Python 3.13 or 3.14')
try:
    import nautilus_trader
except ModuleNotFoundError as error:
    if error.name != 'nautilus_trader':
        raise
    raise RuntimeError(
        "Nautilus execution bridge requires the verified trade-learn[nautilus] dependency"
    ) from error
from nautilus_trader.model import AccountId, ClientOrderId, PositionId, OrderSide, TimeInForce
from nautilus_trader.trading import Strategy as NautilusStrategy
from tradelearn.backtest.models import Position, Trade
from .broker import NautilusBroker
from ._data_period import feed_period_metadata as _feed_period_metadata
from tradelearn.engine import Strategy
from tradelearn.backtest.event_runtime import StrategyRuntime


def number(value):
    return float(value.as_double()) if hasattr(value,'as_double') else float(value)


def _trailing_parameters(request):
    """Compile the initial TL watermark once; NA/venue owns all subsequent trailing."""
    from nautilus_trader.model import TrailingOffsetType
    anchor = Decimal(str(request.stop_price))
    if not anchor.is_finite() or anchor <= 0:
        raise ValueError('positive finite trailing reference price required')
    amount, percent = request.trail_amount, request.trail_percent
    for value in (amount, percent):
        if value is not None and (not Decimal(str(value)).is_finite() or Decimal(str(value)) <= 0):
            raise ValueError('positive finite trailing offset required')
    # Existing TL semantics give absolute amount precedence when both are supplied.
    if amount is not None:
        offset = Decimal(str(amount)); distance = offset
        kind = TrailingOffsetType.PRICE
    elif percent is not None:
        fraction = Decimal(str(percent)); offset = fraction * 10000
        distance = anchor * fraction; kind = TrailingOffsetType.BASIS_POINTS
    else:
        raise ValueError('trailing amount or percent required')
    if not offset.is_finite() or offset <= 0:
        raise ValueError('positive finite trailing offset required')
    trigger = anchor + distance if request.side == 'buy' else anchor - distance
    if not trigger.is_finite() or trigger <= 0:
        raise ValueError('positive finite initial trailing trigger required')
    return trigger, offset, kind


def _trailing_limit_parameters(request):
    """BT keeps price-plimit constant; NA offsets are measured from market basis."""
    trigger, offset, kind = _trailing_parameters(request)
    anchor = Decimal(str(request.stop_price))
    original_limit = anchor if request.limit_price is None else Decimal(str(request.limit_price))
    if not original_limit.is_finite() or original_limit <= 0:
        raise ValueError('positive finite trailing execution limit required')
    difference = anchor - original_limit
    if request.trail_amount is None and difference:
        # Fixed NA shares one offset unit for both prices. A percentage of a
        # changing basis cannot represent BT's constant nonzero price delta.
        raise NotImplementedError('native percent offsets cannot preserve a fixed nonzero BT limit difference')
    limit = trigger - difference
    direction = Decimal(1) if request.side == 'buy' else Decimal(-1)
    limit_offset = offset + direction * (original_limit - anchor) if request.trail_amount is not None else offset
    return trigger, limit, limit_offset, offset, kind


def _require_qualified_expiry(request):
    if request.tif == 'gtd':
        raise NotImplementedError('native GTD expiry boundary is not qualified on pinned NA; explicit deadlines are not admitted')


def _native_route_proof(native, controls):
    """Persist immutable admission flags separately from native command routing."""
    from .native_controls import native_control_proof
    return native_control_proof(native,controls)


def _unclosed_native_orders(cache,strategy_id):
    # Initialized algorithm roots and locally emulated orders are not necessarily
    # in orders_open/inflight. All are outstanding existence witnesses, not ACKs.
    return [order for order in cache.orders(strategy_id=strategy_id) if not order.is_closed]


class _RuntimeBridge:
    def __new__(cls, config, **kwargs):
        return super().__new__(cls, config)

    def __init__(self, config, *, business_cls, feeds, instruments, parameters=None,
                 account_provider=None, subscriptions=(), monitor=None, recorder=None, sender_factory=None,
                 intent_path=None, offline_test=False, runtime_observer_factory=None,
                 account_id=None, intent_identity=None, frontend_broker=None,
                 strategy_args=(), buffer_capacity=512, failure_callback=None):
        if nautilus_trader.__version__ != '2.0.0rc7':
            raise RuntimeError('live bridge requires verified Nautilus 2.0.0rc7')
        if not isinstance(business_cls,type) or not issubclass(business_cls,Strategy):
            raise TypeError('live execution requires tradelearn.engine.Strategy')
        if account_id is None and not offline_test:
            raise ValueError('explicit native account binding required')
        self._account_id = str(account_id) if account_id is not None else None
        if monitor is None:
            if recorder is not None or sender_factory is not None:
                raise ValueError('recorder and sender require explicit monitor settings')
        else:
            expected = monitor.framework_account_id or monitor.account_id
            if self._account_id is None or self._account_id != expected:
                raise ValueError('monitor differs from native bridge account binding')
        if not set(feeds).issubset(instruments) and instruments:
            raise ValueError('every feed needs an explicit Nautilus InstrumentId')
        if runtime_observer_factory is not None and not callable(runtime_observer_factory):
            raise TypeError('runtime_observer_factory must be callable')
        if failure_callback is not None and not callable(failure_callback):
            raise TypeError('failure_callback must be callable')
        self._failure_callback=failure_callback
        self._runtime_observer_factory=runtime_observer_factory
        self._native_strategy_config=config
        self._business_cls=business_cls
        self._feeds=dict(feeds)
        self._instrument_ids=dict(instruments)
        self._parameters=parameters or {}
        self._strategy_args=tuple(strategy_args)
        self._buffer_capacity=buffer_capacity
        self._provided_broker=frontend_broker
        self._account_provider=account_provider
        self._subscriptions=tuple(subscriptions)
        self.frontend=None
        self.frontend_broker=None
        self._position_events={}
        self._position_trade_views={}
        self._next_position_trade_ref=1
        self._tasks=set()
        self._io_lock=asyncio.Lock()
        self._worker_error=None
        self._stopping=False
        self._offline_test=offline_test
        if intent_path is None and not offline_test:
            raise ValueError('live execution requires an identity-bound durable intent_path')
        self.intent_journal=None
        self._recovering={}
        try:
            if intent_path is not None:
                from .intents import IntentJournal
                if not isinstance(intent_identity, str) or not intent_identity or intent_identity.strip() != intent_identity:
                    raise ValueError('explicit durable intent identity required')
                self.intent_journal=IntentJournal(intent_path,identity=intent_identity)
                self._recovering=self.intent_journal.unfinished()
            if monitor is None:
                super().__init__(config)
            else:
                super().__init__(config, monitor=monitor, recorder=recorder, sender_factory=sender_factory)
        except BaseException as primary:
            from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
            for resource in (self.intent_journal, getattr(self,'logger',None)):
                if resource is not None:
                    try:resource.close()
                    except BaseException as cleanup:_safe_cleanup_diagnostic(primary,cleanup)
            raise

    @staticmethod
    def _read_cache(host):
        # Read-only native handle, captured while its Strategy getter is free.
        # This is the same NA cache, not a Python position/account ledger.
        cache = getattr(host, '_callback_cache', None)
        return cache if cache is not None else host.cache

    @staticmethod
    def _strategy_identity(host):
        identity = getattr(host, '_callback_strategy_id', None)
        return identity if identity is not None else host.strategy_id

    def _assert_account_binding(self):
        if not self._account_id:
            raise ValueError('explicit native account binding required')
        cache = _RuntimeBridge._read_cache(self)
        account=cache.account(AccountId(self._account_id))
        if account is None or str(account.id)!=self._account_id:
            raise ValueError('native account is absent or differs from explicit binding')
        for iid in self._instrument_ids.values():
            routed=cache.account_for_venue(iid.venue)
            if routed is None or str(routed.id)!=self._account_id:
                raise ValueError('venue execution account differs from explicit binding')
        return account

    def _owns_native_order(self,order,event=None):
        identity = _RuntimeBridge._strategy_identity(self)
        if order.strategy_id!=identity or order.instrument_id not in self._instrument_ids.values():
            return False
        if order.account_id is not None and str(order.account_id)!=self._account_id:
            return False
        if event is not None:
            if getattr(event,'strategy_id',identity)!=identity:
                return False
            account=getattr(event,'account_id',None)
            if account is not None and str(account)!=self._account_id:
                return False
        if order.account_id is None:
            # Local emulation/algorithm ownership and cancellation/denial can
            # precede venue account assignment. Only an already admitted CID
            # belongs to this explicitly bound frontend; this is not an ACK.
            return (order.status.name in ('INITIALIZED','EMULATED','RELEASED','DENIED','REJECTED','CANCELED') and self.frontend_broker is not None
                    and str(order.client_order_id) in self.frontend_broker._orders)
        return True

    def _native_route(self,client_id=None,params=None):
        self._assert_account_binding()
        if client_id is not None or params:
            raise ValueError('execution client/parameter overrides are unsupported; configure verified Node venue routing')

    def _bound_order(self,value,*,new=False):
        order=value if hasattr(value,'client_order_id') else self.cache.order(ClientOrderId(str(value)))
        if (order is None or order.strategy_id!=self.strategy_id
                or order.instrument_id not in self._instrument_ids.values()
                or (order.account_id is None and not new and not self._owns_native_order(order))
                or (order.account_id is not None and str(order.account_id)!=self._account_id)):
            raise ValueError('native order does not belong to the explicit account/strategy/instrument')
        return order

    def _bound_position(self,value,instrument_id=None):
        position=value if hasattr(value,'instrument_id') else self.cache.position(PositionId(str(value)))
        if (position is None or str(position.account_id)!=self._account_id
                or position.strategy_id!=self.strategy_id
                or position.instrument_id not in self._instrument_ids.values()
                or (instrument_id is not None and position.instrument_id!=instrument_id)):
            raise ValueError('native position does not belong to the explicit account/strategy/instrument')
        return position



    def _before_native_submission(self,order,position_id=None,client_id=None,params=None):
        self._native_route(client_id,params)
        self._bound_order(order,new=True)
        if position_id is not None:self._bound_position(position_id,order.instrument_id)
        if self._stopping and not order.is_reduce_only:
            raise RuntimeError('stopping host only permits native risk reduction')

    def modify_order(self,client_order_id,quantity=None,price=None,trigger_price=None,client_id=None,params=None):
        self._native_route(client_id,params)
        order=self._bound_order(client_order_id)
        if self._stopping:raise RuntimeError('stopping host refuses native order modifications')
        return super().modify_order(order.client_order_id,quantity=quantity,price=price,trigger_price=trigger_price)

    def modify_orders(self,updates,client_id=None,params=None):
        self._native_route(client_id,params)
        raise NotImplementedError('bulk modification is unsupported; use individually bound native orders')

    def cancel_order(self,client_order_id,client_id=None,params=None):
        self._native_route(client_id,params)
        order=self._bound_order(client_order_id)
        return super().cancel_order(order.client_order_id)

    def cancel_orders(self,client_order_ids,client_id=None,params=None):
        self._native_route(client_id,params)
        orders=[self._bound_order(value) for value in client_order_ids]
        return super().cancel_orders([order.client_order_id for order in orders])

    def cancel_all_orders(self,instrument_id,order_side=None,client_id=None,strategy_only=True,params=None):
        self._native_route(client_id,params)
        if not strategy_only or instrument_id not in self._instrument_ids.values():
            raise ValueError('bulk cancel requires this strategy and a bound instrument')
        for order in self.cache.orders_open(instrument_id=instrument_id,strategy_id=self.strategy_id):
            self._bound_order(order)
        return super().cancel_all_orders(instrument_id,order_side=order_side,strategy_only=True)

    def close_position(self,position,client_id=None,tags=None,time_in_force=None,reduce_only=None,quote_quantity=None,params=None):
        self._native_route(client_id,params)
        position=self._bound_position(position)
        if reduce_only is not True or quote_quantity:
            raise ValueError('native close requires explicit reduce_only and underlying quantity')
        return super().close_position(position,tags=tags,time_in_force=time_in_force,reduce_only=True)

    def close_all_positions(self,instrument_id,position_side=None,client_id=None,tags=None,time_in_force=None,reduce_only=None,quote_quantity=None,params=None):
        self._native_route(client_id,params)
        if instrument_id not in self._instrument_ids.values() or reduce_only is not True or quote_quantity:
            raise ValueError('native close requires a bound instrument and explicit reduce_only')
        for position in self.cache.positions_open(instrument_id=instrument_id,strategy_id=self.strategy_id):
            self._bound_position(position)
        return super().close_all_positions(instrument_id,position_side=position_side,tags=tags,time_in_force=time_in_force,reduce_only=True)

    def on_strategy_start(self):
        self._assert_account_binding()
        # Pending command callbacks cannot borrow rc7 Strategy getters. Cache
        # the already-assigned immutable identity before any command is issued.
        self._callback_strategy_id = self.strategy_id
        self._callback_clock = self.clock
        self._bar_check_timers = {}
        self._callback_cache = self.cache
        self._callback_portfolio = self.portfolio if self._account_provider is not None else None
        if self._offline_test:
            try:self.clock.set_time(self.clock.timestamp_ns())
            except ValueError:raise RuntimeError('offline_test requires the native virtual clock') from None
        recovered={}
        for oid,payload in self._recovering.items():
            native=self.lookup_order(oid)
            if native is None or native.strategy_id!=self.strategy_id:
                raise RuntimeError('durable admitted intent lacks authoritative native reconciliation; startup blocked')
            recovered[oid]=(native,payload)
        existing=_unclosed_native_orders(self._callback_cache,self._callback_strategy_id)
        for order in existing:
            if str(order.client_order_id) in recovered:continue
            projection=self._spawn_projection(order)
            if projection is not None and (projection['client_oid'] in recovered
                    or projection['status'] in ('FILLED','CANCELED','EXPIRED','REJECTED','DENIED','VOIDED')):
                continue  # Validated cache group, not an orphan independent intent.
            raise RuntimeError('native pending orders lack durable frontend intent; startup blocked')
        if set(self._feeds)!=set(self._instrument_ids):
            raise ValueError('startup requires explicit instrument bindings')
        for iid in self._instrument_ids.values():
            if self.cache.instrument(iid) is None:
                raise ValueError('required instrument is not loaded')
        provided_broker=getattr(self,'_provided_broker',None)
        self.frontend_broker=provided_broker if provided_broker is not None else NautilusBroker(self)
        self.frontend_broker.bind_gateway(self)
        self.frontend=StrategyRuntime(self._business_cls,feeds=self._feeds,
                                   broker=self.frontend_broker,parameters=self._parameters,
                                   strategy_args=getattr(self,'_strategy_args',()),
                                   capacity=getattr(self,'_buffer_capacity',512),
                                   feed_names=getattr(self,'_feed_names',None),
                                   feed_metadata={**_feed_period_metadata(self._instrument_ids, self._subscriptions),
                                                  **getattr(self, '_live_feed_metadata', {})})
        from .observers import attach_runtime_observer
        attach_runtime_observer(self.frontend,getattr(self,'_runtime_observer_factory',None))
        for oid,(native,payload) in recovered.items():
            from tradelearn.core.broker_contracts import OrderRequest
            request=OrderRequest(**payload['request'])
            self.validate_recovery(request,native,group=payload.get('group'),
                native_controls=payload.get('native_controls'))
            projection=self._spawn_projection(native)
            if projection is not None and projection['status'] in ('FILLED','CANCELED','EXPIRED','REJECTED','DENIED','VOIDED'):
                self._schedule(self._persist_terminal_evidence(oid,
                    dict(status=projection['status'],filled_qty=str(projection['filled_qty']),
                         native_event_id=str(native.last_event.event_id),native_evidence=projection['native_evidence'])))
            elif native.is_closed and projection is None:
                self._schedule(self._persist_terminal(oid,native))
            self.frontend_broker.restore(request,self.frontend.feeds[request.symbol],native)
        self.frontend.start()
        components = getattr(self.frontend,'_cerebro_components',None)
        if components is not None:components.start()
        from .account_refresh import NativeAccountRefresh
        self._account_refresh=NativeAccountRefresh(self)
        self._account_refresh.start()
        for symbol,mode in self._feeds.items():
            if mode=='tick':self.subscribe_quotes(self._instrument_ids[symbol])
        for bar_type in self._subscriptions:self.subscribe_bars(bar_type)

    def on_strategy_stop(self):
        self._stopping=True
        refresh=getattr(self,'_account_refresh',None)
        if refresh is not None:refresh.stop()
        clock=getattr(self,'_callback_clock',None)
        for name in getattr(self,'_bar_check_timers',{}).values():
            if clock is not None and name in clock.timer_names():clock.cancel_timer(name)
        self._bar_check_timers={}
        try:
            if self.frontend is not None:
                primary = None
                try:
                    self.frontend.stop()
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    components=getattr(self.frontend,'_cerebro_components',None)
                    if components is not None:
                        try:components.finish()
                        except BaseException as error:
                            if primary is None:raise
                            from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                            _safe_cleanup_diagnostic(primary,error)
        finally:
            _RuntimeBridge._release_runtime_owner(self)

    def _release_runtime_owner(self):
        """Break Python back-references only after business stop, on its owner."""
        broker = getattr(self, 'frontend_broker', None)
        if broker is not None:
            broker._release_gateway(self)
        logger = getattr(self, 'logger', None)
        if logger is not None and getattr(logger._monotonic_ns, '__self__', None) is self:
            # The SDK method delegates to this exact shared clock. A retained
            # closed logger must not retain an unsendable native Strategy.
            from tradelearn.core.observation_clock import monotonic_ns
            logger._monotonic_ns = monotonic_ns

    def _symbol(self,iid):
        for symbol,bound in self._instrument_ids.items():
            if iid==bound:return symbol
        raise ValueError('unbound instrument event')

    def on_strategy_quote(self,quote):
        try:
            self.frontend.step_market(self._symbol(quote.instrument_id),quote,kind='tick')
        finally:
            if self.frontend.failed:self._stopping=True

    def _request_monitor_account(self):
        refresh=getattr(self,'_account_refresh',None)
        if refresh is not None and refresh.active:
            return refresh.request()
        # Before activation, or for untimed custom providers, retain the SDK's
        # existing observation request policy and configuration unchanged.
        inherited=getattr(super(),'_request_monitor_account',None)
        return inherited() if inherited is not None else False

    def _notify_cashvalue(self):
        from tradelearn.backtest.strategy import Strategy as CoreStrategy
        if type(self.frontend.strategy).notify_cashvalue is CoreStrategy.notify_cashvalue:
            return
        values = self.account()  # One authoritative observation for both fields.
        cash, equity = values.get('cash'), values.get('equity')
        if any(isinstance(value,bool) or value is None or not math.isfinite(float(value))
               for value in (cash,equity)):
            raise ValueError('authoritative finite cash/equity snapshot required')
        self.frontend.invoke_business('notify_cashvalue',float(cash),float(equity))

    def _schedule_bar_checks(self):
        clock=getattr(self,'_callback_clock',None)
        if clock is None:return
        for index,(symbol,plan) in enumerate(getattr(self,'_live_bar_views',{}).items()):
            if not callable(getattr(plan,'next_check_ns',None)):continue
            name=f'__tl_bar_check_{index}'
            if name in clock.timer_names():clock.cancel_timer(name)
            self._bar_check_timers.pop(symbol,None)
            deadline=plan.next_check_ns()
            if deadline is None:continue
            self._bar_check_timers[symbol]=name
            # Past-due completion is checked on the next native clock turn;
            # its logical label and original source ns remain untouched.
            deadline=max(int(deadline),int(clock.timestamp_ns())+1)
            clock.set_time_alert_ns(name,deadline,
                callback=lambda event,symbol=symbol:self._on_bar_check(symbol,event))

    def _on_bar_check(self,symbol,event):
        if self._stopping or self.frontend.failed:return
        import pandas as pd
        self._bar_check_timers.pop(symbol,None)
        try:
            plan=self._live_bar_views[symbol]
            projected=plan.check(int(self._callback_clock.timestamp_ns()))
            if projected is not None:
                label,values,source_ns,count=projected
                dispatch=not plan.clone
                if dispatch:self._notify_cashvalue()
                self.frontend._step_market_projections([(symbol,
                    dict(values,ts=pd.Timestamp(label,unit='ns',tz='UTC')),count)],
                    source_ns,dispatch=dispatch,completion_ns=int(self._callback_clock.timestamp_ns()))
                components=getattr(self.frontend,'_cerebro_components',None)
                if dispatch and components is not None:components.after_market()
            self._schedule_bar_checks()
        except BaseException as error:
            self._worker_error=error
            self.frontend._failed=True
            self.frontend.cleanup(error)
            self._stopping=True
            callback=getattr(self,'_failure_callback',None)
            if callback is not None:
                try:callback()
                except BaseException as secondary:
                    from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                    _safe_cleanup_diagnostic(error,secondary)
            raise

    def on_strategy_bar(self,bar):
        if self._stopping or self.frontend.failed:return
        import pandas as pd
        try:
            row = dict(open=number(bar.open),high=number(bar.high),low=number(bar.low),
                       close=number(bar.close),volume=number(bar.volume))
            stamp = int(bar.ts_event)
            routes = getattr(self,'_source_bar_types',None)
            symbols = ([symbol for symbol,source in routes.items() if source == bar.bar_type]
                       if routes is not None else [self._symbol(bar.bar_type.instrument_id)])
            # An adjusted replay can expose the old logical bar before the
            # current source row. Keep raw/single-turn views on the final phase,
            # so an earlier revision cannot see a future raw observation.
            outputs = []
            for symbol in symbols:
                plan = getattr(self,'_live_bar_views',{}).get(symbol)
                if plan is None:output=[(stamp,row,stamp)]
                elif callable(getattr(plan,'push_observed',None)):
                    clock=getattr(self,'_callback_clock',None)
                    observed=int(clock.timestamp_ns()) if clock is not None else int(bar.ts_init)
                    output=plan.push_observed(stamp,row,observed)
                else:output=plan.push(stamp,row)
                outputs.append((symbol,output))
            phases = max((len(output) for _,output in outputs),default=0)
            deliveries = [[] for _ in range(phases)]
            for symbol,output in outputs:
                offset = phases-len(output)
                for index,projected in enumerate(output):
                    if projected is None:continue
                    label, values, available = projected[:3]
                    logical_count = projected[3] if len(projected) == 4 else None
                    deliveries[offset+index].append((symbol,
                        dict(values,ts=pd.Timestamp(label,unit='ns',tz='UTC')),logical_count))
            for delivery in deliveries:
                self._notify_cashvalue()
                self.frontend._step_market_projections(delivery,stamp)
                components=getattr(self.frontend,'_cerebro_components',None)
                if components is not None:components.after_market()
            self._schedule_bar_checks()
        except BaseException as error:
            self.frontend.cleanup(error)
            callback = getattr(self, '_failure_callback', None)
            if callback is not None:
                try:
                    callback()
                except BaseException as secondary:
                    from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                    _safe_cleanup_diagnostic(error,secondary)
            raise
        finally:
            if self.frontend.failed:self._stopping=True

    def _make_order(self,request,controls):
        self._assert_account_binding()
        from .native_controls import CONTROL_NAMES,factory_control_kwargs
        if set(controls)-CONTROL_NAMES:
            raise ValueError('unsupported execution controls; configure verified Node venue routing')
        position_id=controls.get('position_id')
        if position_id is not None:self._bound_position(position_id,self._instrument_ids[request.symbol])
        iid=self._instrument_ids[request.symbol]
        instrument=self.cache.instrument(iid)
        if instrument is None:raise ValueError('instrument unavailable')
        quantity=instrument.make_qty(request.qty)
        if Decimal(str(quantity)) != Decimal(str(request.qty)):
            raise ValueError('quantity would be rounded by venue precision')
        if self.cache.order(ClientOrderId(request.client_oid)) is not None:
            raise ValueError('native client order ID already exists')
        kwargs=dict(instrument_id=iid,order_side=OrderSide.BUY if request.side=='buy' else OrderSide.SELL,
                    quantity=quantity,time_in_force={'day':TimeInForce.DAY,'gtc':TimeInForce.GTC,'ioc':TimeInForce.IOC,'fok':TimeInForce.FOK,'gtd':TimeInForce.GTD}[request.tif],
                    client_order_id=ClientOrderId(request.client_oid))
        kwargs.update(factory_control_kwargs(request,controls,instrument))
        expiry = getattr(request, 'expire_time_ns', None)
        if request.tif == 'gtd':
            if request.order_type == 'market' or type(expiry) is not int or not 0 < expiry <= 2**64-1:
                raise ValueError('valid GTD expiry and a non-Market order required')
            kwargs['expire_time'] = expiry  # Pinned NA 2rc7 public factory accepts UTC ns integers.
        elif expiry is not None:
            raise ValueError('expiry requires GTD time-in-force')
        def exact_price(value):
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError('positive finite native price required')
            price = instrument.make_price(value)
            if Decimal(str(price)) != Decimal(str(value)):
                raise ValueError('price would be rounded by venue precision')
            return price
        if request.order_type=='limit':
            native=self.order_factory.limit(price=exact_price(request.limit_price),**kwargs)
        elif request.order_type=='stop':
            native=self.order_factory.stop_market(trigger_price=exact_price(request.stop_price),**kwargs)
        elif request.order_type=='stop_limit':
            native=self.order_factory.stop_limit(price=exact_price(request.limit_price),
                trigger_price=exact_price(request.stop_price),**kwargs)
        elif request.order_type=='stop_trail_limit':
            trigger, limit, limit_offset, offset, offset_type = _trailing_limit_parameters(request)
            native=self.order_factory.trailing_stop_limit(
                price=exact_price(limit), limit_offset=limit_offset,
                trigger_price=exact_price(trigger), trailing_offset=offset,
                trailing_offset_type=offset_type, **kwargs)
        elif request.order_type=='stop_trail':
            trigger, offset, offset_type = _trailing_parameters(request)
            native=self.order_factory.trailing_stop_market(
                trigger_price=exact_price(trigger), trailing_offset=offset,
                trailing_offset_type=offset_type, **kwargs)
        elif request.order_type=='market':native=self.order_factory.market(**kwargs)
        else:raise ValueError('unsupported order type')
        return native,controls

    def _make_group(self, entries):
        if getattr(self, '_adapter_supports_groups', None) is False:
            raise ValueError('selected adapter does not support native linked-order groups')
        if self._native_strategy_config.manage_contingent_orders:
            raise ValueError('native linked groups cannot overlay NA-managed contingencies; separately qualified emulation is not enabled')
        if not entries:
            return []
        if len(entries) < 2:
            raise ValueError('a linked native group needs at least two orders')
        requests = [entry['request'] for entry in entries]
        for request in requests:
            _require_qualified_expiry(request)
        ids = [request.client_oid for request in requests]
        if len(set(ids)) != len(ids) or len({request.symbol for request in requests}) != 1:
            raise ValueError('native group requires unique IDs and one instrument')
        positions = {str(entry['controls']['position_id']) for entry in entries
                     if entry['controls'].get('position_id') is not None}
        if len(positions) > 1:
            raise ValueError('native linked group cannot mix position routes')
        native = [self._make_order(entry['request'],dict(entry['controls']))[0] for entry in entries]
        children = {cid:[] for cid in ids}
        peers = {cid:set() for cid in ids}
        for cid,entry in zip(ids,entries):
            parent,oco = entry['parent'],entry['oco']
            if parent is not None:
                if parent not in ids or ids.index(parent) >= ids.index(cid):
                    raise ValueError('parent must precede its child in the same group')
                children[parent].append(cid)
            if oco is not None:
                if oco not in ids or oco == cid:
                    raise ValueError('OCO peer must belong to the same pending group')
                peers[cid].add(oco);peers[oco].add(cid)
        # OCO links are an equivalence class, not independent cancellation callbacks.
        for cid in ids:
            frontier = set(peers[cid])
            while frontier:
                peer = frontier.pop()
                addition = peers[peer] - peers[cid] - {cid}
                peers[cid].update(addition);frontier.update(addition)
        list_id = str(self.order_factory.generate_order_list_id())
        result = []
        for cid,entry,order in zip(ids,entries,native):
            if children[cid] and peers[cid]:
                raise ValueError('a native order cannot combine OTO parent and OCO sibling roles')
            linked = children[cid] or sorted(peers[cid])
            if not linked and entry['parent'] is None:
                raise ValueError('unlinked order cannot enter a native linked group')
            row = type(order).to_dict(order)
            row.update(order_list_id=list_id,parent_order_id=entry['parent'],
                # NA's bracket default is OUO: partial exits resize protection instead of cancelling it.
                contingency_type='OTO' if children[cid] else (('OUO' if entry['parent'] is not None else 'OCO') if peers[cid] else None),
                linked_order_ids=linked or None)
            result.append(type(order).from_dict(row))
        return result

    def validate_group(self,entries):
        return self._make_group(entries)

    def submit_group(self,entries):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        orders = self._make_group(entries)
        if self._stopping or self.frontend.failed:
            raise RuntimeError('execution host is stopping or runtime failed')
        positions = [entry['controls'].get('position_id') for entry in entries
                     if entry['controls'].get('position_id') is not None]
        controls = {'position_id':positions[0]} if positions else {}
        if self._offline_test:
            self.submit_order_list(orders,**controls)
            return
        self._schedule(self._admit_group(entries,orders,controls))

    async def _admit_group(self,entries,orders,controls):
        try:
            async with self._io_lock:
                for entry,order in zip(entries,orders):
                    group = {name:type(order).to_dict(order)[name] for name in
                             ('order_list_id','parent_order_id','linked_order_ids','contingency_type')}
                    await asyncio.to_thread(self.intent_journal.prepare,entry['request'].client_oid,
                        {'request':asdict(entry['request']),
                         'controls':{key:str(value) for key,value in entry['controls'].items()},
                         'group':group,
                         'native_controls':_native_route_proof(order, entry['controls'])})
            if self._stopping or self.frontend.failed:
                for order in orders:self.frontend_broker._unresolved.add(str(order.client_order_id))
                return
            self._assert_account_binding()
            self.submit_order_list(orders,**controls)
        except BaseException:
            for order in orders:self.frontend_broker._unresolved.add(str(order.client_order_id))
            raise

    def validate(self,request,**controls):
        _require_qualified_expiry(request)
        self._make_order(request,dict(controls))

    def submit(self,request,**controls):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        _require_qualified_expiry(request)
        native,controls=self._make_order(request,dict(controls))
        if self._stopping or self.frontend.failed:
            self._stopping=True
            raise RuntimeError('execution host is stopping or runtime failed')
        if self._offline_test:
            self.submit_order(native,**controls)
            return
        self._schedule(self._admit_and_submit(request,native,controls))

    def _schedule(self,coroutine):
        task=asyncio.create_task(coroutine)
        self._tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                self._stopping=True
                self._worker_error=done.exception()
                if self.frontend_broker is not None:
                    self.frontend_broker._unresolved.add('durable-worker-failure')
                callback=getattr(self,'_failure_callback',None)
                if callback is not None:
                    callback()
        task.add_done_callback(finished)
        return task

    async def _admit_and_submit(self,request,native,controls):
        try:
            async with self._io_lock:
                await asyncio.to_thread(self.intent_journal.prepare,request.client_oid,
                                        {'request':asdict(request),'controls':{key:str(value) for key,value in controls.items()},
                                         'native_controls':_native_route_proof(native, controls)})
            if self._stopping or self.frontend.failed:
                self._stopping=True
                self.frontend_broker._unresolved.add(request.client_oid)
                return
            self._assert_account_binding()
            self.submit_order(native,**controls)
        except Exception:
            self.frontend_broker._unresolved.add(request.client_oid)
            order=self.frontend_broker._orders.get(request.client_oid)
            if order is not None:
                order.info['submission_state']='UNKNOWN'
                self.frontend_broker._notify(self.frontend.strategy,order)
            raise

    def _persist_terminal(self,oid,order):
        # Snapshot on the callback owner before queuing the async write. Native
        # orders can be reopened by an authoritative correction in the meantime.
        evidence={'status':order.status.name,'filled_qty':str(order.filled_qty),'native_event_id':str(order.last_event.event_id)}
        return self._persist_terminal_evidence(oid,evidence)

    async def _persist_correction(self,oid,evidence):
        async with self._io_lock:
            await asyncio.to_thread(self.intent_journal.corrected,oid,evidence)

    async def _persist_terminal_evidence(self,oid,evidence):
        async with self._io_lock:
            await asyncio.to_thread(self.intent_journal.terminal,oid,evidence)

    def _retained_native_voids(self,order):
        from .order_corrections import retained_native_voids
        return retained_native_voids(order,_RuntimeBridge._read_cache(self))

    def _spawn_projection(self,order):
        from .algorithm_facts import spawn_projection
        return spawn_projection(_RuntimeBridge._read_cache(self),order,
            strategy_id=_RuntimeBridge._strategy_identity(self),account_id=self._account_id,
            instruments=self._instrument_ids.values())

    async def drain(self):
        self._stopping=True
        errors=[]
        while self._tasks:
            tasks=tuple(self._tasks)
            results=await asyncio.gather(*tasks,return_exceptions=True)
            self._tasks.difference_update(tasks)
            errors.extend(result for result in results if isinstance(result,BaseException))
        if self.intent_journal is not None:
            try:
                await asyncio.to_thread(self.intent_journal.close)
            except BaseException as error:
                errors.append(error)
        if errors:
            from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
            for error in errors[1:]:_safe_cleanup_diagnostic(errors[0],error)
            raise errors[0]
        if self._worker_error is not None:raise RuntimeError('durable intent worker failed') from self._worker_error

    def cancel(self,client_oid):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        order=self.cache.order(ClientOrderId(client_oid))
        self._assert_account_binding()
        if order is None or not self._owns_native_order(order):
            raise ValueError('cancel requires this strategy/account native order')
        self.cancel_order(order)

    @staticmethod
    def _validate_group_recovery(expected,native):
        fields=('order_list_id','parent_order_id','linked_order_ids','contingency_type')
        if not isinstance(expected,dict) or set(expected)!=set(fields):
            raise ValueError('invalid durable linked-order metadata')
        actual=type(native).to_dict(native)
        for name in fields:
            left,right=expected[name],actual[name]
            if name=='linked_order_ids':
                left,right=sorted(left or []),sorted(right or [])
            if left!=right:
                raise ValueError('native linked-order recovery differs from durable group')

    def validate_recovery(self,request,native,*,group=None,native_controls=None):
        from .native_controls import emulation_admission
        admission=emulation_admission(native)
        if admission is not None and native_controls is None:
            raise ValueError('native emulation recovery requires durable admission controls')
        reference=admission if admission is not None else native
        admitted_quantity=reference.quantity
        if (getattr(native,'exec_algorithm_id',None) is not None
                and getattr(native,'exec_spawn_id',None)==native.client_order_id):
            # NA reduces primary quantity as it transfers size into spawned
            # children. Its initialization still proves the admitted request.
            admitted_quantity=native.init_event.quantity
        if group is not None:
            self._validate_group_recovery(group,native)
            # Only an intact native initialization proves a linked child's original size.
            admitted_quantity=native.init_event.quantity
        types={'market':'MARKET','limit':'LIMIT','stop':'STOP_MARKET','stop_limit':'STOP_LIMIT','stop_trail':'TRAILING_STOP_MARKET','stop_trail_limit':'TRAILING_STOP_LIMIT'}
        if request.order_type not in types:
            raise ValueError('unmapped durable order type requires reconciliation')
        expected_type=types[request.order_type]
        if (native.strategy_id!=self.strategy_id or str(native.client_order_id)!=request.client_oid
            or native.instrument_id!=self._instrument_ids[request.symbol]
            or (admission.order_side.name if admission is not None else native.side.name)!=request.side.upper() or reference.order_type.name!=expected_type
            or Decimal(str(admitted_quantity))!=Decimal(str(request.qty))
            or reference.time_in_force.name!=request.tif.upper()):
            raise ValueError('native recovery order differs from durable strategy request')
        if expected_type in ('LIMIT','STOP_LIMIT') and Decimal(str(reference.price))!=Decimal(str(request.limit_price)):
            raise ValueError('native recovery limit price differs from request')
        if expected_type in ('STOP_MARKET','STOP_LIMIT') and Decimal(str(reference.trigger_price))!=Decimal(str(request.stop_price)):
            raise ValueError('native recovery trigger price differs from request')
        if expected_type in ('TRAILING_STOP_MARKET','TRAILING_STOP_LIMIT'):
            trigger, offset, offset_type = _trailing_parameters(request)
            initial = native.init_event
            if (reference.trailing_offset != offset or reference.trailing_offset_type != offset_type
                    or initial.trigger_price is None or Decimal(str(initial.trigger_price)) != trigger):
                raise ValueError('native recovery trailing admission differs from request')
            if expected_type == 'TRAILING_STOP_LIMIT':
                _, limit, limit_offset, _, _ = _trailing_limit_parameters(request)
                if (reference.limit_offset != limit_offset or initial.price is None
                        or Decimal(str(initial.price)) != limit):
                    raise ValueError('native recovery trailing limit admission differs from request')
        expiry = getattr(request, 'expire_time_ns', None)
        if request.tif == 'gtd' and (expiry is None or native.expire_time_ns != expiry):
            raise ValueError('native recovery expiry differs from durable request')
        if native_controls is not None:
            from .native_controls import CONTROL_NAMES,validate_extra_control_proof
            if (not isinstance(native_controls,dict)
                    or not {'reduce_only','position_id'}.issubset(native_controls)
                    or set(native_controls) - CONTROL_NAMES
                    or type(native_controls['reduce_only']) is not bool):
                raise ValueError('invalid durable native route proof')
            admitted_post_only = native_controls.get('post_only', False)
            if (type(admitted_post_only) is not bool
                    or admitted_post_only != (admission.post_only if admission is not None else getattr(native, 'is_post_only', False))):
                raise ValueError('native recovery post-only flag differs from admitted route')
            validate_extra_control_proof(native_controls,request,native)
            if native_controls['reduce_only'] != (admission.reduce_only if admission is not None else native.is_reduce_only):
                raise ValueError('native recovery reduce-only flag differs from admitted route')
            requested = native_controls['position_id']
            if requested is not None:
                actual = native.position_id
                if actual is None and callable(getattr(self.cache,'position_id',None)):
                    actual = self.cache.position_id(native.client_order_id)
                if not isinstance(requested,str) or not requested or str(actual) != requested:
                    raise ValueError('native recovery position route differs from admitted route')
        binding=self._account_id
        if binding is not None:
            if native.account_id is None:
                # Before venue submission there is no broker account ACK. Only
                # a durable admitted route plus this verified single-account
                # host may restore the local pending object, never infer ACK.
                if native.status.name not in ('INITIALIZED','EMULATED','RELEASED','DENIED','REJECTED','CANCELED') or native_controls is None:
                    raise ValueError('native recovery account differs from verified binding')
                self._assert_account_binding()
            elif str(native.account_id)!=binding:
                raise ValueError('native recovery account differs from verified binding')

    def lookup_order(self,client_oid):
        # A native object is an existence/recovery witness, never a synthetic ACK.
        return self.cache.order(ClientOrderId(client_oid))

    def close_controls(self,symbol,*,position_id=None):
        self._assert_account_binding()
        iid=self._instrument_ids[symbol]
        if position_id is not None:
            native=self._bound_position(position_id,iid)
        else:
            positions=[position for position in self.cache.positions_open(instrument_id=iid,strategy_id=self.strategy_id)
                       if str(position.account_id)==self._account_id]
            if not positions:return None
            if len(positions)!=1:
                raise ValueError('hedged close requires an explicit position_id')
            native=self._bound_position(positions[0],iid)
        quantity=number(native.signed_qty)
        if not math.isfinite(quantity) or quantity==0:
            raise ValueError('owned open position quantity required for close')
        return dict(position_id=native.id,size=abs(quantity),side='sell' if quantity>0 else 'buy')

    def position(self,symbol):
        self._assert_account_binding()
        identity = _RuntimeBridge._strategy_identity(self)
        positions=[p for p in _RuntimeBridge._read_cache(self).positions_open(instrument_id=self._instrument_ids[symbol],strategy_id=identity)
                   if str(p.account_id)==self._account_id and p.strategy_id==identity]
        if len(positions)>1:raise ValueError('hedged positions need explicit position_id routing')
        if not positions:return Position()
        native=positions[0]
        size=number(native.signed_qty)
        return Position(size,number(native.avg_px_open))

    def cash_valuation(self,requests):
        """Ephemeral signed asset views from the same native position facts.

        One BT stocklike Data requests raw notional even on a margin account;
        that is not account equity. Cash multi/market views are separately
        qualified. Selected Data close is the explicit valuation price; native
        Position owns quantity/instrument/currency and the notional formula.
        No account or position ledger is reconstructed here.
        """
        from nautilus_trader.model import AccountType,Price
        if requests is not None and any(symbol not in self._instrument_ids for symbol,_ in requests):
            raise ValueError('native subset route differs from bound instruments')
        account=self._assert_account_binding()
        if account.account_type!=AccountType.CASH and (requests is None or len(requests)!=1):
            raise ValueError('native margin scoped valuation requires its distinct capital source')
        balances=account.balances_total()
        if len(balances)!=1:
            raise ValueError('native cash subset requires one authoritative currency; no FX inference')
        currency,cash=next(iter(balances.items()))
        cache=_RuntimeBridge._read_cache(self)
        identity=_RuntimeBridge._strategy_identity(self)
        if requests is None:
            portfolio=getattr(self,'_callback_portfolio',None)
            if portfolio is None:portfolio=self.portfolio
            equities=portfolio.equity(account_id=account.id)
            if set(equities)!={currency}:
                raise ValueError('native cash equity currency differs; no FX inference')
            positions=cache.positions_open(account_id=account.id)
            for venue in {position.instrument_id.venue for position in positions}:
                if portfolio.missing_price_instruments(venue,account_id=account.id):
                    raise ValueError('native cash equity has unpriced positions')
            equity=number(equities[currency]);cash=number(cash)
            if not math.isfinite(equity) or not math.isfinite(cash):
                raise ValueError('finite native cash/equity required')
            return cash,[equity-cash]
        observations=[]
        for symbol,close in requests:
            value=Decimal(str(close))
            if not value.is_finite() or value<=0:
                raise ValueError('positive finite selected native valuation price required')
            price=Price.from_str(format(value,'f'))
            if price.as_decimal()!=value:
                raise ValueError('selected native valuation price loses precision')
            positions=cache.positions_open(instrument_id=self._instrument_ids[symbol],account_id=account.id)
            if any(position.strategy_id!=identity for position in positions):
                raise ValueError('native subset includes a foreign strategy position')
            if len(positions)>1:
                raise ValueError('hedged native cash subset requires an explicit valuation source')
            if not positions:
                observations.append(0.)
                continue
            position=positions[0]
            notional=position.notional_value(price)
            if notional.currency!=currency:
                raise ValueError('native subset currency differs from cash; no FX inference')
            sign=-1. if number(position.signed_qty)<0 else 1.
            observations.append(sign*number(notional))
        return number(cash),observations

    def account(self):
        # Never infer portfolio equity from wallet/buying power or a quote currency.
        if self._account_provider is None:raise ValueError('explicit authoritative account projection required')
        portfolio = getattr(self, '_callback_portfolio', None)
        if portfolio is None:
            portfolio = self.portfolio
        return self._account_provider(_RuntimeBridge._read_cache(self),portfolio)

    def _project_order(self,event):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        order=_RuntimeBridge._read_cache(self).order(event.client_order_id)
        if order is None:raise RuntimeError('order event has no authoritative native cache snapshot')
        if not self._owns_native_order(order,event):return
        status=order.status.name
        if (order.account_id is None and getattr(order,'exec_algorithm_id',None) is not None
                and type(event).__name__=='OrderUpdated'):
            # Native algorithm transfers primary quantity into children before
            # caching each child. This internal transfer is not a venue size
            # amendment of the admitted BT parent. Observe the atomic group on
            # subsequent child execution events, never the intermediate zero.
            return
        if self.frontend_broker is None:
            return  # Startup recovery validates/cache-projects before business begins.
        from .order_fees import cumulative_native_commission
        commission=cumulative_native_commission(order)
        from nautilus_trader.model import OrderFillVoided
        correction=(order,event) if type(event) is OrderFillVoided else None
        projection=self._spawn_projection(order)
        if projection is not None:
            oid=projection.pop('client_oid')
            # Only an existing admitted parent may receive this native group.
            if oid not in self.frontend_broker._orders:
                raise RuntimeError('native execution-spawn lacks admitted frontend primary')
            current=self.frontend_broker._orders[oid]
            if (self._instrument_ids.get(getattr(current.data,'_symbol',current.data._name))!=order.instrument_id
                    or current.isbuy()!=(order.side==OrderSide.BUY)):
                raise ValueError('native execution-spawn differs from admitted frontend scope')
            if (current.info.get('native_evidence')==projection['native_evidence']
                    and type(event).__name__ not in ('OrderModifyRejected','OrderCancelRejected')):
                return  # Nested native callbacks can expose the same completed cache view twice.
            if correction is not None:
                from .order_corrections import native_void_evidence
                correction=(order,event,_RuntimeBridge._read_cache(self))
                evidence=native_void_evidence(correction)
                if self.intent_journal is not None:self._schedule(self._persist_correction(oid,evidence))
            elif projection['status'] in ('FILLED','CANCELED','EXPIRED','REJECTED','DENIED','VOIDED') and self.intent_journal is not None:
                self._schedule(self._persist_terminal_evidence(oid,
                    dict(status=projection['status'],filled_qty=str(projection['filled_qty']),
                         native_event_id=str(event.event_id),native_evidence=projection['native_evidence'])))
            from .notification_evidence import native_event_evidence
            applied=self.frontend_broker.apply(oid,**projection,event_id=str(event.event_id),
                native_ts_ns=int(event.ts_event),native_event=native_event_evidence(event),native_correction=correction)
            components=getattr(self.frontend,'_cerebro_components',None)
            if applied and components is not None:
                components.native_order(event,self.frontend_broker._orders[oid])
            return
        if correction is not None and self.intent_journal is not None:
            from .order_corrections import native_void_evidence
            self._schedule(self._persist_correction(str(event.client_order_id),native_void_evidence(correction)))
        # Native terminal evidence belongs to durable execution, independently of
        # whether the failed/stopped business runtime can accept a notification.
        if order.is_closed and self.intent_journal is not None and correction is None:
            self._schedule(self._persist_terminal(str(event.client_order_id),order))
        from .notification_evidence import notification_evidence, native_event_evidence
        applied=self.frontend_broker.apply(str(event.client_order_id),status=status,
            filled_qty=number(order.filled_qty),avg_price=order.avg_px,
            event_id=str(event.event_id),commission=commission,native_ts_ns=int(event.ts_event),
            native_evidence=notification_evidence(order),total_qty=number(order.quantity),
            native_event=native_event_evidence(event),native_correction=correction)
        components=getattr(self.frontend,'_cerebro_components',None)
        if applied and components is not None:
            components.native_order(event,self.frontend_broker._orders[str(event.client_order_id)])

    def _project_pending_order(self, event, status):
        """Pending callbacks run while rc7 mutably borrows its cache.

        The event is authoritative for this control transition only. Reuse the
        last projected cumulative execution facts; never query/reconstruct the
        native cache or defer this callback past a later rejection/fill.
        """
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        from .notification_evidence import native_event_evidence
        assert_observer_allows_mutation()
        if self.frontend_broker is None:
            return  # Startup reconciliation still validates the authoritative cache.
        if (str(event.account_id) != self._account_id
                or event.strategy_id != self._callback_strategy_id
                or event.instrument_id not in self._instrument_ids.values()):
            return
        oid = str(event.client_order_id)
        order = self.frontend_broker._orders.get(oid)
        if order is None:
            # A native algorithm child belongs to its admitted primary. Use the
            # captured cache handle (not the mutably borrowed Strategy getter)
            # and the same fully scoped native aggregate, never a local fill sum.
            native=_RuntimeBridge._read_cache(self).order(event.client_order_id)
            if (native is not None and getattr(native,'exec_spawn_id',None) is not None
                    and str(native.exec_spawn_id) in self.frontend_broker._orders):
                return self._project_order(event)
            raise RuntimeError('pending native event lacks recovered frontend intent')
        if self._instrument_ids.get(getattr(order.data,'_symbol',order.data._name)) != event.instrument_id:
            raise ValueError('pending native event differs from admitted instrument')
        price, fee = order.executed.price, order.executed.comm
        applied = self.frontend_broker.apply(oid, status=status,
            filled_qty=abs(order.executed.size), total_qty=abs(order.size),
            avg_price=price if math.isfinite(price) else None,
            commission=fee if math.isfinite(fee) else None,
            event_id=str(event.event_id), native_ts_ns=int(event.ts_event),
            native_event=native_event_evidence(event))
        components = getattr(self.frontend, '_cerebro_components', None)
        if applied and components is not None:
            components.native_order(event, order)

    def on_strategy_order_pending_update(self,event):self._project_pending_order(event, 'PENDING_UPDATE')
    def on_strategy_order_pending_cancel(self,event):self._project_pending_order(event, 'PENDING_CANCEL')
    def on_strategy_order_modify_rejected(self,event):self._project_order(event)
    def on_strategy_order_cancel_rejected(self,event):self._project_order(event)
    def on_strategy_order_updated(self,event):self._project_order(event)
    def on_strategy_order_triggered(self,event):self._project_order(event)
    def on_strategy_order_accepted(self,event):self._project_order(event)
    def on_strategy_order_rejected(self,event):self._project_order(event)
    def on_strategy_order_denied(self,event):self._project_order(event)
    def on_strategy_order_canceled(self,event):self._project_order(event)
    def on_strategy_order_expired(self,event):self._project_order(event)
    def on_strategy_fill(self,event):self._project_order(event)
    def on_strategy_order_fill_voided(self,event):
        from .order_corrections import native_void_payload
        try:
            raw=native_void_payload(event)
            broker=getattr(self,'frontend_broker',None)
            seen=getattr(broker,'_void_event_payloads',{})
            for (oid,eid),previous in seen.items():
                if eid!=str(event.event_id) or previous.get('client_order_id')!=str(event.client_order_id):continue
                # A different account/strategy callback is foreign, not an owned replay.
                if any(previous.get(key)!=raw.get(key) for key in ('account_id','strategy_id','instrument_id','trader_id')):return
                if previous!=raw:raise ValueError('conflicting native correction replay')
                return  # An exact old callback must not reinterpret a newer cache.
            self._project_order(event)
        except BaseException as error:
            self._worker_error=error;self._stopping=True
            frontend=getattr(self,'frontend',None)
            if frontend is not None:
                frontend._failed=True
                cleanup=getattr(frontend,'cleanup',None)
                if callable(cleanup):cleanup(error)
            callback=getattr(self,'_failure_callback',None)
            if callback is not None:
                try:callback()
                except BaseException as secondary:
                    from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                    _safe_cleanup_diagnostic(error,secondary)
            raise

    def _project_position(self,event,closed=False):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        from tradelearn.backtest.clock import date2num
        from .position_facts import native_trade_economics
        import pandas as pd
        assert_observer_allows_mutation()
        if str(event.account_id)!=self._account_id or event.strategy_id!=_RuntimeBridge._strategy_identity(self):return
        if event.instrument_id not in self._instrument_ids.values():return
        key=str(event.event_id)
        signature=(type(event).__name__,closed,*(str(getattr(event,name,None)) for name in
            ('account_id','strategy_id','instrument_id','position_id','opening_order_id','closing_order_id',
             'signed_qty','avg_px_open','realized_pnl','currency','entry','ts_opened','ts_event','ts_closed')))
        previous=self._position_events.get(key)
        if previous is not None:
            if previous != signature:
                error=ValueError('conflicting native position event identity')
                self._worker_error=error
                raise error
            return
        origin=self.frontend_broker._orders.get(str(event.opening_order_id))
        if origin is not None and self._instrument_ids.get(
                getattr(origin.data,'_symbol',origin.data._name)) != event.instrument_id:
            raise ValueError('native position differs from admitted opening order scope')
        if origin is None and sum(iid == event.instrument_id for iid in self._instrument_ids.values()) > 1:
            raise ValueError('native position projection lacks its admitted opening feed route')
        symbol=(getattr(origin.data,'_symbol',origin.data._name) if origin is not None
                else self._symbol(event.instrument_id))
        data=self.frontend.feeds[symbol]
        cycle=(str(event.position_id),str(event.opening_order_id))
        opened=type(event).__name__=='PositionOpened'
        state=self._position_trade_views.get(cycle)
        if state is None:
            state=dict(ref=self._next_position_trade_ref,baropen=len(data),
                       dtopen=date2num(pd.Timestamp(getattr(event,'ts_opened',event.ts_event),unit='ns',tz='UTC')))
            self._next_position_trade_ref+=1
            self._position_trade_views[cycle]=state
        facts=native_trade_economics(_RuntimeBridge._read_cache(self),event,closed)
        trade=Trade(ref=state['ref'],data=data,size=number(event.signed_qty),
                    price=number(event.avg_px_open),status=Trade.Closed if closed else Trade.Open,
                    isopen=not closed,isclosed=closed,justopened=opened,
                    long=event.entry==OrderSide.BUY,
                    pnl=facts['pnl'],pnlcomm=facts['pnlcomm'],commission=facts['commission'],value=facts['value'],
                    dtopen=state['dtopen'],baropen=state['baropen'],
                    barlen=len(data)-state['baropen'],barclose=len(data) if closed else 0,
                    dtclose=date2num(pd.Timestamp(event.ts_event,unit='ns',tz='UTC')) if closed else 0.)
        trade.info={'native_position_id':str(event.position_id),'native_event_id':key,
                    'native_economics_source':facts['source'],'pnl_currency':facts['currency'],
                    'value_currency':facts['value_currency']}
        self._position_events[key]=signature
        if closed:self._position_trade_views.pop(cycle,None)
        # BT updates its Trade view through partial changes, but notify_trade
        # and Analyzer trade notices occur only on opening and closing.
        if opened or closed:
            self.frontend.step_broker(trade,kind='trade')
            components=getattr(self.frontend,'_cerebro_components',None)
            if components is not None:components.trade(trade)

    def on_strategy_position_opened(self,event):self._project_position(event)
    def on_strategy_position_changed(self,event):self._project_position(event)
    def on_strategy_position_closed(self,event):self._project_position(event,closed=True)


def _release_exception_native_owners(error):
    """Keep failure identity/trace locations, release completed native-frame locals.

    Engine user strategies are Python objects. Only finished framework frames
    whose self is an official native Strategy can retain its unsendable owner.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for attribute in ('__cause__', '__context__'):
            try:
                pending.append(BaseException.__dict__[attribute].__get__(current))
            except BaseException:
                pass
        trace = BaseException.__dict__["__traceback__"].__get__(current)
        while trace is not None:
            frame = trace.tb_frame
            namespace = frame.f_globals.get('__name__', '')
            if namespace.startswith(('tradelearn_runtime.', 'leafquant_monitor.')):
                owner = frame.f_locals.get('self')
                if isinstance(owner, NautilusStrategy):
                    try:
                        frame.clear()
                    except RuntimeError:
                        pass  # Never clear an executing frame.
                del owner
            trace = trace.tb_next


class ExecutionHost(_RuntimeBridge, NautilusStrategy):
    """Official Nautilus host with no logger, sender or SDK configuration."""
    def submit_order(self,order,position_id=None,client_id=None,params=None):
        self._before_native_submission(order,position_id,client_id,params)
        return super().submit_order(order,position_id=position_id)

    def submit_order_list(self,order_list,position_id=None,client_id=None,params=None):
        for order in order_list:self._before_native_submission(order,position_id,client_id,params)
        return super().submit_order_list(order_list,position_id=position_id)

    on_start = _RuntimeBridge.on_strategy_start
    on_stop = _RuntimeBridge.on_strategy_stop
    on_quote = _RuntimeBridge.on_strategy_quote
    on_bar = _RuntimeBridge.on_strategy_bar
    on_order_pending_update = _RuntimeBridge.on_strategy_order_pending_update
    on_order_pending_cancel = _RuntimeBridge.on_strategy_order_pending_cancel
    on_order_modify_rejected = _RuntimeBridge.on_strategy_order_modify_rejected
    on_order_cancel_rejected = _RuntimeBridge.on_strategy_order_cancel_rejected
    on_order_updated = _RuntimeBridge.on_strategy_order_updated
    on_order_triggered = _RuntimeBridge.on_strategy_order_triggered
    on_order_accepted = _RuntimeBridge.on_strategy_order_accepted
    on_order_rejected = _RuntimeBridge.on_strategy_order_rejected
    on_order_denied = _RuntimeBridge.on_strategy_order_denied
    on_order_canceled = _RuntimeBridge.on_strategy_order_canceled
    on_order_expired = _RuntimeBridge.on_strategy_order_expired
    on_order_filled = _RuntimeBridge.on_strategy_fill
    on_order_fill_voided = _RuntimeBridge.on_strategy_order_fill_voided
    on_position_opened = _RuntimeBridge.on_strategy_position_opened
    on_position_changed = _RuntimeBridge.on_strategy_position_changed
    on_position_closed = _RuntimeBridge.on_strategy_position_closed


def get_execution_host_class(*, logging_enabled=False):
    """Return a subclassable official host; importing the bridge enables no logger."""
    if not logging_enabled:
        return ExecutionHost
    return _logged_execution_host_class()


from functools import cache


@cache
def _logged_execution_host_class():
    from leafquant_monitor.strategy import TradeLogStrategy

    class LoggedExecutionHost(_RuntimeBridge, TradeLogStrategy):
        pass

    return LoggedExecutionHost


def create_execution_host(config, *, monitor=None, **kwargs):
    """Select explicit observation without making logging an execution dependency."""
    host = get_execution_host_class(logging_enabled=monitor is not None)
    return host(config, monitor=monitor, **kwargs)
