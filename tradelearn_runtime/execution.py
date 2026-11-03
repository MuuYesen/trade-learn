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
from tradelearn.engine import Strategy
from tradelearn.engine import StrategyRuntime


def number(value):
    return float(value.as_double()) if hasattr(value,'as_double') else float(value)


class _RuntimeBridge:
    def __new__(cls, config, **kwargs):
        return super().__new__(cls, config)

    def __init__(self, config, *, business_cls, feeds, instruments, parameters=None,
                 account_provider=None, subscriptions=(), monitor=None, recorder=None, sender_factory=None,
                 intent_path=None, offline_test=False, runtime_observer_factory=None,
                 account_id=None, intent_identity=None):
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
            super().__init__(config)
        else:
            expected = monitor.framework_account_id or monitor.account_id
            if self._account_id is None or self._account_id != expected:
                raise ValueError('monitor differs from native bridge account binding')
            super().__init__(config, monitor=monitor, recorder=recorder, sender_factory=sender_factory)
        if not set(feeds).issubset(instruments) and instruments:
            raise ValueError('every feed needs an explicit Nautilus InstrumentId')
        if runtime_observer_factory is not None and not callable(runtime_observer_factory):
            raise TypeError('runtime_observer_factory must be callable')
        self._runtime_observer_factory=runtime_observer_factory
        self._business_cls=business_cls
        self._feeds=dict(feeds)
        self._instrument_ids=dict(instruments)
        self._parameters=parameters or {}
        self._account_provider=account_provider
        self._subscriptions=tuple(subscriptions)
        self.frontend=None
        self.frontend_broker=None
        self._position_events=set()
        self._tasks=set()
        self._io_lock=asyncio.Lock()
        self._worker_error=None
        self._stopping=False
        self._offline_test=offline_test
        if intent_path is None and not offline_test:
            raise ValueError('live execution requires an identity-bound durable intent_path')
        self.intent_journal=None
        self._recovering={}
        if intent_path is not None:
            from .intents import IntentJournal
            if not isinstance(intent_identity, str) or not intent_identity or intent_identity.strip() != intent_identity:
                raise ValueError('explicit durable intent identity required')
            self.intent_journal=IntentJournal(intent_path,identity=intent_identity)
            self._recovering=self.intent_journal.unfinished()

    def _assert_account_binding(self):
        if not self._account_id:
            raise ValueError('explicit native account binding required')
        account=self.cache.account(AccountId(self._account_id))
        if account is None or str(account.id)!=self._account_id:
            raise ValueError('native account is absent or differs from explicit binding')
        for iid in self._instrument_ids.values():
            routed=self.cache.account_for_venue(iid.venue)
            if routed is None or str(routed.id)!=self._account_id:
                raise ValueError('venue execution account differs from explicit binding')
        return account

    def _owns_native_order(self,order,event=None):
        if order.strategy_id!=self.strategy_id or order.instrument_id not in self._instrument_ids.values():
            return False
        if order.account_id is not None and str(order.account_id)!=self._account_id:
            return False
        if event is not None:
            if getattr(event,'strategy_id',self.strategy_id)!=self.strategy_id:
                return False
            account=getattr(event,'account_id',None)
            if account is not None and str(account)!=self._account_id:
                return False
        if order.account_id is None:
            # A local denial can precede native account assignment. It is only
            # ours if the explicitly bound frontend already admitted this CID.
            return (order.status.name in ('DENIED','REJECTED') and self.frontend_broker is not None
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
                or (order.account_id is None and not new)
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
        if self._offline_test:
            try:self.clock.set_time(self.clock.timestamp_ns())
            except ValueError:raise RuntimeError('offline_test requires the native virtual clock') from None
        recovered={}
        for oid,payload in self._recovering.items():
            native=self.lookup_order(oid)
            if native is None or native.strategy_id!=self.strategy_id:
                raise RuntimeError('durable admitted intent lacks authoritative native reconciliation; startup blocked')
            recovered[oid]=(native,payload)
        existing=self.cache.orders_open(strategy_id=self.strategy_id)+self.cache.orders_inflight(strategy_id=self.strategy_id)
        if any(str(order.client_order_id) not in recovered for order in existing):
            raise RuntimeError('native pending orders lack durable frontend intent; startup blocked')
        if set(self._feeds)!=set(self._instrument_ids):
            raise ValueError('startup requires explicit instrument bindings')
        for iid in self._instrument_ids.values():
            if self.cache.instrument(iid) is None:
                raise ValueError('required instrument is not loaded')
        self.frontend_broker=NautilusBroker(self)
        self.frontend=StrategyRuntime(self._business_cls,feeds=self._feeds,
                                   broker=self.frontend_broker,parameters=self._parameters)
        from .observers import attach_runtime_observer
        attach_runtime_observer(self.frontend,getattr(self,'_runtime_observer_factory',None))
        for oid,(native,payload) in recovered.items():
            from tradelearn.core.broker_contracts import OrderRequest
            request=OrderRequest(**payload['request'])
            self.validate_recovery(request,native)
            if native.is_closed:
                self._schedule(self._persist_terminal(oid,native))
            self.frontend_broker.restore(request,self.frontend.feeds[request.symbol],native)
        self.frontend.start()
        for symbol,mode in self._feeds.items():
            if mode=='tick':self.subscribe_quotes(self._instrument_ids[symbol])
        for bar_type in self._subscriptions:self.subscribe_bars(bar_type)

    def on_strategy_stop(self):
        self._stopping=True
        if self.frontend is not None:self.frontend.stop()

    def _symbol(self,iid):
        for symbol,bound in self._instrument_ids.items():
            if iid==bound:return symbol
        raise ValueError('unbound instrument event')

    def on_strategy_quote(self,quote):
        try:
            self.frontend.step_market(self._symbol(quote.instrument_id),quote,kind='tick')
        finally:
            if self.frontend.failed:self._stopping=True

    def on_strategy_bar(self,bar):
        import pandas as pd
        try:
            self.frontend.step_market(self._symbol(bar.bar_type.instrument_id),
                dict(ts=pd.Timestamp(int(bar.ts_event),unit='ns',tz='UTC'),
                     open=number(bar.open),high=number(bar.high),low=number(bar.low),
                     close=number(bar.close),volume=number(bar.volume)))
        finally:
            if self.frontend.failed:self._stopping=True

    def _make_order(self,request,controls):
        self._assert_account_binding()
        if set(controls)-{'reduce_only','position_id'}:
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
                    quantity=quantity,time_in_force={'day':TimeInForce.DAY,'gtc':TimeInForce.GTC,'ioc':TimeInForce.IOC,'fok':TimeInForce.FOK}[request.tif],
                    client_order_id=ClientOrderId(request.client_oid),reduce_only=controls.pop('reduce_only',False))
        if request.order_type=='limit':
            price=instrument.make_price(request.limit_price)
            if Decimal(str(price)) != Decimal(str(request.limit_price)):
                raise ValueError('price would be rounded by venue precision')
            native=self.order_factory.limit(price=price,**kwargs)
        elif request.order_type=='market':native=self.order_factory.market(**kwargs)
        else:raise ValueError('unsupported order type')
        return native,controls

    def validate(self,request,**controls):
        self._make_order(request,dict(controls))

    def submit(self,request,**controls):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
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
        task.add_done_callback(finished)
        return task

    async def _admit_and_submit(self,request,native,controls):
        try:
            async with self._io_lock:
                await asyncio.to_thread(self.intent_journal.prepare,request.client_oid,
                                        {'request':asdict(request),'controls':{key:str(value) for key,value in controls.items()}})
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

    async def _persist_terminal(self,oid,order):
        evidence={'status':order.status.name,'filled_qty':str(order.filled_qty),'native_event_id':str(order.last_event.event_id)}
        async with self._io_lock:
            await asyncio.to_thread(self.intent_journal.terminal,oid,evidence)

    async def drain(self):
        self._stopping=True
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks),return_exceptions=False)
        if self.intent_journal is not None:
            await asyncio.to_thread(self.intent_journal.close)
        if self._worker_error is not None:raise RuntimeError('durable intent worker failed') from self._worker_error

    def cancel(self,client_oid):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        order=self.cache.order(ClientOrderId(client_oid))
        self._assert_account_binding()
        if order is None or not self._owns_native_order(order):
            raise ValueError('cancel requires this strategy/account native order')
        self.cancel_order(order)

    def validate_recovery(self,request,native):
        expected_type='LIMIT' if request.order_type=='limit' else 'MARKET'
        if (native.strategy_id!=self.strategy_id or str(native.client_order_id)!=request.client_oid
            or native.instrument_id!=self._instrument_ids[request.symbol]
            or native.side.name!=request.side.upper() or native.order_type.name!=expected_type
            or Decimal(str(native.quantity))!=Decimal(str(request.qty))
            or native.time_in_force.name!=request.tif.upper()):
            raise ValueError('native recovery order differs from durable strategy request')
        if expected_type=='LIMIT' and Decimal(str(native.price))!=Decimal(str(request.limit_price)):
            raise ValueError('native recovery limit price differs from request')
        binding=self._account_id
        if binding is not None and (native.account_id is None or str(native.account_id)!=binding):
            raise ValueError('native recovery account differs from verified binding')

    def lookup_order(self,client_oid):
        # A native object is an existence/recovery witness, never a synthetic ACK.
        return self.cache.order(ClientOrderId(client_oid))

    def position(self,symbol):
        self._assert_account_binding()
        positions=[p for p in self.cache.positions_open(instrument_id=self._instrument_ids[symbol],strategy_id=self.strategy_id)
                   if str(p.account_id)==self._account_id and p.strategy_id==self.strategy_id]
        if len(positions)>1:raise ValueError('hedged positions need explicit position_id routing')
        if not positions:return Position()
        native=positions[0]
        size=number(native.signed_qty)
        return Position(size,number(native.avg_px_open))

    def account(self):
        # Never infer portfolio equity from wallet/buying power or a quote currency.
        if self._account_provider is None:raise ValueError('explicit authoritative account projection required')
        return self._account_provider(self.cache,self.portfolio)

    def _project_order(self,event):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        order=self.cache.order(event.client_order_id)
        if order is None:raise RuntimeError('order event has no authoritative native cache snapshot')
        if not self._owns_native_order(order,event):return
        status=order.status.name
        if self.frontend_broker is None:
            return  # Startup recovery validates/cache-projects before business begins.
        if status in ('PENDING_CANCEL','PENDING_UPDATE','TRIGGERED'):
            return
        fees=list(order.commissions().values())
        commission=number(fees[0]) if len(fees)==1 else None
        # Native terminal evidence belongs to durable execution, independently of
        # whether the failed/stopped business runtime can accept a notification.
        if order.is_closed and self.intent_journal is not None:
            self._schedule(self._persist_terminal(str(event.client_order_id),order))
        self.frontend_broker.apply(str(event.client_order_id),status=status,
            filled_qty=number(order.filled_qty),avg_price=order.avg_px,
            event_id=str(event.event_id),commission=commission,native_ts_ns=int(event.ts_event))

    def on_strategy_order_accepted(self,event):self._project_order(event)
    def on_strategy_order_rejected(self,event):self._project_order(event)
    def on_strategy_order_denied(self,event):self._project_order(event)
    def on_strategy_order_canceled(self,event):self._project_order(event)
    def on_strategy_order_expired(self,event):self._project_order(event)
    def on_strategy_fill(self,event):self._project_order(event)

    def _project_position(self,event,closed=False):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        if str(event.account_id)!=self._account_id or event.strategy_id!=self.strategy_id:return
        if event.instrument_id not in self._instrument_ids.values():return
        key=str(event.event_id)
        if key in self._position_events:return
        symbol=self._symbol(event.instrument_id)
        trade=Trade(data=self.frontend.feeds[symbol],size=number(event.signed_qty),
                    price=number(event.avg_px_open),status=Trade.Closed if closed else Trade.Open,
                    isopen=not closed,isclosed=closed,
                    pnl=number(event.realized_pnl) if event.realized_pnl is not None else math.nan,
                    pnlcomm=math.nan,commission=math.nan)
        # Position event economics are authoritative; net-of-fee economics need
        # separately qualified complete commission evidence, never assumed zero.
        trade.info={'native_position_id':str(event.position_id),'native_event_id':key}
        self._position_events.add(key)
        self.frontend.step_broker(trade,kind='trade')

    def on_strategy_position_opened(self,event):self._project_position(event)
    def on_strategy_position_changed(self,event):self._project_position(event)
    def on_strategy_position_closed(self,event):self._project_position(event,closed=True)


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
    on_order_accepted = _RuntimeBridge.on_strategy_order_accepted
    on_order_rejected = _RuntimeBridge.on_strategy_order_rejected
    on_order_denied = _RuntimeBridge.on_strategy_order_denied
    on_order_canceled = _RuntimeBridge.on_strategy_order_canceled
    on_order_expired = _RuntimeBridge.on_strategy_order_expired
    on_order_filled = _RuntimeBridge.on_strategy_fill
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
