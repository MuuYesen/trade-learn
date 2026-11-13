"""TradeLearn engine order frontend; execution and economics stay in Nautilus.

Order objects below are notification projections, not an independent matching
or position engine. A submit exception leaves an unresolved intent and is never
converted into a broker rejection.
"""
from itertools import count
from threading import get_ident
import math
import uuid
from tradelearn.backtest.models import Order
from tradelearn.core.broker_contracts import OrderRequest


def _valid_expiry_ns(valid, owner):
    """Preserve TL's inclusive UTC-second deadline; NA expires at its next second."""
    import datetime, numbers
    import pandas as pd
    if isinstance(valid, bool):
        raise ValueError('valid must be an absolute date or a finite relative duration')
    if isinstance(valid, (datetime.timedelta,numbers.Real)):
        if isinstance(valid,numbers.Real) and not math.isfinite(float(valid)):
            raise ValueError('finite relative valid seconds required')
        try:
            line=owner.data.datetime  # Primary strategy clock, never the ordered secondary feed.
            method=getattr(line,'datetime',None)
            base=method(0) if callable(method) else line[0]
        except (AttributeError,IndexError,KeyError) as error:
            raise ValueError('relative order.valid requires a primary submission timestamp') from error
        base=(pd.Timestamp(base,unit='s',tz='UTC') if isinstance(base,numbers.Real) else pd.Timestamp(base))
        deadline=base+(valid if isinstance(valid,datetime.timedelta) else datetime.timedelta(seconds=float(valid)))
    elif isinstance(valid,(datetime.datetime,datetime.date)):
        deadline=pd.Timestamp(valid)
    else:
        raise ValueError('valid must be date, datetime, timedelta or relative seconds')
    if pd.isna(deadline):
        raise ValueError('finite order.valid deadline required')
    deadline=deadline.tz_localize('UTC') if deadline.tzinfo is None else deadline.tz_convert('UTC')
    value=(int(deadline.timestamp())+1)*1_000_000_000
    if not 0 < value <= 2**64-1:
        raise ValueError('order.valid exceeds native nanosecond expiry representation')
    return value


class _ClosedGateway:
    def __getattr__(self, name):
        raise RuntimeError('native execution host has closed')


class NautilusBroker:
    def __init__(self, gateway):
        self.gateway = gateway
        self._refs = count(1)
        self._orders = {}
        self._active_orders = {}
        self._owners = {}
        self._seen = {}
        self._unresolved = set()
        self._recovery_blocks = set()
        self._thread = get_ident()
        self.storage = None
        from tradelearn.engine.comminfo import CommInfoBase
        # BT sizing policy, not native execution fees or an account ledger.
        self.comminfo = {None: CommInfoBase()}

    def bind_gateway(self, gateway):
        """Bind once on the native callback thread, before Runtime construction."""
        if hasattr(self, 'runtime'):
            raise RuntimeError('broker projection cannot be reused across runtimes')
        self.gateway = gateway
        self._thread = get_ident()

    def bind_runtime(self, runtime):
        self.runtime = runtime

    def _release_gateway(self, owner):
        """Retire only this native owner after stop; never revive a closed route."""
        self._check_mutation()
        if self.gateway is owner:
            self.gateway = _ClosedGateway()

    def _notify(self,owner,order):
        oid=order.info['client_oid']
        if order.alive():self._active_orders[oid]=order
        else:self._active_orders.pop(oid,None)
        self.runtime.step_broker(order,kind='order',owner=owner)
        components = getattr(self.runtime,'_cerebro_components',None)
        if components is not None:components.order(order)

    _supports_order_owner = True

    def get_orders_open(self, safe=False, *, owner=None):
        self._check_thread()
        orders = [order for oid, order in self._active_orders.items()
                  if order.alive() and (owner is None or self._owners.get(oid) is owner)]
        if safe:
            orders = [order.clone() for order in orders]
        return tuple(orders) if owner is not None else orders

    def get_orders_history(self):
        self._check_thread()
        return list(self._orders.values())

    def bind(self, owner):
        self.owner = owner

    def _check_thread(self):
        if get_ident() != self._thread:
            raise RuntimeError('broker projection belongs to Nautilus callback thread')

    def _check_mutation(self):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        self._check_thread()

    def _submit(self, owner, data, side, size, price=None, exectype=None, **kwargs):
        self._check_mutation()
        if self._unresolved or self._recovery_blocks:
            raise RuntimeError('unresolved submission blocks new orders until reconciliation')
        unsupported = set(kwargs) - {'client_oid','tif','valid','reduce_only','position_id','pricelimit','trailamount','trailpercent','parent','oco','transmit','info','post_only'}
        if unsupported:
            raise ValueError(f'unsupported live order controls: {sorted(unsupported)}')
        info = dict(kwargs.pop('info', {}))
        if any(key in info for key in ('client_oid', 'submission_state', 'native_status', 'native_evidence')):
            raise ValueError('reserved native order metadata cannot be supplied')
        parent, oco = kwargs.pop('parent', None), kwargs.pop('oco', None)
        transmit = kwargs.pop('transmit', True)
        if type(transmit) is not bool:
            raise TypeError('transmit must be boolean')
        grouped = parent is not None or oco is not None or not transmit
        if grouped and not all(callable(getattr(self.gateway, name, None))
                               for name in ('validate_group', 'submit_group')):
            raise ValueError('native gateway has no linked-order capability')
        if grouped:
            self.gateway.validate_group([])  # reject known unsupported sources before staging
        exectype = Order.Market if exectype is None else exectype
        if exectype not in (Order.Market, Order.Limit, Order.Stop, Order.StopLimit, Order.StopTrail, Order.StopTrailLimit):
            raise ValueError('unsupported live order type')
        if side not in (Order.Buy,Order.Sell) or not math.isfinite(float(size)) or float(size)<=0:
            raise ValueError('positive finite size and explicit side required')
        trail_amount, trail_percent = kwargs.pop('trailamount', None), kwargs.pop('trailpercent', None)
        if exectype in (Order.StopTrail, Order.StopTrailLimit):
            if trail_amount is None and trail_percent is None:
                raise ValueError('trailing amount or percent required')
            for value in (trail_amount, trail_percent):
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError('positive finite trailing offset required')
            if price is None:
                price = float(data.close[0])
        elif trail_amount is not None or trail_percent is not None:
            raise ValueError('trailing offsets require a trailing order')
        if exectype in (Order.Limit, Order.Stop, Order.StopLimit, Order.StopTrail, Order.StopTrailLimit) and (price is None or not math.isfinite(float(price)) or float(price)<=0):
            raise ValueError('positive finite limit/trigger price required')
        pricelimit = kwargs.pop('pricelimit', None)
        if exectype == Order.StopTrailLimit and pricelimit is None:
            pricelimit = price
        if exectype in (Order.StopLimit, Order.StopTrailLimit):
            if pricelimit is None or not math.isfinite(float(pricelimit)) or float(pricelimit) <= 0:
                raise ValueError('positive finite stop-limit execution price required')
        elif pricelimit is not None:
            raise ValueError('pricelimit requires StopLimit or StopTrailLimit')
        if exectype == Order.Market and price is not None:
            raise ValueError('market order cannot contain a limit price')
        tif_value, valid_value = kwargs.pop('tif', None), kwargs.pop('valid', None)
        bt_validity = bool(getattr(owner, '_bt_minperiods', False))
        effective_valid = valid_value
        if bt_validity and (exectype == Order.Market or valid_value == 0):
            # This is fixed BT Market/zero behavior, not a synthetic GTD order.
            effective_valid = None
        if tif_value is not None and effective_valid is not None:
            raise ValueError('tif and valid are mutually exclusive')
        expire_time_ns = None
        if effective_valid is not None and not isinstance(effective_valid,str):
            if exectype == Order.Market:
                raise ValueError('Market order cannot express native GTD expiry')
            deadline = effective_valid
            if bt_validity:
                from tradelearn.backtest.clock import num2date
                deadline = num2date(effective_valid)
            expire_time_ns = _valid_expiry_ns(deadline,owner)
            raise NotImplementedError('native GTD expiry boundary is not qualified on pinned NA; explicit deadlines are not admitted')
        else:
            tif = str(tif_value if tif_value is not None else (effective_valid if effective_valid is not None else 'gtc')).lower()
            if tif not in ('day','gtc','ioc','fok'):
                raise ValueError('unsupported live time-in-force')
        oid = kwargs.pop('client_oid',None) or str(uuid.uuid4())
        if oid in self._orders or self.gateway.lookup_order(oid) is not None:
            raise ValueError('client order ID reuse requires reconciliation, not resubmission')
        request = OrderRequest(data._name,'buy' if side==Order.Buy else 'sell',float(size),
                               {Order.Market:'market',Order.Limit:'limit',Order.Stop:'stop',Order.StopLimit:'stop_limit',Order.StopTrail:'stop_trail',Order.StopTrailLimit:'stop_trail_limit'}[exectype],
                               limit_price=pricelimit if exectype in (Order.StopLimit,Order.StopTrailLimit) else (price if exectype==Order.Limit else None),
                               stop_price=price if exectype in (Order.Stop,Order.StopLimit,Order.StopTrail,Order.StopTrailLimit) else None,
                               tif=tif,client_oid=oid,
                               trail_amount=trail_amount,trail_percent=trail_percent,expire_time_ns=expire_time_ns)
        if callable(getattr(self.gateway,'validate',None)):
            self.gateway.validate(request,**kwargs)
        order = Order._from_request(next(self._refs),request,data=data)
        order._initialize_clock(owner)
        order.info.update(info)
        order.valid = valid_value
        if grouped:
            return self._stage_group(owner, order, request, dict(kwargs), parent, oco, transmit)
        order.status = Order.Submitted
        order.info['submission_state']='SUBMITTED'
        self._orders[oid] = order
        self._owners[oid] = owner
        self._notify(owner,order)
        try:
            self.gateway.submit(request,**kwargs)
        except Exception as exc:
            order.info['submission_state']='UNKNOWN'
            order.info['error_category']=type(exc).__name__
            self._unresolved.add(oid)
            # Returning the unresolved handle preserves engine pending quantities.
            # The next submit is blocked until a real terminal/reconciled report.
            self._notify(owner,order)
            return order
        return order

    def _stage_group(self, owner, order, request, controls, parent, oco, transmit):
        """Buffer only untransmitted intents in the existing projection, never manage fills."""
        root = request.client_oid
        for peer in (parent, oco):
            if peer is None:
                continue
            cid = peer.info.get('client_oid')
            if (self._orders.get(cid) is not peer or self._owners.get(cid) is not owner
                    or not getattr(peer, '_pending_transmit', False) or not peer.alive()
                    or peer.data is not order.data):
                raise ValueError('parent/OCO peer must be this owner/data pending native transmission')
            peer_root = peer._group_root
            if root != request.client_oid and root != peer_root:
                raise ValueError('cannot merge unrelated pending groups')
            root = peer_root
        order.parent, order.oco, order.transmit = parent, oco, transmit
        order._group_root = root
        order._pending_transmit = True
        order._native_request, order._native_controls = request, controls
        orders = [value for value in self._orders.values()
                  if getattr(value, '_pending_transmit', False) and value._group_root == root]
        orders.append(order)
        entries = [dict(request=value._native_request, controls=value._native_controls,
            parent=value.parent.info['client_oid'] if value.parent is not None else None,
            oco=value.oco.info['client_oid'] if value.oco is not None else None) for value in orders]
        if transmit:
            self.gateway.validate_group(entries)  # the complete group is checked before any send
        self._orders[request.client_oid] = order
        self._owners[request.client_oid] = owner
        if not transmit:
            self._notify(owner,order)
            return order
        for value in orders:
            value.status = Order.Submitted
            value.info['submission_state'] = 'SUBMITTED'
        for value in orders:
            if any(not candidate.alive() for candidate in orders):
                return order
            self._notify(owner,value)
        if any(not value.alive() for value in orders):
            return order  # a synchronous cancellation before native admission stays local
        for value in orders:
            value._pending_transmit = False
        try:
            self.gateway.submit_group(entries)
        except Exception as error:
            for value in orders:
                value.info.update(submission_state='UNKNOWN',error_category=type(error).__name__)
                self._unresolved.add(value.info['client_oid'])
                self._notify(owner,value)
        return order

    def apply(self, client_oid, *, status, filled_qty, avg_price=None, event_id=None, commission=None, native_ts_ns=None, native_evidence=None, total_qty=None):
        self._check_mutation()
        if client_oid not in self._orders:
            raise RuntimeError('strategy order event lacks recovered frontend intent')
        payload=(status,filled_qty,avg_price,commission,total_qty)
        key=(client_oid,event_id) if event_id is not None else (client_oid,payload)
        if key in self._seen:
            if self._seen[key] != payload:
                raise ValueError('conflicting native event replay requires reconciliation')
            return False
        order=self._orders[client_oid];owner=self._owners[client_oid]
        mapping={'INITIALIZED':Order.Created,'SUBMITTED':Order.Submitted,'ACCEPTED':Order.Accepted,
                 'PARTIALLY_FILLED':Order.Partial,'FILLED':Order.Completed,'CANCELED':Order.Canceled,
                 'EXPIRED':Order.Expired,'REJECTED':Order.Rejected,'DENIED':Order.Rejected}
        if status in ('TRIGGERED','PENDING_CANCEL','PENDING_UPDATE'):
            mapping[status] = Order.Partial if float(filled_qty) > 0 else Order.Accepted
        if status not in mapping:
            raise ValueError(f'unsupported native order status: {status}')
        qty=float(filled_qty)
        old=abs(order.executed.size)
        old_total=abs(order.size)
        new_total=old_total if total_qty is None else float(total_qty)
        if not math.isfinite(new_total) or new_total <= 0 or new_total < old:
            raise ValueError('invalid native total quantity revision')
        if not math.isfinite(qty) or qty<old or qty>new_total:
            raise ValueError('nonmonotonic cumulative fill')
        if not order.alive() and (mapping[status] != order.status or qty != old or new_total != old_total):
            raise ValueError('conflicting terminal replay requires reconciliation')
        if qty and (avg_price is None or not math.isfinite(float(avg_price))):
            raise ValueError('filled order requires authoritative average price')
        was_alive=order.alive()
        terminal=mapping[status] in (Order.Completed,Order.Canceled,Order.Expired,Order.Rejected)
        if status=='FILLED' and qty!=new_total:
            raise ValueError('full fill requires full cumulative quantity')
        delta=qty-old
        sign=1 if order.isbuy() else -1
        order.size=sign*new_total
        order.status=mapping[status]
        order.executed.size=sign*qty
        order.executed.remsize=order.size-order.executed.size
        if delta > 0 and native_ts_ns is not None:
            order._set_execution_clock(int(native_ts_ns), unit="ns")
        order.executed.price=float(avg_price) if qty else math.nan
        order.executed.value=qty*float(avg_price) if qty else 0
        order.executed.comm=float(commission) if commission is not None else math.nan
        order.info['submission_state']=status
        order.info['native_status']=status
        order.info['native_evidence']=native_evidence
        if native_ts_ns is not None:order.info['native_ts_ns']=int(native_ts_ns)
        # Match the frontend's pending delta without applying synthetic fills.
        pending=owner._pending_size.get(order.data,0)+sign*(new_total-old_total)
        released=(new_total-old) if terminal and was_alive else delta
        owner._pending_size[order.data]=pending-sign*released
        self._seen[key]=payload
        self._unresolved.discard(client_oid)
        if terminal:self._recovery_blocks.discard(client_oid)
        self._notify(owner,order)
        return True

    def restore_projection(self,request,data,native=None):
        self._check_mutation()
        if request.client_oid in self._orders:return self._orders[request.client_oid]
        native=native if native is not None else self.gateway.lookup_order(request.client_oid)
        if native is None:raise RuntimeError('recovery has no authoritative native order')
        if not callable(getattr(self.gateway,'validate_recovery',None)):
            raise RuntimeError('gateway has no qualified native recovery mapping')
        self.gateway.validate_recovery(request,native)
        return self.restore(request,data,native)

    def restore(self,request,data,native):
        self._check_mutation()
        oid=request.client_oid
        if oid in self._orders:raise ValueError('duplicate recovery projection')
        order=Order._from_request(next(self._refs),request,data=data)
        order._initialize_clock(self.owner)
        order.status=Order.Submitted
        self._orders[oid]=order;self._owners[oid]=self.owner
        sign=1 if order.isbuy() else -1
        self.owner._pending_size[data]=self.owner._pending_size.get(data,0)+sign*abs(order.size)
        status=native.status.name
        fees=list(native.commissions().values())
        self.apply(oid,status=status,filled_qty=float(native.filled_qty.as_double()),
                   avg_price=native.avg_px,event_id='recovery:'+str(native.last_event.event_id),
                   commission=float(fees[0].as_double()) if len(fees)==1 else None,
                   native_ts_ns=int(native.ts_last),total_qty=float(native.quantity.as_double()))
        if order.alive():self._recovery_blocks.add(oid)
        return order

    def lookup_order(self, client_oid):
        self._check_thread()
        return self._orders.get(client_oid) or self.gateway.lookup_order(client_oid)

    def cancel(self, order):
        self._check_mutation()
        oid=order if isinstance(order,str) else order.info['client_oid']
        staged = self._orders.get(oid)
        if staged is not None and getattr(staged, '_pending_transmit', False):
            group = [value for value in self._orders.values()
                     if getattr(value, '_pending_transmit', False)
                     and value._group_root == staged._group_root and value.alive()]
            for value in group:
                value._pending_transmit = False
                value.info['cancelled_before_transmit'] = True
                self.apply(value.info['client_oid'],status='CANCELED',filled_qty=0)
            return
        self.gateway.cancel(oid)

    def _close_position(self, owner, data, *, size=None, **kwargs):
        self._check_mutation()
        controls = getattr(self.gateway, 'close_controls', None)
        if not callable(controls):
            return NotImplemented  # Existing private typed-close policy owns its route.
        if kwargs.get('reduce_only', True) is not True:
            raise ValueError('close cannot disable reduce-only protection')
        route = controls(data._name, position_id=kwargs.pop('position_id', None))
        if route is None:return None
        maximum = float(route['size'])
        quantity = maximum if not size else float(size)
        if isinstance(size, bool) or not math.isfinite(quantity) or not 0 < quantity <= maximum:
            raise ValueError('close size must be positive and cannot exceed the owned position')
        kwargs.update(position_id=route['position_id'],reduce_only=True)
        # BT close delegates to the public side method. Preserve user overrides,
        # Engine validity normalization and the common intent observation seam.
        public_submit = getattr(owner, 'buy' if route['side']=='buy' else 'sell', None)
        if callable(public_submit):
            return public_submit(data=data, size=quantity, **kwargs)
        submit = getattr(owner,'_submit_order',None)
        if callable(submit):
            return submit(Order.Buy if route['side']=='buy' else Order.Sell,
                          data=data,size=quantity,**kwargs)
        return self._submit(owner,data,Order.Buy if route['side']=='buy' else Order.Sell,
                            quantity,**kwargs)

    @property
    def positions(self):
        self._check_thread()
        return {data:self.getposition(data) for data in self.owner.datas}

    def getposition(self, data):
        self._check_thread()
        return self.gateway.position(data._name)

    def getcommissioninfo(self, data):
        """Return the BT sizing policy; fills retain their native reported fees."""
        self._check_thread()
        return self.comminfo.get(getattr(data, '_name', None), self.comminfo[None])

    def addcommissioninfo(self, comminfo, name=None):
        self._check_mutation()
        self.comminfo[name] = comminfo

    def setcommission(self, commission=0.0, margin=None, mult=1.0, commtype=None,
                      percabs=True, stocklike=False, interest=0.0,
                      interest_long=False, leverage=1.0, automargin=False, name=None):
        """Configure the ordinary BT sizing estimate, never overwrite venue fees."""
        from tradelearn.engine.comminfo import CommInfoBase
        self.addcommissioninfo(CommInfoBase(commission=commission, margin=margin,
            mult=mult, commtype=commtype, percabs=percabs, stocklike=stocklike,
            interest=interest, interest_long=interest_long, leverage=leverage,
            automargin=automargin), name)

    def _money(self, name):
        self._check_thread()
        value=self.gateway.account().get(name)
        if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
            raise ValueError(f'account {name} unavailable')
        return float(value)

    def getcash(self): return self._money('cash')
    get_cash=getcash
    def getvalue(self, datas=None):
        if datas: raise ValueError('subset valuation is unsupported')
        return self._money('equity')
    get_value=getvalue
