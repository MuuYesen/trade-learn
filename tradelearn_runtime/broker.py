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


class NautilusBroker:
    def __init__(self, gateway):
        self.gateway = gateway
        self._refs = count(1)
        self._orders = {}
        self._owners = {}
        self._seen = set()
        self._unresolved = set()
        self._recovery_blocks = set()
        self._thread = get_ident()
        self.storage = None

    def bind_gateway(self, gateway):
        """Bind once on the native callback thread, before Runtime construction."""
        if hasattr(self, 'runtime'):
            raise RuntimeError('broker projection cannot be reused across runtimes')
        self.gateway = gateway
        self._thread = get_ident()

    def bind_runtime(self, runtime):
        self.runtime = runtime

    def _notify(self,owner,order):
        self.runtime.step_broker(order,kind='order',owner=owner)

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
        unsupported = set(kwargs) - {'client_oid','tif','valid','reduce_only','position_id'}
        if unsupported:
            raise ValueError(f'unsupported live order controls: {sorted(unsupported)}')
        exectype = Order.Market if exectype is None else exectype
        if exectype not in (Order.Market, Order.Limit):
            raise ValueError('unsupported live order type')
        if side not in (Order.Buy,Order.Sell) or not math.isfinite(float(size)) or float(size)<=0:
            raise ValueError('positive finite size and explicit side required')
        if exectype == Order.Limit and (price is None or not math.isfinite(float(price)) or float(price)<=0):
            raise ValueError('positive finite limit price required')
        if exectype == Order.Market and price is not None:
            raise ValueError('market order cannot contain a limit price')
        tif = str(kwargs.pop('tif',kwargs.pop('valid','day'))).lower()
        if tif not in ('day','gtc','ioc','fok'):
            raise ValueError('unsupported live time-in-force')
        oid = kwargs.pop('client_oid',None) or str(uuid.uuid4())
        if oid in self._orders or self.gateway.lookup_order(oid) is not None:
            raise ValueError('client order ID reuse requires reconciliation, not resubmission')
        request = OrderRequest(data._name,'buy' if side==Order.Buy else 'sell',float(size),
                               'limit' if exectype==Order.Limit else 'market',
                               limit_price=price,tif=tif,client_oid=oid)
        if callable(getattr(self.gateway,'validate',None)):
            self.gateway.validate(request,**kwargs)
        order = Order.from_request(next(self._refs),request,data=data)
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

    def apply(self, client_oid, *, status, filled_qty, avg_price=None, event_id=None, commission=None, native_ts_ns=None):
        self._check_mutation()
        if client_oid not in self._orders:
            raise RuntimeError('strategy order event lacks recovered frontend intent')
        key=(client_oid,event_id,status,filled_qty,avg_price,commission)
        if key in self._seen:
            return False
        order=self._orders[client_oid];owner=self._owners[client_oid]
        mapping={'INITIALIZED':Order.Created,'SUBMITTED':Order.Submitted,'ACCEPTED':Order.Accepted,
                 'PARTIALLY_FILLED':Order.Partial,'FILLED':Order.Completed,'CANCELED':Order.Canceled,
                 'EXPIRED':Order.Expired,'REJECTED':Order.Rejected,'DENIED':Order.Rejected}
        if status not in mapping:
            raise ValueError(f'unsupported native order status: {status}')
        qty=float(filled_qty)
        old=abs(order.executed.size)
        if not math.isfinite(qty) or qty<old or qty>order.size:
            raise ValueError('nonmonotonic cumulative fill')
        if not order.alive() and (mapping[status] != order.status or qty != old):
            raise ValueError('conflicting terminal replay requires reconciliation')
        if qty and (avg_price is None or not math.isfinite(float(avg_price))):
            raise ValueError('filled order requires authoritative average price')
        was_alive=order.alive()
        terminal=mapping[status] in (Order.Completed,Order.Canceled,Order.Expired,Order.Rejected)
        if status=='FILLED' and qty!=order.size:
            raise ValueError('full fill requires full cumulative quantity')
        delta=qty-old
        sign=1 if order.isbuy() else -1
        order.status=mapping[status]
        order.executed.size=sign*qty
        order.executed.price=float(avg_price) if qty else math.nan
        order.executed.value=qty*float(avg_price) if qty else 0
        order.executed.comm=float(commission) if commission is not None else math.nan
        order.info['submission_state']=status
        if native_ts_ns is not None:order.info['native_ts_ns']=int(native_ts_ns)
        # Match the frontend's pending delta without applying synthetic fills.
        pending=owner._pending_size.get(order.data,0)
        released=(order.size-old) if terminal and was_alive else delta
        owner._pending_size[order.data]=pending-sign*released
        self._seen.add(key)
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
        order=Order.from_request(next(self._refs),request,data=data)
        order.status=Order.Submitted
        self._orders[oid]=order;self._owners[oid]=self.owner
        sign=1 if order.isbuy() else -1
        self.owner._pending_size[data]=self.owner._pending_size.get(data,0)+sign*order.size
        status=native.status.name
        if status in ('PENDING_CANCEL','PENDING_UPDATE','TRIGGERED'):status='ACCEPTED'
        fees=list(native.commissions().values())
        self.apply(oid,status=status,filled_qty=float(native.filled_qty.as_double()),
                   avg_price=native.avg_px,event_id='recovery:'+str(native.last_event.event_id),
                   commission=float(fees[0].as_double()) if len(fees)==1 else None,
                   native_ts_ns=int(native.ts_last))
        if order.alive():self._recovery_blocks.add(oid)
        return order

    def lookup_order(self, client_oid):
        self._check_thread()
        return self._orders.get(client_oid) or self.gateway.lookup_order(client_oid)

    def cancel(self, order):
        self._check_mutation()
        oid=order if isinstance(order,str) else order.info['client_oid']
        self.gateway.cancel(oid)

    def getposition(self, data):
        self._check_thread()
        return self.gateway.position(data._name)

    def _money(self, name):
        self._check_thread()
        value=self.gateway.account().get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f'account {name} unavailable')
        return float(value)

    def getcash(self): return self._money('cash')
    get_cash=getcash
    def getvalue(self, datas=None):
        if datas: raise ValueError('subset valuation is unsupported')
        return self._money('equity')
    get_value=getvalue
