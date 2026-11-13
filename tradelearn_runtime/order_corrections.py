"""Identity-checked correction evidence from the existing official native order."""
from decimal import Decimal


def native_void_payload(event):
    from nautilus_trader.model import OrderFillVoided
    if type(event) is not OrderFillVoided:
        raise ValueError('actual native fill-void event required')
    return event.to_dict()


def native_void_evidence(pair):
    from nautilus_trader.model import OrderFillVoided
    if not isinstance(pair,tuple) or len(pair) not in (2,3):
        raise ValueError('native correction requires actual order/event pair')
    order,event=pair[:2]
    if type(event) is not OrderFillVoided:
        raise ValueError('actual native fill-void event required')
    if order.last_event.event_id!=event.event_id:
        raise ValueError('native correction is not the current authoritative order event')
    for key in ('trader_id','strategy_id','instrument_id','client_order_id','venue_order_id','account_id'):
        if getattr(order,key)!=getattr(event,key):
            raise ValueError('native correction scope differs from order')
    if order.side!=event.order_side or order.order_type!=event.order_type:
        raise ValueError('native correction type/side differs from order')
    events=order.events() if callable(order.events) else order.events
    if len(events)>4096:raise ValueError('native correction history exceeds bounded proof')
    retained=[item for item in events if item.event_id==event.event_id]
    if len(retained)!=1 or type(retained[0]) is not OrderFillVoided or retained[0].to_dict()!=event.to_dict():
        raise ValueError('native correction lacks exact retained event')
    qty=Decimal(str(event.voided_qty));remaining=Decimal(str(order.filled_qty))
    if not qty.is_finite() or qty<=0 or not remaining.is_finite() or remaining<0:
        raise ValueError('native correction requires positive cumulative void and valid remainder')
    status=order.status.name
    if event.is_reopened:
        if status not in ('ACCEPTED','PARTIALLY_FILLED'):
            raise ValueError('native reopened correction status differs')
    elif status!='VOIDED':raise ValueError('native terminal correction status differs')
    from .order_fees import cumulative_native_commission
    evidence=dict(avg_price=float(order.avg_px) if order.avg_px is not None else None,commission=cumulative_native_commission(order),event_id=str(event.event_id),client_oid=str(order.client_order_id),
        correction_id=str(event.correction_id),trade_id=str(event.trade_id),
        voided_qty=str(event.voided_qty),filled_qty=str(order.filled_qty),
        total_qty=str(order.quantity),native_status=status,is_reopened=bool(event.is_reopened),
        currency=str(event.currency),commission_voided=(str(event.commission_voided) if event.commission_voided is not None else None),
        account_id=str(event.account_id),strategy_id=str(event.strategy_id),
        instrument_id=str(event.instrument_id),native_ts_ns=int(event.ts_event))

    if len(pair)==3:
        from .algorithm_facts import spawn_projection
        projection=spawn_projection(pair[2],order,strategy_id=order.strategy_id,
            account_id=str(order.account_id),instruments=(order.instrument_id,))
        if projection is None:raise ValueError('native correction cache lacks spawned primary')
        evidence.update(child_client_oid=evidence['client_oid'],
            child_native_status=evidence['native_status'],child_is_reopened=evidence['is_reopened'],
            client_oid=projection['client_oid'],native_status=projection['status'],
            total_qty=str(projection['total_qty']),filled_qty=str(projection['filled_qty']),
            avg_price=projection['avg_price'],commission=projection['commission'],
            is_reopened=projection['status'] in ('SUBMITTED','ACCEPTED','PARTIALLY_FILLED'))
    return evidence


def retained_native_voids(order,cache=None):
    """Seed only immutable retained event identities during cache reconciliation.

    No old correction is reapplied to current quantities, fees or durable state.
    """
    from nautilus_trader.model import OrderFillVoided
    members=[order]
    if cache is not None and getattr(order,'exec_spawn_id',None) is not None:
        from .algorithm_facts import spawn_projection
        projection=spawn_projection(cache,order,strategy_id=order.strategy_id,
            account_id=(str(order.account_id) if order.account_id is not None else
                        next((str(item.account_id) for item in cache.orders_for_exec_spawn(order.exec_spawn_id) if item.account_id is not None),None)),
            instruments=(order.instrument_id,))
        if projection is not None:
            members=list(cache.orders_for_exec_spawn(order.exec_spawn_id))
    result={};count=0
    for member in members:
        events=member.events() if callable(member.events) else member.events
        count+=len(events)
        if count>4096:raise ValueError('native correction recovery history exceeds bound')
        for event in events:
            if type(event) is not OrderFillVoided:continue
            if any(getattr(member,key)!=getattr(event,key) for key in
                ('trader_id','strategy_id','instrument_id','client_order_id','venue_order_id','account_id')):
                raise ValueError('native retained correction has foreign scope')
            raw=native_void_payload(event);eid=str(event.event_id)
            if eid in result and result[eid]!=raw:raise ValueError('conflicting retained native correction')
            result[eid]=raw
    return result
