"""Read an NA execution-spawn group as one admitted BT order.

This is a cache view. It owns no fills, child lifecycle, quantities or timers;
NA remains responsible for spawning, routing and canceling its own orders.
"""
from decimal import Decimal


def spawn_projection(cache, native, *, strategy_id, account_id, instruments):
    """Return a group view only for native algorithms which actually spawned.

    Every child must retain the native root ID and the exact admitted scope.
    The aggregate size is the official cache's reduced-primary + children total.
    An INITIALIZED primary is not a venue ACK or an independent outstanding size.
    """
    spawn_id=getattr(native,'exec_spawn_id',None)
    algorithm=getattr(native,'exec_algorithm_id',None)
    if spawn_id is None or algorithm is None:return None
    members=list(cache.orders_for_exec_spawn(spawn_id))
    if len(members)<=1:return None
    if len(members)>4096:raise ValueError('native execution-spawn group exceeds projection bound')
    root=cache.order(spawn_id)
    if root is None or root.client_order_id!=spawn_id:
        raise ValueError('native execution-spawn lacks authoritative primary')
    if root.strategy_id!=strategy_id or root.instrument_id not in instruments:
        raise ValueError('foreign native execution-spawn primary')
    seen=set()
    for order in members:
        cid=str(order.client_order_id)
        if (cid in seen or order.exec_spawn_id!=spawn_id
                or order.exec_algorithm_id!=algorithm or root.exec_algorithm_id!=algorithm
                or order.strategy_id!=strategy_id or order.instrument_id!=root.instrument_id
                or order.side!=root.side
                or (order.account_id is not None and str(order.account_id)!=account_id)
                or (order.account_id is None and order.status.name not in ('INITIALIZED','EMULATED','RELEASED','DENIED','REJECTED')
                    and not (order.client_order_id==spawn_id and order.status.name=='CANCELED'))):
            raise ValueError('conflicting native execution-spawn scope')
        seen.add(cid)
    if str(spawn_id) not in seen:raise ValueError('native execution-spawn cache omits primary')
    total=Decimal(str(cache.exec_spawn_total_quantity(spawn_id,False)))
    filled=Decimal(str(cache.exec_spawn_total_filled_qty(spawn_id,False)))
    admitted=Decimal(str(root.init_event.quantity))
    if total!=admitted:
        raise ValueError('native execution-spawn total differs from admitted primary quantity')
    if not total.is_finite() or not filled.is_finite() or total<=0 or not 0<=filled<=total:
        raise ValueError('invalid native execution-spawn cumulative quantity')
    quantities=[Decimal(str(order.filled_qty)) for order in members]
    if sum(quantities)!=filled:raise ValueError('inconsistent native execution-spawn cache quantities')
    weighted=Decimal(0);fee=Decimal(0);currency=None;fees_known=True
    from .order_fees import cumulative_native_commission
    for order,qty in zip(members,quantities):
        if not qty:continue
        price=Decimal(str(order.avg_px))
        if not price.is_finite():raise ValueError('native execution-spawn fill lacks average price')
        weighted+=qty*price
        amount=cumulative_native_commission(order)
        method=getattr(order,'commissions',None)
        if callable(method):currencies={str(value.currency) for value in method().values()}
        else:currencies={str(event.commission.currency) for event in order.events() if type(event).__name__=='OrderFilled'}
        if amount is None or len(currencies)!=1:
            fees_known=False;continue
        [observed]=currencies
        if currency is not None and currency!=observed:fees_known=False
        currency=observed;fee+=Decimal(str(amount))
    if filled==total:status='FILLED'
    elif root.is_closed and all(order.is_closed for order in members if order is not root):status=root.status.name
    elif filled:status='PARTIALLY_FILLED'
    elif any(order.status.name in ('ACCEPTED','PENDING_CANCEL','PENDING_UPDATE','TRIGGERED') for order in members):status='ACCEPTED'
    else:status='SUBMITTED'
    evidence=dict(type='native_execution_spawn',primary_id=str(spawn_id),
                  primary_status=root.status.name,algorithm_id=str(algorithm),
                  total_qty=str(total),filled_qty=str(filled),
                  members=[dict(client_order_id=str(order.client_order_id),status=order.status.name,
                                quantity=str(order.quantity),filled_qty=str(order.filled_qty),
                                native_event_id=str(order.last_event.event_id)) for order in members])
    return dict(client_oid=str(spawn_id),status=status,total_qty=float(total),
                filled_qty=float(filled),avg_price=float(weighted/filled) if filled else None,
                commission=float(fee) if filled and fees_known else None,native_evidence=evidence)
