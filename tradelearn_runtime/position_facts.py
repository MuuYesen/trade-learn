"""Select the native position for this event, never the next netting cycle."""
from math import fsum,isfinite,nan


def _matches(position,event,closed,*,archived=False):
    if position is None:
        return False
    expected=str(event.position_id)
    actual=str(position.id)
    if actual != expected and not (archived and actual.startswith(expected+'-')):
        return False
    for name in ('account_id','strategy_id','instrument_id','opening_order_id'):
        if str(getattr(position,name,None)) != str(getattr(event,name,None)):
            return False
    if bool(position.is_closed) != closed:
        return False
    if position.ts_last != event.ts_event or position.signed_qty != event.signed_qty:
        return False
    if position.avg_px_open != event.avg_px_open:
        return False
    if str(position.realized_pnl) != str(event.realized_pnl):
        return False
    if closed:
        if position.ts_closed != event.ts_closed or str(position.closing_order_id) != str(event.closing_order_id):
            return False
    return True


def native_position_for_event(cache,event,closed):
    """Read current facts or an identity-checked official archived close.

    Only one matching view is retained. The fixed native snapshot API returns
    that position ID's archive; this does not copy or replay fill history.
    """
    position=cache.position(event.position_id)
    if _matches(position,event,closed):
        return position,'current'
    if not closed:
        return None,'current native position does not match event'
    match=None
    fingerprint=None
    for snapshot in cache.position_snapshots(event.position_id,event.account_id):
        if not _matches(snapshot,event,closed,archived=True):
            continue
        fees=tuple(str(fee) for fee in snapshot.commissions())
        if match is not None and fees != fingerprint:
            return None,'conflicting native closed snapshot commissions'
        match,fingerprint=snapshot,fees
    if match is None:
        return None,'matching native closed snapshot unavailable'
    return match,'closed snapshot'



def native_trade_economics(cache,event,closed):
    """rc7 realized_pnl is net of its native cumulative position commissions.

    Gross PnL is only a projection when same-currency cumulative fees exist.
    The native net fact remains available when those fees cannot be qualified.
    No account value, cash, matching or fill history is reconstructed here.
    """
    net=nan if event.realized_pnl is None else float(event.realized_pnl.as_double())
    if not isfinite(net):
        net=nan
    result=dict(pnl=nan,pnlcomm=net,commission=nan,value=0. if closed and event.signed_qty==0 else nan,
                currency=str(event.currency),value_currency=None)
    if event.realized_pnl is not None and str(event.realized_pnl.currency)!=str(event.currency):
        result.update(pnlcomm=nan,source='native PnL and event currency conflict')
        return result
    position,source=native_position_for_event(cache,event,closed)
    result['source']=source
    if position is None:
        return result
    from nautilus_trader.model import Price
    if not closed:
        notional=position.notional_value(Price.from_str(str(event.avg_px_open)))
        result['value']=float(notional.as_double())
        result['value_currency']=str(notional.currency)
    else:
        result['value_currency']=str(position.settlement_currency)
    fees=list(position.commissions())
    if len(fees)!=1 or any(str(fee.currency)!=str(event.currency) for fee in fees):
        result['source']+='; complete same-currency commissions unavailable'
        return result
    amounts=[float(fee.as_double()) for fee in fees]
    if not all(isfinite(value) for value in amounts):
        result['source']+='; nonfinite native commission'
        return result
    result['commission']=fsum(amounts)
    result['pnl']=net+result['commission']
    return result
