"""Read cumulative fees from fixed native facts; unavailable remains unknown."""
from decimal import Decimal
import math


def cumulative_native_commission(order):
    """Return one-currency complete fees, without reconstructing account value.

    rc7 conditional order classes lack commissions(). Their retained OrderFilled
    events are usable only when quantities cover the current native filled_qty.
    A missing/voided/truncated/mixed-currency history is not a zero fee.
    """
    try:
        method = getattr(order, "commissions", None)
        if callable(method):
            fees = list(method().values())
            if len(fees) == 1:
                value = float(fees[0].as_double())
                return value if math.isfinite(value) else None
            if fees:return None
            # An empty canonical fee map is not generally a known zero. A
            # complete retained native void history can prove that zero.
        events = order.events() if callable(order.events) else order.events
        if len(events) > 4096:
            return None
        if any(type(event).__name__ == 'OrderFillVoided' for event in events):
            return _native_voided_commission(order,events)
        if callable(method):return None
        quantity = Decimal(0)
        total = Decimal(0)
        currency = None
        seen = {}
        for event in events:
            kind = type(event).__name__
            if kind == "OrderFillVoided":
                return None
            if kind != "OrderFilled":
                continue
            fee = event.commission
            event_currency = str(fee.currency)
            amount = Decimal(str(fee.as_double()))
            last_qty = Decimal(str(event.last_qty))
            if not amount.is_finite() or not last_qty.is_finite() or last_qty <= 0:
                return None
            identity = str(event.event_id)
            payload = (last_qty, amount, event_currency)
            if identity in seen:
                if seen[identity] != payload:
                    return None
                continue
            seen[identity] = payload
            if currency is not None and currency != event_currency:
                return None
            currency = event_currency
            quantity += last_qty
            total += amount
        if not seen or quantity != Decimal(str(order.filled_qty)):
            return None
        value = float(total)
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return None


def _native_voided_commission(order,events):
    """Bounded projection of actual native cumulative correction facts only."""
    from nautilus_trader.model import OrderFilled,OrderFillVoided
    fills={};voids={};seen={};currency=None
    try:
        for event in events:
            if type(event) not in (OrderFilled,OrderFillVoided):continue
            for field in ('trader_id','strategy_id','instrument_id','client_order_id','venue_order_id','account_id'):
                if getattr(event,field)!=getattr(order,field):return None
            trade=str(event.trade_id)
            if type(event) is OrderFilled:
                if event.commission is None:return None
                qty=Decimal(str(event.last_qty));fee=Decimal(str(event.commission.as_decimal()))
                code=str(event.commission.currency)
                value=(qty,fee,code)
                if qty<=0 or not qty.is_finite() or not fee.is_finite():return None
                if trade in fills and fills[trade]!=value:return None
                fills[trade]=value
            else:
                if trade not in fills or event.commission_voided is None:return None
                qty=Decimal(str(event.voided_qty));fee=Decimal(str(event.commission_voided.as_decimal()))
                code=str(event.commission_voided.currency)
                value=(trade,qty,fee,code,bool(event.is_reopened))
                identity=str(event.correction_id)
                if identity in seen:
                    if seen[identity]!=value:return None
                    continue
                seen[identity]=value
                if (not qty.is_finite() or qty<=0 or qty>fills[trade][0] or not fee.is_finite()
                        or code!=fills[trade][2] or code!=str(event.currency)):return None
                previous=voids.get(trade,(Decimal(0),Decimal(0)))
                if qty<previous[0]:return None
                voids[trade]=(qty,fee)
            if currency is not None and currency!=code:return None
            currency=code
        if not fills or not seen:return None
        quantity=sum((qty-voids.get(trade,(0,0))[0] for trade,(qty,_,_) in fills.items()),Decimal(0))
        if quantity!=Decimal(str(order.filled_qty)):return None
        fee=sum((amount-voids.get(trade,(0,0))[1] for trade,(_,amount,_) in fills.items()),Decimal(0))
        value=float(fee)
        return value if math.isfinite(value) else None
    except (AttributeError,TypeError,ValueError,ArithmeticError):return None
