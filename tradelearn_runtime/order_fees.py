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
            if len(fees) != 1:
                return None
            value = float(fees[0].as_double())
            return value if math.isfinite(value) else None
        events = order.events() if callable(order.events) else order.events
        if len(events) > 4096:
            return None
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
