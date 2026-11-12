"""Copy public Na cache identity into TL notifications; never infer executions."""

def notification_evidence(order):
    try:
        trades = order.trade_ids
        if len(trades) > 512:
            return None
        return dict(instrument_id=str(order.instrument_id),
            account_id=str(order.account_id) if order.account_id is not None else None,
            strategy_id=str(order.strategy_id), status=order.status.name,
            cumulative_filled_qty=str(order.filled_qty), trade_ids=[str(t) for t in trades])
    except Exception:
        # Optional observation failures must not change submission/projection policy.
        return None
