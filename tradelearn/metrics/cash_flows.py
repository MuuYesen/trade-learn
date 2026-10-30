"""Returns for an explicitly supplied external-flow valuation interval."""
import math


def cash_flow_return(begin, finish, *, net_flow=None, events=(), start=None, end=None,
                     method='end_of_day', allow_insolvency=False):
    """Positive flow denotes a deposit; Modified Dietz is an approximation.

    Callers must establish completeness, currency and account identity. Event
    timestamps and valuation endpoints must be timezone-aware actual instants.
    ``allow_insolvency=True`` explicitly permits a finite return below -100%
    for leveraged-account risk observations. It does not relax the positive
    invested-capital denominator or allow nonfinite inputs/results. The default
    retains the ordinary investment-return domain.
    """
    def number(value):
        if isinstance(value, bool):
            raise ValueError('boolean is not an amount')
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite amount')
        return result

    if type(allow_insolvency) is not bool:
        raise ValueError('allow_insolvency must be boolean')
    begin, finish = number(begin), number(finish)
    if method not in ('end_of_day', 'modified_dietz'):
        raise ValueError('unsupported cash flow method')
    total = weighted = 0.0
    if events:
        if start is None or end is None or start.utcoffset() is None or end.utcoffset() is None or end <= start:
            raise ValueError('aware valuation interval required')
        for instant, amount in events:
            if instant.utcoffset() is None or not start < instant <= end:
                raise ValueError('event outside valuation interval')
            if method == 'end_of_day' and instant != end:
                raise ValueError('intraday flow requires modified_dietz')
            amount = number(amount)
            total += amount
            weight = (end-instant).total_seconds() / (end-start).total_seconds()
            weighted += amount * weight
        if net_flow is not None and not math.isclose(number(net_flow), total, abs_tol=1e-8):
            raise ValueError('cash flow total disagrees with events')
    elif net_flow is not None:
        total = number(net_flow)
        if method == 'modified_dietz' and total:
            raise ValueError('timed events required for modified_dietz')
    else:
        raise ValueError('explicit cash flow required')
    denominator = begin + (weighted if method == 'modified_dietz' else 0)
    if not all(math.isfinite(value) for value in (total, weighted, denominator)):
        raise ValueError('nonfinite cash flow arithmetic')
    if denominator <= 0:
        raise ValueError('positive invested capital required')
    result = (finish - begin - total) / denominator
    if not math.isfinite(result) or (result < -1 and not allow_insolvency):
        raise ValueError('invalid adjusted return')
    return result
