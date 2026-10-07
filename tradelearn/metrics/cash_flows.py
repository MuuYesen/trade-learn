"""Returns for an explicitly supplied external-flow valuation interval."""

import math


def cash_flow_return(
    begin, finish, *, net_flow=None, events=(), start=None, end=None, method="end_of_day"
):
    """Positive flow denotes a deposit; Modified Dietz is an approximation.

    Callers must establish completeness, currency and account identity. Event
    timestamps and valuation endpoints must be timezone-aware actual instants.
    """

    def number(value):
        if isinstance(value, bool):
            raise ValueError("boolean is not an amount")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite amount")
        return result

    begin, finish = number(begin), number(finish)
    if method not in ("end_of_day", "modified_dietz"):
        raise ValueError("unsupported cash flow method")
    total = weighted = 0.0
    if events:
        if (
            start is None
            or end is None
            or start.utcoffset() is None
            or end.utcoffset() is None
            or end <= start
        ):
            raise ValueError("aware valuation interval required")
        for instant, amount in events:
            if instant.utcoffset() is None or not start < instant <= end:
                raise ValueError("event outside valuation interval")
            if method == "end_of_day" and instant != end:
                raise ValueError("intraday flow requires modified_dietz")
            amount = number(amount)
            total += amount
            weighted += amount * (end - instant).total_seconds() / (end - start).total_seconds()
        if net_flow is not None and not math.isclose(number(net_flow), total, abs_tol=1e-8):
            raise ValueError("cash flow total disagrees with events")
    elif net_flow is not None:
        total = number(net_flow)
        if method == "modified_dietz" and total:
            raise ValueError("timed events required for modified_dietz")
    else:
        raise ValueError("explicit cash flow required")
    denominator = begin + (weighted if method == "modified_dietz" else 0)
    if denominator <= 0:
        raise ValueError("positive invested capital required")
    result = (finish - begin - total) / denominator
    if not math.isfinite(result) or result < -1:
        raise ValueError("invalid adjusted return")
    return result
