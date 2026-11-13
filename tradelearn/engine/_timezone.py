"""Declared feed timezone interpretation; internal clocks remain UTC nanoseconds."""
from datetime import timedelta
from zoneinfo import ZoneInfo


def resolve_timezone(value):
    return ZoneInfo(value) if isinstance(value,str) else value


def localize(value,zone):
    if zone is None:return value
    if value.tzinfo is not None:raise ValueError('local timezone input must be naive')
    method=getattr(zone,'localize',None)
    if callable(method):return method(value)
    first=value.replace(tzinfo=zone,fold=0)
    second=value.replace(tzinfo=zone,fold=1)
    # Fixed BT/pytz's unspecified is_dst selects standard time for folds and
    # spring gaps. Keep this choice rather than dropping/reordering source rows.
    if first.utcoffset()!=second.utcoffset() and second.dst()==timedelta() and first.dst()!=timedelta():
        return second
    return first
