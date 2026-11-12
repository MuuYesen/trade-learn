"""One batch event calendar; input feeds retain their own delivered cursors."""

import numpy as np


class BatchClock:
    def __init__(self, datas):
        self._sources = [np.asarray(data._datetime, dtype=np.int64) for data in datas]
        timestamps = self._sources[0]
        for source in self._sources[1:]:
            if not np.array_equal(timestamps, source):
                timestamps = np.union1d(timestamps, source)
        self.timestamps = timestamps
        self.requires_union = not np.array_equal(timestamps, self._sources[0])

    def __len__(self):
        return len(self.timestamps)

    def cursors_at(self, cursor):
        timestamp = self.timestamps[cursor]
        return [int(np.searchsorted(source, timestamp, side="right")) - 1
                for source in self._sources]


def date2num(dt, tz=None):
    """Backtrader Gregorian day number; internal execution clocks stay Unix."""
    import math
    if tz is not None:
        dt = tz.localize(dt) if hasattr(tz, "localize") else dt.replace(tzinfo=tz)
    if getattr(dt, "tzinfo", None) is not None:
        offset = dt.utcoffset()
        if offset is not None:
            dt = dt - offset
    fractions = (dt.hour / 24, dt.minute / 1440, dt.second / 86400,
                 dt.microsecond / 86400000000) if hasattr(dt, "hour") else ()
    return math.fsum((float(dt.toordinal()), *fractions))


def num2date(x, tz=None, naive=True):
    import datetime
    day = int(x)
    hour, fraction = divmod((float(x) - day) * 24, 1)
    minute, fraction = divmod(fraction * 60, 1)
    second, fraction = divmod(fraction * 60, 1)
    microsecond = int(fraction * 1000000)
    if microsecond < 10:
        microsecond = 0
    result = datetime.datetime.combine(datetime.date.fromordinal(day),
        datetime.time(int(hour), int(minute), int(second), microsecond))
    if tz is not None:
        result = result.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
        if naive:
            result = result.replace(tzinfo=None)
    if microsecond > 999990:
        result += datetime.timedelta(microseconds=1000000-microsecond)
    return result
