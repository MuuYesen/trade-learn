"""One batch event calendar; input feeds retain their own delivered cursors."""

import numpy as np


def native_timestamps(data, unit='s', *, availability=False):
    """Adapt the declared clock unit to the same existing native int64 ABI."""
    if availability and hasattr(data, '_delivery_datetime'):
        values = np.asarray(data._delivery_datetime, dtype=np.int64)
        source_unit = getattr(data, '_delivery_unit', 's')
    elif getattr(data, '_clock_unit', 's') == 'ns':
        values = np.asarray(data._clock_timestamps, dtype=np.int64)
        source_unit = 'ns'
    else:
        values = np.asarray(data._datetime, dtype=np.int64)
        source_unit = 's'
    if source_unit == unit:
        return values
    if source_unit == 's' and unit == 'ns':
        bounds = np.iinfo(np.int64)
        if len(values) and (int(values.max()) > bounds.max // 1_000_000_000
                            or int(values.min()) < -((-bounds.min) // 1_000_000_000)):
            raise ValueError('bar time exceeds signed native nanosecond clock range')
        return values * 1_000_000_000
    raise ValueError('native nanosecond clock cannot be truncated to seconds')


class BatchClock:
    def __init__(self, datas):
        self.has_delivery_views=any(getattr(data,'_native_delivery_views',False) for data in datas)
        self.unit = ('ns' if any(getattr(data, '_clock_unit', 's') == 'ns'
                                  or getattr(data, '_delivery_unit', 's') == 'ns' for data in datas) else 's')
        labels = [np.asarray(data._clock_view_timestamps,dtype=np.int64) if hasattr(data,'_clock_view_timestamps')
                  else native_timestamps(data, self.unit) for data in datas]
        self._sources = [native_timestamps(data, self.unit, availability=True) for data in datas]
        self._delivery_calendar = any(not np.array_equal(label, source)
                                      for label, source in zip(labels, self._sources))
        phases = [np.asarray(getattr(data, '_delivery_phase', np.zeros(len(source),dtype=np.int64)))
                  for data,source in zip(datas,self._sources)]
        # An adjusted replay first publishes the previous logical bar, then the
        # incoming raw observation. A registered original feed shares that
        # latter turn; its raw timestamps and rows remain unchanged.
        for child in datas:
            parent=getattr(child,'_replay_source',None)
            parent_index=next((i for i,data in enumerate(datas) if data is parent),None)
            if parent_index is not None:
                projected=getattr(child,'_replay_source_phases',None)
                if projected is None:
                    projected=np.zeros(len(self._sources[parent_index]),dtype=np.int64)
                    np.maximum.at(projected,child._source_observation_positions,child._delivery_phase)
                phases[parent_index]=np.maximum(phases[parent_index],projected)
        explicit_phases = any(np.any(phase) for phase in phases)
        self._search_sources = self._sources
        if explicit_phases:
            dtype = np.dtype([('time', np.int64), ('phase', np.int64)])
            self._search_sources = []
            for data, source, phase in zip(datas, self._sources, phases):
                keys = np.empty(len(source), dtype=dtype)
                keys['time'] = source
                implicit = np.arange(len(source)) - np.searchsorted(source, source, side='left')
                keys['phase'] = np.maximum(implicit, phase)
                self._search_sources.append(keys)
            self._delivery_calendar = True

        # Keep the common aligned path zero-copy. Only duplicate availability
        # needs occurrence ranks; never invent epsilon timestamps for dispatch.
        aligned = all(np.array_equal(self._sources[0], source) for source in self._sources[1:])
        repeated = any(np.any(source[1:] == source[:-1]) for source in self._sources)
        self._ranks = None
        if explicit_phases:
            delivery = np.unique(np.concatenate(self._search_sources))
        elif aligned and not repeated:
            delivery = self._sources[0]
        else:
            unique = np.unique(np.concatenate(self._sources))
            if repeated:
                multiplicity = np.zeros(len(unique), dtype=np.int64)
                for source in self._sources:
                    counts = (np.searchsorted(source, unique, side='right') -
                              np.searchsorted(source, unique, side='left'))
                    np.maximum(multiplicity, counts, out=multiplicity)
                delivery = np.repeat(unique, multiplicity)
                self._ranks = np.concatenate([np.arange(n, dtype=np.int64) for n in multiplicity]) if len(unique) else np.array([], dtype=np.int64)
            else:
                delivery = unique
        self._delivery_keys = delivery
        self._deliveries = delivery['time'] if explicit_phases else delivery
        if self._delivery_calendar:
            # Dispatch follows availability; Strategy time remains the maximum
            # already visible label. Compute one source's cursor vector at a
            # time instead of retaining an events-by-feeds position matrix.
            timestamps = np.full(len(delivery), np.iinfo(np.int64).min, dtype=np.int64)
            for data,label,source in zip(datas,labels,self._search_sources):
                positions = self._source_positions(source, delivery, self._ranks)
                valid = positions >= 0
                logical=getattr(data,'_replay_lengths',None)
                if logical is not None:valid[valid]&=logical[positions[valid]]>0
                timestamps[valid] = np.maximum(timestamps[valid], label[positions[valid]])
            self.timestamps = timestamps
            self.requires_union = True
        else:
            self.timestamps = delivery
            self.requires_union = not np.array_equal(delivery, labels[0])
        self.unixseconds = self.timestamps if self.unit == 's' else self.timestamps / 1_000_000_000.0
        self.strategy_lengths = None
        if any(hasattr(data, '_replay_lengths') for data in datas):
            advanced = np.zeros(len(delivery), dtype=bool)
            for data, source in zip(datas, self._search_sources):
                positions = self._source_positions(source, delivery, self._ranks)
                lengths = np.zeros(len(positions), dtype=np.int64)
                valid = positions >= 0
                logical = getattr(data, '_replay_lengths', None)
                lengths[valid] = positions[valid]+1 if logical is None else logical[positions[valid]]
                advanced |= np.diff(lengths, prepend=0)>0
            self.strategy_lengths = np.cumsum(advanced)


    def __len__(self):
        return len(self.timestamps)

    @staticmethod
    def _source_positions(source, delivery, ranks=None):
        right = np.searchsorted(source, delivery, side='right')
        if ranks is None:
            return right - 1
        left = np.searchsorted(source, delivery, side='left')
        return left - 1 + np.minimum(ranks + 1, right - left)

    def cursors_at(self, cursor):
        timestamp = self._delivery_keys[cursor]
        rank = None if self._ranks is None else self._ranks[cursor]
        return [int(self._source_positions(source, timestamp, rank)) for source in self._search_sources]


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


def bt_date_upper_tick(value, unit='s'):
    """Last raw tick represented by a fixed BT float-day (microsecond input)."""
    import math
    from decimal import Decimal, ROUND_FLOOR, localcontext
    with localcontext() as context:
        context.prec = 50
        current = Decimal.from_float(float(value))
        following = Decimal.from_float(math.nextafter(float(value), math.inf))
        midpoint_us = ((current + following) / 2 - Decimal(719163)) * Decimal(86400000000)
        microseconds = int(midpoint_us.to_integral_value(rounding=ROUND_FLOOR))
    upper_ns = microseconds * 1000 + 999
    return upper_ns if unit == 'ns' else upper_ns // 1_000_000_000


def bt_date_rank(value):
    """Exact ordered positive IEEE BT ordinal, distinct from native source time."""
    import math,struct
    value=float(value)
    if not math.isfinite(value) or value<=0:raise ValueError('finite positive BT date required')
    return struct.unpack('>q',struct.pack('>d',value))[0]
