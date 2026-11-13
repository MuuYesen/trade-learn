"""BT resample feed registration over the existing OHLCV data/clock path."""
from copy import copy

from tradelearn.engine.base import TimeFrame
from tradelearn.backtest.lines import Lines, LineSeries, line_names


class _ClosedBarResampler:
    """One incomplete bar; only completed buckets are emitted before EOF.

    Timestamps are source nanoseconds. EOF is an explicit feed-close signal,
    never inferred from one latest price or a wall-clock observation.
    """
    def __init__(self, period_ns):
        self.period_ns = period_ns
        self._end = None
        self._bar = None
        self._last = None
        self._closed = False
        self._eof_eos = None
        self._last_available = None

    def push(self, stamp, row, available=None):
        if self._closed:
            raise RuntimeError('resample source has closed')
        if self._last is not None and stamp <= self._last:
            raise ValueError('resample requires ordered distinct source bars; revisions need replay')
        previous = self._last
        available = stamp if available is None else available
        self._last = stamp
        self._last_available = available
        self._eof_eos = None
        # BT measures subday points from UTC midnight, not Unix epoch.
        # Non-divisor periods (7 seconds/minutes) expose the difference.
        day = stamp // 86_400_000_000_000 * 86_400_000_000_000
        end = day + -(-(stamp - day) // self.period_ns) * self.period_ns
        output = []
        onedge = stamp == end
        if (previous is not None and day != previous // 86_400_000_000_000 * 86_400_000_000_000
                and self._bar is not None and not onedge):
            # Fixed BT's default daily session ends at 23:59:59.999990.
            # Completion is discovered on this new-day source observation.
            eos = previous // 86_400_000_000_000 * 86_400_000_000_000 + 86_400_000_000_000 - 10_000
            output.append((eos, self._bar, available))
            self._bar = None
            # If EOF follows immediately, BT retains its last session end for
            # the remaining bar as well. Keep both deliveries, even equal labels.
            self._eof_eos = eos
        # BT consumes an exact boundary source bar before delivering the open
        # aggregate, including across a gap or midnight. A non-boundary source
        # which discovers completion delivers the old bar now, not at its label.
        if self._bar is not None and end != self._end and not onedge:
            output.append((self._end, self._bar, available))
            self._bar = None
        elif self._bar is not None and onedge:
            self._end = stamp
        if self._bar is None:
            self._end, self._bar = end, dict(row)
        else:
            self._bar['high'] = max(self._bar['high'], row['high'])
            self._bar['low'] = min(self._bar['low'], row['low'])
            self._bar['close'] = row['close']
            self._bar['volume'] += row['volume']
            for key, value in row.items():
                if key not in {'open', 'high', 'low', 'close', 'volume'}:
                    self._bar[key] = value
        if onedge:
            output.append((stamp, self._bar, available))
            self._bar = None
        return output

    def finish(self):
        self._closed = True
        label = self._eof_eos if self._eof_eos is not None else self._end
        available = max(label, self._last_available) if label is not None else None
        output = [] if self._bar is None else [(label, self._bar, available)]
        self._bar = None
        return output



class _ConfiguredSubdayResampler:
    """One open aggregate for the existing subday options and declared session."""
    def __init__(self,source,timeframe,compression,options):
        self.source=source;self.timeframe=timeframe;self.compression=compression
        self.unit={TimeFrame.MicroSeconds:1000,TimeFrame.Seconds:1_000_000_000,TimeFrame.Minutes:60_000_000_000}.get(timeframe,60_000_000_000)
        self.options=options;self._bar=None;self._last=None;self._last_available=None
        self._closed=False;self._eos=None;self._last_eos=None;self._count=0;self._delivered=None
    def _number(self,stamp):
        import pandas as pd
        from tradelearn.backtest.clock import date2num
        return date2num(pd.Timestamp(stamp,unit='ns'))
    def _datetime(self,stamp,local=False):
        from tradelearn.backtest.clock import num2date
        return num2date(self._number(stamp),tz=self.source._tz if local else None)
    def _point(self,stamp,local=False):
        d=self._datetime(stamp,local)
        point=d.hour*60+d.minute
        rest=d.second+d.microsecond
        if self.unit<60_000_000_000:
            point=point*60+d.second;rest=d.microsecond
        if self.unit==1000:point=point*1_000_000+d.microsecond;rest=0
        return point+self.options['boundoff'],rest
    def _check_eos(self,stamp,exact):
        if self._eos is None:
            end,_=self.source._session_end_at(self._number(stamp))
            import pandas as pd
            self._eos=pd.Timestamp(end).value
        now,end=self._number(stamp),self._number(self._eos)
        reached=now==end or (not exact and now>end and self._bar is not None and self._number(self._last)<=end)
        if reached:self._last_eos=self._eos;self._eos=None
        return reached
    def _merge(self,row,stamp):
        if self._bar is None:self._bar=dict(row)
        else:
            self._bar['high']=max(self._bar['high'],row['high'])
            self._bar['low']=min(self._bar['low'],row['low'])
            self._bar['close']=row['close'];self._bar['volume']+=row['volume']
            for name,value in row.items():
                if name not in {'open','high','low','close','volume'}:self._bar[name]=value
        self._last=stamp
    def _label(self,greater,*,boundary=False):
        if not boundary and not (self.options['bar2edge'] and self.options['adjbartime']):return self._last
        if self._eos is None:return self._last_eos
        import pandas as pd
        from datetime import timedelta
        from ._timezone import localize
        d=self._datetime(self._last,True)
        point,_=self._point(self._last,True)
        point=(point//self.compression+int(boundary or self.options['rightedge']))*self.compression
        days,within=divmod(point*self.unit,86_400_000_000_000)
        seconds,ns=divmod(within,1_000_000_000)
        hour,seconds=divmod(seconds,3600);minute,second=divmod(seconds,60)
        adjusted=d.replace(hour=int(hour),minute=int(minute),second=int(second),microsecond=int(ns//1000))+timedelta(days=int(days))
        label=pd.Timestamp(localize(adjusted,self.source._tz)).value
        return label if not greater or self._number(label)>self._number(self._last) else self._last
    def _emit(self,label,available):
        output=(label,self._bar,available)
        self._bar=None;self._delivered=label
        return output
    def push(self,stamp,row,available=None):
        if self._closed:raise RuntimeError('resample source has closed')
        available=stamp if available is None else available
        self._last_available=available
        if self._delivered is not None and self._number(stamp)<=self._number(self._delivered):
            if self.options['takelate']:
                from tradelearn.backtest.clock import num2date
                import pandas as pd
                late=pd.Timestamp(num2date(self._number(self._delivered)+.000001)).value
                self._merge(row,late)
            return []
        onedge=self._check_eos(stamp,True)
        if not onedge:
            point,rest=self._point(stamp,True)
            onedge=not rest and point%self.compression==0
        if onedge:self._merge(row,stamp)
        output=[]
        if self._bar is not None:
            complete=onedge
            if not complete:
                crossed=self._check_eos(stamp,False)
                if not crossed:
                    old,_=self._point(self._last);new,_=self._point(stamp)
                    crossed=new>old and (not self.options['bar2edge'] or new//self.compression>old//self.compression)
                if crossed:
                    if self.options['bar2edge']:complete=True
                    else:self._count+=1;complete=self._count%self.compression==0
            if complete:output.append(self._emit(stamp if onedge else self._label(True),available))
        if not onedge:self._merge(row,stamp)
        return output
    def finish(self):
        self._closed=True
        if self._bar is None:return []
        label=self._label(False)
        return [self._emit(label,max(label,self._last_available))]


class _ConfiguredCalendarResampler(_ConfiguredSubdayResampler):
    """Calendar completion over the same source/session clock and one aggregate."""
    def __init__(self,source,timeframe,compression,options):
        super().__init__(source,timeframe,compression,options)
        source_frame=getattr(source,'_timeframe',source.p.timeframe)
        source_comp=getattr(source,'_compression',source.p.compression)
        self._componly=source_frame==timeframe and compression%source_comp==0
    def _label(self,greater):
        if self.timeframe!=TimeFrame.Days:return self._last
        if not (self.options['bar2edge'] and self.options['adjbartime']):return self._last
        if self._eos is None:return self._last_eos
        from ._timezone import localize
        import pandas as pd
        d=self._datetime(self._last,True);end=self._datetime(self._eos)
        adjusted=d.replace(hour=end.hour,minute=end.minute,second=end.second,microsecond=end.microsecond)
        label=pd.Timestamp(localize(adjusted,self.source._tz)).value
        return label if not greater or self._number(label)>self._number(self._last) else self._last
    def _crossed(self,stamp):
        if self.timeframe==TimeFrame.Days:return self._check_eos(stamp,False)
        previous=self._datetime(self._last);current=self._datetime(stamp,True)
        if self.timeframe==TimeFrame.Weeks:
            # Fixed BT asks a supplied calendar for its last trading weekday
            # even on a generic crossing check. ISO-week rollover is only the
            # fallback without a calendar; otherwise compression is double-counted.
            if self.source._calendar is not None:
                return bool(self.source._calendar.last_weekday(current.date()))
            return current.isocalendar()[:2]>previous.isocalendar()[:2]
        if self.timeframe==TimeFrame.Months:return (current.year,current.month)>(previous.year,previous.month)
        return current.year>previous.year
    def push(self,stamp,row,available=None):
        if self._closed:raise RuntimeError('resample source has closed')
        available=stamp if available is None else available
        self._last_available=available;onedge=False;check=True
        if self._componly:
            import pandas as pd
            self._last_eos=pd.Timestamp(self.source._session_end_at(self._number(stamp))[0]).value
        elif self.timeframe==TimeFrame.Days:onedge=self._check_eos(stamp,True)
        elif self.source._calendar is not None:
            method={TimeFrame.Weeks:'last_weekday',TimeFrame.Months:'last_monthday',TimeFrame.Years:'last_yearday'}[self.timeframe]
            if getattr(self.source._calendar,method)(self._datetime(stamp,True).date()):
                check=False;self._count+=1;onedge=self._count%self.compression==0
        consumed=self._componly or onedge
        if consumed:self._merge(row,stamp)
        output=[]
        if self._bar is not None:
            # A calendar period edge suppresses generic crossing checks. It
            # does not itself complete a compressed bar until count divides.
            complete=onedge
            if not complete and check and (self._componly or self._crossed(stamp)):
                self._count+=1;complete=self._count%self.compression==0
            if complete:output.append(self._emit(stamp if onedge else self._label(True),available))
        if not consumed:self._merge(row,stamp)
        return output
    def check_source_clock(self,stamp):
        if self._bar is not None and not self._componly and self.timeframe==TimeFrame.Days:
            self._check_eos(stamp,False)


class _CalendarBarResampler:
    """Default UTC daily/ISO-week/month/year completion, one open aggregate."""
    def __init__(self, timeframe, compression, source_timeframe, source_compression):
        self.timeframe = timeframe
        self.compression = compression
        self._componly = (source_timeframe == timeframe and not compression % source_compression)
        self._bar = None
        self._last = None
        self._last_key = None
        self._available = None
        self._count = 0
        self._closed = False
        self._eof_eos = None

    def _key(self, stamp):
        import pandas as pd
        value = pd.Timestamp(stamp, unit='ns', tz='UTC')
        if self.timeframe == TimeFrame.Days:
            return value.date()
        if self.timeframe == TimeFrame.Weeks:
            return tuple(value.isocalendar()[:2])
        if self.timeframe == TimeFrame.Months:
            return value.year, value.month
        return value.year

    def _label(self):
        if self.timeframe == TimeFrame.Days:
            return self._last // 86_400_000_000_000 * 86_400_000_000_000 + 86_400_000_000_000 - 10_000
        return self._last

    def push(self, stamp, row, available=None, *, boundary_key=None):
        if self._closed:
            raise RuntimeError('resample source has closed')
        if self._last is not None and stamp <= self._last:
            raise ValueError('resample requires ordered distinct source bars; revisions need replay')
        available = stamp if available is None else available
        output = []
        self._eof_eos = None
        key = self._key(stamp) if boundary_key is None else boundary_key
        crossed = self._last is not None and key > self._last_key
        if not self._componly and crossed:
            if self.timeframe == TimeFrame.Days:
                self._eof_eos = self._label()
            self._count += 1
            if self._count % self.compression == 0 and self._bar is not None:
                output.append((self._label(), self._bar, available))
                self._bar = None
        self._last_key = key
        self._last, self._available = stamp, available
        if self._bar is None:
            self._bar = dict(row)
        else:
            self._bar['high'] = max(self._bar['high'], row['high'])
            self._bar['low'] = min(self._bar['low'], row['low'])
            self._bar['close'] = row['close']
            self._bar['volume'] += row['volume']
            for key, value in row.items():
                if key not in {'open', 'high', 'low', 'close', 'volume'}:
                    self._bar[key] = value
        if self._componly:
            self._count += 1
            if self._count % self.compression == 0:
                label = self._label()
                output.append((label, self._bar, available))
                self._bar = None
        return output

    def check_source_clock(self):
        # A registered raw time master calls BT's forced check when this feed
        # produced no bar. That re-arms the current daily session-end marker.
        self._eof_eos = None

    def finish(self):
        self._closed = True
        if self._bar is None:
            return []
        label = self._eof_eos if self._eof_eos is not None else self._label()
        result = [(label, self._bar, max(label, self._available))]
        self._bar = None
        return result

def resample_feed(source, *, clone=False, _target=None, **options):
    """Register completed historical bars; live/replay revisions remain separate.

    The preload table is materialized from the same bounded incremental
    completion protocol, not a pandas resample substitute. No second broker,
    indicator engine or event loop is created.
    """
    import pandas as pd
    from tradelearn.engine.datafeed import DataFeed
    defaults = dict(timeframe=TimeFrame.Days, compression=1, bar2edge=True,
                    adjbartime=True, rightedge=True, boundoff=0, takelate=True)
    unknown = options.keys() - defaults.keys()
    if unknown:
        raise TypeError(f'unknown resample parameters: {sorted(unknown)}')
    values = {**defaults, **options}
    nondefault=any(values[key]!=defaults[key] for key in ('bar2edge','adjbartime','rightedge','boundoff','takelate'))
    if type(values['boundoff']) is not int:raise TypeError('resample boundoff must be integer')
    timeframe, compression = values['timeframe'], values['compression']
    if type(compression) is not int or compression <= 0:
        raise ValueError('positive integer resample compression required')
    mapping = {TimeFrame.MicroSeconds:1000,TimeFrame.Seconds: 1_000_000_000,
               TimeFrame.Minutes: 60_000_000_000}
    calendar_frames = (TimeFrame.Days, TimeFrame.Weeks, TimeFrame.Months, TimeFrame.Years)
    if timeframe not in mapping and timeframe not in calendar_frames:
        raise NotImplementedError('subsecond resample incremental delivery is not yet implemented')
    frame = getattr(source, '_frame', None)
    if frame is None or getattr(source, 'is_live', False):
        if getattr(source, 'is_live', False) and getattr(source, '_store', None) is not None:
            from tradelearn_runtime.bar_views import register_resample
            return register_resample(source, values, clone=clone)
        raise NotImplementedError('stream resample delivery is not yet implemented')
    declared_session=(source._tz is not None or getattr(source.p,'sessionstart',None) not in (None,__import__('datetime').time.min)
                      or getattr(source.p,'sessionend',None) not in (None,__import__('datetime').time(23,59,59,999990)))
    configured=(source._calendar is not None or nondefault or declared_session or timeframe==TimeFrame.MicroSeconds or source._frame.index.has_duplicates or not source._frame.index.is_monotonic_increasing)
    builder = (_ConfiguredSubdayResampler(source,timeframe,compression,values) if timeframe in mapping and configured else
               _ClosedBarResampler(mapping[timeframe]*compression) if timeframe in mapping else
               _ConfiguredCalendarResampler(source,timeframe,compression,values) if configured else
               _CalendarBarResampler(timeframe, compression, source.p.timeframe, source.p.compression))
    output = []
    source_availability = getattr(source, '_delivery_datetime', None)
    pending = None
    for offset, (stamp, row) in enumerate(frame.iterrows()):
        available = stamp.value if source_availability is None else int(source_availability[offset]) * (1 if getattr(source, '_delivery_unit', 's') == 'ns' else 1_000_000_000)
        if pending is not None:
            # Fixed BT's clone advances an already buffered future-labelled bar
            # before asking its source for another row. Delivery consumes this
            # source-clock turn, without pretending the pending bar was earlier.
            if stamp.value >= pending[0]:
                output.append((pending[0], pending[1], available))
                pending = None
            continue
        produced = builder.push(stamp.value, row.to_dict(), available)
        if clone and not produced:
            if isinstance(builder,_ConfiguredCalendarResampler):builder.check_source_clock(available)
            elif isinstance(builder,_CalendarBarResampler):builder.check_source_clock()
        for label, values, ready in produced:
            if clone and label > ready:
                pending = (label, values, ready)
            else:
                output.append((label, values, ready))
    completed = len(output)
    tail = builder.finish()
    if pending is not None:
        last_ready = (frame.index[-1].value if source_availability is None else
                      int(source_availability[-1]) * (1 if getattr(source, '_delivery_unit', 's') == 'ns' else 1_000_000_000))
        tail.append((pending[0], pending[1], max(pending[0], last_ready)))
    output.extend(tail)
    # Fixed BT does not nullify its last execution tick when last() delivers an
    # EOF bar. Preserve that quote separately; never overwrite the OHLCV bar.
    execution_rows = [row for _stamp, row, _available in output]
    if tail and completed:
        execution_rows[-1] = output[completed - 1][1]
    stamps = [stamp for stamp, _row, _available in output]
    index = pd.to_datetime(stamps, unit='ns', utc=frame.index.tz is not None)
    if frame.index.tz is not None:
        index = index.tz_convert(frame.index.tz)
    sampled = pd.DataFrame([row for _stamp, row, _available in output], index=index, columns=frame.columns)
    availability = [available for _stamp, _row, available in output]
    # A resampled feed is not the time master while its original source is
    # registered. Future labels wait for the next source observation; EOF is
    # a distinct delivery phase after the final source bar, not an epsilon time.
    phases = [0] * len(output)
    last_available = (int(source_availability[-1]) * (1 if getattr(source, '_delivery_unit', 's') == 'ns' else 1_000_000_000)
                      if source_availability is not None and len(source_availability) else
                      frame.index[-1].value if len(frame) else None)
    if not clone:
        availability = [max(label, available) for (label, _row, _original), available in zip(output, availability)]
    if tail and last_available is not None and availability[-1] == last_available:
        phases[-1] = 1
    fractional_delivery = any(stamp % 1_000_000_000 for stamp in availability)
    target = _target if _target is not None else copy(source) if clone else source
    previous_container = target.lines if _target is not None else None
    previous_lines = ({name: getattr(target.lines,name) for name in line_names(type(target))
                       if hasattr(target.lines,name)} if _target is not None else {})
    parameters = source.p.asdict()
    name = source._name
    target.lines = Lines(target)
    DataFeed.__init__(target, sampled, name=name,_preserve_row_order=True,
                      **{key: value for key, value in parameters.items() if key != 'name'})
    # BT keeps original feed p.timeframe/compression and changes actual metadata.
    target._timeframe, target._compression = timeframe, compression
    target._calendar=source._calendar
    if clone:
        target._eos_source=source
    else:
        vars(target).pop('_eos_source',None)
    import numpy as np
    target._execution_quote_values = np.asarray(
        [[row[name] for name in ('open', 'high', 'low', 'close')] for row in execution_rows],
        dtype=np.float64).reshape((-1, 4))
    target._delivery_phase = np.asarray(phases, dtype=np.int64)
    target._delivery_unit = 'ns' if fractional_delivery else 's'
    target._delivery_datetime = np.asarray(availability, dtype=np.int64)
    if not fractional_delivery:
        target._delivery_datetime = target._delivery_datetime // 1_000_000_000
    for name in line_names(type(target)):
        if name not in {'datetime', 'open', 'high', 'low', 'close', 'volume'}:
            line = LineSeries(target.get_array(name))
            line._cursor_source = target
            setattr(target, name, line)
            setattr(target.lines, name, line)
            target._lines_list.append(line)
    if previous_container is not None:
        replacements = {}
        for name, previous in previous_lines.items():
            current = getattr(target.lines,name)
            previous.__dict__.clear()
            previous.__dict__.update(current.__dict__)
            replacements[id(current)] = previous
            setattr(target,name,previous)
            setattr(target.lines,name,previous)
        target._lines_list = [replacements.get(id(line),line) for line in target._lines_list]
        previous_container.__dict__.update(target.lines.__dict__)
        target.lines = previous_container
    return target
