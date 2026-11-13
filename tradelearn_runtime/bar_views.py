"""Bounded bar projections on the existing native source callback.

There is no source polling, execution state or synthetic EOF here. Closing a
node leaves an incomplete aggregate incomplete.
"""
from tradelearn.engine.base import TimeFrame


def register_resample(source, options, *, clone, kind='resample'):
    store = source._store
    if store._used or source not in store._feeds:
        raise ValueError('live resample requires an unstarted owned subscription')
    if getattr(source, '_live_bar_recipe', None) is not None:
        raise NotImplementedError('chained live bar projections are not yet integrated')
    if getattr(store, '_driver', None) is not None:
        raise NotImplementedError('driver-owned bar transformations require its source lifecycle')
    from tradelearn.backtest.parameters import Params
    recipe = dict(kind=kind, source_period=source.p.asdict(), options=dict(options), clone=clone)
    # Validate/build before mutating the inert subscription.
    make_live_view(recipe)
    from copy import copy
    target = copy(source) if clone else source
    if clone:
        # BT's DataClone has its own default empty name. Cerebro.adddata may
        # subsequently assign an explicit name; it does not inherit the guest's
        # registration alias.
        target._name = ''
        target._eos_source = source
    target._live_bar_recipe = recipe
    # BT's filter changes actual timeframe/compression, not the source feed's
    # constructor params. Filter options remain in this private recipe.
    target.p = target.params = Params(source.p.asdict())
    target._timeframe = options['timeframe']
    target._compression = options['compression']
    if clone:
        store._feeds.append(target)
    return target


def make_live_view(recipe):
    return LiveReplay(recipe) if recipe['kind'] == 'replay' else LiveResample(recipe)


def _source(recipe):
    import pandas as pd
    from tradelearn.engine.datafeed import DataFeed
    source = DataFeed(pd.DataFrame(columns=['open','high','low','close','volume'],
        index=pd.DatetimeIndex([],tz='UTC')), **recipe['source_period'])
    if recipe.get('calendar') is not None:
        source._calendar = recipe['calendar']
    return source


def feed_context(metadata):
    """Reuse the same source/session conversion as the historical BT feed."""
    from datetime import datetime,time,tzinfo
    values = dict(metadata)
    zone = values.get('tz')
    if zone is not None and not isinstance(zone,(str,tzinfo)):
        raise TypeError('feed tz must be a timezone name or tzinfo')
    for key,default in (('sessionstart',time.min),('sessionend',time(23,59,59,999990))):
        value = values.get(key)
        if isinstance(value,datetime):
            value = value.time()
        if value is None:
            value = default
        if not isinstance(value,time):
            raise TypeError(f'{key} must be time or datetime')
        values[key] = value
    return _source(dict(source_period=values))


class LiveReplay:
    def __init__(self, recipe):
        from tradelearn.engine.replaying import (_ConfiguredReplaySubday,
            _ConfiguredReplayCalendar, _ReplayBarBuilder)
        values = recipe['options']
        self.clone = recipe.get('clone',False)
        self.pending = self.published = self.previous = None
        timeframe, compression = values['timeframe'], values['compression']
        source = _source(recipe)
        if timeframe < TimeFrame.Days:
            self.builder = _ConfiguredReplaySubday(source,timeframe,compression,values)
        else:
            from datetime import time
            declared_session = (source._tz is not None or
                getattr(source.p,'sessionstart',None) not in (None,time.min) or
                getattr(source.p,'sessionend',None) not in (None,time(23,59,59,999990)))
            configured = source._calendar is not None or declared_session or any(
                values[name] != default for name,default in
                (('bar2edge',True),('adjbartime',False),('rightedge',True),('boundoff',0),('takelate',True)))
            self.builder = (_ConfiguredReplayCalendar(source,timeframe,compression,values)
                if configured else _ReplayBarBuilder(timeframe,compression,
                    source.p.timeframe,source.p.compression))

    def _publish(self,count,label,values):
        from copy import deepcopy
        if self.published is None or count>self.published[0]:
            self.previous=self.published
        self.published=(count,label,deepcopy(values))

    def _checked_close(self,observed_ns):
        from copy import deepcopy
        state=getattr(self.builder,'state',None)
        if (state is None or state.timeframe>=TimeFrame.Days or state._bar is None or
                not (self.builder.options['bar2edge'] and self.builder.options['adjbartime'])):
            return None
        crossed=state._check_eos(observed_ns,False)
        if not crossed:
            old,_=state._point(state._last);now,_=state._point(observed_ns)
            crossed=now>old and now//self.builder.compression>old//self.builder.compression
        if not crossed:return None
        label=state._label(True)
        result=(self.builder.count,label,deepcopy(state._bar))
        state._bar=None
        return result if state._number(label)>state._number(state._last) else None

    def next_check_ns(self):
        state=getattr(self.builder,'state',None)
        if (self.pending is not None or state is None or state._bar is None or
                state.timeframe>=TimeFrame.Days or not
                (self.builder.options['bar2edge'] and self.builder.options['adjbartime'])):
            return None
        deadline=state._label(True,boundary=True)
        if state._eos is not None:deadline=min(deadline,state._eos)
        return deadline if deadline>state._last else None

    def check(self,observed_ns):
        """Complete only the observed aggregate, on its existing native clock."""
        from copy import deepcopy
        closed=self._checked_close(observed_ns)
        if closed is None:return None
        count,label,values=closed
        self.builder._last_public=self.builder.state._number(label)
        if self.clone:
            self.pending=closed
            self._pending_from_idle=True
            view=(self.published if self.published is not None and self.published[0]<count else self.previous)
            if view is None:view=(0,label,values)
            return (view[1],deepcopy(view[2]),self.last_source_ns,view[0])
        self._publish(count,label,values)
        return (label,values,self.last_source_ns,count)

    def push_observed(self,stamp,row,observed_ns):
        return self.push(stamp,row,check_ns=observed_ns)

    def push(self, stamp, row, *, check_ns=None):
        self.last_source_ns=stamp
        from copy import deepcopy
        from tradelearn.backtest.clock import date2num
        import pandas as pd
        number=lambda value:date2num(pd.Timestamp(value,unit='ns'))
        if self.pending is not None:
            count,label,values=self.pending
            if number(label)>number(stamp):return []
            self.pending=None
            self._publish(count,label,values)
            restored=(label,values,stamp,count)
            if getattr(self,'_pending_from_idle',False):
                self._pending_from_idle=False
                return [restored,*self.push(stamp,row,check_ns=check_ns)]
            closed=self._checked_close(check_ns) if check_ns is not None else None
            if closed is not None:
                length,end,bar=closed
                if number(end)<=number(stamp):
                    self._publish(length,end,bar)
                    return [restored,(end,bar,stamp,length)]
                self.pending=closed
                view=self.previous or (0,end,bar)
                return [restored,(view[1],deepcopy(view[2]),stamp,view[0])]
            # A queued clone revision is visible before the newly arrived raw
            # source. None reserves that final raw phase without replaying the
            # derived bar a second time or consuming another native quote.
            return [restored,None]
        produced = self.builder.push(stamp,row)
        if not isinstance(produced,list):produced=[produced]
        output=[]
        if not produced and self.clone and check_ns is not None:
            closed=self._checked_close(check_ns)
            if closed is not None:
                self.pending=closed
                length,label,values=closed
                view=(self.published if self.published is not None and self.published[0]<length else self.previous)
                if view is None:view=(0,label,values)
                return [(view[1],deepcopy(view[2]),stamp,view[0])]
        for count,label,values in produced:
            if self.clone and number(label)>number(stamp):
                self.pending=(count,label,deepcopy(values))
                view=(self.published if self.published is not None and
                      self.published[0]<count else self.previous)
                if view is None:view=(0,label,values)
                length,visible,bar=view
                output.append((visible,deepcopy(bar),stamp,length))
            else:
                self._publish(count,label,values)
                output.append((label,values,stamp,count))
        return output


class LiveResample:
    def __init__(self, recipe):
        self.clone=recipe.get('clone',False)
        from tradelearn.engine.resampling import (_ClosedBarResampler,
            _CalendarBarResampler, _ConfiguredSubdayResampler, _ConfiguredCalendarResampler)
        values = recipe['options']
        timeframe, compression = values['timeframe'], values['compression']
        source = _source(recipe)
        units = {TimeFrame.MicroSeconds:1000, TimeFrame.Seconds:1_000_000_000,
                 TimeFrame.Minutes:60_000_000_000}
        from datetime import time
        declared_session = (source._tz is not None or
            getattr(source.p,'sessionstart',None) not in (None,time.min) or
            getattr(source.p,'sessionend',None) not in (None,time(23,59,59,999990)))
        configured = source._calendar is not None or declared_session or any(values[name] != default for name, default in
            (('bar2edge',True),('adjbartime',True),('rightedge',True),('boundoff',0),('takelate',True)))
        if timeframe in units:
            self.builder = (_ConfiguredSubdayResampler(source,timeframe,compression,values)
                if configured or timeframe == TimeFrame.MicroSeconds else
                _ClosedBarResampler(units[timeframe]*compression))
        else:
            self.builder = (_ConfiguredCalendarResampler(source,timeframe,compression,values)
                if configured else _CalendarBarResampler(timeframe,compression,
                    source.p.timeframe,source.p.compression))
        self.pending = None

    def next_check_ns(self):
        if self.pending is not None or self.builder._bar is None:return None
        if hasattr(self.builder,'_end'):return self.builder._end
        if (getattr(self.builder,'timeframe',TimeFrame.Days)>=TimeFrame.Days or
                not self.builder.options['bar2edge']):return None
        deadline=self.builder._label(True,boundary=True)
        if self.builder._eos is not None:deadline=min(deadline,self.builder._eos)
        return deadline if deadline>self.builder._last else None

    def check(self,observed_ns):
        deadline=self.next_check_ns()
        if deadline is None or observed_ns<deadline:return None
        label=(self.builder._end if hasattr(self.builder,'_end') else self.builder._label(True))
        from copy import deepcopy
        values=deepcopy(self.builder._bar)
        self.builder._bar=None
        if hasattr(self.builder,'_delivered'):self.builder._delivered=label
        if self.clone:
            self.pending=(label,values,self.last_source_ns)
            return None  # Completed clone still waits for its raw master.
        return (label,values,self.last_source_ns,None)

    def push(self, stamp, row):
        self.last_source_ns=stamp
        output = []
        if self.pending is not None:
            label, values, ready = self.pending
            if label <= stamp:
                output.append((label, values, stamp))
                self.pending = None
            else:
                return output
        for label, values, ready in self.builder.push(stamp,row,stamp):
            if label > stamp:
                if self.pending is not None:
                    raise RuntimeError('live bar projection has multiple unobservable future bars')
                self.pending = (label,values,ready)
            else:
                output.append((label,values,ready))
        return output
