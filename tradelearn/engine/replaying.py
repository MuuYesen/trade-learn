"""Current-bar views over the existing bounded OHLCV completion protocol.

This builder does not dispatch strategies, calculate indicators or execute
orders. The shared Runtime must replace revisions at their logical bar index.
"""
from copy import deepcopy

from tradelearn.engine.base import TimeFrame
from tradelearn.engine.resampling import _CalendarBarResampler


class _ReplaySubdayBuilder:
    """Bounded BT replay aggregation; source timestamps are never replaced.

    The fixed BT historical clone invokes its filter check against the current
    wall clock between source deliveries. With default adjbartime=False that
    check only re-arms EOS, never delivers or changes a business timestamp.
    This occurs when target metadata sorts the clone before its raw source.
    Capture that private check clock once for a historical preparation run.
    """
    def __init__(self,timeframe,compression,check_ns=None):
        self.unit=1_000_000_000 if timeframe==TimeFrame.Seconds else 60_000_000_000
        self.compression=compression
        self.check_ns=check_ns
        self._bar=None;self._last=None;self._next_eos=None
        self.count=0;self._first=True;self.closed=False

    def push(self,stamp,row):
        if self.closed:raise RuntimeError('replay source has closed')
        if self._last is not None and stamp<=self._last:
            raise ValueError('replay requires ordered distinct source bars; late revisions remain separate')
        day=86_400_000_000_000
        if self._next_eos is None:self._next_eos=stamp//day*day+day-10_000
        point,rest=divmod(stamp%day,self.unit)
        onedge=stamp==self._next_eos or (not rest and point%self.compression==0)
        crossed=(stamp>self._next_eos and self._last is not None and self._last<=self._next_eos)
        if onedge or crossed:self._next_eos=None
        oldpoint=None if self._last is None else (self._last%day)//self.unit
        over=crossed or (oldpoint is not None and point>oldpoint and point//self.compression>oldpoint//self.compression)
        if self._first or (over and not onedge):
            self.count+=1;self._bar=dict(row)
        else:
            self._bar['high']=max(self._bar['high'],row['high'])
            self._bar['low']=min(self._bar['low'],row['low'])
            self._bar['close']=row['close'];self._bar['volume']+=row['volume']
            for name,value in row.items():
                if name not in {'open','high','low','close','volume'}:self._bar[name]=value
        self._last=stamp;self._first=onedge
        if (not onedge and self.check_ns is not None and self._next_eos is not None
                and self.check_ns>self._next_eos and stamp<=self._next_eos):
            self._next_eos=None
        return self.count,stamp,deepcopy(self._bar)

    def finish(self):
        self.closed=True;self._bar=None


class _ConfiguredReplaySubday:
    """Replay revisions over the same bounded boundary/session primitives."""
    def __init__(self,source,timeframe,compression,options,check_ns=None):
        from .resampling import _ConfiguredSubdayResampler
        self.state=_ConfiguredSubdayResampler(source,timeframe,compression,options)
        self.options=options;self.compression=compression;self.check_ns=check_ns
        self.count=0;self.closed=False;self._last=None;self._last_public=None

    def push(self,stamp,row):
        if self.closed:raise RuntimeError('replay source has closed')
        state=self.state
        if self._last_public is not None and state._number(stamp)<=self._last_public:
            if not self.options['takelate']:return []
            # Fixed BT's replay public date advances by one millionth of a
            # Gregorian day for an admitted late row. This is a logical label,
            # never a rewrite of the incoming source clock or quote.
            from tradelearn.backtest.clock import num2date
            import pandas as pd
            number=self._last_public+.000001
            label=pd.Timestamp(num2date(number)).value
            if state._bar is None:self.count+=1
            state._merge(row,label)
            self._last=stamp;self._last_public=number
            return [(self.count,label,deepcopy(state._bar))]
        onedge=state._check_eos(stamp,True)
        if not onedge:
            point,rest=state._point(stamp,True)
            onedge=not rest and point%self.compression==0
        crossed=False
        if state._bar is not None and not onedge:
            crossed=state._check_eos(stamp,False)
            if not crossed:
                old,_=state._point(state._last);current,_=state._point(stamp)
                crossed=current>old and (not self.options['bar2edge'] or current//self.compression>old//self.compression)
        complete=False
        if crossed:
            if self.options['bar2edge']:complete=True
            else:
                state._count+=1;complete=state._count%self.compression==0
        output=[]
        if complete and self.options['bar2edge'] and self.options['adjbartime']:
            output.append((self.count,state._label(True),deepcopy(state._bar)))
        if state._bar is None or complete:
            self.count+=1;state._bar=None
        state._merge(row,stamp)
        value=deepcopy(state._bar)
        if onedge:state._bar=None
        self._last=stamp
        if (not onedge and self.check_ns is not None and state._eos is not None
                and self.check_ns>state._eos and stamp<=state._eos):
            state._last_eos=state._eos;state._eos=None
        output.append((self.count,stamp,value))
        self._last_public=state._number(stamp)
        return output

    def finish(self):
        self.closed=True;self.state._bar=None


class _ReplayBarBuilder:
    def __init__(self,timeframe,compression,source_timeframe,source_compression=1,*,check_ns=None):
        units={TimeFrame.Seconds:1_000_000_000,TimeFrame.Minutes:60_000_000_000}
        self.builder=(_ReplaySubdayBuilder(timeframe,compression,check_ns) if timeframe in units else
                      _CalendarBarResampler(timeframe,compression,source_timeframe,source_compression))
        self.count=0
        self.closed=False
        self._daily_session=timeframe==TimeFrame.Days and source_timeframe<TimeFrame.Days
        self._next_eos=None
        self._session_sequence=0

    def push(self,stamp,row):
        if self.closed:raise RuntimeError('replay source has closed')
        if isinstance(self.builder,_ReplaySubdayBuilder):
            return self.builder.push(stamp,row)
        had_open=self.builder._bar is not None
        if self._daily_session:
            # Fixed BT replay establishes the next session end lazily from the
            # incoming raw bar. Sparse one-observation days therefore differ
            # from closed-bar resampling. Keep that real source-clock behavior.
            if self._next_eos is None:
                self._next_eos=stamp//86_400_000_000_000*86_400_000_000_000+86_400_000_000_000-10_000
            if (stamp>self._next_eos and had_open
                    and self.builder._last<=self._next_eos):
                self._session_sequence+=1
                self._next_eos=None
            completed=self.builder.push(stamp,row,boundary_key=self._session_sequence)
        else:
            completed=self.builder.push(stamp,row)
        current=self.builder._bar
        if current is not None:
            if not had_open or completed:self.count+=1
            value=current
        else:
            if not completed:raise RuntimeError('replay completion lacks a current bar')
            if not had_open or len(completed)>1:self.count+=1
            value=completed[-1][1]
        return self.count,stamp,deepcopy(value)

    def finish(self):
        self.closed=True
        self.builder.finish()  # Already-visible revision does not create an EOF bar.


def _clone_late_replay_events(frame,builder,source_availability,source_unit,check_ns):
    """Prepare one queued clone revision over the already observed source.

    A future labelled revision withdraws one visible bar, including the empty
    first-bar state. Source loading and forced historical checks use the same
    bounded builder; this never owns strategy or execution dispatch.
    """
    from tradelearn.backtest.clock import date2num
    import pandas as pd
    pending=None;published=None;previous=None;ready=0;last_quote_origin=None

    def preceding(length,stamp,held):
        view=published if published is not None and published[0]<length else previous
        # BT withdraws the cursor, not the already observed in-progress bar.
        # Keep that actual accumulator and its label without inventing history.
        return view if view is not None else (0,held[1],deepcopy(held[2]))

    def checked_close(origin,ready):
        state=builder.state
        if not (builder.options['bar2edge'] and builder.options['adjbartime']) or state._bar is None:return None
        if state._eos is None:
            state._eos=pd.Timestamp(state.source._session_end_at(state._number(state._last))[0]).value
        crossed=state._check_eos(check_ns,False)
        if not crossed:
            old,_=state._point(state._last);new,_=state._point(check_ns)
            crossed=new>old and new//builder.compression>old//builder.compression
        if not crossed:return None
        label=state._label(True)
        queued=((builder.count,label,deepcopy(state._bar),origin,ready)
                if state._number(label)>state._number(state._last) else None)
        state._bar=None
        return queued

    for offset,(stamp,row) in enumerate(frame.iterrows()):
        ready=(stamp.value if source_availability is None else
               int(source_availability[offset])*(1 if source_unit=='ns' else 1_000_000_000))
        if pending is not None:
            length,label,bar,origin,seen=pending
            if date2num(pd.Timestamp(label,unit='ns'))<=date2num(stamp):
                yield length,label,bar,origin,max(label,seen),2*offset
                if published is None or length>published[0]:previous=published
                published=(length,label,bar)
                pending=checked_close(origin,ready)
                if pending is not None:yield *preceding(pending[0],stamp.value,pending),origin,ready,2*offset+1
            # Advancing the clone's queued buffer consumes this source turn.
            continue
        produced=builder.push(stamp.value,row.to_dict())
        if produced:last_quote_origin=offset
        for length,label,bar in produced:
            if date2num(pd.Timestamp(label,unit='ns'))>date2num(stamp):
                pending=(length,label,bar,offset,ready)
                yield *preceding(length,stamp.value,pending),offset,ready,2*offset+1
            else:
                yield length,label,bar,offset,ready,2*offset+1
                if published is None or length>published[0]:previous=published
                published=(length,label,bar)
        if not produced:
            # takelate=False may exhaust the clone's available source row;
            # its ordinary historical check can still force an adjusted close.
            pending=checked_close(last_quote_origin,ready)
            if pending is not None:yield *preceding(pending[0],stamp.value,pending),last_quote_origin,ready,2*offset+1
    if pending is not None:
        length,label,bar,origin,seen=pending
        yield length,label,bar,origin,max(label,seen,ready),2*len(frame)


def replay_feed(source,*,clone=False,**options):
    """Prepare historical revision storage; public dispatch is integrated separately."""
    import numpy as np
    import pandas as pd
    from copy import copy
    from types import SimpleNamespace
    from tradelearn.backtest.data import SharedBarBuffer
    from tradelearn.engine.datafeed import DataFeed
    from tradelearn.backtest.lines import Lines,LineSeries,line_names
    defaults=dict(timeframe=TimeFrame.Days,compression=1,bar2edge=True,
                  adjbartime=False,rightedge=True,boundoff=0,takelate=True)
    unknown=options.keys()-defaults.keys()
    if unknown:raise TypeError(f'unknown replay parameters: {sorted(unknown)}')
    values={**defaults,**options}
    nondefault=any(values[name]!=defaults[name] for name in ('bar2edge','adjbartime','rightedge','boundoff','takelate'))
    if type(values['boundoff']) is not int:raise TypeError('replay boundoff must be integer')
    timeframe,compression=values['timeframe'],values['compression']
    if type(compression) is not int or compression<=0:raise ValueError('positive integer replay compression required')
    if timeframe not in (TimeFrame.MicroSeconds,TimeFrame.Seconds,TimeFrame.Minutes,TimeFrame.Days,TimeFrame.Weeks,TimeFrame.Months,TimeFrame.Years):
        raise NotImplementedError('raw tick/subsecond replay is not yet integrated')
    frame=getattr(source,'_frame',None)
    if frame is None or getattr(source,'is_live',False):raise NotImplementedError('stream replay source integration is not yet implemented')
    check_ns=(pd.Timestamp.now(tz="UTC").value if clone and
              (timeframe,compression)<(source.p.timeframe,source.p.compression) else None)
    declared_session=(source._tz is not None or getattr(source.p,'sessionstart',None) not in (None,__import__('datetime').time.min)
                      or getattr(source.p,'sessionend',None) not in (None,__import__('datetime').time(23,59,59,999990)))
    from tradelearn.backtest.clock import date2num
    raw_dates=np.asarray([date2num(stamp) for stamp in frame.index],dtype=np.float64)
    late_source=bool(np.any(raw_dates[1:]<=raw_dates[:-1]))
    configured=nondefault or declared_session or timeframe==TimeFrame.MicroSeconds or late_source
    if configured and timeframe>=TimeFrame.Days:
        raise NotImplementedError('configured calendar replay still requires its calendar revision projection')
    if source._calendar is not None:
        raise NotImplementedError('calendar replay still requires its source-start calendar projection')
    builder=(_ConfiguredReplaySubday(source,timeframe,compression,values,check_ns) if configured else
             _ReplayBarBuilder(timeframe,compression,source.p.timeframe,source.p.compression,check_ns=check_ns))
    needs_views=bool(values['adjbartime'] and values['bar2edge'] or late_source)
    if needs_views and check_ns is not None:
        raise NotImplementedError('adjusted replay clone check/queued-close projection remains in progress')
    rows=[];lengths=[];labels=[];source_positions=[];source_turns=[];available=[]
    source_availability=getattr(source,'_delivery_datetime',None)
    if clone and late_source:
        events=_clone_late_replay_events(frame,builder,source_availability,getattr(source,'_delivery_unit','s'),pd.Timestamp.now(tz='UTC').value)
    else:
        def observed():
            for offset,(stamp,row) in enumerate(frame.iterrows()):
                results=builder.push(stamp.value,row.to_dict())
                if not isinstance(results,list):results=[results]
                ready=(stamp.value if source_availability is None else int(source_availability[offset])*(1 if getattr(source,'_delivery_unit','s')=='ns' else 1_000_000_000))
                for length,label,bar in results:yield length,label,bar,offset,ready,2*offset+1
        events=observed()
    for length,label,bar,offset,ready,turn in events:
        rows.append(bar);lengths.append(length);labels.append(label);source_positions.append(offset)
        source_turns.append(turn);available.append(ready)
    builder.finish()
    physical=frame.index.take(source_positions)
    sampled=pd.DataFrame(rows,index=physical,columns=frame.columns)
    target=copy(source) if clone else source
    parameters=source.p.asdict();name=source._name
    target.lines=Lines(target)
    DataFeed.__init__(target,sampled,name=name,_preserve_row_order=True,**{k:v for k,v in parameters.items() if k!='name'})
    target._timeframe,target._compression=timeframe,compression
    for name in line_names(type(target)):
        if name not in {'datetime','open','high','low','close','volume'}:
            line=LineSeries(target.get_array(name),buffer=target._bar_buffer,buffer_name=name)
            line._cursor_source=target
            setattr(target,name,line);setattr(target.lines,name,line);target._lines_list.append(line)
    target._replay_lengths=np.asarray(lengths,dtype=np.int64)
    target._replay_visible=SimpleNamespace(_cursor=-1,_bt_line_views=True,_empty_current=None,_empty_epoch_ns=None)
    target._replay_values=dict(target._arrays)
    size=max(lengths,default=0)
    target._replay_lines={name:np.full(size,np.nan) for name in target._arrays}
    target._bar_buffer=SharedBarBuffer(target._replay_visible,target._replay_lines)
    target._replay_epoch_ns=np.zeros(size,dtype=np.int64)
    for line in target._lines_list:
        field=line._buffer_name
        if field in target._replay_lines:
            line._values=target._replay_lines[field]
            line._buffer=target._bar_buffer
            line._cursor_source=target._replay_visible
    target.datetime._epoch_ns_values=target._replay_epoch_ns
    target.datetime._empty_epoch_ns_source=target._replay_visible
    target._execution_quote_values=frame[['open','high','low','close']].iloc[source_positions].to_numpy(dtype=np.float64,copy=True)
    target._source_observation_positions=np.asarray(source_positions,dtype=np.int64)
    target._native_delivery_views=needs_views
    if clone and needs_views:target._replay_source=source
    if needs_views:
        from tradelearn.backtest.clock import date2num
        target._clock_unit='ns'
        target._clock_timestamps=physical.as_unit('ns').asi8.copy()
        target._clock_view_timestamps=np.asarray(labels,dtype=np.int64)
        target._replay_label_ns=target._clock_view_timestamps
        target._replay_values['datetime']=target._clock_view_timestamps/1_000_000_000.
        dates=np.asarray([date2num(pd.Timestamp(int(label),unit='ns')) for label in labels],dtype=np.float64)
        target._delivery_date_ranks=dates.view(np.int64)
        target._delivery_unit='ns'
        target._delivery_datetime=np.asarray(available,dtype=np.int64)
        target._delivery_phase=np.arange(len(available))-np.searchsorted(target._delivery_datetime,target._delivery_datetime,side='left')
        if clone:
            source_ready=(np.asarray(source_availability,dtype=np.int64)*(1 if getattr(source,'_delivery_unit','s')=='ns' else 1_000_000_000)
                          if source_availability is not None else frame.index.as_unit('ns').asi8)
            # Quote provenance is not the source delivery turn. A rejected
            # late row can force a new delivery using a previously observed
            # quote; keep each physical source turn and before-turn release.
            events_by_turn={}
            for event,(turn,ready) in enumerate(zip(source_turns,available)):
                origin=turn//2
                if origin<len(source_ready) and ready==int(source_ready[origin]):
                    events_by_turn.setdefault(turn,[]).append(event)
            parent_phases=np.zeros(len(frame),dtype=np.int64)
            phase=0;previous_ready=None
            for origin,ready in enumerate(source_ready):
                if ready!=previous_ready:phase=0
                before=events_by_turn.get(2*origin,())
                for extra,event in enumerate(before):target._delivery_phase[event]=phase+extra
                phase+=len(before)
                turns=events_by_turn.get(2*origin+1,())
                for extra,event in enumerate(turns):target._delivery_phase[event]=phase+extra
                parent_phases[origin]=phase+max(0,len(turns)-1)
                phase+=max(1,len(turns));previous_ready=ready
            if len(frame):
                tail=[event for event,(turn,ready) in enumerate(zip(source_turns,available))
                      if turn==2*len(frame) and ready==int(source_ready[-1])]
                for extra,event in enumerate(tail):target._delivery_phase[event]=phase+extra
            target._replay_source_phases=parent_phases


    target.replaying=True
    return target
