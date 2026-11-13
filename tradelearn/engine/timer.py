"""Bar-clock scheduling state for the existing Backtrader timer interface.

No worker thread, sleep, wall clock or execution authority is owned here.
"""
from datetime import date,datetime,time,timedelta
from types import SimpleNamespace
from tradelearn.backtest.clock import date2num,num2date

SESSION_TIME,SESSION_START,SESSION_END=range(3)


class _DateSelector:
    def __init__(self,days,carry):
        self.days=tuple(days);self.carry=carry;self.period=None;self.pending=()
    def accepts(self,period,day):
        if not self.days:return True
        carried=False
        if period!=self.period:
            carried=self.carry and bool(self.pending)
            self.period=period;self.pending=self.days
        earlier=sum(value<day for value in self.pending)
        # Compatibility with the fixed BT mask: when skipped entries precede
        # a later pending day, its selection consumes that next entry as well.
        # Do not silently substitute a different calendar policy.
        selected=earlier<len(self.pending) and any(value<=day for value in self.pending)
        self.pending=self.pending[earlier+int(selected):]
        return selected or (self.carry and (carried or earlier>0))


class Timer:
    """One finite per-registration schedule, checked only on actual bar delivery."""
    SESSION_TIME,SESSION_START,SESSION_END=SESSION_TIME,SESSION_START,SESSION_END
    def __init__(self,*args,**kwargs):
        defaults=dict(tid=None,owner=None,strats=False,when=None,offset=timedelta(),repeat=timedelta(),
                      weekdays=(),weekcarry=False,monthdays=(),monthcarry=True,allow=None,tzdata=None,cheat=False)
        self.kwargs={key:value for key,value in kwargs.items() if key not in defaults}
        self.p=self.params=SimpleNamespace(**(defaults|{key:value for key,value in kwargs.items() if key in defaults}))
        self.args=args
    def start(self,data):
        p=self.p
        if not isinstance(p.offset,timedelta) or not isinstance(p.repeat,timedelta):
            raise TypeError('timer offset/repeat must be timedelta')
        if p.repeat<timedelta():raise ValueError('negative timer repeat would not advance')
        self._reference=p.tzdata if p.tzdata is not None else (data if isinstance(p.when,int) else None)
        if p.when in (SESSION_START,SESSION_END):
            field='sessionstart' if p.when==SESSION_START else 'sessionend'
            default=time.min if p.when==SESSION_START else time(23,59,59,999990)
            self._time=getattr(getattr(self._reference,'p',None),field,None) or default
        else:self._time=p.when
        if not isinstance(self._time,time):raise TypeError('timer when must be time or SESSION_START/END')
        self._zone=getattr(getattr(self._reference,'p',None),'tz',None) if hasattr(self._reference,'p') else self._reference
        from ._timezone import resolve_timezone
        self._zone=resolve_timezone(self._zone)
        self._session_end=getattr(getattr(self._reference,'p',None),'sessionend',None) or time(23,59,59,999990)
        self._eos=None
        self._month=_DateSelector(p.monthdays,p.monthcarry);self._week=_DateSelector(p.weekdays,p.weekcarry)
        self._date=None;self._blocked=None;self._target=None;self._deadline=None;self.lastwhen=None
    def check(self,dt):
        now=num2date(dt);day=now.date()
        if self._blocked==day:return False
        if self._eos is None or now>self._eos:
            provider=getattr(self._reference,'_getnexteos',None)
            self._eos=provider()[0] if callable(provider) else datetime.combine(day,self._session_end)
            self._target=None;self._deadline=None
        if self._date!=day:
            # Calendar EOS can lie beyond a date boundary. BT keeps the pending
            # scheduled target until EOS resets it, even while day filters move
            # forward. Do not silently discard a due prior-day notification.
            self._date=day
            allowed=self._month.accepts(day.month,day.day)
            if allowed:allowed=self._week.accepts(day.isocalendar().week,day.isoweekday())
            if allowed and self.p.allow is not None:allowed=self.p.allow(day)
            if not allowed:
                self._blocked=day;self._target=None;self._deadline=None
                return False
        if self._target is None:
            self._target=datetime.combine(day,self._time)+self.p.offset
            from ._timezone import localize
            self._deadline=date2num(localize(self._target,self._zone))
        if dt<self._deadline:return False
        self.lastwhen=self._target
        if not self.p.repeat:
            self._blocked=day;self._target=None
        else:
            steps=max(1,(now-self._target)//self.p.repeat+1)
            following=self._target+steps*self.p.repeat
            if following>self._eos:self._blocked=day;self._target=None
            else:
                # Fixed BT repeat advances against the UTC check date and then
                # localizes only the displayed next scheduled time.
                self._deadline=date2num(following)
                self._target=num2date(self._deadline,tz=self._zone)
        return True
