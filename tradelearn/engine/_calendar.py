"""Use explicitly supplied BT calendar providers without importing Backtrader."""
from datetime import tzinfo


def resolve_calendar(value):
    if value is None:
        return None
    if isinstance(value,type):
        value=value()
    if isinstance(value,str):
        import pandas_market_calendars as calendars
        value=calendars.get_calendar(value)
    if not callable(getattr(value,'schedule',None)):
        raise TypeError('calendar must provide schedule(day, tz)')
    if callable(getattr(value,'valid_days',None)):
        return _MarketCalendar(value)
    return value


class CalendarTimezone(tzinfo):
    """BT calendar providers expect localize; internal clocks keep stdlib zones."""
    def __init__(self,zone):self.zone=zone
    def localize(self,value):
        from ._timezone import localize
        return localize(value,self.zone)
    def utcoffset(self,value):
        return self.zone.utcoffset(value.replace(tzinfo=self.zone) if value is not None else None)
    def dst(self,value):
        return self.zone.dst(value.replace(tzinfo=self.zone) if value is not None else None)
    def tzname(self,value):
        return self.zone.tzname(value.replace(tzinfo=self.zone) if value is not None else None)
    def fromutc(self,value):
        return self.zone.fromutc(value.replace(tzinfo=self.zone)).replace(tzinfo=self)


class _MarketCalendar:
    """A bounded adapter over the provider's authoritative UTC session table."""
    def __init__(self,provider):
        self.provider=provider
        self._sessions=None
        self._window=None

    def _table(self,day):
        import datetime as dt
        import pandas as pd
        stamp=pd.Timestamp(day)
        if stamp.tzinfo is not None:stamp=stamp.tz_convert('UTC').tz_localize(None)
        date=stamp.date()
        if self._window is None or not self._window[0]<=date<=self._window[1]:
            start=date-dt.timedelta(days=2);end=date+dt.timedelta(days=365)
            table=self.provider.schedule(start_date=start,end_date=end)
            if not isinstance(table,pd.DataFrame) or not {'market_open','market_close'}<=set(table.columns) or len(table)>368:
                raise ValueError('calendar requires a bounded session table with open/close')
            table=table[['market_open','market_close']].copy()
            for field in table:
                if table[field].isna().any():raise ValueError('calendar session time is missing')
                values=pd.to_datetime(table[field])
                if values.dt.tz is None:raise ValueError('calendar session time must carry a timezone')
                table[field]=values.dt.tz_convert('UTC').dt.tz_localize(None)
            if not table.index.is_unique or not table.index.is_monotonic_increasing or (table.market_close<table.market_open).any():
                raise ValueError('calendar session table is conflicting')
            self._sessions=table
            self._window=(date,end-dt.timedelta(days=2))
        return stamp,self._sessions

    def schedule(self,day,tz=None):
        stamp,table=self._table(day)
        eligible=table.loc[table.market_close>=stamp]
        if eligible.empty:raise ValueError('calendar has no session in bounded horizon')
        row=eligible.iloc[0]
        return row.market_open.to_pydatetime(),row.market_close.to_pydatetime()

    def _nextday(self,day):
        import pandas as pd
        stamp,table=self._table(day)
        dates=pd.DatetimeIndex(table.index)
        if dates.tz is not None:dates=dates.tz_convert('UTC').tz_localize(None)
        later=dates[dates.normalize()>stamp.normalize()]
        if later.empty:raise ValueError('calendar has no next session in bounded horizon')
        return later[0].to_pydatetime()

    def last_weekday(self,day):
        return bool(day.isocalendar()[:2]!=self._nextday(day).isocalendar()[:2])

    def last_monthday(self,day):
        following=self._nextday(day)
        return bool((day.year,day.month)!=(following.year,following.month))

    def last_yearday(self,day):
        return bool(day.year!=self._nextday(day).year)
