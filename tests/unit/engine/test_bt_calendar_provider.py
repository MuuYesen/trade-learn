"""P03 actual public feed/global calendars; no runtime dependency on Backtrader."""
import datetime as dt
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def calendar(close=16):
    return official.TradingCalendar(open=dt.time(9),close=dt.time(close),
        earlydays=[(dt.date(2026,1,30),dt.time(9),dt.time(min(close,15)))])


def run(api,kind,when,override=False,*,tz=None,timeframe=None,aliases=False,eos=True):
    index=pd.to_datetime(['2026-01-29 14:59','2026-01-29 16:00',
        '2026-01-30 14:59','2026-01-30 15:00','2026-02-02 14:59','2026-02-02 16:00'])
    if tz=='Asia/Tokyo':index=index-pd.Timedelta(hours=12)
    frame=pd.DataFrame({name:[10.,20.,30.,40.,50.,60.] for name in ('open','high','low','close','volume')},index=index)
    out=[];views=[]
    class Business(api.Strategy):
        def prenext(self):self.next()
        def next(self):
            out.append((self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,
                d.close[0] if len(d) else None,d._getnexteos()[1] if len(d) and eos else None) for d in self.datas),
                tuple((line[0] if len(d) else None) for d,line in views)))
    c=api.Cerebro(stdstats=False,runonce=False)
    if when=='before':assert c.addcalendar(calendar()) is None
    options={'calendar':calendar(17)} if override or when=='feed' else {}
    data=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,tz=tz,**options)
    if kind=='raw':c.adddata(data)
    else:
        if kind=='clone':c.adddata(data)
        c.resampledata(data,timeframe=timeframe or api.TimeFrame.Days,compression=1)
    if aliases:views.extend((d,d.datetime) for d in c.datas)
    if when=='after':assert c.addcalendar(calendar()) is None
    c.addstrategy(Business);c.run();return out


@pytest.mark.parametrize('kind',['raw','resample','clone'])
@pytest.mark.parametrize('when',['feed','before','after'])
def test_declared_or_global_calendar_and_late_registration_match_fixed_bt(kind,when):
    assert run(tl,kind,when)==run(official,kind,when)


@pytest.mark.parametrize('when',['before','after'])
@pytest.mark.parametrize('kind',['raw','resample','clone'])
def test_feed_calendar_overrides_global_even_when_global_added_later(kind,when):
    assert run(tl,kind,when,override=True)==run(official,kind,when,override=True)


def test_global_calendar_class_is_instantiated_once():
    class Calendar:
        def schedule(self,day,tz=None):return day,day
    engine=tl.Cerebro()
    assert engine.addcalendar(Calendar) is None
    assert isinstance(engine._tradingcal,Calendar)
    assert engine.calendars==[engine._tradingcal]


def test_invalid_calendar_rejected_without_running():
    with pytest.raises(TypeError,match='calendar'):
        tl.feeds.PandasData(dataname=pd.DataFrame(columns=['open','high','low','close','volume'],index=pd.DatetimeIndex([])),calendar=object())


@pytest.mark.parametrize('kind',['raw','resample','clone'])
@pytest.mark.parametrize('tz',['America/New_York','Asia/Tokyo'])
def test_calendar_provider_receives_bt_localize_timezone_without_changing_raw_time(kind,tz):
    assert run(tl,kind,'after',tz=tz,eos=False)==run(official,kind,'after',tz=tz,eos=False)


@pytest.mark.parametrize('kind',['resample','clone'])
@pytest.mark.parametrize('timeframe',[official.TimeFrame.Weeks,official.TimeFrame.Months,official.TimeFrame.Years])
def test_calendar_period_end_predicates_and_eof_match_bt(kind,timeframe):
    assert run(tl,kind,'after',timeframe=timeframe)==run(official,kind,'after',timeframe=timeframe)


@pytest.mark.parametrize('kind',['resample','clone'])
def test_late_calendar_preserves_already_bound_public_line_identity(kind):
    assert run(tl,kind,'after',aliases=True)==run(official,kind,'after',aliases=True)


def test_named_and_pandas_calendar_providers_use_real_holidays_early_close_and_utc():
    mcal=pytest.importorskip('pandas_market_calendars')
    from tradelearn.engine._calendar import resolve_calendar
    day=dt.datetime(2026,11,27,16)
    for argument in ('NYSE',mcal.get_calendar('NYSE')):
        provider=resolve_calendar(argument)
        assert provider.schedule(day)==(dt.datetime(2026,11,27,14,30),dt.datetime(2026,11,27,18))
        assert provider.schedule(dt.datetime(2026,11,26,15))==(dt.datetime(2026,11,27,14,30),dt.datetime(2026,11,27,18))
        assert provider.last_weekday(dt.date(2026,11,27)) is True
        assert provider.last_weekday(dt.date(2026,11,25)) is False
        assert provider.last_monthday(dt.date(2026,11,30)) is True
        assert provider.last_yearday(dt.date(2026,12,31)) is True


@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('where',['feed','before','after'])
def test_named_nyse_daily_delivery_matches_bt_calendar_facts(clone,where):
    pytest.importorskip('pandas_market_calendars')
    def execute(api):
        index=pd.to_datetime(['2026-11-25 15:00','2026-11-25 21:00',
                              '2026-11-27 15:00','2026-11-27 18:00','2026-11-30 15:00','2026-11-30 21:00'])
        frame=pd.DataFrame({n:[10.,20.,30.,40.,50.,60.] for n in ('open','high','low','close','volume')},index=index)
        # Fixed BT PandasMarketCalendar fails on pandas3 date-vs-Timestamp.
        # Use its real plain-calendar provider with the same actual NYSE facts;
        # do not patch BT, its source or its acceptance threshold.
        provider=('NYSE' if api is tl else official.TradingCalendar(open=dt.time(9,30),close=dt.time(16),
            holidays=[dt.date(2026,11,26)],earlydays=[(dt.date(2026,11,27),dt.time(9,30),dt.time(13))]))
        c=api.Cerebro(stdstats=False,runonce=False)
        if where=='before':c.addcalendar(provider)
        d=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,tz='America/New_York',
                              **({'calendar':provider} if where=='feed' else {}))
        if clone:c.adddata(d)
        c.resampledata(d,timeframe=api.TimeFrame.Days)
        if where=='after':c.addcalendar(provider)
        rows=[]
        class Business(api.Strategy):
            def prenext(self):self.next()
            def next(self):rows.append((self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.close[0] if len(d) else None,d.volume[0] if len(d) else None) for d in self.datas)))
        c.addstrategy(Business);c.run();return rows
    assert execute(tl)==execute(official)


@pytest.mark.parametrize('kind',['missing','oversized','naive','reversed','duplicate'])
def test_market_calendar_invalid_or_unbounded_provider_table_is_rejected(kind):
    from tradelearn.engine._calendar import resolve_calendar
    index=pd.date_range('2026-01-01',periods=1)
    table=pd.DataFrame({'market_open':pd.to_datetime(['2026-01-01 09:00'],utc=True),
                        'market_close':pd.to_datetime(['2026-01-01 16:00'],utc=True)},index=index)
    if kind=='missing':table=table.drop(columns='market_close')
    elif kind=='oversized':table=pd.concat([table]*369,ignore_index=True)
    elif kind=='naive':table.market_close=table.market_close.dt.tz_localize(None)
    elif kind=='reversed':table.market_close=table.market_open-pd.Timedelta(hours=1)
    else:table=pd.concat([table,table])
    class Provider:
        def valid_days(self,*args):raise AssertionError('calendar session table is authoritative')
        def schedule(self,**kwargs):return table
    with pytest.raises(ValueError,match='calendar'):
        resolve_calendar(Provider()).schedule(dt.datetime(2026,1,1,10))


def test_pandas_calendar_provider_must_also_supply_schedule():
    from tradelearn.engine._calendar import resolve_calendar
    class Incomplete:
        def valid_days(self,*args):return []
    with pytest.raises(TypeError,match='calendar'):
        resolve_calendar(Incomplete())


@pytest.mark.parametrize('mode',['raw','resample','clone'])
@pytest.mark.parametrize('cheat',[False,True])
def test_calendar_session_end_timer_uses_actual_early_close_same_bt_phase(mode,cheat):
    pytest.importorskip('pandas_market_calendars')
    def execute(api):
        index=pd.to_datetime(['2026-11-25 20:54','2026-11-25 20:56','2026-11-25 21:00',
            '2026-11-27 17:54','2026-11-27 17:56','2026-11-27 18:00',
            '2026-11-30 20:54','2026-11-30 20:56','2026-11-30 21:00'])
        frame=pd.DataFrame({n:list(range(10,19)) for n in ('open','high','low','close','volume')},index=index)
        c=api.Cerebro(stdstats=False,runonce=False)
        d=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,tz='America/New_York')
        if mode in ('raw','clone'):c.adddata(d)
        if mode!='raw':c.resampledata(d,timeframe=api.TimeFrame.Days)
        provider=('NYSE' if api is tl else official.TradingCalendar(open=dt.time(9,30),close=dt.time(16),
            holidays=[dt.date(2026,11,26)],earlydays=[(dt.date(2026,11,27),dt.time(9,30),dt.time(13))]))
        c.addcalendar(provider)
        c.add_timer(when=api.timer.SESSION_END,offset=dt.timedelta(minutes=-5),cheat=cheat)
        out=[]
        class Business(api.Strategy):
            def notify_timer(self,timer,when,*args,**kwargs):
                out.append((when,len(self),self.datetime[0] if len(self) else None,
                            tuple((len(d),d.datetime[0] if len(d) else None) for d in self.datas)))
        c.addstrategy(Business);c.run();return out
    assert execute(tl)==execute(official)
