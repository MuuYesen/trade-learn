"""Feed session declarations provide date bounds/EOS; they do not clip raw bars."""
from datetime import date,time,timedelta
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_feed_timezone import profile


@pytest.mark.parametrize('tz',[None,'America/New_York','Asia/Tokyo'])
@pytest.mark.parametrize('with_bounds',[False,True])
def test_declared_session_times_normalize_date_bounds_without_dropping_unbounded_bars(tz,with_bounds):
    index=pd.date_range('2026-01-01 00:00',periods=36,freq='h',tz='UTC')
    kw=dict(tz=tz,sessionstart=time(9),sessionend=time(16))
    if with_bounds:kw.update(fromdate=date(2026,1,1),todate=date(2026,1,1))
    actual,d=profile(tl,index,**kw);expected,_=profile(official,index,**kw)
    assert actual==expected
    if not with_bounds:assert len(actual)==len(index)
    assert d.p.sessionstart==time(9) and d.p.sessionend==time(16)


def eos_profile(api,tz):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            d=self.data
            end,number=d._getnexteos()
            rows.append((d.datetime[0],end.isoformat(),number,d.num2date().isoformat()))
    index=pd.to_datetime(['2026-03-06 14:00','2026-03-06 22:00','2026-03-09 13:00','2026-03-09 21:00'])
    frame=pd.DataFrame({n:[10.]*4 for n in ('open','high','low','close','volume')},index=index)
    c=api.Cerebro(stdstats=False,runonce=False)
    c.adddata(api.feeds.PandasData(dataname=frame,tz=tz,sessionstart=time(9),sessionend=time(16)))
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('tz',[None,'America/New_York','Asia/Tokyo'])
def test_default_calendar_eos_and_feed_date_helpers_match_bt_across_dst(tz):
    assert eos_profile(tl,tz)==eos_profile(official,tz)


def timer_session(api,kind):
    rows=[]
    class Business(api.Strategy):
        def notify_timer(self,timer,when,*args,**kwargs):rows.append((when.isoformat(),len(self),self.data.datetime.datetime().isoformat()))
    index=pd.to_datetime(['2026-01-01 13:59','2026-01-01 14:00','2026-01-01 14:35','2026-01-01 20:00','2026-01-01 21:00','2026-01-02 14:00'])
    frame=pd.DataFrame({n:[10.]*len(index) for n in ('open','high','low','close','volume')},index=index)
    c=api.Cerebro(stdstats=False,runonce=False)
    feed=api.feeds.PandasData(dataname=frame,tz='America/New_York',sessionstart=time(9),sessionend=time(16));c.adddata(feed)
    c.add_timer(api.timer.SESSION_START if kind=='start' else api.timer.SESSION_END,offset=timedelta() if kind=='start' else timedelta(hours=-1),repeat=timedelta(minutes=30),strats=True)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['start','end'])
def test_timer_uses_the_same_declared_feed_timezone_and_session(kind):
    assert timer_session(tl,kind)==timer_session(official,kind)
