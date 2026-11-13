"""Configured calendar replay revises the same logical bar, as fixed BT."""
import datetime as dt
import math
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


class Calendar:
    def schedule(self,day,tz=None):
        day=day.replace(tzinfo=None)
        while True:
            end=dt.datetime.combine(day.date(),dt.time(18 if day.date()==dt.date(2026,3,6) else 21))
            if day.weekday()<5 and day<=end:return dt.datetime.combine(day.date(),dt.time(13,30)),end
            day=dt.datetime.combine(day.date()+dt.timedelta(days=1),dt.time())


def run(api,frame_name,compression,flag,clone,calendar):
    target=getattr(api.TimeFrame,frame_name)
    times=(['2026-03-06 14:30','2026-03-06 14:31','2026-03-06 18:01','2026-03-09 13:30','2026-03-09 13:31','2026-03-09 21:01','2026-03-10 13:30','2026-03-10 13:31','2026-03-10 21:01'] if frame_name=='Days' else
           ['2025-12-29 16:00','2025-12-30 16:00','2025-12-31 16:00','2026-01-02 16:00','2026-01-05 16:00','2026-01-30 16:00','2026-02-02 16:00','2026-12-31 16:00','2027-01-04 16:00'])
    values=[100.,107.,94.,115.,81.,120.,99.,109.,88.]
    frame=pd.DataFrame(dict(open=values,high=[x+4 for x in values],low=[x-6 for x in values],close=[x+1 for x in values],volume=range(1,10)),index=pd.to_datetime(times))
    options={'adjbartime':True} if flag=='adjust' else {'bar2edge':False} if flag=='unbounded' else {'boundoff':2} if flag=='offset' else {'takelate':False}
    rows=[]
    def value(x):return None if math.isnan(float(x)) else float(x)
    class Business(api.Strategy):
        def __init__(self):self.average=api.indicators.SMA(self.datas[-1].close,period=2) if api is official else api.talib.SMA(self.datas[-1].close,timeperiod=2)
        def record(self,phase):rows.append((phase,len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,*[value(getattr(d,k)[0]) if len(d) else None for k in ['open','high','low','close','volume']]) for d in self.datas),len(self.average),value(self.average[0])))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=api.Cerebro(stdstats=False,runonce=False)
    raw=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes if frame_name=='Days' else api.TimeFrame.Days,**({'calendar':Calendar()} if calendar is True else {}))
    if clone:c.adddata(raw,name='raw')
    targetfeed=c.replaydata(raw,timeframe=target,compression=compression,name='sample',**options)
    references=(targetfeed.lines,targetfeed.close,targetfeed.datetime)
    if calendar=='global':c.addcalendar(Calendar())
    c.addstrategy(Business);c.run()
    if api is tl:assert all(a is b for a,b in zip(references,(targetfeed.lines,targetfeed.close,targetfeed.datetime)))
    return rows


@pytest.mark.parametrize('frame_name',['Days','Weeks','Months','Years'])
@pytest.mark.parametrize('compression',[1,2])
@pytest.mark.parametrize('flag',['adjust','unbounded','offset','rejectlate'])
@pytest.mark.parametrize('clone',[False,True])
def test_configured_calendar_replay_rows_and_sma_match_fixed_bt(frame_name,compression,flag,clone):
    assert run(tl,frame_name,compression,flag,clone,False)==run(official,frame_name,compression,flag,clone,False)


@pytest.mark.parametrize('flag',['adjust','rejectlate'])
@pytest.mark.parametrize('clone',[False,True])
def test_provided_daily_calendar_replay_preserves_source_session_and_logical_revisions(flag,clone):
    assert run(tl,'Days',1,flag,clone,True)==run(official,'Days',1,flag,clone,True)


@pytest.mark.parametrize('flag',['adjust','rejectlate'])
@pytest.mark.parametrize('clone',[False,True])
def test_global_calendar_registered_after_replay_refreshes_same_line_objects(flag,clone):
    assert run(tl,'Days',1,flag,clone,'global')==run(official,'Days',1,flag,clone,'global')
