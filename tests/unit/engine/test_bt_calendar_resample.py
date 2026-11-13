"""Existing resample registration: calendar boundaries use real fixed BT calls."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def calendar_calls(api,timeframe,compression,clone):
    rows=[]
    index=pd.date_range('2025-12-20',periods=40,freq='D')
    frame=pd.DataFrame({n:[float(i+10) for i in range(len(index))] for n in ('open','high','low','close','volume')},index=index)
    c=api.Cerebro(stdstats=False,runonce=False)
    feed=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Days)
    if clone:c.adddata(feed,name='raw')
    c.resampledata(feed,name='sampled',timeframe=getattr(api.TimeFrame,timeframe),compression=compression)
    class Business(api.Strategy):
        def prenext(self):self.next()
        def next(self):rows.append((len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.open[0] if len(d) else None,d.close[0] if len(d) else None,d.volume[0] if len(d) else None) for d in self.datas)))
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('timeframe',['Days','Weeks','Months','Years'])
@pytest.mark.parametrize('compression',[1,2,3])
@pytest.mark.parametrize('clone',[False,True])
def test_default_calendar_bar_completion_and_delivery_match_fixed_bt(timeframe,compression,clone):
    assert calendar_calls(tl,timeframe,compression,clone)==calendar_calls(official,timeframe,compression,clone)


def intraday_to_day(api,compression,clone,extra):
    rows=[]
    index=pd.to_datetime(['2025-12-31 23:58','2025-12-31 23:59','2026-01-01 00:01']+(['2026-01-01 00:02','2026-01-01 00:03'] if extra else []))
    frame=pd.DataFrame({n:[float(i+10) for i in range(len(index))] for n in ('open','high','low','close','volume')},index=index)
    c=api.Cerebro(stdstats=False,runonce=False)
    feed=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(feed,name='raw')
    c.resampledata(feed,name='sampled',timeframe=api.TimeFrame.Days,compression=compression)
    class Business(api.Strategy):
        def prenext(self):self.next()
        def next(self):rows.append((len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.open[0] if len(d) else None,d.close[0] if len(d) else None,d.volume[0] if len(d) else None) for d in self.datas)))
    c.addstrategy(Business);c.run();return rows

@pytest.mark.parametrize('compression',[1,2])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('extra',[False,True])
def test_intraday_to_daily_default_eos_and_immediate_eof(compression,clone,extra):
    assert intraday_to_day(tl,compression,clone,extra)==intraday_to_day(official,compression,clone,extra)
