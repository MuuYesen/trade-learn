"""Configured native-source replay uses the fixed BT calendar revision rules."""
import pandas as pd
import pytest
import backtrader as bt
from tradelearn.engine.base import TimeFrame
from tradelearn_runtime.bar_views import LiveReplay
from tests.unit.engine.test_bt_replay_calendar_configured import Calendar


@pytest.mark.parametrize('frame_name',['Days','Weeks','Months','Years'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('flag',['adjust','unbounded','offset','rejectlate'])
def test_live_calendar_recipe_matches_fixed_bt_without_source_eof(frame_name,clone,flag):
    times=(['2026-03-06 14:30','2026-03-06 14:31','2026-03-06 18:01','2026-03-09 13:30','2026-03-09 13:31','2026-03-09 21:01','2026-03-10 13:30'] if frame_name=='Days' else ['2025-12-29 16:00','2025-12-30 16:00','2025-12-31 16:00','2026-01-02 16:00','2026-01-05 16:00','2026-01-30 16:00','2026-02-02 16:00'])
    values=[100.,107.,94.,115.,81.,120.,99.]
    df=pd.DataFrame(dict(open=values,high=[x+4 for x in values],low=[x-6 for x in values],close=[x+1 for x in values],volume=range(1,8)),index=pd.to_datetime(times))
    options=dict(timeframe=getattr(TimeFrame,frame_name),compression=1,bar2edge=True,adjbartime=False,rightedge=True,boundoff=0,takelate=True)
    options.update({'adjbartime':True} if flag=='adjust' else {'bar2edge':False} if flag=='unbounded' else {'boundoff':2} if flag=='offset' else {'takelate':False})
    gold=[]
    class Business(bt.Strategy):
        def next(self):
            d=self.datas[-1];gold.append((len(d),d.datetime[0],*[float(getattr(d,k)[0]) for k in ['open','high','low','close','volume']]))
    c=bt.Cerebro(stdstats=False,runonce=False,preload=False)
    params=dict(timeframe=bt.TimeFrame.Minutes if frame_name=='Days' else bt.TimeFrame.Days,compression=1,**({'calendar':Calendar()} if frame_name=='Days' else {}))
    raw=bt.feeds.PandasData(dataname=df,**params)
    if clone:c.adddata(raw,name='raw')
    target=c.replaydata(raw,name='sample',**options);target._last=lambda datamaster=None:False
    c.addstrategy(Business);c.run()
    live=LiveReplay(dict(kind='replay',source_period=params,calendar=Calendar() if frame_name=='Days' else None,options=options))
    result=[]
    for stamp,row in df.iterrows():
        for label,bar,available,count in live.push(stamp.value,{k:float(row[k]) for k in ['open','high','low','close','volume']}):
            result.append((count,bt.date2num(pd.Timestamp(label).to_pydatetime()),*[bar[k] for k in ['open','high','low','close','volume']]))
    assert result==gold
