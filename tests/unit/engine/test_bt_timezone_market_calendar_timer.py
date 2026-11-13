"""Independent P03 counterexamples: numeric order clock and retained timer target."""
import datetime as dt
import pandas as pd
import backtrader as official
import pytest
from tradelearn import engine as tl


@pytest.mark.parametrize('zone',['UTC','America/New_York','Asia/Tokyo'])
@pytest.mark.parametrize('mode',['raw','resample','clone'])
def test_order_creation_uses_bt_utc_ordinal_and_not_localized_display(zone,mode):
    def execute(api):
        index=pd.to_datetime(['2026-03-06 09:30','2026-03-06 09:31','2026-03-06 09:37','2026-03-06 16:00',
            '2026-03-09 09:30','2026-03-09 09:31','2026-03-09 09:37','2026-03-09 16:00'])
        values=[100.,110.,90.,120.,80.,140.,95.,105.]
        frame=pd.DataFrame({'open':values,'high':[v+4 for v in values],'low':[v-6 for v in values],
                           'close':[v+1 for v in values],'volume':list(range(1,9))},index=index)
        c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(12500);c.broker.setcommission(commission=.002)
        raw=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,tzinput='America/New_York',tz=zone,
                               sessionstart=dt.time(9,30),sessionend=dt.time(16))
        if mode=='clone':c.adddata(raw)
        target=c.adddata(raw) if mode=='raw' else c.resampledata(raw,timeframe=api.TimeFrame.Minutes,compression=7)
        if api is tl:
            from tradelearn.backtest.clock import native_timestamps
            frozen=native_timestamps(target,'ns').copy()
        rows=[];orders=[]
        class Business(api.Strategy):
            def __init__(self):self.sent=False
            def next(self):
                rows.append((self.datetime[0],target.datetime[0],target.datetime.datetime().isoformat()))
                if not self.sent:self.sent=True;self.buy(data=target,size=.125)
            def notify_order(self,o):orders.append((o.status,o.created.dt,o.executed.size,o.executed.price,o.executed.comm))
        c.addstrategy(Business);c.run()
        if api is tl:assert (native_timestamps(target,'ns')==frozen).all()
        return rows,orders,c.broker.getcash()
    got,want=execute(tl),execute(official)
    assert got[0]==want[0]
    assert len(got[1])==len(want[1])
    for left,right in zip(got[1],want[1]):
        assert left[:4]==right[:4]
        assert left[4]==pytest.approx(right[4],abs=1e-12)
    assert got[2]==pytest.approx(want[2],abs=1e-9)


class ProvidedSession:
    def __init__(self,early=True):self.early=early
    def schedule(self,day,tz=None):
        day=day.replace(tzinfo=None)
        while True:
            end=dt.datetime.combine(day.date(),dt.time(18 if self.early and day.date()==dt.date(2026,11,27) else 21))
            if day.weekday()<5 and day<=end:return dt.datetime.combine(day.date(),dt.time(14,30)),end
            day=dt.datetime.combine(day.date()+dt.timedelta(days=1),dt.time())


@pytest.mark.parametrize('override',[False,True])
def test_default_session_end_target_survives_date_change_until_provider_eos(override):
    def execute(api):
        index=pd.to_datetime(['2026-11-27 14:30','2026-11-27 16:00','2026-11-27 17:59','2026-11-27 18:01',
                             '2026-11-30 14:30','2026-11-30 20:59','2026-11-30 21:01','2026-12-01 14:30'])
        frame=pd.DataFrame({n:list(range(10,18)) for n in ('open','high','low','close','volume')},index=index)
        c=api.Cerebro(stdstats=False,runonce=False)
        raw=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,
                              **({'calendar':ProvidedSession(early=False)} if override else {}))
        c.adddata(raw);c.resampledata(raw,timeframe=api.TimeFrame.Days)
        c.addcalendar(ProvidedSession());c.add_timer(api.timer.SESSION_END,strats=True)
        events=[]
        class Business(api.Strategy):
            def notify_timer(self,timer,when,*args,**kwargs):events.append((when.isoformat(),len(self),len(self.data)))
        c.addstrategy(Business);c.run();return events
    assert execute(tl)==execute(official)
