"""Existing BT feed timezone/session/calendar facts on the native bar clock."""
from datetime import datetime,time,timedelta
from types import SimpleNamespace as NS
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as bt


class Sessions:
    def __init__(self,early=True):self.early=early
    def schedule(self,day,tz=None):
        day=day.replace(tzinfo=None)
        while True:
            end=datetime.combine(day.date(),time(18 if self.early and day.date()==datetime(2026,3,6).date() else 21))
            if day.weekday()<5 and day<=end:return datetime.combine(day.date(),time(13,30)),end
            day=datetime.combine(day.date()+timedelta(days=1),time())


def build(api,rows,notices):
    class Business(api.Strategy):
        def next(self):rows.append((len(self),tuple((len(d),d.datetime[0],d.datetime.datetime().isoformat(),d.close[0],d.volume[0],d.num2date().isoformat(),d.date2num(d.datetime.datetime()),d.p.timeframe,d.p.compression,d._timeframe,d._compression) for d in self.datas)))
        def notify_timer(self,timer,when,*args,**kwargs):
            notices.append((kwargs['marker'],when.isoformat(),len(self),len(self.data),self.data.datetime[0],self.data.datetime.datetime().isoformat()))
    return Business


@pytest.mark.parametrize('transform',['dailyreplay','dailyreplayclone'])
@pytest.mark.parametrize('flag',['adjust','unbounded'])
@pytest.mark.parametrize('zone,mode',[(None,'global'),('America/New_York','override')])
def test_actual_native_configured_calendar_replay_matches_fixed_bt(tmp_path,monkeypatch,zone,mode,transform,flag,_period='Days',_compression=1,_stamps=None,_calendar_type=Sessions):
    pytest.importorskip('nautilus_trader')  # This real-native case requires the optional ABI.
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tests.unit.engine.test_nautilus_store import store
    stamps=pd.to_datetime(['2026-03-06 14:30Z','2026-03-06 14:31Z','2026-03-06 18:01Z','2026-03-06 21:01Z','2026-03-09 13:30Z','2026-03-09 13:31Z','2026-03-09 20:01Z','2026-03-09 21:01Z'])
    if _stamps is not None:stamps=pd.to_datetime(_stamps,utc=True)
    prices=[1000.+10*i for i in range(len(stamps))]
    frame=pd.DataFrame({k:prices for k in ['open','high','low','close','volume']},index=stamps)
    rows=[];notices=[];gold_rows=[];gold_notices=[];errors=[]
    def options():return dict(tz=zone,sessionstart=time(9,30),sessionend=time(16),**({'calendar':_calendar_type(False)} if mode=='override' else {}))
    reference=official.Cerebro(stdstats=False,runonce=False,preload=False)
    raw=official.feeds.PandasData(dataname=frame,timeframe=official.TimeFrame.Minutes,**options())
    if transform.endswith('clone'):reference.adddata(raw)
    reference.replaydata(raw,timeframe=getattr(official.TimeFrame,_period),compression=_compression,**({'adjbartime':True} if flag=='adjust' else {'bar2edge':False}))
    for d in reference.datas:d._last=lambda datamaster=None:False
    if mode!='none':reference.addcalendar(_calendar_type())
    reference.add_timer(official.timer.SESSION_START,strats=True,marker='start')
    reference.add_timer(official.timer.SESSION_END,strats=True,marker='end')
    reference.addstrategy(build(official,gold_rows,gold_notices));reference.run()
    ins=TestInstrumentProvider.btcusdt_perp_binance();typ=BarType.from_str(str(ins.id)+'-1-MINUTE-LAST-EXTERNAL')
    n=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False));n.add_venue(ins.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))));n.add_instrument(ins)
    n.add_data([Bar(typ,*[ins.make_price(v)]*4,ins.make_qty(v),stamp.value,stamp.value) for stamp,v in zip(stamps,prices)])
    value=store(tmp_path);engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker())
    raw=value.getdata(typ,**options())
    if transform.endswith('clone'):engine.adddata(raw)
    engine.replaydata(raw,timeframe=getattr(bt.TimeFrame,_period),compression=_compression,**({'adjbartime':True} if flag=='adjust' else {'bar2edge':False}))
    if mode!='none':engine.addcalendar(_calendar_type())
    engine.add_timer(bt.timer.SESSION_START,strats=True,marker='start')
    engine.add_timer(bt.timer.SESSION_END,strats=True,marker='end')
    engine.addstrategy(build(bt,rows,notices))
    class ReplayNode:
        def add_strategy(self,actor):self.actor=actor;n.add_strategy(actor)
        def handle(self):return NS(stop=lambda:None)
        async def run_async(self):n.run()
        def dispose(self):
            nonlocal n
            errors.append(self.actor._worker_error)
            try:n.dispose()
            finally:self.actor=None;n=None
    monkeypatch.setattr(value,'_build_node',ReplayNode)
    engine.run()
    assert errors==[None] and not value._host.frontend.failed
    assert rows==gold_rows and notices==gold_notices



class PeriodCalendar(Sessions):
    def following(self,day):
        day+=timedelta(days=1)
        while day.weekday()>4:day+=timedelta(days=1)
        return day
    def last_weekday(self,day):return day.isocalendar()[:2]!=self.following(day).isocalendar()[:2]
    def last_monthday(self,day):return (day.year,day.month)!=(self.following(day).year,self.following(day).month)
    def last_yearday(self,day):return day.year!=self.following(day).year


@pytest.mark.parametrize('period,days',[('Weeks',['2026-11-26 14:30Z','2026-11-27 14:30Z']),('Months',['2026-11-27 14:30Z','2026-11-30 14:30Z']),('Years',['2026-12-30 14:30Z','2026-12-31 14:30Z'])])
@pytest.mark.parametrize('compression',[1,2])
@pytest.mark.parametrize('clone',[False,True])
def test_native_calendar_compression_edge_does_not_close_before_compressed_count(tmp_path,monkeypatch,period,days,compression,clone):
    test_actual_native_configured_calendar_replay_matches_fixed_bt(tmp_path,monkeypatch,'America/New_York','global',
        'dailyreplayclone' if clone else 'dailyreplay','adjust',_period=period,
        _compression=compression,_stamps=days,_calendar_type=PeriodCalendar)

@pytest.mark.parametrize('period,days',[('Weeks',['2026-11-27 14:30Z','2026-11-27 15:00Z','2026-11-27 16:00Z','2026-11-30 14:30Z']),('Months',['2026-11-30 14:30Z','2026-11-30 15:00Z','2026-11-30 16:00Z','2026-12-01 14:30Z']),('Years',['2026-12-31 14:30Z','2026-12-31 15:00Z','2026-12-31 16:00Z','2027-01-04 14:30Z'])])
@pytest.mark.parametrize('clone',[False,True])
def test_native_calendar_compression_period_continuation_matches_fixed_bt(tmp_path,monkeypatch,period,days,clone):
    test_actual_native_configured_calendar_replay_matches_fixed_bt(tmp_path,monkeypatch,'America/New_York','global',
        'dailyreplayclone' if clone else 'dailyreplay','adjust',_period=period,
        _compression=2,_stamps=days,_calendar_type=PeriodCalendar)
