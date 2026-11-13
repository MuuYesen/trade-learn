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


@pytest.mark.parametrize('transform',['raw','dailyclone'])
@pytest.mark.parametrize('zone,mode',[(None,'none'),('America/New_York','none'),(None,'global'),('America/New_York','global'),('America/New_York','override')])
def test_actual_native_sessions_dst_and_late_global_calendar_match_fixed_bt(tmp_path,monkeypatch,zone,mode,transform):
    pytest.importorskip('nautilus_trader')  # This real-native case requires the optional ABI.
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tests.unit.engine.test_nautilus_store import store
    stamps=pd.to_datetime(['2026-03-06 14:30Z','2026-03-06 14:31Z','2026-03-06 18:01Z','2026-03-06 21:01Z','2026-03-09 13:30Z','2026-03-09 13:31Z','2026-03-09 20:01Z','2026-03-09 21:01Z'])
    prices=[1000.+10*i for i in range(len(stamps))]
    frame=pd.DataFrame({k:prices for k in ['open','high','low','close','volume']},index=stamps)
    rows=[];notices=[];gold_rows=[];gold_notices=[];errors=[]
    def options():return dict(tz=zone,sessionstart=time(9,30),sessionend=time(16),**({'calendar':Sessions(False)} if mode=='override' else {}))
    reference=official.Cerebro(stdstats=False,runonce=False,preload=False)
    raw=official.feeds.PandasData(dataname=frame,timeframe=official.TimeFrame.Minutes,**options());reference.adddata(raw)
    if transform=='dailyclone':reference.resampledata(raw,timeframe=official.TimeFrame.Days)
    for d in reference.datas:d._last=lambda datamaster=None:False
    if mode!='none':reference.addcalendar(Sessions())
    reference.add_timer(official.timer.SESSION_START,strats=True,marker='start')
    reference.add_timer(official.timer.SESSION_END,strats=True,marker='end')
    reference.addstrategy(build(official,gold_rows,gold_notices));reference.run()
    ins=TestInstrumentProvider.btcusdt_perp_binance();typ=BarType.from_str(str(ins.id)+'-1-MINUTE-LAST-EXTERNAL')
    n=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False));n.add_venue(ins.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))));n.add_instrument(ins)
    n.add_data([Bar(typ,*[ins.make_price(v)]*4,ins.make_qty(v),stamp.value,stamp.value) for stamp,v in zip(stamps,prices)])
    value=store(tmp_path);engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker())
    raw=value.getdata(typ,**options());engine.adddata(raw)
    if transform=='dailyclone':engine.resampledata(raw,timeframe=bt.TimeFrame.Days)
    if mode!='none':engine.addcalendar(Sessions())
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


@pytest.mark.parametrize('options',[dict(tz=False),dict(tz=3),dict(sessionstart='09:30'),dict(sessionend=16),dict(calendar=object())])
def test_bad_live_view_metadata_fails_before_registering_a_subscription(tmp_path,options):
    from tests.unit.engine.test_nautilus_store import store
    value=store(tmp_path)
    with pytest.raises((TypeError,ValueError)):
        value.getdata('BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL',**options)
    assert value._feeds==[] and not value._used
