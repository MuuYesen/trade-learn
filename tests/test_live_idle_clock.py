"""Real NA Clock alert against fixed BT no-source check with the same clock input."""
from types import SimpleNamespace as NS
from datetime import datetime, timezone
import asyncio
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as bt


@pytest.mark.parametrize('profile',['default','left','rawtime','countonly'])
@pytest.mark.parametrize('kind',['replay','resample'])
@pytest.mark.parametrize('with_indicators',[False,True])
@pytest.mark.parametrize('clone',[False,True])
def test_native_clock_closes_observed_replay_without_new_bar(tmp_path,monkeypatch,clone,with_indicators,kind,profile):
    pytest.importorskip('nautilus_trader')  # This real-native case requires the optional ABI.
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType,QuoteTick
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tests.unit.engine.test_nautilus_store import store
    stamps=[pd.Timestamp('2026-01-02T09:01:00Z'),pd.Timestamp('2026-01-02T09:09:00Z')]
    check=pd.Timestamp('2026-01-02T09:06:00Z').value
    options=dict(adjbartime=True)
    if profile=='left':options['rightedge']=False
    elif profile=='rawtime':options['adjbartime']=False
    elif profile=='countonly':options['bar2edge']=False
    frame=pd.DataFrame({key:[1000.,1100.] for key in ('open','high','low','close')},index=pd.DatetimeIndex(stamps));frame['volume']=10.
    gold=[];actual=[];errors=[];sources=[];stopped=[];gold_orders=[];actual_orders=[]
    def business(api,rows):
        class Counter(api.Indicator):
            lines=('calls',)
            def __init__(self):self.n=0
            def next(self):self.n+=1;self.lines.calls[0]=self.n
            prenext=next;nextstart=next
        class Business(api.Strategy):
            def __init__(self):
                self.sent=False
                self.counts=[Counter(d) for d in self.datas] if with_indicators else []
            def record(self):
                rows.append((clock['ns'],len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,float(d.close[0]) if len(d) else None) for d in self.datas),tuple((len(c),float(c[0]) if len(c) else None) for c in self.counts)))
                if profile=='default' and not clone and clock['ns']==check and not self.sent:
                    self.sent=True;self.buy(size=.125)
            def notify_order(self,order):
                if order.status==order.Completed:
                    (gold_orders if api is official else actual_orders).append((order.executed.size,order.executed.price,order.executed.comm))
            prenext=record;nextstart=record;next=record
        return Business
    clock={'ns':stamps[0].value}
    class ObservedClock(datetime):
        @classmethod
        def utcnow(cls):return datetime.fromtimestamp(clock['ns']/1e9,timezone.utc).replace(tzinfo=None)
    import backtrader.resamplerfilter as filters
    monkeypatch.setattr(filters,'datetime',ObservedClock)
    class IdleFeed(official.feeds.PandasData):
        def __init__(self):
            super().__init__();self.turn=0
        def _load(self):
            self.turn+=1
            if self.turn==2:
                clock['ns']=check
                return None
            loaded=super()._load()
            if loaded:clock['ns']=stamps[self._idx].value
            return loaded
    reference=official.Cerebro(stdstats=False,runonce=False,preload=False)
    raw=IdleFeed(dataname=frame,timeframe=official.TimeFrame.Minutes)
    if clone:reference.adddata(raw,name='raw')
    target=(reference.replaydata if kind=='replay' else reference.resampledata)(raw,timeframe=official.TimeFrame.Minutes,compression=7,name='sample',**options)
    target._last=lambda datamaster=None:False
    reference.addstrategy(business(official,gold));reference.run()
    ins=TestInstrumentProvider.btcusdt_perp_binance();typ=BarType.from_str(str(ins.id)+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(ins.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))));native.add_instrument(ins)
    events=[Bar(typ,*[ins.make_price(1000)]*4,ins.make_qty(10),stamps[0].value,stamps[0].value),
        QuoteTick(ins.id,ins.make_price(1000),ins.make_price(1001),ins.make_qty(1),ins.make_qty(1),check,check),
        Bar(typ,*[ins.make_price(1100)]*4,ins.make_qty(10),stamps[1].value,stamps[1].value)]
    value=store(tmp_path,buffer_capacity=3);engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker());raw=value.getdata(typ)
    if clone:engine.adddata(raw,name='raw')
    (engine.replaydata if kind=='replay' else engine.resampledata)(raw,timeframe=bt.TimeFrame.Minutes,compression=7,name='sample',**options)
    engine.addstrategy(business(bt,actual))
    class Node:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return NS(stop=lambda:None)
        async def run_async(self):
            for event in events:
                clock['ns']=event.ts_init
                native.clear_data();native.add_data([event]);native.run(streaming=True)
                while self.actor._tasks:await asyncio.gather(*tuple(self.actor._tasks))
                sources.append(self.actor.frontend.context.ts_event_ns if self.actor.frontend.context is not None else None)
            native.end()
            stopped.append([name for name in self.actor._callback_clock.timer_names() if name.startswith('__tl_bar_check_')])
        def dispose(self):
            nonlocal native
            errors.append(self.actor._worker_error)
            try:
                native.dispose()
            finally:
                # This fixture owns the closure retaining the unsendable simulator.
                self.actor=None
                native=None
    monkeypatch.setattr(value,'_build_node',Node)
    engine.run()
    assert errors==[None]
    if sources[1] is not None:assert sources[1]==stamps[0].value  # Clock/quote arrival never replaces source bar time.
    assert stopped==[[]]
    assert actual==gold
    if kind=='replay':
        assert actual_orders==gold_orders
    elif not clone and profile=='default':
        # Historical BackBroker gates Market on its ordered Data advancing.
        # BT live brokers transmit immediately; NA owns this venue fill.
        # Keep the observed economic distinction explicit, never rewrite fills.
        assert gold_orders==[]
        assert actual_orders==[(.125,1100.,0.)]
    if profile=='default' and kind=='replay' and not clone:assert actual_orders==[(.125,1100.,0.)]


@pytest.mark.parametrize('kind',['replay','resample'])
def test_rightedge_false_idle_deadline_is_a_future_boundary_not_past_label(kind):
    from tradelearn_runtime.bar_views import make_live_view
    stamp=pd.Timestamp('2026-01-02T09:01:00Z').value
    plan=make_live_view(dict(kind=kind,clone=False,source_period=dict(timeframe=bt.TimeFrame.Minutes,compression=1),options=dict(timeframe=bt.TimeFrame.Minutes,compression=7,bar2edge=True,adjbartime=True,rightedge=False,boundoff=0,takelate=True)))
    plan.push(stamp,dict(open=1000.,high=1000.,low=1000.,close=1000.,volume=1.))
    assert plan.next_check_ns()==pd.Timestamp('2026-01-02T09:06:00Z').value
