"""Same observation clock on BT clone checks and fixed native input, without fake source timestamps."""
from types import SimpleNamespace as NS
import asyncio
import math
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as bt


def strategy(api,rows,indicator_order):
    class Counter(api.Indicator):
        lines=('calls',)
        def __init__(self):self.count=0
        def next(self):
            self.count+=1;self.lines.calls[0]=self.count
            indicator_order.append((self.data._name,len(self.data),self.count))
        prenext=next
        nextstart=next
    class Business(api.Strategy):
        def __init__(self):
            self.counters=[Counter(d) for d in self.datas]
            self.means=[api.indicators.SMA(d.close,period=2) if api is official else api.talib.SMA(d.close,timeperiod=2) for d in self.datas]
        def record(self,phase):
            def number(value):return None if math.isnan(float(value)) else float(value)
            rows.append((phase,len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,float(d.close[0]) if len(d) else None,len(self.means[i]),number(self.means[i][0]),len(self.counters[i]),number(self.counters[i][0])) for i,d in enumerate(self.datas))))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    return Business


@pytest.mark.parametrize('offsets',[[0,1,1,8,9,16],[0,4,2,8,9,16]])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('takelate',[False,True])
def test_actual_na_admitted_late_replay_does_not_rewrite_source_clock(tmp_path,monkeypatch,offsets,clone,takelate):
    pytest.importorskip('nautilus_trader')  # This real-native case requires the optional ABI.
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.config import DataEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tests.unit.engine.test_nautilus_store import store
    stamps=[pd.Timestamp('2026-01-02T09:00:00Z')+pd.Timedelta(minutes=i) for i in offsets]
    prices=[1000.,1070.,940.,1150.,810.,1200.]
    frame=pd.DataFrame({k:prices for k in ['open','high','low','close']},index=pd.DatetimeIndex(stamps));frame['volume']=10.
    gold=[];actual=[];errors=[];gold_order=[];actual_order=[]
    # BT's clone check reads utcnow, whereas BacktestEngine uses ts_init.
    # Supply the SAME observation-clock input to the oracle; source dates and
    # all resampler logic remain the untouched fixed BT implementation.
    from datetime import datetime,timezone
    import backtrader.resamplerfilter as filters
    receive=stamps[0].value;arrival=[]
    for stamp in stamps:
        receive=max(receive+1000,stamp.value);arrival.append(receive)
    clock={'ns':arrival[0]}
    class ObservedClock(datetime):
        @classmethod
        def utcnow(cls):return datetime.fromtimestamp(clock['ns']/1e9,timezone.utc).replace(tzinfo=None)
    class ObservedFeed(official.feeds.PandasData):
        def _load(self):
            loaded=super()._load()
            if loaded:clock['ns']=arrival[self._idx]
            return loaded
    monkeypatch.setattr(filters,'datetime',ObservedClock)
    reference=official.Cerebro(stdstats=False,runonce=False,preload=False)
    raw=ObservedFeed(dataname=frame,timeframe=official.TimeFrame.Minutes)
    if clone:reference.adddata(raw,name='raw')
    target=reference.replaydata(raw,timeframe=official.TimeFrame.Minutes,compression=7,adjbartime=True,takelate=takelate,name='sample')
    target._last=lambda datamaster=None:False
    reference.addstrategy(strategy(official,gold,gold_order));reference.run()
    ins=TestInstrumentProvider.btcusdt_perp_binance();typ=BarType.from_str(str(ins.id)+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False,data_engine=DataEngineConfig(validate_data_sequence=False)))
    native.add_venue(ins.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))));native.add_instrument(ins)
    # The event clock is genuinely late; only the distinct observation clock
    # advances. This is explicit NA sequence policy, not its default behavior.
    receive=stamps[0].value
    bars=[]
    for stamp,price in zip(stamps,prices):
        receive=max(receive+1000,stamp.value)
        bars.append(Bar(typ,*[ins.make_price(price)]*4,ins.make_qty(10),stamp.value,receive))
    value=store(tmp_path,buffer_capacity=3);engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker())
    raw=value.getdata(typ)
    if clone:engine.adddata(raw,name='raw')
    engine.replaydata(raw,timeframe=bt.TimeFrame.Minutes,compression=7,adjbartime=True,takelate=takelate,name='sample')
    engine.addstrategy(strategy(bt,actual,actual_order))
    class Node:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return NS(stop=lambda:None)
        async def run_async(self):
            for bar in bars:
                native.clear_data();native.add_data([bar]);native.run(streaming=True)
                while self.actor._tasks:await asyncio.gather(*tuple(self.actor._tasks))
            native.end()
        def dispose(self):
            nonlocal native
            errors.append(self.actor._worker_error)
            try:native.dispose()
            finally:self.actor=None;native=None
    monkeypatch.setattr(value,'_build_node',Node)
    engine.run()
    assert errors==[None]
    assert actual==gold
    assert actual_order==gold_order


def test_unpublished_idle_clone_can_accept_a_current_source_before_its_boundary():
    from tradelearn_runtime.bar_views import LiveReplay
    options=dict(timeframe=bt.TimeFrame.Minutes,compression=7,bar2edge=True,adjbartime=True,rightedge=True,boundoff=0,takelate=False)
    plan=LiveReplay(dict(kind='replay',clone=True,source_period=dict(timeframe=bt.TimeFrame.Minutes,compression=1),options=options))
    stamp=lambda text:pd.Timestamp('2026-01-02T'+text+'Z').value
    row=lambda price,qty:dict(open=price,high=price+4,low=price-6,close=price+1,volume=qty)
    plan.push_observed(stamp('09:00:00'),row(100.,1.),stamp('09:00:01'))
    plan.push_observed(stamp('09:04:00'),row(107.,2.),stamp('09:04:01'))
    assert plan.push_observed(stamp('09:02:00'),row(94.,3.),stamp('09:05:00'))==[]
    withdrawn=plan.check(stamp('09:06:00'));assert withdrawn[3]==0
    output=plan.push_observed(stamp('09:05:00'),row(115.,4.),stamp('09:06:01'))
    assert len(output)==1
    label,values,source,count=output[0]
    assert label==source==stamp('09:05:00') and count==1
    assert values==dict(open=100.,high=119.,low=94.,close=116.,volume=7.)


@pytest.mark.parametrize('rejected',['09:03:00','09:04:00'])
def test_rejected_preclose_clone_source_consumes_the_next_restore_turn(rejected):
    from tradelearn_runtime.bar_views import LiveReplay
    options=dict(timeframe=bt.TimeFrame.Minutes,compression=7,bar2edge=True,adjbartime=True,rightedge=True,boundoff=0,takelate=False)
    plan=LiveReplay(dict(kind='replay',clone=True,source_period=dict(timeframe=bt.TimeFrame.Minutes,compression=1),options=options))
    stamp=lambda text:pd.Timestamp('2026-01-02T'+text+'Z').value
    row=lambda price,qty:dict(open=price,high=price+4,low=price-6,close=price+1,volume=qty)
    plan.push_observed(stamp('09:00:00'),row(100.,1.),stamp('09:00:01'))
    plan.push_observed(stamp('09:04:00'),row(107.,2.),stamp('09:04:01'))
    assert plan.push_observed(stamp('09:02:00'),row(94.,3.),stamp('09:05:00'))==[]
    plan.check(stamp('09:06:00'))
    assert plan.push_observed(stamp(rejected),row(115.,4.),stamp('09:06:01'))==[]
    output=plan.push_observed(stamp('09:08:00'),row(81.,5.),stamp('09:08:01'))
    assert output[-1] is None  # Actual raw master advances, queued derived only restores.
    assert output[0][3]==1 and output[0][1]['close']==108. and output[0][1]['volume']==3.
    [current]=plan.push_observed(stamp('09:09:00'),row(88.,6.),stamp('09:09:01'))
    assert current[3]==2 and current[1]['close']==89. and current[1]['volume']==6.
