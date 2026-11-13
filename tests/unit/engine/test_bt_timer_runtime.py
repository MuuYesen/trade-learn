"""Cerebro timers are ordinary shared-runtime callbacks, including during warmup."""
from datetime import time,timedelta
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def profile(api,repeat,notify_strats,with_indicator,cheat=False,stream=False,when=None,offset=timedelta()):
    trace=[]
    class Driver(api.Cerebro):
        def notify_timer(self,timer,when,*args,**kwargs):trace.append(('engine',timer.p.tid,when.isoformat(),args,kwargs))
    class Business(api.Strategy):
        def __init__(self):
            if with_indicator:
                if api is official:self.average=api.indicators.SMA(self.data,period=3)
                else:
                    from tradelearn.indicators import talib
                    self.average=talib.SMA(self.data.close,timeperiod=3)
        def notify_timer(self,timer,when,*args,**kwargs):
            import math
            value=float(self.average[0]) if with_indicator and len(self) else None
            if value is not None and math.isnan(value):value=None
            trace.append(('timer',when.isoformat(),len(self),len(self.data),args,kwargs,value,
                          self.datetime.datetime().isoformat() if len(self) else None))
        def record(self,name):trace.append((name,len(self),self.data.datetime.datetime().isoformat()))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=Driver(stdstats=False,runonce=False)
    index=pd.to_datetime(['2026-01-01 08:59','2026-01-01 09:00','2026-01-01 09:06','2026-01-01 09:09','2026-01-02 09:20'])
    f=pd.DataFrame(dict(open=10.,high=11.,low=9.,close=10.,volume=1.),index=index)
    c.adddata(api.feeds.PandasData(dataname=f,timeframe=api.TimeFrame.Minutes))
    timer=c.add_timer(time(9) if when is None else when,offset=offset,repeat=repeat,strats=notify_strats,cheat=cheat,marker=42)
    assert timer.p.tid==0
    c.addstrategy(Business)
    if stream:
        from tradelearn.backtest.event_runtime import StrategyRuntime
        from tradelearn_runtime.components import LiveComponents
        runtime=StrategyRuntime(Business,feeds={'A':'bar'},capacity=3)
        components=LiveComponents(c,runtime)
        try:
            runtime.start()
            for stamp,row in f.iterrows():
                runtime.step_market('A',dict(ts=stamp.tz_localize('UTC'),**row.to_dict()))
                components.after_market()
            runtime.stop();components.finish()
        finally:runtime.cleanup()
    else:c.run()
    return trace


@pytest.mark.parametrize('repeat',[timedelta(),timedelta(minutes=5)])
@pytest.mark.parametrize('notify_strats',[False,True])
@pytest.mark.parametrize('with_indicator',[False,True])
def test_normal_timer_callback_owner_strategy_warmup_and_sparse_repeat_match_bt(repeat,notify_strats,with_indicator):
    assert profile(tl,repeat,notify_strats,with_indicator)==profile(official,repeat,notify_strats,with_indicator)


@pytest.mark.parametrize('with_indicator',[False,True])
@pytest.mark.parametrize('repeat',[timedelta(),timedelta(minutes=5)])
def test_cheat_timer_uses_original_pre_match_phase_without_a_second_execution_loop(with_indicator,repeat):
    assert profile(tl,repeat,True,with_indicator,cheat=True)==profile(official,repeat,True,with_indicator,cheat=True)


def timer_orders(api,cheat,order_kind):
    trace=[]
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def notify_timer(self,timer,when,*args,**kwargs):
            trace.append(('timer',len(self),len(self.data),self.data.close[0]))
            if not self.sent:
                self.sent=True
                self.buy(size=.25,exectype=api.Order.Market if order_kind=='market' else api.Order.Limit,price=None if order_kind=='market' else 35.)
        def notify_order(self,o):trace.append(('order',o.getstatusname(),o.executed.size,o.executed.price,len(self),len(self.data)))
        def next(self):trace.append(('next',len(self),self.broker.getcash(),self.broker.getvalue()))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000)
    f=pd.DataFrame({n:[10.,20.,30.,40.] for n in ('open','high','low','close','volume')},index=pd.to_datetime(['2026-01-01 08:59','2026-01-01 09:00','2026-01-01 09:06','2026-01-01 09:09']))
    c.adddata(api.feeds.PandasData(dataname=f,timeframe=api.TimeFrame.Minutes))
    c.add_timer(time(9),strats=True,cheat=cheat);c.addstrategy(Business);c.run();return trace


@pytest.mark.parametrize('cheat',[False,True])
@pytest.mark.parametrize('order_kind',['market','limit'])
def test_timer_order_matching_and_public_notifications_match_fixed_bt(cheat,order_kind):
    assert timer_orders(tl,cheat,order_kind)==timer_orders(official,cheat,order_kind)


@pytest.mark.parametrize('with_indicator',[False,True])
@pytest.mark.parametrize('repeat',[timedelta(),timedelta(minutes=5)])
def test_native_shared_stream_timer_keeps_bound_capacity_and_bt_normal_phase(with_indicator,repeat):
    assert profile(tl,repeat,True,with_indicator,stream=True)==profile(official,repeat,True,with_indicator)


@pytest.mark.parametrize('when,offset',[(official.timer.SESSION_START,timedelta()),(official.timer.SESSION_END,timedelta(hours=-15))])
@pytest.mark.parametrize('stream',[False,True])
def test_default_data_session_timer_uses_bt_session_reference_and_offset(when,offset,stream):
    assert profile(tl,timedelta(),True,True,stream=stream,when=when,offset=offset)==profile(official,timedelta(),True,True,when=when,offset=offset)
