import pandas as pd
import pytest
from tradelearn.indicators.base import FunctionIndicator
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.engine import Strategy


def test_persistent_function_advances_once_per_event_and_shares_tuple_outputs():
    calls=[]
    def step(buffers,state,params):
        calls.append(buffers['close'][-1]); total=state.get('total',0)+buffers['close'][-1]
        state['total']=total
        return total,total*2
    f=FunctionIndicator('sum',lambda close: pd.DataFrame({'sum':close.cumsum(),'twice':close.cumsum()*2}),
                        {},('close',),step)
    f.output_names=('sum','twice')
    class S(Strategy):
        def __init__(self): self.values=[]; self.output=f(self.data.close)
        def next(self):
            first=self.output.sum[0]
            assert self.output.sum[0]==first
            self.values.append((first,self.output.twice[0]))
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(4):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=i+1,high=i+1,low=i+1,close=i+1,volume=1))
    assert calls==[1,2,3,4]
    assert r.strategy.values==[(1,2),(3,6),(6,12),(10,20)]
    assert r.feeds['A']._history is None
    r.stop()


def test_line_comparison_expression_is_live_in_both_profiles():
    from tradelearn.lite import Backtest,Strategy as LiteStrategy
    from tradelearn.engine import Cerebro,feeds
    frame=pd.DataFrame({k:[1.,3.,2.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3,tz='UTC'))
    class E(Strategy):
        def __init__(self):self.signal=self.data.close>2;self.values=[]
        def next(self):self.values.append(bool(self.signal))
    class L(LiteStrategy):
        def init(self):self.signal=self.data.close>2;self.values=[]
        def next(self):self.values.append(bool(self.signal))
    c=Cerebro();c.adddata(feeds.PandasData(dataname=frame));c.addstrategy(E)
    assert c.run()[0].values==[False,True,False]
    assert Backtest(frame,L).run().strategy.values==[False,True,False]


def test_native_state_warmup_survives_retained_window_rollover():
    f=FunctionIndicator('sum',lambda close,period=4: close.cumsum(),{'period':4},('close',),
        lambda buffers,state,params: state.setdefault('values',[]).append(buffers['close'][-1]) or sum(state['values']),warmup=4)
    class S(Strategy):
        def __init__(self):self.output=f(self.data.close,period=4);self.values=[]
        def next(self):self.values.append(self.output[0])
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(5):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=1,high=1,low=1,close=1,volume=1))
    assert r.strategy.values==[4,5]
    r.stop()


def test_late_persistent_state_requires_available_prefix():
    f=FunctionIndicator('sum',lambda close:close.cumsum(),{},('close',),
        lambda buffers,state,params: state.setdefault('total',0) or 0)
    r=StrategyRuntime(Strategy,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(3):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=1,high=1,low=1,close=1,volume=1))
    with pytest.raises(ValueError,match='discarded|replay'):f(r.feeds['A'].close)
    r.stop()


def test_stream_lifecycle_length_counts_deliveries_while_history_is_bounded():
    class S(Strategy):
        def __init__(self):self.counts=[]
        def next(self):self.counts.append((len(self),len(self.data),len(self.data.close),len(self.data.close.to_series())))
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(5):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=1,high=1,low=1,close=1,volume=1))
    assert r.strategy.counts[-1]==(5,5,5,2)
    assert len(r.feeds['A']._frame)==2
    r.stop()


@pytest.mark.parametrize('bands',[None,2.])
def test_configured_vwap_output_arity_matches_native_batch(bands):
    import numpy as np
    from tradelearn.indicators import tv
    frame=pd.DataFrame({k:[1.,3.,2.,4.,6.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=5,tz='UTC'))
    class S(Strategy):
        def __init__(self):
            self.output=tv.vwap(self.data.close,self.data.volume,self.data.close<2,stdev_mult=bands)
            self.values=[]
        def next(self):
            names=('vwap','upper','lower') if bands is not None else ('vwap',)
            self.values.append(tuple(getattr(self.output,k)[0] for k in names) if bands is not None else (self.output[0],))
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for ts,row in frame.iterrows():r.step_market('A',dict(ts=ts,**row))
    r.stop()
    expected=tv.vwap(frame.close,frame.volume,frame.close<2,stdev_mult=bands)
    np.testing.assert_allclose(r.strategy.values,np.asarray(expected).reshape(len(frame),-1),equal_nan=True)


def test_source_on_custom_shared_indicator_keeps_persistent_graph():
    import numpy as np
    from tradelearn.engine import Indicator
    from tradelearn.indicators import talib
    class Mid(Indicator):
        lines=('mid',)
        def __init__(self):self.lines.mid=(self.data.high+self.data.low)/2
    class S(Strategy):
        def __init__(self):self.output=talib.SMA(Mid(self.data),timeperiod=3);self.values=[]
        def next(self):self.values.append(self.output[0])
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(5):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=i+1,high=i+2,low=i,close=i+1,volume=1))
    r.stop()
    np.testing.assert_allclose(r.strategy.values,[2.,3.,4.])
    assert r.feeds['A']._history_materializations==0


def test_custom_indicator_uses_bt_warmup_and_nextstart_once():
    from tradelearn.engine import Indicator
    calls=[]
    class Custom(Indicator):
        lines=('value',)
        def __init__(self):self.addminperiod(2)
        def prenext(self):calls.append('prenext')
        def nextstart(self):calls.append('nextstart');self.next()
        def next(self):calls.append('next');self.value[0]=self.data.close[0]
    class S(Strategy):
        def __init__(self):self.indicator=Custom(self.data)
    r=StrategyRuntime(S,feeds={'A':'bar'});r.start()
    for i in range(3):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=1,high=1,low=1,close=1,volume=1))
    r.stop()
    assert calls==['prenext','nextstart','next','next']
