"""Shared lifecycle releases owned backend state once and preserves output facts."""
import pandas as pd
import pytest
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.engine import Strategy
from tradelearn.indicators.base import FunctionIndicator,StreamBinding


def bar(i):
    return dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=i+1,high=i+1,low=i+1,close=i+1,volume=1)


@pytest.mark.parametrize('failure',['none','business','stop','both'])
def test_owned_bindings_reset_once_after_normal_or_exceptional_stop(monkeypatch,failure):
    resets=[];closed=[]
    original=StreamBinding.reset
    def reset(binding):resets.append(id(binding));return original(binding)
    monkeypatch.setattr(StreamBinding,'reset',reset)
    class Backend:
        def update(self,value):return value*2
        def close(self):closed.append(1)
    f=FunctionIndicator('owned',lambda close:close*2,{},('close',),stream_factory=lambda params:Backend())
    primary=LookupError('business');secondary=RuntimeError('stop')
    class S(Strategy):
        def __init__(self):self.line=f(self.data.close)
        def next(self):
            if failure in ('business','both'):raise primary
        def stop(self):
            assert self.line[0]==2
            if failure in ('stop','both'):raise secondary
    runtime=StrategyRuntime(S,feeds={'A':'bar'});runtime.start()
    if failure in ('business','both'):
        with pytest.raises(LookupError) as caught:runtime.step_market('A',bar(0))
        assert caught.value is primary
        if failure=='both':assert any('stop' in note for note in primary.__notes__)
    else:
        runtime.step_market('A',bar(0))
        if failure=='stop':
            with pytest.raises(RuntimeError) as caught:runtime.stop()
            assert caught.value is secondary
        else:runtime.stop()
    runtime.stop();runtime.stop()
    assert len(resets)==1 and closed==[1]
    assert runtime.strategy.line[0]==2


def test_all_bindings_release_even_when_one_close_fails():
    closed=[]
    class Backend:
        def __init__(self,key):self.key=key
        def update(self,value):return value
        def close(self):
            closed.append(self.key)
            if self.key=='failing':raise RuntimeError('release failed')
    a=FunctionIndicator('a',lambda close:close,{},('close',),stream_factory=lambda params:Backend('normal'))
    b=FunctionIndicator('b',lambda close:close,{},('close',),stream_factory=lambda params:Backend('failing'))
    class S(Strategy):
        def __init__(self):self.a=a(self.data.close);self.b=b(self.data.close)
    runtime=StrategyRuntime(S,feeds={'A':'bar'});runtime.start();runtime.step_market('A',bar(0))
    with pytest.raises(RuntimeError,match='release failed'):runtime.stop()
    runtime.stop()
    assert sorted(closed)==['failing','normal']
    assert runtime.strategy.a[0]==runtime.strategy.b[0]==1
