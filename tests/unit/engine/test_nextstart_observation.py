"""Every delivered eligible callback is visible to existing next subscribers."""
import pandas as pd
import pytest
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.engine import Strategy


@pytest.mark.parametrize('custom',[False,True])
def test_nextstart_and_next_emit_one_common_business_observation(custom):
    class S(Strategy):
        def __init__(self):self.calls=[]
        def next(self):self.calls.append('next')
    if custom:
        def nextstart(self):self.calls.append('nextstart')
        S.nextstart=nextstart
    runtime=StrategyRuntime(S,feeds={'A':'bar'})
    observed=[]
    runtime.add_observer(lambda event:observed.append(event.name),events=('business.next',),phases=('after',))
    runtime.start()
    for i in range(2):runtime.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=1,high=1,low=1,close=1,volume=1))
    runtime.stop()
    assert runtime.strategy.calls==(['nextstart','next'] if custom else ['next','next'])
    assert observed==['business.next','business.next']
