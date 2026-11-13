"""Strategy line minperiod is not a BT per-data eligibility override."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime


def minperiod_trace(api,period,indicator_period=None,when='init',stream=False):
    trace=[]
    class Minimum(api.Indicator):
        lines=('sample',)
        def __init__(self):self.addminperiod(indicator_period)
        def next(self):self.lines.sample[0]=self.data.close[0]
    class Business(api.Strategy):
        def __init__(self):
            if indicator_period:self.minimum=Minimum(self.data)
            if when=='init':assert self.addminperiod(period) is None
        def start(self):
            if when=='start':assert self.addminperiod(period) is None
        def prenext(self):trace.append(('prenext',len(self)))
        def nextstart(self):trace.append(('nextstart',len(self)))
        def next(self):trace.append(('next',len(self)))
    frame=pd.DataFrame({n:[10.,20.,30.,40.,50.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=5))
    if stream:
        runtime=StrategyRuntime(Business,feeds={'A':'bar'})
        try:
            runtime.start()
            for stamp,row in frame.iterrows():runtime.step_market('A',dict(ts=stamp.tz_localize('UTC'),**row.to_dict()))
            runtime.stop()
        finally:runtime.cleanup()
    else:
        c=api.Cerebro(stdstats=False,runonce=False);c.adddata(api.feeds.PandasData(dataname=frame));c.addstrategy(Business);c.run()
    return trace


@pytest.mark.parametrize('period',[1,2,4])
@pytest.mark.parametrize('indicator_period',[None,2,3])
@pytest.mark.parametrize('when',['init','start'])
@pytest.mark.parametrize('stream',[False,True])
def test_strategy_addminperiod_preserves_fixed_bt_data_and_indicator_owned_eligibility(period,indicator_period,when,stream):
    assert minperiod_trace(tl,period,indicator_period,when,stream)==minperiod_trace(official,period,indicator_period,when)
