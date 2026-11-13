"""Imperative indicators consume every replay revision without rollback of user state."""
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def profile(api,kind,clone,period):
    trace=[]
    class Sample(api.Indicator):
        lines=('sample',)
        def __init__(self):self.addminperiod(period);self.calls=0
        def record(self,phase):
            self.calls+=1
            self.lines.sample[0]=self.data.close[0]+self.calls
            trace.append(('indicator',phase,len(self),len(self.data),self.calls,self.data.close[0]))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    class Business(api.Strategy):
        def __init__(self):self.indicator=Sample(self.datas[-1])
        def record(self,phase):trace.append(('strategy',phase,len(self),len(self.datas[-1]),self.indicator[0]))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=api.Cerebro(stdstats=False,runonce=False)
    s=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(s,name='raw')
    c.replaydata(s,timeframe=api.TimeFrame.Minutes,compression=2)
    c.addstrategy(Business);c.run();return trace


@pytest.mark.parametrize('kind',['dense','sparse','midnight'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('period',[1,2,3])
def test_replay_custom_indicator_phases_follow_bt_every_revision(kind,clone,period):
    assert profile(tl,kind,clone,period)==profile(official,kind,clone,period)
