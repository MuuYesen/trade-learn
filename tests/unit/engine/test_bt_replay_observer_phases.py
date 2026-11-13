"""Observer phase and revision history use the strategy's logical replay index."""
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def profile(api,kind,clone):
    trace=[]
    class Sample(api.Observer):
        lines=('measured',)
        def __init__(self):self.calls=0
        def record(self,phase):
            self.calls+=1;self.measured[0]=self.data.close[0]+self.calls
            trace.append(('observer',phase,len(self),len(self.data),self.measured[0],self.measured[-1] if len(self)>1 else None))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    class Business(api.Strategy):
        def next(self):trace.append(('strategy',len(self),len(self.data),self.data.close[0]))
    c=api.Cerebro(stdstats=False,runonce=False);c.addobserver(Sample)
    s=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(s,name='raw')
    c.replaydata(s,timeframe=api.TimeFrame.Minutes,compression=2)
    c.addstrategy(Business);c.run();return trace


@pytest.mark.parametrize('kind',['dense','sparse','midnight'])
@pytest.mark.parametrize('clone',[False,True])
def test_replay_observer_uses_bt_revision_phase_and_logical_history(kind,clone):
    assert profile(tl,kind,clone)==profile(official,kind,clone)
