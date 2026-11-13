"""P02 existing replaydata dispatches revisions through the sole shared Runtime."""
import math
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def replay_runtime_profile(api,kind,timeframe,compression,clone=False,with_indicator=True):
    trace=[]
    def number(value):
        value=float(value)
        return None if math.isnan(value) else value
    class Business(api.Strategy):
        def __init__(self):
            self.replayed=self.datas[-1]
            if with_indicator:
                if api is official:self.average=api.indicators.EMA(self.replayed,period=2)
                else:
                    from tradelearn.indicators import talib
                    self.average=talib.EMA(self.replayed.close,timeperiod=2)
        def record(self,phase):
            d=self.replayed
            row=(phase,len(self),len(d),self.datetime.datetime().isoformat(),d.datetime.datetime().isoformat(),
                 d.open[0],d.high[0],d.low[0],d.close[0],d.volume[0],
                 d.close[-1] if len(d)>1 else None,list(d.close.get(size=min(2,len(d)))))
            if with_indicator:
                row+= (number(self.average[0]),number(self.average[-1]) if len(d)>1 else None,
                       [number(x) for x in self.average.get(size=min(2,len(d)))])
            trace.append(row)
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=api.Cerebro(stdstats=False,runonce=False)
    source=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(source,name='raw')
    c.replaydata(source,timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run();return trace


@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('kind',['dense','sparse','midnight','days'])
@pytest.mark.parametrize('timeframe,compression',[(official.TimeFrame.Minutes,2),(official.TimeFrame.Minutes,7),(official.TimeFrame.Days,1),(official.TimeFrame.Days,2),(official.TimeFrame.Weeks,1),(official.TimeFrame.Months,1),(official.TimeFrame.Years,1),(official.TimeFrame.Seconds,7)])
def test_existing_replay_entry_preserves_bt_logical_clock_and_persistent_indicator_revision(kind,timeframe,compression,clone):
    actual=replay_runtime_profile(tl,kind,timeframe,compression,clone)
    expected=replay_runtime_profile(official,kind,timeframe,compression,clone)
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected):
        assert got[:12]==want[:12]  # Exact clocks, logical sizes and OHLCV/history.
        for a,b in zip((*got[12:14],*got[14]),(*want[12:14],*want[14])):
            if b is None:assert a is None
            else:assert a==pytest.approx(b,rel=1e-12,abs=1e-12)

