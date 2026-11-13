"""BT event mode revisits held source clocks; default runonce does not."""
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as candidate

@pytest.mark.parametrize('runonce',[True,False])
@pytest.mark.parametrize('reverse',[False,True])
def test_sparse_imperative_indicator_matches_fixed_bt_runonce_profile(runonce,reverse):
    frame=pd.DataFrame({key:[2.,4.,6.] for key in ('open','high','low','close')},
                       index=pd.date_range('2026-01-01',periods=3,tz='UTC')).assign(volume=1.)
    def run(api):
        calls=[]
        class Counter(api.Indicator):
            lines=('value',)
            def __init__(self):self.count=0
            def next(self):
                self.count+=1;self.lines.value[0]=self.count
                calls.append((self.data._name,len(self.data),self.count))
        class Business(api.Strategy):
            def __init__(self):
                self.rows=[];self.counters={}
                for feed in reversed(self.datas) if reverse else self.datas:
                    self.counters[feed._name]=Counter(feed)
            def next(self):
                self.rows.append((len(self.data0),len(self.data1),
                                  self.counters['A'][0],self.counters['B'][0]))
        engine=api.Cerebro(stdstats=False,runonce=runonce)
        engine.adddata(api.feeds.PandasData(dataname=frame),name='A')
        engine.adddata(api.feeds.PandasData(dataname=frame.iloc[[0,2]]),name='B')
        engine.addstrategy(Business)
        result=engine.run()[0]
        # runonce precomputes one complete Indicator at a time; event mode follows
        # registration order per turn. Both public callback traces are retained.
        return result.rows,calls
    expected=run(official);actual=run(candidate)
    assert actual[0]==expected[0]
    if not runonce:assert actual[1]==expected[1]

@pytest.mark.parametrize('runonce',[True,False])
def test_held_source_event_callbacks_do_not_advance_shared_ema_state_twice(runonce):
    frame=pd.DataFrame({key:[2.,4.,6.,8.,10.] for key in ('open','high','low','close')},
                       index=pd.date_range('2026-01-01',periods=5,tz='UTC')).assign(volume=1.)
    def run(api):
        class Business(api.Strategy):
            def __init__(self):
                self.rows=[]
                self.average=(api.indicators.EMA(self.data1,period=2) if api is official
                              else api.talib.EMA(self.data1.close,timeperiod=2))
            def next(self):self.rows.append((len(self.data0),len(self.data1),self.average[0]))
        engine=api.Cerebro(stdstats=False,runonce=runonce)
        engine.adddata(api.feeds.PandasData(dataname=frame),name='A')
        engine.adddata(api.feeds.PandasData(dataname=frame.iloc[[0,2,4]]),name='B')
        engine.addstrategy(Business)
        return engine.run()[0].rows
    assert run(candidate)==run(official)
