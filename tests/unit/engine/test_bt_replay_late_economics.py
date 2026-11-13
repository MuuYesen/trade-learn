"""Late replay phases retain actual BT order and indicator/observer semantics."""
import math
import pytest
import backtrader as official
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_late import frame_for


def profile(api,kind,clone,order_kind,coc,indicator_raw=False):
    trace=[]
    def clean(value):return 'NaN' if isinstance(value,float) and math.isnan(value) else value
    class Sample(api.Indicator):
        lines=('sample',)
        def __init__(self):self.calls=0;self.addminperiod(2)
        def record(self,phase):
            self.calls+=1;self.sample[0]=self.data.close[0]+self.calls
            trace.append(('indicator',phase,len(self),len(self.data),self.calls,self.sample[0]))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    class Business(api.Strategy):
        def __init__(self):
            self.count=0;clock=self.datas[0 if indicator_raw else -1];self.custom=Sample(clock);self.average=api.talib.SMA(clock.close,timeperiod=2)
        def notify_order(self,o):trace.append(('order',o.getstatusname(),o.created.dt,o.executed.dt,o.executed.price,o.executed.size,o.executed.comm))
        def record(self,phase):
            self.count+=1;d=self.datas[-1]
            trace.append(('business',phase,len(self),len(d),d.datetime[0],clean(self.custom[0]),clean(self.average[0]),self.broker.getcash(),self.broker.getvalue()))
            if self.count==2:
                kw=dict(data=d,size=.125)
                if order_kind=='limit':kw.update(exectype=api.Order.Limit,price=1000.)
                elif order_kind=='stop':kw.update(exectype=api.Order.Stop,price=35.)
                self.buy(**kw)
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    class Audit(api.Observer):
        lines=('equity',)
        _stclock=True
        def record(self,phase):
            self.equity[0]=self._owner.broker.getvalue();trace.append(('observer',phase,len(self),self.equity[0]))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(12500);c.broker.setcommission(commission=.002);c.broker.set_coc(coc)
    raw=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(raw)
    c.replaydata(raw,timeframe=api.TimeFrame.Minutes,compression=2);c.addstrategy(Business);c.addobserver(Audit);c.run()
    return trace


@pytest.mark.parametrize('kind',['duplicate','late','late_edge','quantized'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('order_kind',['market','limit','stop'])
@pytest.mark.parametrize('coc',[False,True])
def test_late_replay_orders_indicator_revisions_and_observer_account_phase_match_bt(kind,clone,order_kind,coc):
    assert profile(tl,kind,clone,order_kind,coc)==profile(official,kind,clone,order_kind,coc)


@pytest.mark.parametrize('kind',['duplicate','late','late_edge','quantized'])
@pytest.mark.parametrize('order_kind',['market','limit','stop'])
@pytest.mark.parametrize('coc',[False,True])
def test_raw_indicator_beside_late_clone_does_not_double_consume_held_raw_quote(kind,order_kind,coc):
    assert profile(tl,kind,True,order_kind,coc,True)==profile(official,kind,True,order_kind,coc,True)
