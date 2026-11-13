"""Replay revisions retain raw-bar matching and existing BT notification phases."""
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def profile(api,kind,compression,clone,order_kind):
    trace=[]
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def notify_order(self,o):
            trace.append(('order',o.getstatusname(),o.executed.size,o.executed.price,
                          o.executed.comm,len(self),len(self.datas[-1])))
        def next(self):
            d=self.datas[-1]
            trace.append(('next',len(self),len(d),d.datetime.datetime().isoformat(),d.close[0],self.broker.getcash(),self.broker.getvalue()))
            if not self.sent:
                self.sent=True
                if order_kind=='market':self.buy(data=d,size=.25)
                elif order_kind=='limit':self.buy(data=d,size=.25,exectype=api.Order.Limit,price=35)
                else:self.buy(data=d,size=.25,exectype=api.Order.Stop,price=35)
    c=api.Cerebro(stdstats=False,runonce=False)
    c.broker.setcash(10000);c.broker.setcommission(commission=.001)
    source=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(source,name='raw')
    c.replaydata(source,timeframe=api.TimeFrame.Minutes,compression=compression)
    c.addstrategy(Business);c.run()
    return trace


@pytest.mark.parametrize('kind',['dense','sparse','midnight'])
@pytest.mark.parametrize('compression',[2,7])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('order_kind',['market','limit','stop'])
def test_replay_uses_original_bar_execution_quotes_and_bt_order_phases(kind,compression,clone,order_kind):
    assert profile(tl,kind,compression,clone,order_kind)==profile(official,kind,compression,clone,order_kind)
