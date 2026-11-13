"""BT admits a submission batch before delivering its accepted snapshots."""
import pytest
import pandas as pd
import backtrader as official
from tradelearn import engine as tl


def trace(api,count,multi,kind,cash):
    events=[]
    class Business(api.Strategy):
        def __init__(self):self.ids={};self.submitted=0
        def next(self):
            if len(self)==1:
                for j in range(count):
                    options={} if kind=='market' else dict(exectype=api.Order.Limit,price=50.)
                    order=self.buy(data=self.datas[j if multi else 0],size=.125,**options)
                    self.ids[order.ref]=j
        def notify_order(self,order):
            if order.status==order.Submitted:self.submitted+=1
            events.append((self.ids[order.ref],order.getstatusname(),len(self),self.submitted,
                order.executed.size,order.executed.price,order.executed.comm))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(cash)
    frame=pd.DataFrame({k:[100.,101.,102.,103.] for k in ('open','high','low','close','volume')},index=pd.date_range('2026-02-02',periods=4))
    for _ in range(count if multi else 1):c.adddata(api.feeds.PandasData(dataname=frame.copy()))
    c.addstrategy(Business);c.run();return events


@pytest.mark.parametrize('count,multi',[(1,False),(2,False),(2,True),(3,False)])
@pytest.mark.parametrize('kind',['market','limit'])
@pytest.mark.parametrize('cash',[10000.,20.])
def test_submission_batch_precedes_acceptance_without_reordering_fills(count,multi,kind,cash):
    assert trace(tl,count,multi,kind,cash)==trace(official,count,multi,kind,cash)
