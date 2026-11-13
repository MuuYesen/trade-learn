"""Withdrawing a public cursor does not erase an already observed native quote."""
import math
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


def profile(api,late,unique,kind,coc):
    rows=[];notices=[]
    def number(x):return 'NaN' if isinstance(x,float) and math.isnan(x) else x
    class Business(api.Strategy):
        def __init__(self):self.count=0;self.sent=False
        def record(self,phase):
            self.count+=1;d=self.datas[-1]
            rows.append((phase,len(self),len(self.data0),len(d),number(d.datetime[0]),
                *[number(getattr(d,n)[0]) for n in ['open','high','low','close','volume']],self.broker.getcash(),number(self.broker.getvalue())))
            if not self.sent and self.count>=3 and len(d):
                self.sent=True;kw=dict(data=d,size=.125)
                if kind=='limit':kw.update(exectype=api.Order.Limit,price=105.)
                elif kind=='stop':kw.update(exectype=api.Order.Stop,price=105.)
                self.buy(**kw)
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
        def notify_order(self,o):notices.append((o.getstatusname(),o.created.dt,o.executed.dt,o.executed.price,o.executed.size,o.executed.comm,len(self),len(self.datas[-1])))
    values=[100.,107.,94.,115.,81.,120.]
    times=['09:00','09:04','09:02','09:08','09:09','09:16'] if late else ['09:00','09:01','09:01','09:08','09:09','09:16']
    f=pd.DataFrame(dict(open=values,high=[v+4 for v in values],low=[v-6 for v in values],close=[v+1 for v in values],volume=range(1,7)),index=pd.to_datetime(['2026-01-02 '+x for x in times]))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(12500);c.broker.setcommission(commission=.002);c.broker.set_coc(coc)
    raw=api.feeds.PandasData(dataname=f,timeframe=api.TimeFrame.Minutes);c.adddata(raw,name='raw')
    c.replaydata(raw,timeframe=api.TimeFrame.Minutes,compression=7,adjbartime=True,**({'name':'sample'} if unique else {}))
    c.addstrategy(Business);c.run();return rows,notices


@pytest.mark.parametrize('late',[False,True])
@pytest.mark.parametrize('unique',[False,True])
@pytest.mark.parametrize('kind',['market','limit','stop'])
@pytest.mark.parametrize('coc',[False,True])
def test_known_inprogress_bar_and_quote_remain_available_at_zero_visible_length_for_one_native_broker_turn(late,unique,kind,coc):
    assert profile(tl,late,unique,kind,coc)==profile(official,late,unique,kind,coc)
