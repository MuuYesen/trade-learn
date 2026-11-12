"""Actual paired fixed-BT calls, including callback order and closed cost basis."""
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


def lifecycle_calls(api,side):
    calls=[]
    class Business(api.Strategy):
        params=(('stake',2),)
        def __init__(self):
            calls.append(('init',self.p.stake,self.data is self.data0,self.data is self.datas[0]))
        def start(self):calls.append(('start',len(self),self.broker.getcash()))
        def prenext(self):calls.append(('prenext',len(self)))
        def nextstart(self):calls.append(('nextstart',len(self)));self.next()
        def next(self):
            calls.append(('next',len(self),len(self.data),len(self.data.close),
                self.data.close[0],self.position.size,self.position.price,
                self.getpositionbyname('A').size))
            if len(self)==1:getattr(self,side)(size=self.p.stake)
            elif len(self)==3:self.close()
        def notify_order(self,order):
            calls.append(('order',len(self),order.getstatusname(),order.isbuy(),
                order.size,order.executed.size,order.executed.price))
        def notify_trade(self,trade):
            calls.append(('trade',len(self),trade.isopen,trade.isclosed,
                trade.size,trade.price,trade.pnl,trade.pnlcomm))
        def stop(self):calls.append(('stop',len(self),self.broker.getcash(),self.broker.getvalue()))
    frame=pd.DataFrame({k:[10.,11.,12.,13.,14.] for k in ['open','high','low','close']},
        index=pd.date_range('2026-01-01',periods=5));frame['volume']=1000
    cerebro=api.Cerebro(stdstats=False);cerebro.broker.setcash(1000)
    cerebro.adddata(api.feeds.PandasData(dataname=frame),name='A')
    cerebro.addstrategy(Business,stake=3);cerebro.run()
    return calls


@pytest.mark.parametrize('side',['buy','sell'])
def test_real_bt_strategy_params_lines_orders_trade_notifications_and_lifecycle(side):
    assert lifecycle_calls(tl,side)==lifecycle_calls(official,side)
