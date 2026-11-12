"""Same input, independent facade syntax, one execution and indicator fact set."""
import pandas as pd
from tradelearn import engine,lite
from tradelearn.indicators import talib


def test_same_input_replay_indicator_orders_account_and_notifications():
    frame=pd.DataFrame(dict(open=[10.,11.,12.,13.,14.],high=[11.,12.,13.,14.,15.],
        low=[9.,10.,11.,12.,13.],close=[10.,11.,12.,13.,14.],volume=[100.]*5),
        index=pd.date_range('2026-01-01',periods=5,tz='UTC'))
    def record(s):
        s.seen.append((len(s),s.data.close[0],s.average[0],s.broker.getcash(),s.broker.getvalue()))
    class E(engine.Strategy):
        params=(('quantity',2),)
        def __init__(self):self.seen=[];self.notifications=[];self.average=talib.SMA(self.data.close,timeperiod=2)
        def next(self):
            record(self)
            if len(self)==2:self.buy(size=self.p.quantity)
            elif len(self)==3:self.close()
        def notify_order(self,o):self.notifications.append((o.getstatusname(),o.isbuy(),o.size,o.executed.size))
    class L(lite.Strategy):
        quantity=2
        def init(self):self.seen=[];self.notifications=[];self.average=talib.SMA(self.data.close,timeperiod=2)
        def next(self):
            record(self)
            if len(self)==2:self.buy(size=self.quantity)
            elif len(self)==3:self.close()
        def notify_order(self,o):self.notifications.append((o.getstatusname(),o.isbuy(),o.size,o.executed.size))
    c=engine.Cerebro();c.broker.setcash(1000);c.adddata(engine.feeds.PandasData(dataname=frame));c.addstrategy(E,quantity=2)
    e=c.run()[0];l=lite.Backtest(frame,L,cash=1000).run().strategy
    assert e.seen==l.seen
    assert e.notifications==l.notifications
    assert e.broker.getvalue()==l.broker.getvalue()==1002
    assert l.orders==()
