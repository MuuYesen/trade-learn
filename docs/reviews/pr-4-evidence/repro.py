import datetime as dt
import pandas as pd
from tradelearn.engine import Cerebro, Strategy
from tradelearn.backtest.models import Order
from tradelearn.backtest.broker import RustBroker

def bars():
    return pd.DataFrame({'open':[10.,10.,10.,8.,8.], 'high':[11.,11.,11.,9.,9.], 'low':[9.5,9.5,9.5,7.,7.], 'close':[10.,10.,10.,8.,8.], 'volume':[1000.]*5}, index=pd.date_range('2026-01-01', periods=5,tz='UTC'))
class Cancel(Strategy):
    def init(self):
        self.n=0; self.order=None; self.events=[]
    def next(self):
        self.n+=1
        if self.n==1:
            self.order=self.buy(size=1,price=9,exectype=Order.Limit)
        if self.n==2:
            self.cancel(self.order)
    def notify_order(self,order):
        self.events.append(order.getstatusname())
for count in (1,2):
    c=Cerebro(match_mode='exact')
    for i in range(count): c.adddata(bars())
    c.addstrategy(Cancel)
    s=c.run()[0]
    print('CANCEL',count,'feeds','status',s.order.getstatusname(),'position',s.position.size,'events',s.events)

class D: pass
d=D()
b=RustBroker(cash=100)
b._now_datetime=lambda data: dt.datetime(2026,1,3,tzinfo=dt.timezone.utc)
b._orders=[Order(ref=1,data=d,ordtype=Order.Buy,size=1,status=Order.Accepted,valid=None),Order(ref=2,data=d,ordtype=Order.Buy,size=1,status=Order.Accepted,valid=dt.datetime(2026,1,2,tzinfo=dt.timezone.utc))]
b._expire_valid_orders(0)
print('EXPIRY with GTC first',[o.getstatusname() for o in b._orders])

class Expire(Strategy):
    def init(self): self.n=0; self.order=None; self.events=[]
    def next(self):
        self.n+=1
        if self.n==1: self.order=self.buy(size=1,price=9,exectype=Order.Limit,valid=dt.datetime(2026,1,3,tzinfo=dt.timezone.utc))
    def notify_order(self,order): self.events.append(order.getstatusname())
c=Cerebro(match_mode='exact'); c.adddata(bars()); c.addstrategy(Expire)
s=c.run()[0]
print('EXPIRED FILL',s.order.getstatusname(),'position',s.position.size,'events',s.events)

class Bracket(Strategy):
    def init(self): self.n=0; self.orders=None
    def next(self):
        self.n+=1
        if self.n==1: self.orders=self.buy_bracket(size=1,exectype=Order.Market,stopprice=9,limitprice=11)
df=bars(); df['high']=12.; df['low']=8.; df['open']=10.; df['close']=10.
c=Cerebro(match_mode='exact'); c.adddata(df); c.addstrategy(Bracket)
s=c.run()[0]
print('OCO same bar',[o.getstatusname() for o in s.orders],'position',s.position.size)
