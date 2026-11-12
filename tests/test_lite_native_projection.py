"""Real TL broker projection driven by explicit native gateway facts; no venue connection."""
from types import SimpleNamespace
import pandas as pd
import pytest
from tradelearn_runtime.broker import NautilusBroker
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.backtest.models import Order,Position
from tradelearn.lite import Strategy

class Gateway:
    def __init__(self): self.requests=[];self.cancelled=[];self.failure=False
    def lookup_order(self,oid): return None
    def submit(self,request,**kw):
        self.requests.append(request)
        if self.failure: raise TimeoutError('unknown submit')
    def cancel(self,oid): self.cancelled.append(oid)
    def position(self,symbol): return Position()
    def account(self): return dict(cash=1000,equity=1000)


def runtime(strategy=Strategy):
    gateway=Gateway(); broker=NautilusBroker(gateway)
    r=StrategyRuntime(strategy,feeds={'A':'bar'},broker=broker,capacity=4)
    r.start();r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC'),open=10,high=11,low=9,close=10,volume=100))
    return r,gateway,broker


def test_partial_cancel_unknown_and_recovery_stay_active_until_terminal():
    r,g,b=runtime();s=r.strategy
    order=s.buy(size=4,tag='entry');oid=order.info['client_oid']
    assert s.orders==(order,) and s.position().size==0
    b.apply(oid,status='PARTIALLY_FILLED',filled_qty=1,avg_price=10)
    assert s.orders==(order,) and s._pending_size[s.datas[0]]==3
    s.cancel(order)
    assert s.orders==(order,) and g.cancelled==[oid]
    b.apply(oid,status='CANCELED',filled_qty=1,avg_price=10)
    assert s.orders==() and s._pending_size[s.datas[0]]==0
    assert order.info['tag']=='entry'
    g.failure=True
    unknown=s.buy(size=2)
    assert unknown in s.orders and unknown.info['submission_state']=='UNKNOWN'
    with pytest.raises(RuntimeError,match='reconciliation'):s.buy(size=1)
    assert s._pending_size[s.datas[0]]==2
    b.apply(unknown.info['client_oid'],status='REJECTED',filled_qty=0)
    assert s.orders==() and s._pending_size[s.datas[0]]==0
    r.stop()


def test_live_target_weights_repeats_do_not_use_subset_valuation():
    r,g,b=runtime();s=r.strategy
    assert len(s.target_weights({'A':.5}))==1
    assert s.target_weights({'A':.5})==[]
    r.stop()
