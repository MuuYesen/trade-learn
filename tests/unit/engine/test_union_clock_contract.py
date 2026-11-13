"""Match frozen BT event ordering with real staggered input feeds."""
import pandas as pd
import pytest
from tradelearn import engine as tl


def frame(days):
    return pd.DataFrame({k:[10.+i for i in range(len(days))] for k in ['open','high','low','close','volume']},index=pd.to_datetime(days))


def replay(bt):
    class S(bt.Strategy):
        def __init__(self):self.trace=[];self.clock=self.datetime
        def capture(self,phase):self.trace.append((phase,len(self),self.clock.datetime(0),len(self.datas[0]),len(self.datas[1])))
        def prenext(self):self.capture('pre')
        def nextstart(self):self.capture('first')
        def next(self):self.capture('next')
    c=bt.Cerebro(stdstats=False)
    c.adddata(bt.feeds.PandasData(dataname=frame(['2026-01-01','2026-01-03'])),name='A')
    c.adddata(bt.feeds.PandasData(dataname=frame(['2026-01-02','2026-01-04'])),name='B')
    c.addstrategy(S)
    return c.run()[0]


def test_staggered_batch_clock_delivers_union_and_retains_feed_lengths():
    import backtrader as bt
    expected=replay(bt)
    actual=replay(tl)
    assert actual.trace==expected.trace
    assert len(actual.stats.equity)==4
    assert list(actual.stats.equity.index)==list(pd.date_range('2026-01-01',periods=4))


def test_secondary_market_order_fills_on_its_next_actual_bar_after_primary_ends():
    class S(tl.Strategy):
        def next(self):
            if len(self)==2:self.order=self.buy(data=self.datas[1],size=1)
    c=tl.Cerebro(stdstats=False)
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-01','2026-01-03'])),name='A')
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-02','2026-01-04'])),name='B')
    c.addstrategy(S);s=c.run()[0]
    assert s.order.status==tl.Order.Completed
    assert pd.Timestamp(tl.num2date(s.order.executed.dt),tz='UTC')==pd.Timestamp('2026-01-04',tz='UTC')
    assert s.order.executed.price==11


def test_union_indicator_consumes_only_real_feed_bars_and_clock_has_no_future():
    import numpy as np
    from tradelearn.indicators import talib
    class S(tl.Strategy):
        def __init__(self):
            self.ma=talib.SMA(self.data.close,timeperiod=2);self.values=[]
            assert self.datetime.to_series().empty and len(self.datetime[:])==0
            with pytest.raises(ValueError,match='future'):self.datetime[1]
        def next(self):self.values.append(self.ma[0])
    c=tl.Cerebro(stdstats=False)
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-01','2026-01-03'])),name='A')
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-02','2026-01-04'])),name='B')
    c.addstrategy(S);s=c.run()[0]
    np.testing.assert_allclose(s.values,[10.5,10.5])
    assert len(s.datetime.to_series())==4


def test_target_relative_validity_is_distinct_from_global_submit_clock():
    from datetime import timedelta
    class S(tl.Strategy):
        def next(self):
            if len(self)==2:self.order=self.buy(size=1,price=11,exectype=tl.Order.Limit,valid=timedelta(days=1))
    c=tl.Cerebro(stdstats=False)
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-01','2026-01-03'])),name='A')
    c.adddata(tl.feeds.PandasData(dataname=frame(['2026-01-02','2026-01-04'])),name='B')
    c.addstrategy(S);s=c.run()[0]
    assert pd.Timestamp(s.order.created_ts,unit='s')==pd.Timestamp('2026-01-02')
    assert s.order.status==tl.Order.Expired  # BT timedelta uses the target feed clock, Jan1+1day.


def test_engine_lite_replay_share_async_feed_facts_after_primary_finishes():
    import numpy as np
    from tradelearn import lite
    from tradelearn.indicators import talib
    inputs={'A':frame(['2026-01-01','2026-01-03']),
            'B':frame(['2026-01-02','2026-01-04','2026-01-05'])}
    def setup(strategy):strategy.ma=talib.SMA(strategy.data.close,timeperiod=2);strategy.values=[]
    def record(strategy):
        day=pd.Timestamp(strategy.runtime_context.ts_event_ns,unit='ns',tz='UTC').day
        strategy.values.append((day,strategy.ma[0]))
        return day
    class E(tl.Strategy):
        def __init__(self):setup(self)
        def next(self):
            if record(self)==3:self.buy(data=self.datas[1],size=1)
    class L(lite.Strategy):
        def init(self):setup(self)
        def next(self):
            if record(self)==3:self.buy(ticker='B',size=1)
    c=tl.Cerebro(stdstats=False);c.broker.setcash(10000)
    for name,bars in inputs.items():c.adddata(tl.feeds.PandasData(dataname=bars),name=name)
    c.addstrategy(E);e=c.run()[0]
    result=lite.Backtest(inputs,L,cash=10000).run()
    assert e.values==result.strategy.values==[(3,10.5),(4,10.5),(5,10.5)]
    np.testing.assert_allclose(e.stats.equity,result.equity)
    assert e.stats.summary['final_value']==result.summary['final_value']==10001


def test_batch_subsecond_timestamps_preserve_both_events_and_exact_context():
    bars=frame(['2026-01-01','2026-01-02'])
    bars.index=pd.date_range('2026-01-01',periods=2,freq='500ms')
    class Business(tl.Strategy):
        def __init__(self):self.seen=[]
        def next(self):self.seen.append((len(self),self.runtime_context.ts_event_ns,self.data.close[0]))
    c=tl.Cerebro(stdstats=False);c.adddata(tl.feeds.PandasData(dataname=bars));c.addstrategy(Business)
    s=c.run()[0]
    assert s.seen==[(1,bars.index[0].value,10.),(2,bars.index[1].value,11.)]
