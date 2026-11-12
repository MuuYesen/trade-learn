import numpy as np
import pandas as pd
import pytest

import tradelearn.engine as bt
import tradelearn.lite as tl
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.backtest.lines import LineSeries
from tradelearn.backtest.models import Order, Position
from tradelearn.backtest.strategy import Strategy as CoreStrategy


def bars():
    return pd.DataFrame(dict(open=[10.,11.,12.,13.], high=[11.,12.,13.,14.],
        low=[9.,10.,11.,12.], close=[10.,11.,12.,13.], volume=[100.]*4),
        index=pd.date_range('2026-01-01', periods=4, tz='UTC'))


def test_lite_exports_shared_indicator_and_lines():
    assert tl.Indicator is bt.Indicator
    class S(tl.Strategy):
        def init(self):
            assert isinstance(self.data.close, LineSeries)
            self.average = (self.data.close + self.data.close(-1)) / 2
        def next(self):
            assert self.average[0] == (self.data.close[0]+self.data.close[-1])/2
    tl.Backtest(bars(), S).run()


def test_lite_position_is_filled_and_orders_are_active():
    class S(tl.Strategy):
        def init(self): self.states=[]
        def next(self):
            if len(self)==1:
                self.order=self.buy(size=2)
                self.states.append((self.position().size,len(self.orders)))
            if len(self)==2:
                self.states.append((self.position().size,len(self.orders)))
    s=tl.Backtest(bars(),S).run().strategy
    assert s.states == [(0,1),(2,0)]


def test_lite_target_repeated_same_bar_is_idempotent():
    class S(tl.Strategy):
        def next(self):
            if len(self)==1:
                first=self.target_weights({'Asset': .5})
                second=self.target_weights({'Asset': .5})
                assert len(first)==1
                assert second==[]
    tl.Backtest(bars(),S,cash=1000).run()


def test_lite_order_path_uses_common_submit_and_cancel():
    assert tl.Strategy.cancel is CoreStrategy.cancel
    class S(tl.Strategy):
        def init(self): self.events=[]; __import__("tradelearn.core.strategy_events", fromlist=["add_observer"]).add_observer(self, self.observe)
        def observe(self,event): self.events.append(event.name)
        def next(self):
            if len(self)==1:
                order=self.buy(size=1,limit=1)
                self.cancel(order)
                assert self.position().size==0
                assert self.orders==()
        def notify_order(self,order): pass
    s=tl.Backtest(bars(),S).run().strategy
    assert 'intent.submit_order' in s.events


def test_lite_streaming_runtime_uses_same_indicator_and_init():
    class Twice(tl.Indicator):
        lines=('value',)
        def next(self): self.lines.value[0]=self.data.close[0]*2
    class S(tl.Strategy):
        scale=3
        def init(self): self.values=[]; self.ind=Twice(self.data)
        def next(self): self.values.append((self.data.close[0],self.ind[0],self.scale))
    r=StrategyRuntime(S,feeds={'A':'bar'},parameters={'scale':4},capacity=4)
    r.start()
    for ts,row in bars().iterrows(): r.step_market('A',dict(ts=ts,**row))
    r.stop()
    assert r.strategy.values==[(10.,20.,4),(11.,22.,4),(12.,24.,4),(13.,26.,4)]


def test_lite_rejects_retired_vector_I():
    assert not hasattr(tl.Strategy,'I')


def test_lite_class_parameters_cannot_overwrite_framework_methods():
    class S(tl.Strategy):
        threshold=1
    with pytest.raises(AttributeError):tl.Backtest(bars(),S).run(buy=3)


def test_lite_removed_noop_and_duplicate_conveniences():
    for name in ['prepare_data','target_percent','target_equal','history_panel']:
        assert not hasattr(tl.Strategy,name),name


def test_lite_window_keeps_naive_source_clock_without_timezone_guess():
    frame=bars();frame.index=frame.index.tz_localize(None)
    class S(tl.Strategy):
        def next(self):
            snapshot=self.window(2)
            assert snapshot.index.tz is None
            assert snapshot.index[-1]==frame.index[len(self)-1]
    tl.Backtest(frame,S).run()


def test_lite_stream_records_are_bounded_by_retained_capacity():
    class S(tl.Strategy):
        def next(self):self.record(value=self.data.close[0])
    r=StrategyRuntime(S,feeds={'A':'bar'},capacity=2);r.start()
    for i in range(5):r.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'s'),open=i+1,high=i+1,low=i+1,close=i+1,volume=1))
    assert r.strategy._records['value'].tolist()==[4.,5.]
    r.stop()


def test_contingent_bracket_exits_do_not_cancel_entry_target_intent():
    class S(tl.Strategy):
        def next(self):
            if len(self)==1:
                self.entry=self.buy(size=5,sl=8,tp=15)
                assert self.target_weights({'Asset':.05})==[]
    tl.Backtest(bars(),S,cash=1000).run()
