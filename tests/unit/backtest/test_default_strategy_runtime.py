"""The default/native research paths share the public lifecycle coordinator."""
import pandas as pd
import pytest
from tradelearn.engine import Cerebro, Strategy, StrategyRuntime
from tradelearn.backtest import engine
from tradelearn.backtest.models import Order


def _cleanup_messages(error, recorded_warnings):
    return list(getattr(error, '__notes__', ())) + [str(item.message) for item in recorded_warnings]


@pytest.mark.parametrize('runner', ['single', 'multi', 'close', 'python'])
@pytest.mark.parametrize('cash_notify', [False, True])
def test_default_runner_owns_strategy_lifecycle(runner, cash_notify, monkeypatch):
    if runner == 'python':
        monkeypatch.setattr(engine, '_build_clocked_multi_data_runner', lambda datas: None)
    observed = []
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: lambda event: observed.append((event.name, event.phase)),)
        def init(self):
            self.trace = [('init', self._runtime)]
            self.notifications = []
            self.cash = []
            self.addminperiod(2)
        def start(self): self.trace.append(('start', self._runtime))
        def next(self):
            self.trace.append(('next', self._runtime))
            if len(self.data) == 2: self.buy(size=1)
        def notify_order(self, order): self.notifications.append(('order', self._runtime))
        def notify_trade(self, trade): self.notifications.append(('trade', self._runtime))
        def stop(self): self.trace.append(('stop', self._runtime))
    if cash_notify:
        Probe.notify_cashvalue = lambda self, cash, value: self.cash.append(self._runtime)
    c = Cerebro(trade_on_close=runner == 'close')
    bars = pd.DataFrame(dict(open=[1.,2.,3.], high=[1.,2.,3.], low=[1.,2.,3.], close=[1.,2.,3.], volume=[10.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    c.adddata(bars, name='a')
    if runner in ('multi', 'python'): c.adddata(bars, name='b')
    c.addstrategy(Probe)
    strategy = c.run()[0]
    assert isinstance(strategy._runtime, StrategyRuntime)
    assert [name for name, _ in strategy.trace] == ['init','start','next','next','stop']
    assert all(runtime is strategy._runtime for _, runtime in strategy.trace)
    assert {name for name, _ in strategy.notifications} == {'order', 'trade'}
    assert all(runtime is strategy._runtime for _, runtime in strategy.notifications)
    for callback in ('order', 'trade'):
        assert observed.count(('business.notify_' + callback, 'after')) == sum(
            name == callback for name, _ in strategy.notifications
        )
    assert observed.count(('lifecycle.init', 'after')) == 1
    assert observed.count(('lifecycle.start', 'after')) == 1
    assert observed.count(('lifecycle.stop', 'after')) == 1
    assert len(strategy.cash) == (3 if cash_notify else 0)
    assert all(runtime is strategy._runtime for runtime in strategy.cash)
    assert observed.count(('business.notify_cashvalue', 'after')) == len(strategy.cash)

def test_batch_indicator_is_precomputed_once_and_not_recomputed_per_prefix():
    from tradelearn.engine import Indicator
    from tradelearn.indicators.base import FunctionIndicator
    calls = []
    def mean(values):
        calls.append(len(values))
        return values.rolling(2).mean()
    sma = FunctionIndicator('mean', mean)
    class Mean(Indicator):
        lines = ('value',)
        def __init__(self): self.lines.value = sma(self.data.close)
    class Probe(Strategy):
        def init(self):
            self.mean = Mean(self.data)
            self.seen = []
            self.addminperiod(2)
        def next(self): self.seen.append(self.mean[0])
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.seen == [3., 5.]
    assert calls == [3]


def test_lite_uses_batch_runtime_with_original_I_setup():
    from tradelearn.lite import Backtest, Strategy as LiteStrategy
    class Probe(LiteStrategy):
        def init(self):
            self.mean = self.I(lambda s: pd.Series(s).rolling(2).mean(), self.data.close)
            self.addminperiod(2)
            self.seen = []
        def next(self): self.seen.append(float(self.mean[0]))
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    result = Backtest(bars, Probe).run()
    assert result.strategy._runtime.profile == 'batch'
    assert result.strategy.seen == [3., 5.]

@pytest.mark.parametrize('on_close', [False, True])
def test_warmup_callback_orders_use_the_same_backend_buffer(on_close):
    class Probe(Strategy):
        def init(self): self.addminperiod(3)
        def prenext(self):
            if len(self.data) == 1: self.order = self.buy(size=1)
    c = Cerebro(trade_on_close=on_close)
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.order.status == s.order.Completed
    assert s.order.executed.price == (2. if on_close else 4.)

def test_initialization_broker_notifications_share_runtime_before_start():
    class Probe(Strategy):
        def init(self):
            self.notified = []
            self.order = self.buy(size=1)
        def notify_order(self, order): self.notified.append(order.status)
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.order.Completed in s.notified

def test_stop_can_cancel_orders_and_dispatch_terminal_notification():
    class Probe(Strategy):
        def init(self): self.notified = []
        def next(self):
            if len(self.data) == 1: self.order = self.buy(size=1, price=.1, exectype=Order.Limit)
        def notify_order(self, order): self.notified.append(order.status)
        def stop(self): self.cancel(self.order)
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.order.Canceled in s.notified

@pytest.mark.parametrize('runner', ['single', 'multi', 'close', 'python'])
def test_default_stateful_indicator_nested_and_dynamic_current_output(runner, monkeypatch):
    if runner == 'python':
        monkeypatch.setattr(engine, '_build_clocked_multi_data_runner', lambda datas: None)
    from tradelearn.engine import Indicator
    class Counter(Indicator):
        lines = ('value',)
        def __init__(self): self.calls = 0
        def next(self):
            self.calls += 1
            self.lines.value[0] = self.data.close[0] * self.calls
    class Outer(Indicator):
        lines = ('value',)
        def __init__(self): self.inner = Counter(self.data)
        def next(self): self.lines.value[0] = self.inner[0] + 1
    class Probe(Strategy):
        def init(self):
            self.counter = Counter(self.data)
            self.outer = Outer(self.data)
            self.seen = []
        def next(self):
            dynamic = Counter(self.data)
            self.seen.append((self.counter.calls, self.counter[0], self.counter[-1], self.outer[0], dynamic[0], dynamic[-1]))
    c = Cerebro(trade_on_close=runner == 'close')
    if runner in ('multi', 'python'):
        c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')), name='secondary')
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert [row[0] for row in s.seen] == [1, 2, 3]
    assert [row[1] for row in s.seen] == [2., 8., 18.]
    assert [row[3] for row in s.seen] == [3., 9., 19.]
    assert [row[4] for row in s.seen] == [2., 4., 6.]
    assert pd.isna(s.seen[0][2]) and s.seen[1][2] == 2.
    assert all(pd.isna(row[5]) for row in s.seen)

def test_default_dynamic_function_indicator_uses_visible_prefix():
    from tradelearn.indicators.base import FunctionIndicator
    calls = []
    def mean(values, period=2):
        calls.append(len(values))
        return values.rolling(period).mean()
    sma = FunctionIndicator('dynamic_mean', mean)
    class Probe(Strategy):
        def init(self): self.seen = []
        def next(self):
            line = sma(self.data.close, period=2)
            self.seen.append(line[0])
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert calls == [2, 3]
    assert pd.isna(s.seen[0]) and s.seen[1:] == [3., 5.]

def test_core_strategy_constructor_receives_bound_data_and_broker():
    from tradelearn.backtest.strategy import Strategy as CoreStrategy
    class Probe(CoreStrategy):
        def __init__(self):
            super().__init__()
            self.constructed = (self.data is not None, len(self.datas), self.broker is not None)
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    r = StrategyRuntime(Probe, feeds={'a':'bar'}, broker=object())
    assert s.constructed == r.strategy.constructed == (True, 1, True)

def test_engine_constructor_user_args_and_multiple_feeds_remain_bound():
    class Probe(Strategy):
        def __init__(self, label):
            super().__init__()
            self.constructed = (label, self.data0 is self.datas[0], self.data1 is self.datas[1], self.broker is not None)
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    c = Cerebro()
    c.adddata(bars, name='a')
    c.adddata(bars, name='b')
    c.addstrategy(Probe, 'user-label')
    s = c.run()[0]
    r = StrategyRuntime(Probe, strategy_args=('user-label',), feeds={'a':'bar','b':'bar'}, broker=object())
    assert s.constructed == r.strategy.constructed == ('user-label', True, True, True)

def test_batch_function_depending_on_stateful_output_remains_incremental():
    from tradelearn.engine import Indicator
    from tradelearn.indicators.base import FunctionIndicator
    combine = FunctionIndicator('combine', lambda price, state: price + state)
    class Counter(Indicator):
        lines = ('value',)
        def next(self): self.lines.value[0] = len(self.data)
    class Probe(Strategy):
        def init(self):
            self.counter = Counter(self.data)
            self.combined = combine(self.data.close, self.counter)
            self.seen = []
        def next(self): self.seen.append(self.combined[0])
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.seen == [3., 6., 9.]

def test_default_business_lines_expose_only_current_prefix():
    class Probe(Strategy):
        def init(self): self.seen = []
        def next(self):
            line = self.data.close
            self.seen.append((line.to_series().tolist(), list(line[:]), list(line.get(size=9))))
            with pytest.raises(ValueError): line[1]
            with pytest.raises(ValueError): line.get(ago=1)
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    expected = [[2.], [2.,4.], [2.,4.,6.]]
    assert s.seen == [(values, values, values) for values in expected]

def test_default_nested_indicator_output_alias_tracks_child_once():
    from tradelearn.engine import Indicator
    class Counter(Indicator):
        lines = ('value',)
        def __init__(self): self.calls = 0
        def next(self):
            self.calls += 1
            self.lines.value[0] = self.calls
    class Alias(Indicator):
        lines = ('value',)
        def __init__(self):
            self.inner = Counter(self.data)
            self.lines.value = self.inner
    class Probe(Strategy):
        def init(self): self.alias, self.seen = Alias(self.data), []
        def next(self): self.seen.append((self.alias[0], self.alias.inner.calls))
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    assert s.seen == [(1.,1),(2.,2),(3.,3)]

def test_default_stateful_indicator_prenext_and_warmup_match_stream():
    from tradelearn.engine import Indicator
    class Warm(Indicator):
        lines = ('value',)
        def __init__(self): self.addminperiod(2)
        def prenext(self): self.lines.value[0] = -1.
        def next(self): self.lines.value[0] = self.data.close[0] * 2
    class Probe(Strategy):
        def init(self): self.warm, self.seen = Warm(self.data), []
        def prenext(self): self.seen.append(('pre', self.warm[0]))
        def next(self): self.seen.append(('next', self.warm[0]))
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    r = StrategyRuntime(Probe, feeds={'a':'bar'})
    r.start()
    for i, value in enumerate([2.,4.,6.]):
        r.step_market('a',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'d'),open=value,high=value,low=value,close=value,volume=1.))
    assert s.seen == r.strategy.seen == [('pre',-1.),('next',8.),('next',12.)]

def test_compiled_function_warmup_uses_same_rule_as_stream_without_manual_period():
    from tradelearn.indicators.base import FunctionIndicator
    mean = FunctionIndicator('warm_mean', lambda values, period=2: values.rolling(period).mean())
    class Probe(Strategy):
        def init(self): self.mean, self.seen = mean(self.data.close), []
        def next(self): self.seen.append(self.mean[0])
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    r = StrategyRuntime(Probe, feeds={'a':'bar'})
    r.start()
    for i, value in enumerate([2.,4.,6.]):
        r.step_market('a',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'d'),open=value,high=value,low=value,close=value,volume=1.))
    assert s.seen == r.strategy.seen == [3.,5.]

def test_sparse_secondary_warmup_counts_received_bars_not_primary_cursor():
    from tradelearn.engine import Indicator
    class Warm(Indicator):
        lines = ('value',)
        def __init__(self): self.addminperiod(2)
        def next(self): self.lines.value[0] = self.data.close[0] * 2
    class Probe(Strategy):
        def init(self): self.warm, self.seen = Warm(self.data1), []
        def next(self): self.seen.append((len(self.data0), self.warm[0]))
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    c = Cerebro()
    c.adddata(bars,name='a')
    c.adddata(bars.iloc[[0,2]],name='b')
    c.addstrategy(Probe)
    s = c.run()[0]
    r = StrategyRuntime(Probe, feeds={'a':'bar','b':'bar'})
    r.start()
    for i,value in enumerate([2.,4.,6.]):
        bar = dict(ts=bars.index[i],open=value,high=value,low=value,close=value,volume=1.)
        r.step_market('a',bar)
        if i != 1: r.step_market('b',bar)
    assert s.seen == r.strategy.seen == [(3,12.)]

def test_strategy_setup_runs_once_after_construction_in_both_profiles():
    class Probe(Strategy):
        def __init__(self):
            super().__init__()
            self.trace = ['constructed']
        def _setup(self): self.trace.append('setup')
        def init(self): self.trace.append('init')
        def start(self): self.trace.append('start')
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC')))
    c.addstrategy(Probe)
    s = c.run()[0]
    r = StrategyRuntime(Probe, feeds={'a':'bar'})
    r.start()
    assert s.trace == r.strategy.trace == ['constructed','setup','init','start']

def test_batch_context_retains_actual_active_source_bars_without_invented_fields():
    class Probe(Strategy):
        def init(self): self.contexts = []
        def next(self): self.contexts.append(self.runtime_context)
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    c = Cerebro()
    c.adddata(bars,name='a')
    c.adddata(bars.iloc[[0,2]],name='b')
    c.addstrategy(Probe)
    s = c.run()[0]
    first, second, third = s.contexts
    assert first.event['active_symbols'] == ('a','b')
    assert second.event['symbol'] == 'a' and second.event['close'] == 4.
    assert third.event['active_symbols'] == ('a','b')
    assert dict(first.event['bars'][1][1]) == dict(symbol='b',ts=bars.index[0],open=2.,high=2.,low=2.,close=2.,volume=1.)
    assert first.ts_event_ns == bars.index[0].value
    assert [context.sequence for context in s.contexts] == [1,2,3]
    with pytest.raises(TypeError): first.event['bars'][0][1]['close'] = 99
    assert 'observed_time' not in first.event and 'event_id' not in first.event

def test_explicit_market_groups_match_default_clock_trace_and_preserve_sparse_feed():
    from tradelearn.engine import Indicator
    from tradelearn.indicators.base import FunctionIndicator
    from tradelearn.core.strategy_events import ObserverSpec
    mean = FunctionIndicator('group_mean', lambda values, period=2: values.rolling(period).mean())
    class Counter(Indicator):
        lines = ('value',)
        def __init__(self): self.calls = 0
        def next(self):
            self.calls += 1
            self.lines.value[0] = self.calls
    def intent_observer(strategy):
        return ObserverSpec(lambda event: strategy.intents.append((event.details['side'],event.details['data']['_name'],event.details['size'],event.details['price'],event.details['exectype'])), events=frozenset({'intent.submit_order'}), phases=('before',))
    class Probe(Strategy):
        strategy_observer_factories = (intent_observer,)
        def init(self):
            self.a, self.b = Counter(self.data0), Counter(self.data1)
            self.mean, self.trace, self.intents = mean(self.data0.close), [], []
        def record(self, phase):
            context = self.runtime_context
            event = context.event
            active = event.get('active_symbols', (event.get('symbol'),))
            clean = lambda value: None if pd.isna(value) else float(value)
            self.trace.append((phase,context.sequence,context.symbol,tuple(active),len(self.data0),len(self.data1),clean(self.a[0]),clean(self.b[0]),clean(self.mean[0])))
        def prenext(self): self.record('pre')
        def next(self):
            self.record('next')
            if len(self.data0) == 2: self.buy(size=1, price=.1, exectype=Order.Limit)
    class IntentBroker:
        def _submit(self, owner, data, side, size, price, exectype, **kwargs):
            return Order(ref=1,data=data,ordtype=side,size=size,price=price,exectype=exectype)
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01', periods=3, tz='UTC'))
    c = Cerebro()
    c.adddata(bars,name='a')
    c.adddata(bars.iloc[[0,2]],name='b')
    c.addstrategy(Probe)
    default = c.run()[0]
    grouped = StrategyRuntime(Probe,feeds={'a':'bar','b':'bar'},broker=IntentBroker())
    grouped.start()
    ungrouped = StrategyRuntime(Probe,feeds={'a':'bar','b':'bar'},broker=IntentBroker())
    ungrouped.start()
    for i,value in enumerate([2.,4.,6.]):
        bar = dict(ts=bars.index[i],open=value,high=value,low=value,close=value,volume=1.)
        group = [('a',bar)] + ([('b',bar)] if i != 1 else [])
        grouped.step_market_batch(group)
        for symbol,event in group: ungrouped.step_market(symbol,event)
    assert default.trace == grouped.strategy.trace
    assert default.intents == grouped.strategy.intents == [(Order.Buy,'a',1.,.1,Order.Limit)]
    assert [row[0] for row in default.trace] == ['pre','next','next']
    assert len(ungrouped.strategy.trace) == 5
    assert ungrouped.strategy.trace != grouped.strategy.trace
    assert len(grouped.feeds['a']) == 3 and len(grouped.feeds['b']) == 2
    assert len(ungrouped.feeds['a']) == 3 and len(ungrouped.feeds['b']) == 2


def test_market_group_rejects_mixed_times_duplicate_symbols_and_partial_updates():
    runtime = StrategyRuntime(Strategy,feeds={'a':'bar','b':'bar'})
    runtime.start()
    bar = dict(ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.)
    for events in ([], [('a',bar),('a',bar)], [('a',bar),('b',dict(bar,ts=bar['ts']+pd.Timedelta(1,'s')))], [('a',bar),('b',dict(bar,volume=-1.))]):
        with pytest.raises(ValueError): runtime.step_market_batch(events)
        assert len(runtime.feeds['a']) == len(runtime.feeds['b']) == 0

@pytest.mark.parametrize('runner', ['single','multi','close','python'])
@pytest.mark.parametrize('stop_fails', [False,True])
def test_default_failure_stops_once_preserves_original_error_and_does_not_build_stats(runner, stop_fails, monkeypatch, recwarn):
    from tradelearn.core.strategy_events import ObserverSpec
    import traceback
    primary, cleanup_error = LookupError('primary business error'), RuntimeError('stop cleanup error')
    trace = []
    def observer(event):
        if event.name == 'lifecycle.closed': trace.append('closed')
        else: raise ValueError('observer failure cannot replace business error')
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(observer,events=frozenset({'business.next','lifecycle.closed'})),)
        def start(self): trace.append('start')
        def next(self):
            trace.append('next')
            raise primary
        def stop(self):
            trace.append('stop')
            if stop_fails: raise cleanup_error
    if runner == 'python': monkeypatch.setattr(engine,'_build_clocked_multi_data_runner',lambda datas:None)
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3), index=pd.date_range('2026-01-01',periods=3,tz='UTC'))
    c = Cerebro(trade_on_close=runner == 'close')
    c.adddata(bars,name='a')
    if runner in ('multi','python'): c.adddata(bars,name='b')
    c.addstrategy(Probe)
    with pytest.warns(RuntimeWarning,match='observer'):
        with pytest.raises(LookupError) as raised: c.run()
    assert raised.value is primary
    assert any(frame.name == 'next' for frame in traceback.extract_tb(raised.value.__traceback__))
    assert trace == ['start','next','stop','closed']
    assert getattr(c,'stats',None) is None
    if stop_fails: assert any('stop cleanup error' in note for note in _cleanup_messages(primary, recwarn))

@pytest.mark.parametrize('phase', ['setup','init','start','notify'])
def test_partial_start_and_broker_callback_failure_share_runtime_cleanup(phase, recwarn):
    primary, trace = LookupError(phase), []
    class Probe(Strategy):
        def _setup(self):
            if phase == 'setup': raise primary
        def init(self):
            if phase == 'init': raise primary
        def start(self):
            if phase == 'start': raise primary
        def notify_order(self, order): raise primary
        def stop(self):
            trace.append('stop')
            raise RuntimeError('secondary cleanup failure')
    runtime = StrategyRuntime(Probe,feeds={'a':'bar'})
    with pytest.raises(LookupError) as raised:
        runtime.start()
        runtime.step_broker(object())
    runtime.stop()
    assert raised.value is primary
    assert trace == ['stop']
    assert any('secondary cleanup failure' in note for note in _cleanup_messages(primary, recwarn))


def test_normal_stop_failure_is_visible_and_closed_event_still_emitted():
    from tradelearn.core.strategy_events import ObserverSpec
    failure, closed = RuntimeError('normal stop error'), []
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(closed.append,events=frozenset({'lifecycle.closed'})),)
        def stop(self): raise failure
    runtime = StrategyRuntime(Probe,feeds={'a':'bar'})
    runtime.start()
    with pytest.raises(RuntimeError) as raised: runtime.stop()
    runtime.stop()
    assert raised.value is failure and len(closed) == 1

def test_grouped_dynamic_secondary_indicator_sees_current_bar_once():
    from tradelearn.engine import Indicator
    from tradelearn.core.contracts import StreamBar
    class Current(Indicator):
        lines = ('value',)
        def next(self): self.lines.value[0] = self.data.close[0]
    class Probe(Strategy):
        def init(self): self.values = []
        def next(self):
            dynamic = Current(self.data1)
            self.values.append((dynamic[0],len(dynamic.lines.value.to_series())))
    r = StrategyRuntime(Probe,feeds={'a':'bar','b':'bar'})
    r.start()
    ts = pd.Timestamp('2026-01-01',tz='UTC')
    r.step_market_batch([StreamBar(ts,'a',2.,2.,2.,2.,1.),StreamBar(ts,'b',10.,10.,10.,10.,1.)])
    assert r.strategy.values == [(10.,1)]

@pytest.mark.parametrize('phase', ['setup','init','start','notify_cashvalue'])
def test_default_partial_lifecycle_errors_cleanup_without_masking(phase, recwarn):
    primary, trace = LookupError(phase), []
    class Probe(Strategy):
        def _setup(self):
            if phase == 'setup': raise primary
        def init(self):
            if phase == 'init': raise primary
        def start(self):
            if phase == 'start': raise primary
        def notify_cashvalue(self, cash, value):
            if phase == 'notify_cashvalue': raise primary
        def stop(self):
            trace.append('stop')
            raise RuntimeError('default cleanup error')
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3),index=pd.date_range('2026-01-01',periods=3,tz='UTC')))
    c.addstrategy(Probe)
    with pytest.raises(LookupError) as raised: c.run()
    assert raised.value is primary and trace == ['stop']
    assert any('default cleanup error' in note for note in _cleanup_messages(primary, recwarn))
    assert getattr(c,'stats',None) is None


def test_component_failure_uses_the_same_runtime_cleanup():
    primary, trace = LookupError('component failed'), []
    class Probe(Strategy):
        def stop(self): trace.append('stop')
    class Analyzer:
        is_streaming = True
        def on_bar(self, bar): raise primary
    c = Cerebro()
    c.adddata(pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3),index=pd.date_range('2026-01-01',periods=3,tz='UTC')))
    c.addstrategy(Probe)
    c.addanalyzer(Analyzer)
    with pytest.raises(LookupError) as raised: c.run()
    assert raised.value is primary and trace == ['stop']


@pytest.mark.parametrize('fails', [False,True])
def test_closing_closed_and_failed_runtime_reject_new_orders_before_pending_or_intent(fails):
    from tradelearn.core.strategy_events import ObserverSpec
    submitted, intents = [], []
    primary = LookupError('failed strategy')
    class Broker:
        def _submit(self, *args, **kwargs): submitted.append(args)
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(intents.append,events=frozenset({'intent.submit_order'})),)
        def next(self):
            if fails: raise primary
        def stop(self):
            assert not self._runtime.accepts_new_orders
            with pytest.raises(RuntimeError,match='new orders'): self.buy(size=1)
    runtime = StrategyRuntime(Probe,feeds={'a':'bar'},broker=Broker())
    runtime.start()
    if fails:
        with pytest.raises(LookupError): runtime.step_market('a',dict(ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.))
    else: runtime.stop()
    with pytest.raises(RuntimeError,match='new orders'): runtime.strategy.buy(size=1)
    assert not runtime.accepts_new_orders
    assert submitted == intents == []
    assert runtime.strategy._pending_size == {}

@pytest.mark.parametrize('failure_phase', ['source','poller','decode'])
def test_event_driver_primary_error_survives_stop_and_observer_failures(failure_phase, recwarn):
    from tradelearn.core.strategy_events import ObserverSpec
    primary, trace = LookupError(failure_phase), []
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(lambda event: (_ for _ in ()).throw(ValueError('observer cleanup failure')),events=frozenset({'lifecycle.closed'})),)
        def start(self): trace.append('start')
        def stop(self):
            trace.append('stop')
            raise RuntimeError('driver stop cleanup failure')
    def source():
        raise primary
        yield
    def poller(): raise primary
    class BadBar:
        def keys(self): return ('ts',)
        def __getitem__(self,key): raise primary
    c = Cerebro(mode='live',live_poller=poller) if failure_phase == 'poller' else Cerebro(mode='paper',event_bars=source() if failure_phase == 'source' else [BadBar()])
    c.adddata(pd.DataFrame(dict(open=[2.],high=[2.],low=[2.],close=[2.],volume=[1.]),index=pd.date_range('2026-01-01',periods=1,tz='UTC')),name='a')
    c.addstrategy(Probe)
    with pytest.warns(RuntimeWarning,match='observer'):
        with pytest.raises(LookupError) as raised: c.run()
    assert raised.value is primary and trace == ['start','stop']
    assert any('driver stop cleanup failure' in note for note in _cleanup_messages(primary, recwarn))

@pytest.mark.parametrize('runner', ['single','multi','close','python'])
def test_default_runstop_from_warmup_stops_before_another_callback(runner,monkeypatch):
    trace = []
    class Probe(Strategy):
        def init(self): self.addminperiod(3)
        def prenext(self):
            trace.append('pre')
            self.cerebro.runstop()
        def next(self): trace.append('next')
        def stop(self): trace.append('stop')
    if runner == 'python': monkeypatch.setattr(engine,'_build_clocked_multi_data_runner',lambda datas:None)
    bars = pd.DataFrame(dict(open=[2.,4.,6.],high=[2.,4.,6.],low=[2.,4.,6.],close=[2.,4.,6.],volume=[1.]*3),index=pd.date_range('2026-01-01',periods=3,tz='UTC'))
    c = Cerebro(trade_on_close=runner == 'close')
    c.adddata(bars,name='a')
    if runner in ('multi','python'): c.adddata(bars,name='b')
    c.addstrategy(Probe)
    c.run()
    assert trace == ['pre','stop']


def test_ungrouped_arrivals_and_explicit_groups_keep_distinct_market_counts():
    class Probe(Strategy):
        def init(self): self.windows = []
        def next(self): self.windows.append((len(self.data0),len(self.data1)))
    grouped = StrategyRuntime(Probe,feeds={'a':'bar','b':'bar'})
    single = StrategyRuntime(Probe,feeds={'a':'bar','b':'bar'})
    grouped.start()
    single.start()
    for i,value in enumerate([2.,4.,6.]):
        event = dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(i,'d'),open=value,high=value,low=value,close=value,volume=1.)
        group = [('a',event)] + ([('b',event)] if i != 1 else [])
        grouped.step_market_batch(group)
        for symbol,bar in group: single.step_market(symbol,bar)
    assert grouped.strategy.windows == [(1,1),(2,1),(3,2)]
    assert single.strategy.windows == [(1,0),(1,1),(2,1),(3,1),(3,2)]

@pytest.mark.parametrize('phase',['on_start','start','on_broker_event','on_end','stop'])
def test_event_component_failures_share_cleanup_even_before_strategy_start(phase):
    from tradelearn.core.strategy_events import ObserverSpec
    from tradelearn.core.broker_events import BrokerEvent
    primary, stopped, closed = LookupError(phase), [], []
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(closed.append,events=frozenset({'lifecycle.closed'})),)
        def stop(self): stopped.append('stop')
    class Analyzer:
        def check(self,name):
            if name == phase: raise primary
        def on_start(self): self.check('on_start')
        def start(self): self.check('start')
        def on_broker_event(self,event): self.check('on_broker_event')
        def on_end(self,stats): self.check('on_end')
        def stop(self): self.check('stop')
        def get_analysis(self): return {}
    bar = dict(symbol='a',ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.)
    c = Cerebro(mode='paper',event_bars=[bar],broker_event_poller=lambda:[BrokerEvent('status',order_id=1)])
    c.adddata(pd.DataFrame(dict(open=[2.],high=[2.],low=[2.],close=[2.],volume=[1.]),index=pd.date_range('2026-01-01',periods=1,tz='UTC')),name='a')
    c.addstrategy(Probe)
    c.addanalyzer(Analyzer)
    with pytest.raises(LookupError) as raised: c.run()
    assert raised.value is primary and stopped == ['stop'] and len(closed) == 1


def test_event_components_use_runtime_context_and_observer_cannot_invoke_them():
    from tradelearn.core.strategy_events import ObserverSpec
    from tradelearn.core.broker_events import BrokerEvent
    from tradelearn.engine.base import _G
    calls, forbidden = [], []
    class Probe(Strategy):
        strategy_observer_factories = (lambda strategy: ObserverSpec(lambda event: strategy._runtime.invoke_component(lambda: forbidden.append('ran')),events=frozenset({'business.next'})),)
    class Analyzer:
        def record(self,name): calls.append((name,_G.current_strategy is self.strategy,_G.current_data is self.strategy.data,self.strategy._runtime))
        def on_start(self): self.record('on_start')
        def start(self): self.record('start')
        def on_broker_event(self,event): self.record('on_broker_event')
        def on_end(self,stats): self.record('on_end')
        def stop(self): self.record('stop')
        def get_analysis(self): return {}
    bar = dict(symbol='a',ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.)
    c = Cerebro(mode='paper',event_bars=[bar],broker_event_poller=lambda:[BrokerEvent('status',order_id=1)])
    c.adddata(pd.DataFrame(dict(open=[2.],high=[2.],low=[2.],close=[2.],volume=[1.]),index=pd.date_range('2026-01-01',periods=1,tz='UTC')),name='a')
    c.addstrategy(Probe)
    c.addanalyzer(Analyzer)
    with pytest.warns(RuntimeWarning,match='observer'): strategy = c.run()[0]
    assert [name for name,*_ in calls] == ['on_start','start','on_broker_event','on_end','stop']
    assert all(strategy_bound and data_bound and runtime is strategy._runtime for _,strategy_bound,data_bound,runtime in calls)
    assert forbidden == [] and strategy.observation_failures().total == 1
    assert strategy._runtime.invoke_component(lambda *,value:value,value=3) == 3

@pytest.mark.parametrize('primary_kind',['no_note','bad_note','bad_note_lookup','normal'])
@pytest.mark.parametrize('bad_str',[False,True])
def test_cleanup_diagnostics_never_mask_primary_on_python310_or_broken_handlers(primary_kind,bad_str):
    import traceback
    import warnings
    class NoNote(LookupError): add_note = None
    class BadNote(LookupError):
        def add_note(self,note): raise RuntimeError('broken note handler')
    class BadLookup(LookupError):
        def __getattribute__(self,name):
            if name == 'add_note': raise RuntimeError('broken capability lookup')
            return super().__getattribute__(name)
    class BadString(RuntimeError):
        def __str__(self): raise RuntimeError('broken cleanup error rendering')
    primary = {'no_note':NoNote,'bad_note':BadNote,'bad_note_lookup':BadLookup,'normal':LookupError}[primary_kind]('primary failure')
    cleanup = BadString() if bad_str else RuntimeError('cleanup failure')
    trace = []
    class Probe(Strategy):
        def next(self): raise primary
        def stop(self):
            trace.append('stop')
            raise cleanup
    runtime = StrategyRuntime(Probe,feeds={'a':'bar'})
    runtime.start()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        with pytest.raises(LookupError) as raised:
            runtime.step_market('a',dict(ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.))
    assert raised.value is primary and trace == ['stop']
    assert any(frame.name == 'next' for frame in traceback.extract_tb(primary.__traceback__))
    diagnostics = list(getattr(primary,'__notes__',())) + [str(item.message) for item in recorded]
    assert any('cleanup' in text for text in diagnostics)
    if bad_str: assert any('unprintable' in text for text in diagnostics)


def test_cleanup_diagnostic_warning_failure_cannot_replace_primary(monkeypatch):
    import warnings
    primary = LookupError('primary')
    primary.add_note = None
    class Probe(Strategy):
        def next(self): raise primary
        def stop(self): raise RuntimeError('cleanup failure')
    runtime = StrategyRuntime(Probe,feeds={'a':'bar'})
    runtime.start()
    def broken_warning(*args,**kwargs): raise RuntimeError('diagnostic handler failure')
    monkeypatch.setattr(warnings,'warn',broken_warning)
    with pytest.raises(LookupError) as raised:
        runtime.step_market('a',dict(ts=pd.Timestamp('2026-01-01',tz='UTC'),open=2.,high=2.,low=2.,close=2.,volume=1.))
    assert raised.value is primary


def test_runtime_source_remains_python310_syntax_compatible():
    import ast
    from pathlib import Path
    import tradelearn.backtest.event_runtime as module
    ast.parse(Path(module.__file__).read_text(),feature_version=(3,10))
