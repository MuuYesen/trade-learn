from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.backtest.event_runner import EventRunner
from tradelearn.core.broker_events import BrokerEvent, BrokerEventPump
from tradelearn.engine import Cerebro, Strategy


def bar(symbol='AAPL', close=10):
    return dict(ts=pd.Timestamp('2026-01-01', tz='UTC'), symbol=symbol,
                open=close, high=close, low=close, close=close, volume=1)


class Lifecycle(Strategy):
    def __init__(self):
        super().__init__()
        self.calls = []

    def start(self):
        self.calls.append(('start', self.data._name))

    def next(self):
        self.calls.append(('next', self.data._name, self.data.close[0]))

    def stop(self):
        self.calls.append(('stop',))


def test_runner_lazily_starts_runtime_and_fixes_first_symbol():
    strategy = Lifecycle()
    runner = EventRunner(strategy)
    runner.on_bar(bar())
    assert strategy.calls == [('start', 'AAPL'), ('next', 'AAPL', 10)]
    assert runner.runtime.strategy is strategy
    with pytest.raises((KeyError, ValueError)):
        runner.on_bar(bar('MSFT'))
    runner.stop()
    with pytest.raises(RuntimeError):
        runner.on_bar(bar())
    assert strategy.calls[-1] == ('stop',)


def test_runner_configured_multisymbol_preserves_feed_identity():
    strategy = Lifecycle()
    runner = EventRunner(strategy, feeds={'AAPL': 'bar', 'MSFT': 'bar'})
    runner.on_bar(bar('AAPL'))
    runner.on_bar(bar('MSFT', 20))
    assert [feed._name for feed in strategy.datas] == ['AAPL', 'MSFT']
    assert [feed.close[0] for feed in strategy.datas] == [10, 20]


def test_cerebro_runtime_binds_named_feeds_before_strategy_init():
    class Constructed(Lifecycle):
        def __init__(self):
            super().__init__()
            self.constructed_feed = self.data

    cerebro = Cerebro(mode='paper', event_bars=[bar()])
    frame = pd.DataFrame([dict(open=1, high=1, low=1, close=1, volume=1)],
                         index=pd.to_datetime(['2026-01-01'], utc=True))
    cerebro.adddata(frame, name='AAPL')
    cerebro.addstrategy(Constructed)
    [strategy] = cerebro.run()
    assert strategy.constructed_feed is strategy.data
    assert strategy.calls == [('start', 'AAPL'), ('next', 'AAPL', 10), ('stop',)]


def test_runner_broker_notifications_and_pump_subscribers_each_receive_once():
    class Orders(Lifecycle):
        def __init__(self):
            super().__init__()
            self.orders = []

        def notify_order(self, order):
            self.orders.append(order)

    order = object()
    event = BrokerEvent('fill', order_id=1, payload=order)
    external = []
    pump = BrokerEventPump(lambda: [event])
    pump.on_fill(external.append)
    strategy = Orders()
    runner = EventRunner(strategy, broker_event_pump=pump)
    runner.on_bar(bar())
    assert external == [order]
    assert strategy.orders == [order]


def test_runner_keeps_explicit_rolling_buffer():
    from tradelearn.backtest.data import RollingBarBuffer

    buffer = RollingBarBuffer(4)
    runner = EventRunner(Lifecycle(), buffer=buffer)
    runner.on_bar(bar())
    assert runner.buffer is buffer
    assert buffer.cursor == 0


def test_named_live_feed_polls_after_start_and_stops_on_failure():
    lifecycle = []

    class Failing(Lifecycle):
        def start(self):
            lifecycle.append('start')

        def stop(self):
            lifecycle.append('stop')

    def poller():
        assert lifecycle == ['start']
        raise RuntimeError('source failed')

    cerebro = Cerebro(mode='live', live_poller=poller)
    frame = pd.DataFrame([dict(open=1, high=1, low=1, close=1, volume=1)],
                         index=pd.to_datetime(['2026-01-01'], utc=True))
    cerebro.adddata(frame, name='AAPL')
    cerebro.addstrategy(Failing)
    with pytest.raises(RuntimeError, match='source failed'):
        cerebro.run()
    assert lifecycle == ['start', 'stop']
