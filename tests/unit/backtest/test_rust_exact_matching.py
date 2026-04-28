from __future__ import annotations

import pandas as pd
import pytest

from tradelearn import _rust
from tradelearn.backtest.core.brokers.rust import RustBroker
from tradelearn.backtest.core.models import Order
from tradelearn.compat.backtrader.base import LineSeries
from tradelearn.compat.backtrader import Cerebro, Strategy
from tradelearn.compat.backtesting.strategy import BacktestingDataProxy, IndicatorProxy, PositionProxy


def _match(
    side: str,
    order_type: str,
    *,
    limit_price: float | None = None,
    stop_price: float | None = None,
    open_: float = 10.0,
    high: float = 12.0,
    low: float = 8.0,
    close: float = 11.0,
) -> tuple[float, float, float, float] | None:
    return _rust.match_order_fill(
        1,
        "data0",
        side,
        order_type,
        2.0,
        limit_price,
        stop_price,
        0,
        1,
        open_,
        high,
        low,
        close,
        1000.0,
        False,
        0.0,
    )


def test_rust_match_order_fill_uses_exact_bar_rules() -> None:
    assert _match("buy", "limit", limit_price=9.0) == (2.0, 9.0, 0.0, 0.0)
    assert _match("sell", "limit", limit_price=11.0) == (-2.0, 11.0, 0.0, 0.0)
    assert _match("buy", "stop", stop_price=11.5) == (2.0, 10.0, 0.0, 0.0)
    assert _match("sell", "stop", stop_price=8.5) == (-2.0, 10.0, 0.0, 0.0)
    assert _match("buy", "stop_limit", limit_price=9.5, stop_price=11.5) == (
        2.0,
        9.5,
        0.0,
        0.0,
    )


def test_exact_broker_routes_order_prices_to_rust_exact_engine() -> None:
    data = pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class BuyStop(Strategy):
        def __init__(self) -> None:
            self.filled_prices: list[float] = []

        def init(self) -> None:
            pass

        def next(self) -> None:
            if not self.position and not self.filled_prices:
                self.buy(size=2, price=11.5, exectype=Order.Stop)

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.filled_prices.append(order.executed.price)

    cerebro = Cerebro(match_mode="exact")
    cerebro.broker.setcash(100.0)
    cerebro.adddata(data)
    cerebro.addstrategy(BuyStop)

    [strategy] = cerebro.run()

    assert strategy.filled_prices == [10.0]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 10.0


def test_smart_broker_still_routes_to_rust_engine() -> None:
    data = pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class BuyStop(Strategy):
        def __init__(self) -> None:
            self.filled_prices: list[float] = []

        def init(self) -> None:
            pass

        def next(self) -> None:
            if not self.position and not self.filled_prices:
                self.buy(size=2, price=11.5, exectype=Order.Stop)

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.filled_prices.append(order.executed.price)

    cerebro = Cerebro(match_mode="smart")
    cerebro.broker.setcash(100.0)
    cerebro.adddata(data)
    cerebro.addstrategy(BuyStop)

    [strategy] = cerebro.run()

    assert strategy.filled_prices == [11.5]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 11.5


def test_bt_match_mode_is_not_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported match_mode"):
        RustBroker(match_mode="bt")


def test_rust_broker_caches_cash_and_position_within_bar() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.cash_calls = 0
            self.position_calls = 0

        def get_cash(self) -> float:
            self.cash_calls += 1
            return 100.0

        def get_position(self) -> tuple[float, float]:
            self.position_calls += 1
            return 2.0, 10.0

    engine = FakeEngine()
    broker = RustBroker(match_mode="exact")
    broker._engine = engine
    broker._close_prices = [12.0]
    broker._curr_idx = 0

    assert broker.getcash() == 100.0
    assert broker.getcash() == 100.0
    assert broker.getposition().size == 2.0
    assert broker.getposition().price == 10.0
    assert broker.getvalue() == 124.0
    assert broker.getvalue() == 124.0

    assert engine.cash_calls == 1
    assert engine.position_calls == 1


def test_rust_broker_uses_combined_step_snapshot_when_available() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.collect_calls = 0

        def step_open_collect(self, cursor: int, fill_start_idx: int):
            self.collect_calls += 1
            assert cursor == 0
            assert fill_start_idx == 0
            return [], 100.0, 2.0, 10.0

        def step_open(self, cursor: int) -> None:
            raise AssertionError("step_open should be covered by step_open_collect")

        def get_new_fills(self, start_idx: int):
            raise AssertionError("fills should come from step_open_collect")

        def get_cash(self) -> float:
            raise AssertionError("cash should come from step_open_collect")

        def get_position(self) -> tuple[float, float]:
            raise AssertionError("position should come from step_open_collect")

    engine = FakeEngine()
    broker = RustBroker(match_mode="exact")
    broker._engine = engine
    broker._close_prices = [12.0]

    broker.step(0)

    assert broker.getcash() == 100.0
    assert broker.getposition().size == 2.0
    assert broker.getposition().price == 10.0
    assert broker.getvalue() == 124.0
    assert engine.collect_calls == 1


def test_rust_broker_can_buffer_order_submissions_for_rust_driven_callbacks() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.submissions: list[tuple] = []

        def submit_order(self, *args):
            self.submissions.append(args)
            return 42

    engine = FakeEngine()
    broker = RustBroker(match_mode="exact")
    broker._engine = engine

    broker.begin_order_buffering()
    order = broker.buy(object(), object(), size=2.0, price=9.5, exectype=Order.Limit)

    assert order.status == Order.Accepted
    assert order.ref == 1
    assert engine.submissions == []

    broker.flush_order_buffer()

    assert order.ref == 42
    assert broker._orders_by_ref[42] is order
    assert engine.submissions == [("buy", "limit", 2.0, 9.5, None)]


def test_rust_broker_can_drain_buffer_without_reentering_engine() -> None:
    class FakeEngine:
        def submit_order(self, *args):
            raise AssertionError("drain_order_buffer must not call the Rust engine")

    broker = RustBroker(match_mode="exact")
    broker._engine = FakeEngine()

    broker.begin_order_buffering()
    order = broker.sell(object(), object(), size=3.0, price=8.0, exectype=Order.Stop)

    drained = broker.drain_order_buffer()

    assert drained == [(1, "sell", "stop", 3.0, None, 8.0)]
    assert order.ref == 1
    assert broker._orders_by_ref[1] is order

    broker.bind_rust_order_ref(1, 77)

    assert order.ref == 77
    assert 1 not in broker._orders_by_ref
    assert broker._orders_by_ref[77] is order


def test_backtest_engine_buffers_orders_while_strategy_next_runs() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    )

    class BufferAwareStrategy(Strategy):
        def __init__(self) -> None:
            self.saw_buffering = False

        def next(self) -> None:
            if not self.saw_buffering:
                self.saw_buffering = self.broker._buffer_order_submissions
                self.buy(size=1)

    cerebro = Cerebro(match_mode="exact")
    cerebro.adddata(data)
    cerebro.addstrategy(BufferAwareStrategy)

    [strategy] = cerebro.run()

    assert strategy.saw_buffering is True
    assert strategy.position.size == 1.0


def test_rust_bar_loop_submits_drained_orders_after_python_callback() -> None:
    engine = _rust.RustBacktestEngine(
        [1, 2],
        [10.0, 11.0],
        [10.0, 11.0],
        [10.0, 11.0],
        [10.0, 11.0],
        [1000.0, 1000.0],
        100.0,
        0.0,
        False,
        False,
        False,
        0.0,
        0.0,
        False,
        False,
        False,
    )

    class BrokerRefSink:
        def __init__(self) -> None:
            self.bound: list[tuple[int, int]] = []

        def bind_rust_order_ref(self, provisional_ref: int, rust_ref: int) -> None:
            self.bound.append((provisional_ref, rust_ref))

    broker = BrokerRefSink()
    seen: list[tuple[int, list]] = []

    def on_bar(cursor, fills, cash, size, price):
        seen.append((cursor, fills))
        if cursor == 0:
            return [(99, "buy", "market", 1.0, None, None)]
        return []

    engine.run_bar_loop(broker, on_bar, 0, 2)

    assert broker.bound == [(99, 1)]
    assert seen[0] == (0, [])
    assert seen[1][0] == 1
    assert seen[1][1][0][:5] == (1, "buy", 1.0, 11.0, 0.0)
    assert engine.get_position() == (1.0, 11.0)


def test_backtest_engine_does_not_advance_data_twice_from_strategy_attrs() -> None:
    class CountingDataFeed:
        def __init__(self) -> None:
            self.advance_calls: list[int] = []
            self._datetime = [1, 2, 3]
            self._open = [10.0, 11.0, 12.0]
            self._high = [10.0, 11.0, 12.0]
            self._low = [10.0, 11.0, 12.0]
            self._close = [10.0, 11.0, 12.0]
            self._volume = [1000.0, 1000.0, 1000.0]

        def _advance(self, cursor: int) -> None:
            self.advance_calls.append(cursor)

        def buflen(self) -> int:
            return 3

    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    data = CountingDataFeed()
    cerebro = Cerebro(match_mode="exact")
    cerebro.adddata(data)
    cerebro.addstrategy(NoopStrategy)

    cerebro.run()

    assert data.advance_calls == [0, 1, 2]


def test_line_series_previous_value_before_start_is_nan() -> None:
    line = LineSeries([1.0, 2.0, 3.0])
    line._advance(0)

    assert pd.isna(line[-1])
    assert line[0] == 1.0


def test_backtesting_position_proxy_uses_broker_size_fast_path() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.size_calls = 0

        def get_position_size(self) -> float:
            self.size_calls += 1
            return 2.0

        def getposition(self):
            raise AssertionError("PositionProxy should use get_position_size")

    class FakeStrategy:
        def __init__(self) -> None:
            self.broker = FakeBroker()

    strategy = FakeStrategy()
    position = PositionProxy(strategy)

    assert bool(position) is True
    assert position.size == 2.0
    assert strategy.broker.size_calls == 2


def test_backtesting_position_proxy_refreshes_after_broker_swap() -> None:
    class FakeBroker:
        def __init__(self, size: float) -> None:
            self._size = size

        def get_position_size(self) -> float:
            return self._size

    class FakeStrategy:
        def __init__(self) -> None:
            self.broker = FakeBroker(0.0)

    strategy = FakeStrategy()
    position = PositionProxy(strategy)

    assert bool(position) is False
    strategy.broker = FakeBroker(3.0)
    assert bool(position) is True
    assert position.size == 3.0


def test_backtesting_position_proxy_can_use_rust_state_fast_path() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.state_calls = 0

        def _get_rust_state(self):
            self.state_calls += 1
            return 0, 100.0, 4.0, 10.0

    class FakeStrategy:
        def __init__(self) -> None:
            self.broker = FakeBroker()

        def getposition(self):
            raise AssertionError("PositionProxy should use the Rust state getter")

    strategy = FakeStrategy()
    position = PositionProxy(strategy)

    assert bool(position) is True
    assert position.size == 4.0
    assert strategy.broker.state_calls == 2


def test_backtesting_indicator_proxy_relative_indexing() -> None:
    class Feed:
        _cursor = 2

    indicator = IndicatorProxy([10.0, 20.0, 30.0], Feed())

    assert indicator[-1] == 30.0
    assert indicator[-2] == 20.0
    assert indicator[0] == 10.0

    Feed._cursor = 0
    assert indicator[-1] == 10.0
    with pytest.raises(IndexError):
        indicator[-2]


def test_backtesting_data_proxy_reuses_line_proxy_after_cursor_starts() -> None:
    class Feed:
        def __init__(self) -> None:
            self._cursor = 0
            self.arrays = {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [1.0, 2.0, 3.0],
            }

        def get_array(self, name: str):
            return self.arrays[name]

    proxy = BacktestingDataProxy(Feed())

    first = proxy.Close
    second = proxy.Close

    assert isinstance(first, IndicatorProxy)
    assert first is second


def test_backtesting_data_proxy_caches_extra_line_proxy_after_cursor_starts() -> None:
    class Feed:
        def __init__(self) -> None:
            self._cursor = 0
            self.arrays = {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [1.0, 2.0, 3.0],
                "factor": [4.0, 5.0, 6.0],
            }

        def get_array(self, name: str):
            return self.arrays[name]

    feed = Feed()
    proxy = BacktestingDataProxy(feed)

    first = proxy.Factor
    second = proxy.Factor
    assert isinstance(first, IndicatorProxy)
    assert first is second

    feed.arrays["factor"] = [7.0, 8.0, 9.0]
    assert proxy.Factor is not first


def test_cashvalue_notification_only_runs_when_strategy_overrides_hook() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class NoCashHook(Strategy):
        def next(self) -> None:
            pass

    class CashHook(Strategy):
        def __init__(self) -> None:
            self.cash_values: list[tuple[float, float]] = []

        def notify_cashvalue(self, cash: float, value: float) -> None:
            self.cash_values.append((cash, value))

        def next(self) -> None:
            pass

    no_hook = Cerebro(match_mode="exact")
    no_hook.adddata(data)
    no_hook.addstrategy(NoCashHook)
    [no_hook_strategy] = no_hook.run()

    with_hook = Cerebro(match_mode="exact")
    with_hook.adddata(data)
    with_hook.addstrategy(CashHook)
    [with_hook_strategy] = with_hook.run()

    assert "cash_values" not in no_hook_strategy.__dict__
    assert len(with_hook_strategy.cash_values) == 3


def test_smart_matching_prefers_stop_loss_when_exit_orders_are_ambiguous() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 12.0],
            "low": [9.5, 9.5, 8.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class BracketExit(Strategy):
        def __init__(self) -> None:
            self.completed: list[float] = []
            self.bracket_submitted = False

        def init(self) -> None:
            pass

        def next(self) -> None:
            if not self.position and not self.completed:
                self.buy(size=2)
            elif self.position and not self.bracket_submitted:
                self.sell(size=2, price=8.5, exectype=Order.Stop)
                self.sell(size=2, price=11.5, exectype=Order.Limit)
                self.bracket_submitted = True

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.completed.append(order.executed.price)

    cerebro = Cerebro(match_mode="smart")
    cerebro.broker.setcash(100.0)
    cerebro.adddata(data)
    cerebro.addstrategy(BracketExit)

    [strategy] = cerebro.run()

    assert strategy.completed == [10.0, 8.5]
    assert strategy.position.size == 0.0
