from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradelearn import _rust
from tradelearn.backtest import broker as broker_module
from tradelearn.backtest.broker import RustBroker
from tradelearn.backtest.data import DataContainer, RollingBarBuffer
from tradelearn.backtest.engine import _build_bar_advancers, _build_data_advance_plan
from tradelearn.backtest.lines import LineSeries
from tradelearn.backtest.models import Order, Stats
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.compat.backtesting.backtest import Backtest
from tradelearn.compat.backtesting.strategy import (
    BacktestingDataProxy,
    IndicatorProxy,
    PositionProxy,
)
from tradelearn.compat.backtesting.strategy import Strategy as BacktestingStrategy
from tradelearn.compat.backtrader import Cerebro, DataFeed, Strategy
from tradelearn.compat.backtrader import indicators as btind
from tradelearn.core import Fill, OrderRequest


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
    broker.bind_engine(engine)
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
    broker.bind_engine(engine)
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
    broker.bind_engine(engine)

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
    broker.bind_engine(FakeEngine())

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


def test_rust_broker_reuses_empty_order_buffer_result() -> None:
    broker = RustBroker(match_mode="exact")
    broker.bind_engine(object())

    broker.begin_order_buffering()
    first = broker.drain_order_buffer()
    broker.begin_order_buffering()
    second = broker.drain_order_buffer()

    assert first == ()
    assert first is second
    assert broker._order_submit_buffer == []


def test_rust_broker_can_bind_order_refs_in_batch() -> None:
    broker = RustBroker(match_mode="exact")
    broker.bind_engine(object())

    broker.begin_order_buffering()
    first = broker.buy(object(), object(), size=1.0)
    second = broker.sell(object(), object(), size=2.0)

    broker.bind_rust_order_refs([(first.ref, 11), (second.ref, 12)])

    assert first.ref == 11
    assert second.ref == 12
    assert 1 not in broker._orders_by_ref
    assert 2 not in broker._orders_by_ref
    assert broker._orders_by_ref[11] is first
    assert broker._orders_by_ref[12] is second


def test_rust_broker_processes_rust_fills_with_batch_sync() -> None:
    data = object()

    class FakeEngine:
        def get_new_fills(self, start_idx: int):
            assert start_idx == 0
            return [
                (1, "buy", 1.0, 10.0, 0.0, 0.0, 0.0),
                (2, "sell", -1.0, 11.0, 0.0, 0.0, 1.0),
            ]

    class FakeStrategy:
        def __init__(self) -> None:
            self._pending_size = {data: 0.0}
            self.orders = []
            self.trades = []

        def notify_order(self, order) -> None:
            self.orders.append((order.ref, order.status, order.executed.price))

        def notify_trade(self, trade) -> None:
            self.trades.append((trade.size, trade.price, trade.pnl))

    broker = RustBroker(match_mode="exact")
    broker.bind_engine(FakeEngine())
    first = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)
    second = Order(ref=2, data=data, ordtype=Order.Sell, size=1.0)
    broker._orders_by_ref = {1: first, 2: second}

    strategy = FakeStrategy()

    broker.process_fills(strategy, 1)

    assert strategy.orders == [
        (1, Order.Completed, 10.0),
        (2, Order.Completed, 11.0),
    ]
    assert strategy.trades == [(1.0, 10.0, 0.0), (0.0, 11.0, 1.0)]
    assert broker._last_fill_idx == 2


def test_rust_broker_processes_compact_columnar_fill_batch() -> None:
    data = object()

    class FakeEngine:
        def get_new_fills_compact(self, start_idx: int):
            assert start_idx == 0
            return (
                [1, 2],
                [1.0, -1.0],
                [10.0, 11.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            )

    class FakeStrategy:
        def __init__(self) -> None:
            self._pending_size = {data: 0.0}
            self.orders = []
            self.trades = []

        def notify_order(self, order) -> None:
            self.orders.append((order.ref, order.status, order.executed.price))

        def notify_trade(self, trade) -> None:
            self.trades.append((trade.size, trade.price, trade.pnl))

    broker = RustBroker(match_mode="exact")
    broker.bind_engine(FakeEngine())
    first = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)
    second = Order(ref=2, data=data, ordtype=Order.Sell, size=1.0)
    broker._orders_by_ref = {1: first, 2: second}

    strategy = FakeStrategy()

    broker.process_fills(strategy, 1)

    assert strategy.orders == [
        (1, Order.Completed, 10.0),
        (2, Order.Completed, 11.0),
    ]
    assert strategy.trades == [(1.0, 10.0, 0.0), (0.0, 11.0, 1.0)]
    assert broker._last_fill_idx == 2


def test_rust_broker_caches_fills_frame_until_new_fill() -> None:
    data = object()
    broker = RustBroker(match_mode="exact")
    order = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)

    first = broker.fills_frame()
    second = broker.fills_frame()
    assert first is second

    broker._record_fill(order, 1.0, 10.0, 0.0)
    third = broker.fills_frame()

    assert third is not first
    assert third["size"].tolist() == [1.0]


def test_rust_broker_requires_rust_engine_for_fill_processing() -> None:
    broker = RustBroker(match_mode="exact")
    broker._pending_orders = [Order(ref=1, data=object(), ordtype=Order.Buy, size=1.0)]

    try:
        broker.process_fills(object(), 0)
    except RuntimeError as exc:
        assert "RustBroker requires a Rust engine" in str(exc)
    else:
        raise AssertionError("RustBroker should not run Python matching fallback")


def test_rust_broker_skips_fill_object_when_no_fill_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = object()

    class StrategySink:
        def __init__(self) -> None:
            self._pending_size = {data: 0.0}

        def notify_order(self, order) -> None:
            pass

    def fail_namespace(*args, **kwargs):
        raise AssertionError("fill object should be lazy when no analyzer consumes it")

    monkeypatch.setattr(broker_module, "SimpleNamespace", fail_namespace)
    broker = RustBroker(match_mode="exact")
    order = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)
    broker._orders_by_ref = {1: order}

    broker._process_rust_fills_batch(
        StrategySink(),
        ([1], [1.0], [10.0], [0.0], [0.0], [0.0]),
    )


def test_rust_broker_skips_trade_object_when_no_trade_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = object()

    class StrategySink:
        def __init__(self) -> None:
            self._pending_size = {data: 0.0}

        def notify_order(self, order) -> None:
            pass

    def fail_trade(*args, **kwargs):
        raise AssertionError("trade object should be lazy when no callback consumes it")

    monkeypatch.setattr(broker_module, "Trade", fail_trade)
    broker = RustBroker(match_mode="exact")
    order = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)
    broker._orders_by_ref = {1: order}

    broker._process_rust_fills_batch(
        StrategySink(),
        ([1], [1.0], [10.0], [0.0], [0.0], [0.0]),
    )


def test_rust_broker_records_closed_trade_flags_in_fill_ledger() -> None:
    data = object()

    class StrategySink:
        def __init__(self) -> None:
            self._pending_size = {data: 0.0}

        def notify_order(self, order) -> None:
            pass

    broker = RustBroker(match_mode="exact")
    first = Order(ref=1, data=data, ordtype=Order.Buy, size=1.0)
    second = Order(ref=2, data=data, ordtype=Order.Sell, size=1.0)
    broker._orders_by_ref = {1: first, 2: second}

    broker._process_rust_fills_batch(
        StrategySink(),
        ([1, 2], [1.0, -1.0], [10.0, 11.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]),
    )

    fills = broker.fills_frame()

    assert fills["trade_closed"].tolist() == [False, True]
    assert fills["pnl"].tolist() == [0.0, 1.0]


def test_backtesting_run_uses_lazy_stats_without_materializing_fills(monkeypatch) -> None:
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [10.0, 11.0, 12.0, 13.0],
            "Low": [10.0, 11.0, 12.0, 13.0],
            "Close": [10.0, 11.0, 12.0, 13.0],
            "Volume": [100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    class RoundTrip(BacktestingStrategy):
        def init(self) -> None:
            pass

        def next(self) -> None:
            if len(self.data) == 1:
                self.buy(size=1)
            elif len(self.data) == 3:
                self.position().close()

    def fail_fills_frame(self):
        raise AssertionError("fills_frame should be lazy during Backtest.run()")

    monkeypatch.setattr(RustBroker, "fills_frame", fail_fills_frame)

    stats = Backtest(data, RoundTrip, cash=1000, commission=0.0).run()

    assert stats["# Trades"] == 1
    assert stats["Win Rate [%]"] == 100.0


def test_stats_lazy_artifacts_materialize_once() -> None:
    calls = {"fills": 0, "returns": 0}
    fills = pd.DataFrame({"size": [1.0], "price": [10.0]})
    returns = pd.Series([0.0, 0.01], name="returns")

    def fills_factory() -> pd.DataFrame:
        calls["fills"] += 1
        return fills

    def returns_factory() -> pd.Series:
        calls["returns"] += 1
        return returns

    stats = Stats(
        returns=returns_factory,
        equity=lambda: pd.Series([100.0, 101.0], name="equity"),
        trades=lambda: pd.DataFrame(),
        positions=lambda: pd.DataFrame(),
        orders=lambda: pd.DataFrame(),
        summary={"final_value": 101.0},
        analyzers={},
        config={},
        fills=fills_factory,
    )

    assert stats.fills is fills
    assert stats.fills is fills
    assert stats.returns is returns
    assert stats.returns is returns
    assert calls == {"fills": 1, "returns": 1}


def test_backtesting_strategy_uses_public_basic_submit_fast_path() -> None:
    class FakeData:
        _cursor = 0

        def get_array(self, name: str):
            return np.array([10.0])

    class FakeBroker:
        commission_ratio = 0.0

        def __init__(self) -> None:
            self.basic_calls = []

        def getvalue(self) -> float:
            return 1000.0

        def submit_basic(self, *args):
            self.basic_calls.append(args)
            return "order"

        def _submit(self, *args, **kwargs):
            raise AssertionError("backtesting facade should use submit_basic when available")

    strategy = BacktestingStrategy()
    data = FakeData()
    strategy.datas = [data]
    strategy.broker = FakeBroker()
    strategy._setup()

    order = strategy.buy(size=1)

    assert order == "order"
    assert len(strategy.broker.basic_calls) == 1
    assert strategy.broker.basic_calls[0][2] == Order.Buy


def test_backtest_engine_calls_strategy_setup_once() -> None:
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.0, 11.0, 12.0],
            "Low": [10.0, 11.0, 12.0],
            "Close": [10.0, 11.0, 12.0],
            "Volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    class SetupCounter(BacktestingStrategy):
        setup_calls = 0

        def _setup(self) -> None:
            type(self).setup_calls += 1
            super()._setup()

        def init(self) -> None:
            pass

        def next(self) -> None:
            pass

    SetupCounter.setup_calls = 0

    Backtest(data, SetupCounter, cash=1000).run()

    assert SetupCounter.setup_calls == 1


def test_broker_submit_paths_preserve_submitted_and_accepted_notifications() -> None:
    class Owner(CoreStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = []

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    for submit_name in ("_submit", "_submit_basic", "submit_basic"):
        owner = Owner()
        broker = RustBroker(match_mode="exact")
        submit = getattr(broker, submit_name)

        submit(owner, object(), Order.Buy, 1.0, None, Order.Market)

        assert owner.statuses == [Order.Submitted, Order.Accepted]


def test_backtest_order_adapts_to_core_order_request_and_fill() -> None:
    request = OrderRequest(
        symbol="data0",
        side="sell",
        qty=3.0,
        order_type="limit",
        limit_price=8.0,
        tif=Order.DAY,
        client_oid="client-1",
    )
    data = object()

    order = Order.from_request(7, request, data=data)
    fill = order.to_fill(qty=-3.0, price=8.0, commission=0.5, broker_oid="broker-7")

    assert order.ref == 7
    assert order.ordtype == Order.Sell
    assert order.exectype == Order.Limit
    assert order.price == 8.0
    assert order.time_in_force == Order.DAY
    assert order.info["client_oid"] == "client-1"
    assert isinstance(fill, Fill)
    assert fill.symbol == "data0"
    assert fill.qty == -3.0
    assert fill.price == 8.0
    assert fill.commission == 0.5

    stop_limit = Order.from_request(
        8,
        OrderRequest(
            symbol="data0",
            side="buy",
            qty=1.0,
            order_type="stop_limit",
            stop_price=9.0,
            limit_price=9.5,
        ),
        data=data,
    )

    assert stop_limit.exectype == Order.StopLimit
    assert stop_limit.price == 9.0
    assert stop_limit.pricelimit == 9.5


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
            raise AssertionError("run_bar_loop should batch order ref binding")

        def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
            self.bound.extend(bindings)

    broker = BrokerRefSink()
    seen: list[tuple[int, object]] = []

    def on_bar(cursor, fills, cash, size, price):
        seen.append((cursor, fills))
        if cursor == 0:
            return [
                (99, "buy", "market", 1.0, None, None),
                (100, "buy", "limit", 1.0, 10.5, None),
            ]
        return []

    engine.run_bar_loop(broker, on_bar, 0, 2)

    assert broker.bound == [(99, 1), (100, 2)]
    assert seen[0] == (0, None)
    assert seen[1][0] == 1
    order_ids, sizes, prices, comms, slippages, pnls = seen[1][1]
    assert (order_ids[0], sizes[0], prices[0], comms[0], slippages[0], pnls[0]) == (
        1,
        1.0,
        11.0,
        0.0,
        0.0,
        0.0,
    )
    assert engine.get_position() == (1.0, 11.0)


def test_rust_bar_loop_allows_none_when_callback_has_no_orders() -> None:
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
        def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
            raise AssertionError("no order refs should be bound when callbacks return None")

    seen: list[int] = []

    def on_bar(cursor, fills, cash, size, price):
        seen.append(cursor)
        return None

    engine.run_bar_loop(BrokerRefSink(), on_bar, 0, 2)

    assert seen == [0, 1]


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


def test_backtrader_datafeed_binds_ohlcv_lines_as_direct_attrs() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
        },
        index=pd.to_datetime(["2026-01-01"], utc=True),
    )
    feed = DataFeed(data)

    assert feed.__dict__["close"] is feed.lines.close
    assert feed.__dict__["open"] is feed.lines.open


def test_bar_advance_plan_deduplicates_data_and_indicator_attrs() -> None:
    class Advancer:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def _advance(self, cursor: int) -> None:
            self.calls.append(cursor)

    class StrategyWithAttrs:
        def __init__(self, data: Advancer, indicator: Advancer) -> None:
            self.data = data
            self.indicator = indicator

    data = Advancer()
    indicator = Advancer()
    strategy = StrategyWithAttrs(data, indicator)

    plan = _build_bar_advancers(strategy, [data], [indicator])
    for advance in plan:
        advance(3)

    assert data.calls == [3]
    assert indicator.calls == [3]


def test_bar_advance_plan_skips_empty_strategy_line_root() -> None:
    class EmptyLines:
        def __len__(self) -> int:
            return 0

    class StrategyWithEmptyLines:
        def __init__(self) -> None:
            self.lines = EmptyLines()
            self.calls: list[int] = []

        def _advance(self, cursor: int) -> None:
            self.calls.append(cursor)

    strategy = StrategyWithEmptyLines()

    assert _build_bar_advancers(strategy, [], []) == ()
    assert strategy.calls == []


def test_strategy_pre_next_runs_static_bar_advance_plan() -> None:
    calls: list[int] = []
    strategy = CoreStrategy()
    strategy._set_bar_advancers((calls.append,))

    strategy._pre_next(7)

    assert calls == [7]


def test_backtest_engine_calls_strategy_pre_next_hook() -> None:
    class CountingDataFeed:
        def __init__(self) -> None:
            self._datetime = [1, 2]
            self._open = [10.0, 11.0]
            self._high = [10.0, 11.0]
            self._low = [10.0, 11.0]
            self._close = [10.0, 11.0]
            self._volume = [1000.0, 1000.0]
            self.advance_calls: list[int] = []

        def _advance(self, cursor: int) -> None:
            self.advance_calls.append(cursor)

        def buflen(self) -> int:
            return 2

    class PreNextStrategy(Strategy):
        seen: list[int] = []

        def _pre_next(self, cursor: int) -> None:
            type(self).seen.append(cursor)
            super()._pre_next(cursor)

        def next(self) -> None:
            pass

    data = CountingDataFeed()
    PreNextStrategy.seen = []
    cerebro = Cerebro(match_mode="exact")
    cerebro.adddata(data)
    cerebro.addstrategy(PreNextStrategy)

    cerebro.run()

    assert PreNextStrategy.seen == [0, 1]
    assert data.advance_calls == [0, 1]


def test_rust_primary_clock_plan_aligns_secondary_latest_at_or_before() -> None:
    plan = _rust.RustPrimaryClockPlan([1, 2, 3], [[1, 3], [2]])

    assert plan.len() == 3
    assert plan.cursors_at(0) == [0, 0, -1]
    assert plan.cursors_at(1) == [1, 0, 0]
    assert plan.cursors_at(2) == [2, 1, 0]


def test_rust_primary_clock_plan_accepts_numpy_timestamp_arrays() -> None:
    plan = _rust.RustPrimaryClockPlan(
        np.array([1, 2, 3], dtype=np.int64),
        [
            np.array([1, 3], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ],
    )

    assert plan.cursors_at(0) == [0, 0, -1]
    assert plan.cursors_at(1) == [1, 0, 0]
    assert plan.cursors_at(2) == [2, 1, 0]


def test_rust_bar_runner_drives_multi_data_cursor_batches() -> None:
    engine = _rust.RustBacktestEngine(
        [1, 2, 3],
        [10.0, 11.0, 12.0],
        [10.0, 11.0, 12.0],
        [10.0, 11.0, 12.0],
        [10.0, 11.0, 12.0],
        [1000.0, 1000.0, 1000.0],
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
    runner = _rust.RustBarRunner([1, 2, 3], [[1, 3], [2]])
    seen: list[tuple[int, list[int]]] = []

    class BrokerRefSink:
        def bind_rust_order_refs(self, bindings: list[tuple[int, int]]) -> None:
            raise AssertionError("no bindings expected")

    def on_bar(cursor, data_cursors, fills, cash, size, price):
        seen.append((cursor, data_cursors))
        return None

    runner.run(engine, BrokerRefSink(), on_bar, 0, 3)

    assert runner.len() == 3
    assert runner.cursors_at(2) == [2, 1, 0]
    assert seen == [(0, [0, 0, -1]), (1, [1, 0, 0]), (2, [2, 1, 0])]


def test_data_advance_plan_uses_rust_primary_clock_cursors() -> None:
    primary = DataContainer(
        pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        )
    )
    secondary = DataContainer(
        pd.DataFrame(
            {
                "open": [10.0, 30.0],
                "high": [10.0, 30.0],
                "low": [10.0, 30.0],
                "close": [10.0, 30.0],
                "volume": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True),
        )
    )

    plan = _build_data_advance_plan([primary, secondary])

    assert plan.cursors_at(0) == [0, 0]
    assert plan.cursors_at(1) == [1, 0]
    assert plan.cursors_at(2) == [2, 1]


def test_backtest_engine_uses_rust_plan_for_secondary_data_clock() -> None:
    primary = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )
    secondary = pd.DataFrame(
        {
            "open": [100.0, 130.0],
            "high": [100.0, 130.0],
            "low": [100.0, 130.0],
            "close": [100.0, 130.0],
            "volume": [1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True),
    )

    class MultiDataStrategy(Strategy):
        def __init__(self) -> None:
            self.rows: list[tuple[float, float]] = []

        def next(self) -> None:
            self.rows.append((self.data.close[0], self.datas[1].close[0]))

    cerebro = Cerebro(match_mode="exact")
    cerebro.adddata(primary)
    cerebro.adddata(secondary)
    cerebro.addstrategy(MultiDataStrategy)

    [strategy] = cerebro.run()

    assert strategy.rows == [(10.0, 100.0), (11.0, 100.0), (12.0, 130.0)]


def test_shared_bar_buffer_reuses_data_container_arrays() -> None:
    data = DataContainer(
        pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [3.0, 4.0],
                "volume": [5.0, 6.0],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        )
    )
    buffer = data.shared_bar_buffer()

    data._advance(1)

    assert buffer.arrays["close"] is data.get_array("close")
    assert buffer.cursor == 1
    assert buffer.value("close") == 4.0
    assert buffer.value("close", ago=1) == 3.0


def test_rolling_bar_buffer_appends_live_bars_and_keeps_recent_window() -> None:
    buffer = RollingBarBuffer(capacity=2)

    buffer.append(
        {
            "datetime": pd.Timestamp("2026-01-01", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 10.0,
        }
    )
    buffer.append(
        {
            "datetime": pd.Timestamp("2026-01-02", tz="UTC"),
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 20.0,
        }
    )
    buffer.append(
        {
            "datetime": pd.Timestamp("2026-01-03", tz="UTC"),
            "open": 3.0,
            "high": 3.0,
            "low": 3.0,
            "close": 3.0,
            "volume": 30.0,
        }
    )

    assert buffer.cursor == 1
    assert buffer.value("close") == 3.0
    assert buffer.value("close", ago=1) == 2.0
    assert buffer.arrays["close"].tolist() == [2.0, 3.0]


def test_backtrader_datafeed_lines_read_shared_bar_buffer_without_copy() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [3.0, 4.0],
            "volume": [5.0, 6.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    )
    feed = DataFeed(frame)

    feed._advance(1)

    assert feed.close._buffer is feed.shared_bar_buffer()
    assert feed.close[0] == 4.0
    assert feed.close[-1] == 3.0
    assert feed.datetime[0] == pd.Timestamp("2026-01-02", tz="UTC")


def test_line_series_previous_value_before_start_is_nan() -> None:
    line = LineSeries([1.0, 2.0, 3.0])
    line._advance(0)

    assert pd.isna(line[-1])
    assert line[0] == 1.0


def test_line_series_hot_path_indexing_preserves_relative_semantics() -> None:
    line = LineSeries([1.0, 2.0, 3.0, 4.0])

    line._advance(2)

    assert line[0] == 3.0
    assert line[-1] == 2.0
    assert line[-2] == 1.0
    assert pd.isna(line[-3])
    assert line[object()] is line
    assert line[1:3].tolist() == [2.0, 3.0]


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


def test_backtesting_position_proxy_uses_public_current_position_size() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.size_calls = 0

        def current_position_size(self) -> float:
            self.size_calls += 1
            return 4.0

        def _get_rust_state(self):
            raise AssertionError("PositionProxy should not read Rust private state")

    class FakeStrategy:
        def __init__(self) -> None:
            self.broker = FakeBroker()

        def getposition(self):
            raise AssertionError("PositionProxy should use the Rust state getter")

    strategy = FakeStrategy()
    position = PositionProxy(strategy)

    assert bool(position) is True
    assert position.size == 4.0
    assert strategy.broker.size_calls == 2


def test_backtesting_indicator_proxy_relative_indexing() -> None:
    class Feed:
        _cursor = 2

    indicator = IndicatorProxy([10.0, 20.0, 30.0], Feed())

    assert indicator[-1] == 30.0
    assert indicator.current() == 30.0
    assert indicator[-2] == 20.0
    assert indicator.previous() == 20.0
    assert indicator[0] == 10.0

    Feed._cursor = 0
    assert indicator[-1] == 10.0
    assert indicator.current() == 10.0
    with pytest.raises(IndexError):
        indicator[-2]
    with pytest.raises(IndexError):
        indicator.previous()


def test_backtesting_data_proxy_exposes_fixed_line_proxy_before_cursor_starts() -> None:
    class Feed:
        def __init__(self) -> None:
            self._cursor = -1
            self.arrays = {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [4.0, 5.0, 6.0],
                "volume": [1.0, 2.0, 3.0],
            }

        def get_array(self, name: str):
            return self.arrays[name]

    proxy = BacktestingDataProxy(Feed())

    assert isinstance(proxy.close, IndicatorProxy)
    assert proxy.close is proxy.close
    assert pd.Series(proxy.close).tolist() == [4.0, 5.0, 6.0]


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

    first = proxy.close
    second = proxy.close

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

    first = proxy.factor
    second = proxy.factor
    assert isinstance(first, IndicatorProxy)
    assert first is second

    feed.arrays["factor"] = [7.0, 8.0, 9.0]
    assert proxy.factor is not first


def test_backtesting_strategy_init_runs_once() -> None:
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.0, 11.0, 12.0],
            "Low": [10.0, 11.0, 12.0],
            "Close": [10.0, 11.0, 12.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class InitCountingStrategy(BacktestingStrategy):
        init_calls = 0

        def init(self) -> None:
            type(self).init_calls += 1
            assert isinstance(self.data, BacktestingDataProxy)

        def next(self) -> None:
            pass

    InitCountingStrategy.init_calls = 0

    Backtest(data, InitCountingStrategy, cash=1000.0).run()

    assert InitCountingStrategy.init_calls == 1


def test_backtesting_strategy_I_caches_batch_indicator_results() -> None:
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [10.0, 11.0, 12.0, 13.0],
            "Low": [10.0, 11.0, 12.0, 13.0],
            "Close": [10.0, 11.0, 12.0, 13.0],
            "Volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
    )
    calls = {"count": 0}

    def counted_indicator(close, period: int = 2):
        calls["count"] += 1
        return pd.Series(close).rolling(period).mean()

    class CachedIndicatorStrategy(BacktestingStrategy):
        cache_seen = False

        def init(self) -> None:
            self.first = self.I(counted_indicator, self.data.close, period=2)
            self.second = self.I(counted_indicator, self.data.close, period=2)

        def next(self) -> None:
            type(self).cache_seen = self._batch_indicator_cache is not None

    CachedIndicatorStrategy.cache_seen = False
    bt = Backtest(data, CachedIndicatorStrategy, cash=1000.0)
    bt.run()

    assert calls["count"] == 1
    assert CachedIndicatorStrategy.cache_seen is True


def test_backtrader_indicators_use_strategy_batch_cache_without_strategy_i_api() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
    )

    class CachedBtIndicatorStrategy(Strategy):
        cache_seen = False

        def __init__(self) -> None:
            assert not hasattr(type(self), "I")
            self.first = btind.SMA(self.data.close, period=2)
            self.second = btind.SMA(self.data.close, period=2)

        def next(self) -> None:
            type(self).cache_seen = hasattr(self, "_bt_indicator_batch_cache")

    CachedBtIndicatorStrategy.cache_seen = False
    cerebro = Cerebro()
    cerebro.adddata(DataFeed(data))
    cerebro.addstrategy(CachedBtIndicatorStrategy)

    cerebro.run()

    assert CachedBtIndicatorStrategy.cache_seen is True


def test_backtesting_strategy_I_does_not_merge_distinct_lambda_indicators() -> None:
    data = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.0, 11.0, 12.0],
            "Low": [10.0, 11.0, 12.0],
            "Close": [10.0, 11.0, 12.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class LambdaIndicatorStrategy(BacktestingStrategy):
        first_seen = None
        second_seen = None

        def init(self) -> None:
            self.first = self.I(lambda: pd.Series([1.0, 1.0, 1.0]))
            self.second = self.I(lambda: pd.Series([2.0, 2.0, 2.0]))

        def next(self) -> None:
            type(self).first_seen = self.first[-1]
            type(self).second_seen = self.second[-1]

    LambdaIndicatorStrategy.first_seen = None
    LambdaIndicatorStrategy.second_seen = None

    Backtest(data, LambdaIndicatorStrategy, cash=1000.0).run()

    assert LambdaIndicatorStrategy.first_seen == 1.0
    assert LambdaIndicatorStrategy.second_seen == 2.0


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
