from __future__ import annotations

import pandas as pd

from tradelearn.backtest.broker import RustBroker
from tradelearn.backtest.engine import _build_equity_from_fills
from tradelearn.backtest.models import Order
from tradelearn.engine import Strategy


class DummyData:
    _name = "AAA"


def test_equity_from_fills_tracks_position_size_per_data() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    fills = pd.DataFrame(
        {
            "datetime": [index[0], index[0]],
            "data": ["AAA", "BBB"],
            "size": [5.0, 2.0],
            "price": [10.0, 20.0],
            "commission": [0.0, 0.0],
        }
    )
    frame = pd.DataFrame(
        {
            ("AAA", "close"): [10.0, 11.0],
            ("BBB", "close"): [20.0, 22.0],
        },
        index=index,
    )

    strategy = Strategy()
    strategy.broker = RustBroker(cash=100.0)

    equity = _build_equity_from_fills(strategy, index, fills, frame)

    assert equity.tolist() == [100.0, 109.0]


def test_completed_rust_partial_fill_clears_pending_order_size() -> None:
    data = DummyData()
    strategy = Strategy()
    broker = RustBroker(cash=100.0)
    strategy.broker = broker
    order = Order(ref=1, data=data, ordtype=Order.Buy, size=10.0)
    broker._orders_by_ref[1] = order
    strategy._pending_size[data] = 10.0

    broker._process_rust_fills_batch(
        strategy,
        ([1], [3.0], [10.0], [0.0], [0.0], [0.0]),
    )

    assert strategy._pending_size[data] == 0.0
    assert strategy.getposition(data).size == 3.0


def test_unfilled_rust_market_order_clears_pending_size() -> None:
    data = DummyData()
    strategy = Strategy()
    broker = RustBroker(cash=100.0)
    strategy.broker = broker
    order = Order(
        ref=1,
        data=data,
        ordtype=Order.Buy,
        size=10.0,
        exectype=Order.Market,
        status=Order.Accepted,
    )
    broker._orders.append(order)
    broker._orders_by_ref[1] = order
    strategy._pending_size[data] = 10.0

    broker._reconcile_unfilled_market_orders(strategy, set())

    assert order.status == Order.Margin
    assert strategy._pending_size[data] == 0.0
