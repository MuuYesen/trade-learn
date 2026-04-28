from __future__ import annotations

import pandas as pd
import pytest

from tradelearn import _rust
from tradelearn.backtest.core.brokers.rust import RustBroker
from tradelearn.backtest.core.models import Order
from tradelearn.compat.backtrader import Cerebro, Strategy


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
