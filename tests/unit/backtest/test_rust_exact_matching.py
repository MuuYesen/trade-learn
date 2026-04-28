from __future__ import annotations

import pandas as pd

from tradelearn import _rust
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


def test_smart_broker_passes_order_prices_to_rust_exact_engine() -> None:
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

    class BuyLimit(Strategy):
        def __init__(self) -> None:
            self.filled_prices: list[float] = []

        def init(self) -> None:
            pass

        def next(self) -> None:
            if not self.position and not self.filled_prices:
                self.buy(size=2, price=9.5, exectype=Order.Limit)

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.filled_prices.append(order.executed.price)

    cerebro = Cerebro(match_mode="smart")
    cerebro.broker.setcash(100.0)
    cerebro.adddata(data)
    cerebro.addstrategy(BuyLimit)

    [strategy] = cerebro.run()

    assert strategy.filled_prices == [9.5]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 9.5
