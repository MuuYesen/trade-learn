from __future__ import annotations

from tradelearn.backtest.models import Order, Position
from tradelearn.backtest.strategy import Strategy


class DummyData:
    _name = "primary"
    close = [10.0]


class DummyBroker:
    def __init__(self) -> None:
        self.orders = []

    def _submit(self, owner, data, side, size, price=None, exectype=None, **kwargs):
        order = Order(
            ref=len(self.orders) + 1,
            data=data,
            ordtype=side,
            size=float(size),
            price=price,
            pricelimit=kwargs.get("pricelimit"),
            exectype=exectype or Order.Market,
            parent=kwargs.get("parent"),
            oco=kwargs.get("oco"),
            transmit=kwargs.get("transmit", True),
        )
        self.orders.append(order)
        return order

    def cancel(self, order):
        order.status = Order.Canceled

    def getvalue(self, datas=None) -> float:
        if datas is not None:
            return 0.0
        return 1_000.0

    def getposition(self, data=None):
        return Position(size=0.0, price=0.0)


def _strategy() -> Strategy:
    strategy = Strategy()
    strategy.data = DummyData()
    strategy.datas = [strategy.data]
    strategy.broker = DummyBroker()
    return strategy


def test_core_strategy_order_target_helpers_are_shared_runtime_logic() -> None:
    strategy = _strategy()

    strategy.order_target_size(target=3)
    strategy.order_target_value(target=20.0, price=10.0)
    strategy.order_target_percent(target=0.0, price=10.0)

    assert [order.size for order in strategy.broker.orders[:2]] == [3.0, 2.0]
    assert strategy.broker.orders[0].isbuy()
    assert strategy.broker.orders[1].isbuy()


def test_core_strategy_bracket_helpers_preserve_relationships() -> None:
    strategy = _strategy()

    main, stop, limit = strategy.buy_bracket(size=1, stopprice=9.0, limitprice=12.0)

    assert main.parent is None
    assert stop.parent is main
    assert limit.parent is main
    assert limit.oco is stop
    assert stop.exectype == Order.Stop
    assert limit.exectype == Order.Limit
