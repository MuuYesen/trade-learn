from __future__ import annotations

from tradelearn.backtest.models import Order, Position
from tradelearn.engine import Strategy


class RecordingBroker:
    def __init__(self) -> None:
        self.submissions = []

    def _submit(self, owner, data, side, size, price, exectype, **kwargs):
        self.submissions.append((owner, data, side, size, price, exectype, kwargs))
        return self.submissions[-1]


def test_strategy_buy_sell_share_submit_order_path() -> None:
    data = object()
    strategy = Strategy()
    strategy.data = data
    strategy.broker = RecordingBroker()

    buy_order = strategy.buy(size=2, price=10.0, exectype=Order.Limit, valid="day")
    sell_order = strategy.sell(size=3, price=11.0, exectype=Order.Stop)

    assert buy_order[2:6] == (Order.Buy, 2.0, 10.0, Order.Limit)
    assert buy_order[6] == {"valid": "day"}
    assert sell_order[2:6] == (Order.Sell, 3.0, 11.0, Order.Stop)
    assert strategy._pending_size[data] == -1.0


def test_strategy_submit_order_is_shared_core_order_entrypoint() -> None:
    data = object()
    strategy = Strategy()
    strategy.data = data
    strategy.broker = RecordingBroker()

    submitted = strategy.submit_order(Order.Sell, size=4, price=9.5, exectype=Order.Limit)

    assert submitted[2:6] == (Order.Sell, 4.0, 9.5, Order.Limit)
    assert strategy._pending_size[data] == -4.0


def test_strategy_position_uses_broker_before_fallback_state() -> None:
    data = object()
    broker_position = Position(size=5.0, price=12.0)

    class PositionBroker(RecordingBroker):
        def getposition(self, requested_data):
            assert requested_data is data
            return broker_position

    strategy = Strategy()
    strategy.data = data
    strategy.broker = PositionBroker()
    strategy._positions[data] = Position(size=99.0, price=1.0)

    assert strategy.getposition() is broker_position

    strategy.broker = None
    assert strategy.getposition().size == 99.0


def test_strategy_close_uses_current_position_not_pending_size() -> None:
    data = object()

    class PositionBroker(RecordingBroker):
        def getposition(self, requested_data):
            assert requested_data is data
            return Position(size=1.0, price=10.0)

    strategy = Strategy()
    strategy.data = data
    strategy.broker = PositionBroker()
    strategy._pending_size[data] = 7.0

    strategy.close()

    _owner, _data, side, size, _price, _exectype, _kwargs = strategy.broker.submissions[0]
    assert side == Order.Sell
    assert size == 1.0
