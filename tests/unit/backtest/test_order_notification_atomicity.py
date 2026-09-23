"""Notifications cannot cancel an order already resolved by the current native batch."""
import datetime as dt

import pytest

from test_order_lifecycle import RecordingStrategy, _run, _terminal
from tradelearn.backtest.models import Order


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
@pytest.mark.parametrize("trigger", ["expiry", "fill", "fill-peer"])
def test_native_batch_terminal_outcomes_are_atomic(
    match_mode, feed_count, trade_on_close, trigger,
):
    class CancelDuringNotification(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.expiring = self.buy(size=1, price=1, exectype=Order.Limit,
                                         valid=dt.timedelta(days=2))
                self.first_fill = self.buy(size=2, price=9, exectype=Order.Limit)
                self.second_fill = self.buy(size=3, price=9, exectype=Order.Limit)

        def notify_order(self, order):
            super().notify_order(order)
            if trigger == "expiry" and order.status == Order.Expired:
                self.cancel(self.first_fill)
            elif trigger == "fill" and order.status == Order.Completed:
                self.cancel(self.expiring)
            elif trigger == "fill-peer" and order.status == Order.Completed and order is self.first_fill:
                self.cancel(self.second_fill)

    s = _run(CancelDuringNotification, match_mode, feed_count, trade_on_close)
    _terminal(s, s.expiring, Order.Expired)
    _terminal(s, s.first_fill, Order.Completed)
    _terminal(s, s.second_fill, Order.Completed)
    assert s.position.size == 5
    assert s._pending_size.get(s.data, 0) == 0


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
def test_submitted_notification_can_cancel_before_routing(
    match_mode, feed_count, trade_on_close,
):
    class CancelSubmitted(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.order = self.buy(size=2)

        def notify_order(self, order):
            super().notify_order(order)
            if order.status == Order.Submitted:
                self.cancel(order)

    s = _run(CancelSubmitted, match_mode, feed_count, trade_on_close)
    _terminal(s, s.order, Order.Canceled)
    assert s.statuses(s.order) == [Order.Submitted, Order.Canceled]
    assert s.position.size == 0
    assert s._pending_size.get(s.data, 0) == 0


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
def test_parent_canceled_during_submission_cancels_later_bracket_children(
    match_mode, feed_count, trade_on_close,
):
    class CancelParent(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.bracket = self.buy_bracket(size=1, price=9, exectype=Order.Limit,
                                               stopprice=7, limitprice=11)

        def notify_order(self, order):
            super().notify_order(order)
            if order.status == Order.Submitted and order.parent is None:
                self.cancel(order)

    s = _run(CancelParent, match_mode, feed_count, trade_on_close)
    for order in s.bracket:
        _terminal(s, order, Order.Canceled)
        assert s.statuses(order) == [Order.Submitted, Order.Canceled]
    assert s.position.size == 0
    assert s._pending_size.get(s.data, 0) == 0
