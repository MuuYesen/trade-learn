"""Orders created by lifecycle callbacks must wait for actual native matching."""
import datetime as dt

import pytest

from test_order_lifecycle import RecordingStrategy, _run, _terminal
from tradelearn.backtest.models import Order


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
def test_deferred_market_child_waits_for_parent(match_mode, feed_count, trade_on_close):
    class Deferred(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.parent_order = self.buy(size=1, price=9, exectype=Order.Limit)
                self.child = self.sell(size=1, parent=self.parent_order)

    s = _run(Deferred, match_mode, feed_count, trade_on_close)
    _terminal(s, s.parent_order, Order.Completed)
    _terminal(s, s.child, Order.Completed)
    assert Order.Margin not in s.statuses(s.child)
    assert s.position.size == 0
    assert s._pending_size.get(s.data, 0) == 0


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
def test_expiry_callback_can_submit_market_replacement(match_mode, feed_count, trade_on_close):
    class Replace(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.original = self.buy(size=1, price=9, exectype=Order.Limit,
                                         valid=dt.timedelta(days=1))

        def notify_order(self, order):
            super().notify_order(order)
            if order.status == Order.Expired:
                self.replacement = self.buy(size=2)

    s = _run(Replace, match_mode, feed_count, trade_on_close)
    _terminal(s, s.original, Order.Expired)
    _terminal(s, s.replacement, Order.Completed)
    assert Order.Margin not in s.statuses(s.replacement)
    assert s.position.size == 2
    assert s._pending_size.get(s.data, 0) == 0


@pytest.mark.parametrize("match_mode", ["exact", "smart"])
@pytest.mark.parametrize("feed_count", [1, 2])
@pytest.mark.parametrize("trade_on_close", [False, True])
def test_native_cash_rejection_is_terminal(match_mode, feed_count, trade_on_close):
    class InsufficientCash(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.order = self.buy(size=100_000)

    s = _run(InsufficientCash, match_mode, feed_count, trade_on_close)
    assert s.order.status == Order.Margin
    assert s.statuses(s.order).count(Order.Margin) == 1
    assert s.position.size == 0
    assert s._pending_size.get(s.data, 0) == 0
