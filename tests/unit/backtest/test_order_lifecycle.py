"""Lifecycle regressions across the real Python facade / Rust bar-loop boundary."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


@pytest.fixture(params=["exact", "smart"])
def match_mode(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=["single", "multi"])
def feed_count(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["next-open", "on-close"])
def trade_on_close(request):
    return request.param


def _bars(*, wide=False):
    # Pending buy limits first become marketable on Jan 4, after expiry/cancel.
    return pd.DataFrame(
        {
            "open": [10., 10., 10., 8., 8., 8.],
            "high": [12.] * 6 if wide else [11., 11., 11., 9., 9., 9.],
            "low": [7.] * 6 if wide else [9.5, 9.5, 9.5, 7., 7., 7.],
            "close": [10., 10., 10., 8., 8., 8.],
            "volume": [1000.] * 6,
        },
        index=pd.date_range("2026-01-01", periods=6, tz="UTC"),
    )


class RecordingStrategy(Strategy):
    def init(self):
        self.bar = 0
        self.events = []
        self.positions = []

    def next(self):
        self.bar += 1
        self.positions.append(self.position.size)
        self.act()

    def notify_order(self, order):
        # Keep object identity and immutable status snapshots: refs may be bound
        # to Rust IDs after Submitted/Accepted callbacks have already happened.
        self.events.append((order, order.status, self.bar))

    def statuses(self, order):
        return [status for event_order, status, _ in self.events if event_order is order]


def _run(strategy, match_mode, feed_count, trade_on_close, *, wide=False):
    cerebro = Cerebro(match_mode=match_mode, trade_on_close=trade_on_close)
    cerebro.broker.setcash(10_000)
    for i in range(feed_count):
        cerebro.adddata(_bars(wide=wide), name=f"asset{i}")
    cerebro.addstrategy(strategy)
    [result] = cerebro.run()
    if feed_count == 2:
        assert result.getposition(result.datas[1]).size == 0
    return result


def _terminal(strategy, order, status):
    assert order.status == status
    terminal = [s for s in strategy.statuses(order)
                if s in (Order.Completed, Order.Canceled, Order.Expired)]
    assert terminal == [status]


def test_active_cancel_never_fills(match_mode, feed_count, trade_on_close):
    class Cancel(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.order = self.buy(size=1, price=9, exectype=Order.Limit)
            elif self.bar == 2:
                self.cancel(self.order)

    s = _run(Cancel, match_mode, feed_count, trade_on_close)
    _terminal(s, s.order, Order.Canceled)
    assert s.position.size == 0
    assert s.positions == [0] * 6
    assert s.order.executed.size == 0


def test_submit_cancel_same_callback_has_no_ghost_fill_or_ref_collision(
    match_mode, feed_count, trade_on_close,
):
    class CancelAndReplace(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.canceled = self.buy(size=7)
                self.cancel(self.canceled)
                self.replacement = self.buy(size=2)
            elif self.bar == 3:
                self.later = self.buy(size=3)

    s = _run(CancelAndReplace, match_mode, feed_count, trade_on_close)
    _terminal(s, s.canceled, Order.Canceled)
    _terminal(s, s.replacement, Order.Completed)
    _terminal(s, s.later, Order.Completed)
    assert len({s.canceled.ref, s.replacement.ref, s.later.ref}) == 3
    assert s.canceled.executed.size == 0
    assert s.replacement.executed.size == 2
    assert s.later.executed.size == 3
    assert s.position.size == 5
    assert max(s.positions) <= 5


@pytest.mark.parametrize("gtc_first", [False, True])
def test_gtc_and_expiring_orders_have_independent_deadlines(
    match_mode, feed_count, trade_on_close, gtc_first,
):
    class MixedDeadlines(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.orders_by_kind = {}
                kinds = ["gtc", "expires"] if gtc_first else ["expires", "gtc"]
                for kind in kinds:
                    self.orders_by_kind[kind] = self.buy(
                        size=1, price=9, exectype=Order.Limit,
                        valid=None if kind == "gtc" else dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
                    )

    s = _run(MixedDeadlines, match_mode, feed_count, trade_on_close)
    _terminal(s, s.orders_by_kind["expires"], Order.Expired)
    _terminal(s, s.orders_by_kind["gtc"], Order.Completed)
    assert s.orders_by_kind["expires"].executed.size == 0
    assert s.position.size == 1


@pytest.mark.parametrize("valid", [
    dt.datetime(2026, 1, 3, tzinfo=dt.timezone.utc),
    dt.date(2026, 1, 3),
    dt.timedelta(days=1),
    86400,
], ids=["datetime", "date", "timedelta", "seconds"])
def test_expiry_is_based_on_creation_and_happens_before_matching(
    match_mode, feed_count, trade_on_close, valid,
):
    class Expires(RecordingStrategy):
        def act(self):
            # Created on Jan 2; all four forms mean a Jan 3 deadline.
            # On Jan 4 the limit becomes marketable for the very first time.
            if self.bar == 2:
                self.order = self.buy(size=1, price=9, exectype=Order.Limit, valid=valid)

    s = _run(Expires, match_mode, feed_count, trade_on_close)
    _terminal(s, s.order, Order.Expired)
    assert s.order.executed.size == 0
    assert s.position.size == 0
    assert s.positions == [0] * 6


def test_bracket_oco_cancels_sibling_before_same_bar_match(
    match_mode, feed_count, trade_on_close,
):
    class Bracket(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.bracket = self.buy_bracket(
                    size=1, exectype=Order.Market, stopprice=9, limitprice=11,
                )

    s = _run(Bracket, match_mode, feed_count, trade_on_close, wide=True)
    parent, stop, limit = s.bracket
    _terminal(s, parent, Order.Completed)
    assert sorted([stop.status, limit.status]) == sorted([Order.Completed, Order.Canceled])
    for order in (stop, limit):
        _terminal(s, order, order.status)
    assert s.position.size == 0
    assert min(s.positions) >= 0
    assert sum(abs(o.executed.size) for o in (stop, limit)) == 1


def test_explicit_non_bracket_oco_pair_only_fills_once(
    match_mode, feed_count, trade_on_close,
):
    class ExplicitOCO(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.entry = self.buy(size=1)
            elif self.bar == 2:
                self.stop_order = self.sell(size=1, price=9, exectype=Order.Stop)
                self.limit = self.sell(size=1, price=11, exectype=Order.Limit, oco=self.stop_order)

    s = _run(ExplicitOCO, match_mode, feed_count, trade_on_close, wide=True)
    _terminal(s, s.entry, Order.Completed)
    assert sorted([s.stop_order.status, s.limit.status]) == sorted([Order.Completed, Order.Canceled])
    for order in (s.stop_order, s.limit):
        _terminal(s, order, order.status)
    assert s.position.size == 0
    assert min(s.positions) >= 0


def test_cancel_parent_cancels_deferred_children(match_mode, feed_count, trade_on_close):
    class CancelBracket(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.bracket = self.buy_bracket(
                    size=1, price=9, exectype=Order.Limit, stopprice=7, limitprice=11,
                )
            elif self.bar == 2:
                self.cancel(self.bracket[0])

    s = _run(CancelBracket, match_mode, feed_count, trade_on_close)
    for order in s.bracket:
        _terminal(s, order, Order.Canceled)
        assert order.executed.size == 0
    assert s.position.size == 0
    assert s.positions == [0] * 6


def test_order_tag_and_info_survive_in_stats(match_mode, feed_count, trade_on_close):
    class Tagged(RecordingStrategy):
        def act(self):
            if self.bar == 1:
                self.order = self.buy(size=1, tag="entry-alpha", info={"signal": "alpha", "rank": 2})

    s = _run(Tagged, match_mode, feed_count, trade_on_close)
    _terminal(s, s.order, Order.Completed)
    assert s.position.size == 1
    row = s.stats.orders.set_index("ref").loc[s.order.ref]
    assert row["tag"] == "entry-alpha"
    assert row["info"]["signal"] == "alpha"
    assert row["info"]["rank"] == 2


def test_public_order_execution_constants_retain_existing_values():
    assert {name: getattr(Order, name) for name in (
        "Market", "Limit", "Stop", "StopLimit", "Close", "StopTrail", "StopTrailLimit",
    )} == {
        "Market": 1, "Limit": 2, "Stop": 3, "StopLimit": 4,
        "Close": 5, "StopTrail": 6, "StopTrailLimit": 7,
    }
