"""Real-order coverage of native and Python-driven Rust runner paths."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tradelearn.backtest import engine as engine_module
from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


class Recorder(Strategy):
    def init(self):
        self.bar = 0
        self.events = []
        self.target = self.datas[-1]

    def next(self):
        self.bar += 1
        self.act()

    def notify_order(self, order):
        self.events.append((order, order.status, self.bar, order.data.datetime[0]))

    def terminal_events(self, order):
        return [status for candidate, status, _, _ in self.events
                if candidate is order and status in (Order.Completed, Order.Canceled, Order.Expired)]


def _frame(prices, *, asynchronous=False, highs=None, lows=None):
    dates = ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-06", "2026-01-07", "2026-01-08"]
    index = pd.to_datetime(dates, utc=True) if asynchronous else pd.date_range("2026-01-01", periods=len(prices), tz="UTC")
    return pd.DataFrame({"open": prices, "high": highs or prices,
                         "low": lows or prices, "close": prices,
                         "volume": [1000.] * len(prices)}, index=index)


def _run(monkeypatch, strategy, runner, on_close, mode, target_frame):
    cerebro = Cerebro(match_mode=mode, trade_on_close=on_close)
    cerebro.broker.setcash(10_000)
    if runner != "single":
        # The target feed has gaps; repeated primary ticks must neither replay
        # its previous bar nor seed its orders using the primary asset's price.
        cerebro.adddata(_frame([1000.] * 8), name="clock")
    if runner == "python-multi":
        monkeypatch.setattr(engine_module, "_build_clocked_multi_data_runner", lambda datas: None)
    cerebro.adddata(target_frame, name="target")
    cerebro.addstrategy(strategy)
    [s] = cerebro.run()
    if runner != "single":
        assert s.getposition(s.datas[0]).size == 0
    return s


@pytest.mark.parametrize("runner", ["single", "native-multi", "python-multi"])
@pytest.mark.parametrize("on_close", [False, True])
@pytest.mark.parametrize("mode", ["exact", "smart"])
@pytest.mark.parametrize("scenario", ["initial-gap", "ratchet"])
def test_trailing_percent_without_price_uses_visible_target_close(
    monkeypatch, runner, on_close, mode, scenario,
):
    class Trailing(Recorder):
        def act(self):
            if self.bar == 1:
                self.entry = self.buy(data=self.target, size=1)
            elif self.bar == 2:
                self.exit_order = self.sell(data=self.target, size=1,
                                            exectype=Order.StopTrail, trailpercent=0.1)

    if scenario == "initial-gap":
        # The seed is the Jan 2 close (100), so the next bar's gap to 80
        # crosses the 90 stop immediately. Seeding from that future bar loses it.
        frame = _frame([100., 100., 80., 80., 80., 80.], asynchronous=runner != "single")
        expected_price = 80.
        expected_date = frame.index[2]
    else:
        # High=120 raises tomorrow's stop to 108, but low=100 must not
        # retrospectively trigger it on the same bar. Next open=105 does.
        frame = _frame([100., 100., 115., 105., 105., 105.],
                       highs=[100., 100., 120., 110., 110., 110.],
                       lows=[100., 100., 100., 100., 100., 100.],
                       asynchronous=runner != "single")
        expected_price = 105.
        expected_date = frame.index[3]
    s = _run(monkeypatch, Trailing, runner, on_close, mode, frame)
    assert s.terminal_events(s.entry) == [Order.Completed]
    assert s.terminal_events(s.exit_order) == [Order.Completed]
    fill_timestamp = next(timestamp for order, status, _, timestamp in s.events
                          if order is s.exit_order and status == Order.Completed)
    assert pd.Timestamp(fill_timestamp) == expected_date
    assert s.exit_order.executed.price == pytest.approx(expected_price)
    assert s.getposition(s.target).size == 0


@pytest.mark.parametrize("on_close", [False, True])
@pytest.mark.parametrize("mode", ["exact", "smart"])
def test_python_multi_fallback_cancellation_and_expiry(monkeypatch, on_close, mode):
    class Lifecycle(Recorder):
        def act(self):
            if self.bar == 1:
                self.immediate = self.buy(data=self.target, size=7)
                self.cancel(self.immediate)
                self.active = self.buy(data=self.target, size=3, price=90, exectype=Order.Limit)
                self.expiring = self.buy(data=self.target, size=5, price=90, exectype=Order.Limit,
                                         valid=dt.datetime(2026, 1, 3, tzinfo=dt.timezone.utc))
                self.gtc = self.buy(data=self.target, size=1, price=90, exectype=Order.Limit)
            elif self.bar == 2:
                self.cancel(self.active)

    frame = _frame([100., 100., 80., 80., 80., 80.], asynchronous=True)
    s = _run(monkeypatch, Lifecycle, "python-multi", on_close, mode, frame)
    for order in (s.immediate, s.active):
        assert order.status == Order.Canceled
        assert order.executed.size == 0
        assert s.terminal_events(order) == [Order.Canceled]
    assert s.expiring.status == Order.Expired
    assert s.expiring.executed.size == 0
    assert s.terminal_events(s.expiring) == [Order.Expired]
    assert s.terminal_events(s.gtc) == [Order.Completed]
    assert s.gtc.executed.size == 1
    assert s.getposition(s.target).size == 1
    assert len({order.ref for order in (s.immediate, s.active, s.expiring, s.gtc)}) == 4
