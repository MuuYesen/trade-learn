"""Small deterministic Backtrader oracle gate for shared lifecycle semantics.

Expiry interpretation and trailing-watermark policies intentionally remain in
TradeLearn-specific tests; this oracle covers next-open stops, cancel, and OCO.
"""
from __future__ import annotations

import pandas as pd
import pytest

bt = pytest.importorskip("backtrader")

from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


def _bars(scenario):
    rows = [[100., 101., 99., 100.]] * 5
    if scenario == "buy-stop-intrabar":
        rows[1] = [100., 112., 99., 110.]
    elif scenario == "buy-stop-gap":
        rows[1] = [115., 116., 114., 115.]
    elif scenario == "sell-stop-intrabar":
        rows[1] = [100., 101., 88., 90.]
    elif scenario == "sell-stop-gap":
        rows[1] = [85., 86., 84., 85.]
    elif scenario == "cancel":
        rows[3] = [80., 85., 75., 80.]
    elif scenario == "bracket":
        rows[2] = [100., 112., 88., 100.]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                         index=pd.date_range("2026-01-01", periods=5))
    frame["volume"] = 1000.
    return frame


def _run(scenario, *, backtrader):
    base = bt.Strategy if backtrader else Strategy
    order_type = bt.Order if backtrader else Order

    class OracleStrategy(base):
        def __init__(self):
            self.bar = 0
            self.named_orders = {}
            self.terminals = []

        def next(self):
            self.bar += 1
            if self.bar == 1:
                if scenario.startswith("buy-stop"):
                    self.named_orders["entry"] = self.buy(size=2, price=110., exectype=order_type.Stop)
                elif scenario.startswith("sell-stop"):
                    self.named_orders["entry"] = self.sell(size=2, price=90., exectype=order_type.Stop)
                elif scenario == "cancel":
                    self.named_orders["entry"] = self.buy(size=2, price=90., exectype=order_type.Limit)
                else:
                    orders = self.buy_bracket(size=2, exectype=order_type.Market,
                                              stopprice=90., limitprice=110.)
                    self.named_orders = dict(zip(("entry", "stop", "limit"), orders))
            elif self.bar == 2 and scenario == "cancel":
                self.cancel(self.named_orders["entry"])

        def notify_order(self, order):
            if order.status in (order.Completed, order.Canceled, order.Expired):
                # TradeLearn exposes an unsigned executed quantity; normalize
                # direction explicitly to compare the economic fill with BT.
                quantity = abs(float(order.executed.size)) * (1 if order.isbuy() else -1)
                self.terminals.append((order.ref, order.getstatusname(),
                                       quantity, float(order.executed.price)))

    if backtrader:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.PandasData(dataname=_bars(scenario)))
    else:
        cerebro = Cerebro(match_mode="exact", trade_on_close=False, stdstats=False)
        cerebro.adddata(_bars(scenario))
    cerebro.broker.setcash(10_000.)
    cerebro.broker.setcommission(commission=0.)
    cerebro.addstrategy(OracleStrategy)
    [s] = cerebro.run()
    names = {order.ref: name for name, order in s.named_orders.items()}
    return {
        # OCO notification order differs between adapters. Sort without
        # deduplicating so missing or repeated terminal events still fail.
        "terminals": sorted((names[ref], status, size, price)
                            for ref, status, size, price in s.terminals),
        "statuses": {name: order.getstatusname() for name, order in s.named_orders.items()},
        "position": float(s.position.size),
        "cash": float(cerebro.broker.getcash()),
    }


@pytest.mark.parametrize("scenario,expected_price,expected_position", [
    ("buy-stop-intrabar", 110., 2.),
    ("buy-stop-gap", 115., 2.),
    ("sell-stop-intrabar", 90., -2.),
    ("sell-stop-gap", 85., -2.),
])
def test_next_open_stop_execution_matches_backtrader(scenario, expected_price, expected_position):
    oracle = _run(scenario, backtrader=True)
    actual = _run(scenario, backtrader=False)
    assert oracle["terminals"] == [("entry", "Completed", expected_position, expected_price)]
    assert actual == oracle


def test_active_cancel_matches_backtrader_without_later_fill():
    oracle = _run("cancel", backtrader=True)
    actual = _run("cancel", backtrader=False)
    assert oracle["terminals"] == [("entry", "Canceled", 0., 0.)]
    assert oracle["position"] == 0
    assert oracle["cash"] == 10_000.
    assert actual == oracle


def test_bracket_both_touches_matches_backtrader_stop_first_oco():
    oracle = _run("bracket", backtrader=True)
    actual = _run("bracket", backtrader=False)
    assert oracle["terminals"] == sorted([
        ("entry", "Completed", 2., 100.),
        ("stop", "Completed", -2., 90.),
        ("limit", "Canceled", 0., 0.),
    ])
    assert oracle["position"] == 0
    assert oracle["cash"] == 9980.
    assert actual == oracle
