"""Available funds exclude short-sale proceeds and 1:1 entry-cost collateral."""
import pandas as pd
import pytest

from tradelearn.backtest import engine as engine_module
from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


CASES = [
    ("weighted-short-cost", [[("sell", 3, 0)], [("sell", 2, 0)],
                              [("buy", 1, 0), ("buy", 4, 1)]],
     [10, 10, 20, 15, 15], 0, 1, [Order.Completed] * 4, [-4, 4], 115),
    ("weighted-short-cost-limit", [[("sell", 3, 0)], [("sell", 2, 0)],
                                    [("buy", 1, 0), ("buy", 4.4, 1)]],
     [10, 10, 20, 15, 15], 0, 1, [Order.Completed] * 3 + [Order.Margin], [-4, 0], 155),
    ("closing-commission-cash", [[("buy", 1, 0)], [("sell", 1, 0)]],
     [10, 10, 100, 100, 100], 2, 1, [Order.Completed, Order.Margin], [1, 0], 70),
    # actions per strategy bar: (side, size, target index)
    ("oversized-short", [[("sell", 11, 0)]], [10] * 5, 0, 1,
     [Order.Margin], [0, 0], 100),
    ("repeated-short", [[("sell", 10, 0)], [("sell", 1, 0)]], [10] * 5, 0, 1,
     [Order.Completed, Order.Margin], [-10, 0], 200),
    ("same-batch-short", [[("sell", 6, 0), ("sell", 5, 0)]], [10] * 5, 0, 1,
     [Order.Completed, Order.Margin], [-6, 0], 160),
    ("buy-other-collateral", [[("sell", 5, 0)], [("buy", 6, 1)]], [10] * 5, 0, 1,
     [Order.Completed, Order.Margin], [-5, 0], 150),
    ("partial-cover-releases", [[("sell", 10, 0)], [("buy", 5, 0)], [("sell", 5, 0)]],
     [10] * 5, 0, 1, [Order.Completed] * 3, [-10, 0], 200),
    ("full-cover-releases", [[("sell", 10, 0)], [("buy", 10, 0)], [("buy", 10, 1)]],
     [10] * 5, 0, 1, [Order.Completed] * 3, [0, 10], 0),
    ("long-short-reversal", [[("buy", 5, 0)], [("sell", 15, 0)]], [10] * 5, 0, 1,
     [Order.Completed] * 2, [-10, 0], 200),
    ("oversized-long-short-reversal", [[("buy", 5, 0)], [("sell", 16, 0)]],
     [10] * 5, 0, 1, [Order.Completed, Order.Margin], [5, 0], 50),
    ("short-long-reversal", [[("sell", 5, 0)], [("buy", 15, 0)]], [10] * 5, 0, 1,
     [Order.Completed] * 2, [10, 0], 0),
    ("oversized-short-long-reversal", [[("sell", 5, 0)], [("buy", 16, 0)]],
     [10] * 5, 0, 1, [Order.Completed, Order.Margin], [-5, 0], 150),
    ("short-commission", [[("sell", 10, 0)]], [10] * 5, .01, 1,
     [Order.Margin], [0, 0], 100),
    ("contract-multiplier", [[("sell", 2, 0)]], [10] * 5, 0, 10,
     [Order.Margin], [0, 0], 100),
    ("multiplier-cover", [[("sell", 1, 0)], [("buy", 1, 0)]], [10] * 5, 0, 10,
     [Order.Completed] * 2, [0, 0], 100),
    ("commission-boundary", [[("sell", 9, 0)], [("buy", .9, 1)]], [10] * 5, .01, 1,
     [Order.Completed] * 2, [-9, .9], 180.01),
    ("commission-over-boundary", [[("sell", 9, 0)], [("buy", .91, 1)]], [10] * 5, .01, 1,
     [Order.Completed, Order.Margin], [-9, 0], 189.1),
    ("loss-cover-reduces-risk", [[("sell", 10, 0)], [("buy", 1, 0)],
                                 [("buy", 1, 1), ("buy", 1, 0)]],
     [10, 10, 30, 30, 30], 0, 1,
     [Order.Completed, Order.Completed, Order.Margin, Order.Completed], [-8, 0], 140),
    ("insolvent-cover", [[("sell", 10, 0)], [("buy", 10, 0)]],
     [10, 10, 30, 30, 30], 0, 1, [Order.Completed, Order.Margin], [-10, 0], 200),
]


@pytest.mark.parametrize("runner", ["single", "native-multi", "python-multi"])
@pytest.mark.parametrize("mode", ["exact", "smart"])
@pytest.mark.parametrize("on_close", [False, True])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_margin_available_funds(monkeypatch, runner, mode, on_close, case):
    _, actions, prices, commission, mult, statuses, positions, cash = case
    if runner == "single" and any(target for batch in actions for _, _, target in batch):
        pytest.skip("cross-asset scenario needs multiple feeds")

    class Orders(Strategy):
        def init(self):
            self.orders = []
            self.terminal = []

        def next(self):
            if len(self) <= len(actions):
                for side, size, target in actions[len(self) - 1]:
                    self.orders.append(getattr(self, side)(data=self.datas[target], size=size))

        def notify_order(self, order):
            if order.status in (Order.Completed, Order.Margin):
                self.terminal.append((order, order.status))

    c = Cerebro(match_mode=mode, trade_on_close=on_close)
    if on_close:
        prices = prices[1:] + prices[-1:]
    c.broker.setcash(100)
    c.broker.setcommission(commission=commission, mult=mult)
    dates = pd.date_range("2026-01-01", periods=len(prices), tz="UTC")
    for i in range(1 if runner == "single" else 2):
        p = prices if i == 0 else [10] * len(prices)
        c.adddata(pd.DataFrame(dict(open=p, high=p, low=p, close=p, volume=100), index=dates), name=str(i))
    if runner == "python-multi":
        monkeypatch.setattr(engine_module, "_build_clocked_multi_data_runner", lambda datas: None)
    c.addstrategy(Orders)
    [s] = c.run()
    assert [order.status for order in s.orders] == statuses
    assert [s.getposition(data).size for data in s.datas] == positions[:len(s.datas)]
    assert s.broker.getcash() == pytest.approx(cash)
    assert len(s.terminal) == len(s.orders)
    assert all(s._pending_size.get(data, 0) == pytest.approx(0) for data in s.datas)
