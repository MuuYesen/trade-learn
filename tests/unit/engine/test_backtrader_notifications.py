from __future__ import annotations

import pandas as pd

import tradelearn.engine as bt


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )


def test_notify_order_receives_backtrader_style_order_helpers() -> None:
    class BuyAndClose(bt.Strategy):
        def __init__(self) -> None:
            self.orders: list[tuple[str, bool, bool, bool, float]] = []

        def next(self) -> None:
            if len(self.orders) == 0:
                self.buy(size=1)
            elif self.position:
                self.close()

        def notify_order(self, order) -> None:
            self.orders.append(
                (
                    order.getstatusname(),
                    order.isbuy(),
                    order.issell(),
                    order.alive(),
                    order.executed.price,
                )
            )

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars()))
    cerebro.addstrategy(BuyAndClose)

    [strategy] = cerebro.run()

    assert bt.Order.Cancelled == bt.Order.Canceled
    assert strategy.orders == [
        ("Submitted", True, False, True, 0.0),
        ("Accepted", True, False, True, 0.0),
        ("Completed", True, False, False, 10.0),
        ("Submitted", False, True, True, 0.0),
        ("Accepted", False, True, True, 0.0),
        ("Completed", False, True, False, 11.0),
    ]


def test_notify_trade_receives_backtrader_style_trade_status() -> None:
    class BuyAndClose(bt.Strategy):
        def __init__(self) -> None:
            self.trades: list[tuple[str, int, bool, bool, float]] = []

        def next(self) -> None:
            if not self.position and len(self.trades) == 0:
                self.buy(size=1)
            elif self.position:
                self.close()

        def notify_trade(self, trade) -> None:
            self.trades.append(
                (
                    trade.getstatusname(),
                    trade.status,
                    trade.isopen,
                    trade.isclosed,
                    trade.pnl,
                )
            )

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars()))
    cerebro.addstrategy(BuyAndClose)

    [strategy] = cerebro.run()

    assert bt.Trade.Open == 1
    assert bt.Trade.Closed == 2
    assert strategy.trades == [
        ("Open", bt.Trade.Open, True, False, 0.0),
        ("Closed", bt.Trade.Closed, False, True, 1.0),
    ]
