from __future__ import annotations

import pandas as pd

from tradelearn.backtest import Analyzer, Cerebro, Strategy


def synthetic_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )


def test_small_golden_smoke_covers_trades_pnl_and_equity_chain() -> None:
    class BuyThenClose(Strategy):
        def next(self) -> None:
            if self.data.close[0] == 10.5:
                self.buy(size=2)
            elif self.data.close[0] == 12.5:
                self.close()

    class SmokeAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.trades = []
            self.equity = []

        def on_trade(self, trade) -> None:
            self.trades.append(
                {
                    "size": trade.size,
                    "price": trade.price,
                    "pnl": trade.pnl,
                    "isclosed": trade.isclosed,
                }
            )

        def on_bar(self, bar) -> None:
            position = self.strategy.position
            cash = self.strategy.broker.getcash()
            self.equity.append(cash + position.size * bar.close)

        def get_analysis(self) -> dict[str, object]:
            return {"trades": self.trades, "equity": self.equity}

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(synthetic_bars(), name="synthetic")
    cerebro.addstrategy(BuyThenClose)
    cerebro.addanalyzer(SmokeAnalyzer, name="smoke")

    [strategy] = cerebro.run()
    analysis = strategy.analyzers["smoke"].get_analysis()

    assert analysis["trades"] == [
        {"size": 2.0, "price": 11.0, "pnl": 0.0, "isclosed": False},
        {"size": 0.0, "price": 13.0, "pnl": 4.0, "isclosed": True},
    ]
    assert analysis["equity"] == [100.0, 101.0, 103.0, 104.0]
    assert strategy.broker.getcash() == 104.0
