from __future__ import annotations

import pandas as pd
import pytest

import tradelearn.engine as bt


def test_engine_facade_exports_cn_stock_commission() -> None:
    assert bt.CNAStockCommission.__name__ == "CNAStockCommission"
    assert "CNAStockCommission" in bt.__all__


def test_engine_applies_cn_stock_commission_minimum_transfer_and_stamp_tax() -> None:
    class RoundTrip(bt.Strategy):
        def next(self) -> None:
            if len(self.data) == 1:
                self.buy(size=100)
            elif len(self.data) == 2:
                self.sell(size=100)

    bars = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "volume": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
        },
        index=pd.date_range("2025-01-01", periods=4),
    )
    cerebro = bt.Cerebro(trade_on_close=False, commission=bt.CNAStockCommission())
    cerebro.setcash(100_000.0)
    cerebro.adddata(bars)
    cerebro.addstrategy(RoundTrip)

    [strategy] = cerebro.run()
    fills = strategy.stats.fills

    assert fills["side"].tolist() == ["buy", "sell"]
    assert fills["commission"].tolist() == [5.02, 6.02]
    assert strategy.stats.summary["final_value"] == pytest.approx(99_988.96)
