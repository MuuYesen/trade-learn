from __future__ import annotations

import pandas as pd
import pytest

import tradelearn.engine as bt


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC"),
    )


class BuyOnce(bt.Strategy):
    def next(self) -> None:
        self.buy(size=1)


def test_backtest_reports_missing_rust_extension_before_matching() -> None:
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=_bars(), name="demo"))
    cerebro.addstrategy(BuyOnce)

    with pytest.raises(RuntimeError, match="tradelearn._rust.*maturin develop"):
        cerebro.run()
