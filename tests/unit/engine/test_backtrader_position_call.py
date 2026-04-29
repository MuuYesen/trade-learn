from __future__ import annotations

import pandas as pd

import tradelearn.engine as bt


def test_backtrader_position_object_is_callable_extension() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
    )
    seen = {}

    class PositionCallStrategy(bt.Strategy):
        def next(self) -> None:
            pos = self.position()
            seen["same_size"] = pos.size == self.position.size
            if not pos:
                self.buy(size=1)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.DataFeed(data))
    cerebro.addstrategy(PositionCallStrategy)
    cerebro.run()

    assert seen["same_size"] is True
