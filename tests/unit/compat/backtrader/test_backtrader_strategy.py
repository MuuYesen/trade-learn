from __future__ import annotations

import pandas as pd

import tradelearn.compat.backtrader as bt
from tradelearn.compat.backtrader import Cerebro


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


def test_backtrader_strategy_import_path_preserves_params_and_line_indexing() -> None:
    class RecordingStrategy(bt.Strategy):
        params = (("threshold", 10.5),)

        def __init__(self) -> None:
            self.threshold_seen = self.p.threshold
            self.values: list[tuple[float, float | None]] = []

        def next(self) -> None:
            previous = None if len(self.values) == 0 else self.data.close[-1]
            self.values.append((self.data.close[0], previous))

    cerebro = Cerebro()
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)

    [strategy] = cerebro.run()

    assert strategy.threshold_seen == 11.0
    assert strategy.p.threshold == 11.0
    assert strategy.data._name == "daily"
    assert strategy.values == [(10.0, None), (11.0, 10.0), (12.0, 11.0)]


def test_backtrader_strategy_module_exports_order_and_line_types() -> None:
    line = bt.LineSeries([1.0, 2.0, 3.0])
    line._advance(1)

    assert line[0] == 2.0
    assert line[-1] == 1.0
    assert bt.Order.Completed == 4
