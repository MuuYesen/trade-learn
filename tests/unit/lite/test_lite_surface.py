from __future__ import annotations

import pandas as pd

from tradelearn.lite import Backtest, Strategy


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
    )


def test_lite_uses_backtrader_bar_indexing_with_1x_position_call() -> None:
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.line = self.I(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=self.data.index))
            self.sma = self.data.close.ta.sma(length=2)
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["close_now"] = self.data.close[0]
                seen["close_prev"] = self.data.close[-1]
                seen["line_now"] = self.line[0]
                seen["line_prev"] = self.line[-1]
                seen["sma_now"] = self.sma[0]
                assert not self.position()
                self.buy(size=1)
            elif len(self.data) == 4:
                assert self.position()
                self.position().close()

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert stats["# Trades"] == 1
    assert seen == {
        "close_now": 12.0,
        "close_prev": 11.0,
        "line_now": 3.0,
        "line_prev": 2.0,
        "sma_now": 11.5,
    }
