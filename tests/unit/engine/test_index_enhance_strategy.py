from __future__ import annotations

import pandas as pd

from tradelearn.engine import Cerebro, IndexEnhanceStrategy


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-29", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def test_index_enhance_strategy_rebalances_from_cross_section_universe() -> None:
    class MonthlyTopClose(IndexEnhanceStrategy):
        rebalance_freq = "monthly"

        def __init__(self) -> None:
            super().__init__()
            self.universe_snapshots: list[pd.DataFrame] = []

        def rebalance(self, dt: pd.Timestamp, universe: pd.DataFrame) -> pd.Series:
            self.universe_snapshots.append(universe.copy())
            winner = str(universe["close"].idxmax())
            return pd.Series({winner: 0.5})

    cerebro = Cerebro(trade_on_close=True)
    cerebro.setcash(100_000.0)
    cerebro.adddata(_frame([10, 11, 12, 13, 14, 15]), name="AAA")
    cerebro.adddata(_frame([20, 19, 18, 17, 16, 15]), name="BBB")
    cerebro.addstrategy(MonthlyTopClose)

    strategy = cerebro.run()[0]

    assert len(strategy.universe_snapshots) == 2
    first = strategy.universe_snapshots[0]
    assert first.index.name == "symbol"
    assert set(first.index) == {"AAA", "BBB"}
    assert {"open", "high", "low", "close", "volume", "datetime"}.issubset(first.columns)
    assert first.loc["BBB", "close"] == 20.0
    assert len(strategy.broker._orders) >= 1


def test_index_enhance_strategy_integer_rebalance_frequency() -> None:
    class EveryTwoBars(IndexEnhanceStrategy):
        rebalance_freq = 2

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[pd.Timestamp] = []

        def rebalance(self, dt: pd.Timestamp, universe: pd.DataFrame) -> dict[str, float]:
            self.calls.append(dt)
            return {"AAA": 0.25}

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_frame([10, 11, 12, 13, 14]), name="AAA")
    cerebro.addstrategy(EveryTwoBars)

    strategy = cerebro.run()[0]

    assert len(strategy.calls) == 3
