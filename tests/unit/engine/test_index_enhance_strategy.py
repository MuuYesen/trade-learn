from __future__ import annotations

import pandas as pd

from tradelearn.engine import Cerebro, IndexEnhanceStrategy
from tradelearn.ml import EqualWeightOptimizer, ModelAdapter, StrategyPipeline, TopKSelector


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


def test_index_enhance_strategy_targets_weights_from_user_next() -> None:
    class MonthlyTopClose(IndexEnhanceStrategy):
        rebalance_freq = "monthly"

        def __init__(self) -> None:
            super().__init__()
            self.universe_snapshots: list[pd.DataFrame] = []

        def next(self) -> None:
            if not self.should_rebalance():
                return
            universe = self.current_universe()
            self.universe_snapshots.append(universe.copy())
            winner = str(universe["close"].idxmax())
            self.target_weights(pd.Series({winner: 0.5}))

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

        def next(self) -> None:
            if not self.should_rebalance():
                return
            self.calls.append(self.current_datetime())
            self.target_weights({"AAA": 0.25})

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_frame([10, 11, 12, 13, 14]), name="AAA")
    cerebro.addstrategy(EveryTwoBars)

    strategy = cerebro.run()[0]

    assert len(strategy.calls) == 3


def test_index_enhance_strategy_can_consume_pipeline_weights() -> None:
    class PipelineWeights(IndexEnhanceStrategy):
        rebalance_freq = 1

        def __init__(self) -> None:
            super().__init__()
            self.pipeline = StrategyPipeline(
                [
                    ("model", ModelAdapter(score_column="close")),
                    ("selector", TopKSelector(k=1)),
                    ("optimizer", EqualWeightOptimizer(gross=0.5)),
                ]
            )

        def next(self) -> None:
            if not self.should_rebalance():
                return
            universe = self.current_universe()
            self.target_weights(self.pipeline.predict_weights(universe).as_weight_dict())

    cerebro = Cerebro(trade_on_close=True)
    cerebro.setcash(100_000.0)
    cerebro.adddata(_frame([10, 10, 10]), name="AAA")
    cerebro.adddata(_frame([20, 20, 20]), name="BBB")
    cerebro.addstrategy(PipelineWeights)

    strategy = cerebro.run()[0]

    assert strategy.getpositionbyname("BBB").size > 0
