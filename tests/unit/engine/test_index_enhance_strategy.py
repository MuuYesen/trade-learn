from __future__ import annotations

from tests.unit.backtest._engine_config import feed as _as_feed

from tests.unit.backtest._engine_config import configured_engine

import pandas as pd

from tradelearn.engine import Cerebro, Strategy
from tradelearn.research import ResearchResult
from tradelearn.backtest.targets import target_weights
from tradelearn.research.run import bind_research_result
import tradelearn.research.portfolio as pf


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


def test_engine_strategy_targets_weights_from_user_next() -> None:
    class MonthlyTopClose(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.score_snapshots: list[pd.Series] = []

        def next(self) -> None:
            scores = pd.Series(
                {data._name: float(data.close[0]) for data in self.datas},
                name="close",
            )
            self.score_snapshots.append(scores.copy())
            winner = str(scores.idxmax())
            target_weights(self, pd.Series({winner: 0.5}))

    cerebro = configured_engine(trade_on_close=True)
    cerebro.broker.setcash(100_000.0)
    cerebro.adddata(_as_feed(_frame([10, 11, 12, 13, 14, 15])), name="AAA")
    cerebro.adddata(_as_feed(_frame([20, 19, 18, 17, 16, 15])), name="BBB")
    cerebro.addstrategy(MonthlyTopClose)

    strategy = cerebro.run()[0]

    assert len(strategy.score_snapshots) == 6
    first = strategy.score_snapshots[0]
    assert set(first.index) == {"AAA", "BBB"}
    assert first.loc["BBB"] == 20.0
    assert len(strategy.broker._orders) >= 1


def test_engine_strategy_allows_user_defined_rebalance_frequency() -> None:
    class EveryTwoBars(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def next(self) -> None:
            if len(self) % 2 != 0:
                return
            self.calls += 1
            target_weights(self, {"AAA": 0.25})

    cerebro = configured_engine(trade_on_close=True)
    cerebro.adddata(_as_feed(_frame([10, 11, 12, 13, 14])), name="AAA")
    cerebro.addstrategy(EveryTwoBars)

    strategy = cerebro.run()[0]

    assert strategy.calls == 2


def test_engine_strategy_can_consume_research_weight_functions() -> None:
    class ResearchWeights(Strategy):
        def next(self) -> None:
            scores = pd.Series(
                {data._name: float(data.close[0]) for data in self.datas},
                name="close",
            )
            selected = pf.select_top(scores, k=1)
            weights = pf.equal_weight(selected, gross=0.5)
            target_weights(self, weights.to_dict())

    cerebro = configured_engine(trade_on_close=True)
    cerebro.broker.setcash(100_000.0)
    cerebro.adddata(_as_feed(_frame([10, 10, 10])), name="AAA")
    cerebro.adddata(_as_feed(_frame([20, 20, 20])), name="BBB")
    cerebro.addstrategy(ResearchWeights)

    strategy = cerebro.run()[0]

    assert strategy.getpositionbyname("BBB").size > 0


def test_engine_strategy_can_consume_research_result_current_weights() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-29", periods=3, freq="D", tz="UTC"), ["AAA", "BBB"]],
        names=["timestamp", "symbol"],
    )
    research = ResearchResult(
        name="engine-research-weights",
        weights=pd.Series([0.0, 0.5, 0.0, 0.5, 0.0, 0.5], index=index),
    )

    class ResearchWeights(Strategy):
        params=(("research_result",None),)
        def __init__(self):
            self.research_result=bind_research_result(self.p.research_result,self)
        def next(self) -> None:
            target_weights(self, self.research_result.weights[0])

    cerebro = configured_engine(trade_on_close=True)
    cerebro.broker.setcash(100_000.0)
    cerebro.adddata(_as_feed(_frame([10, 10, 10])), name="AAA")
    cerebro.adddata(_as_feed(_frame([20, 20, 20])), name="BBB")
    cerebro.addstrategy(ResearchWeights, research_result=research)

    strategy = cerebro.run()[0]

    assert strategy.research_result.raw is research
    assert not hasattr(strategy,"research_results_history")
    assert strategy.getpositionbyname("BBB").size > 0
