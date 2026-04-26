from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.backtest import Analyzer, Cerebro, Strategy


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


def test_cerebro_runs_strategy_with_params_and_current_bar_index() -> None:
    class RecordingStrategy(Strategy):
        params = (("threshold", 10.5),)

        def __init__(self) -> None:
            self.seen_in_init = self.p.threshold
            self.values: list[tuple[float, float | None]] = []

        def next(self) -> None:
            previous = None if len(self.values) == 0 else self.data.close[-1]
            self.values.append((self.data.close[0], previous))

    cerebro = Cerebro()
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)

    [strategy] = cerebro.run()

    assert strategy.seen_in_init == 11.0
    assert strategy.p.threshold == 11.0
    assert strategy.params.threshold == 11.0
    assert strategy.data is strategy.datas[0]
    assert strategy.data._name == "daily"
    assert strategy.values == [(10.0, None), (11.0, 10.0), (12.0, 11.0)]


def test_analyzer_receives_strategy_and_bar_lifecycle() -> None:
    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    class CloseAnalyzer(Analyzer):
        params = (("scale", 1.0),)

        def __init__(self) -> None:
            self.started = False
            self.values: list[float] = []
            self.ended = False

        def on_start(self) -> None:
            self.started = self.strategy is not None

        def on_bar(self, bar) -> None:
            self.values.append(bar.close * self.p.scale)

        def on_end(self, stats) -> None:
            self.ended = stats["bars"] == 3

        def get_analysis(self) -> dict[str, object]:
            return {"started": self.started, "values": self.values, "ended": self.ended}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(CloseAnalyzer, scale=2.0, name="close")

    [strategy] = cerebro.run()

    assert strategy.analyzers["close"].get_analysis() == {
        "started": True,
        "values": [20.0, 22.0, 24.0],
        "ended": True,
    }


def test_strategy_params_must_be_tuple_pairs() -> None:
    class BadParams(Strategy):
        params = {"fast": 10}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(BadParams)

    with pytest.raises(ValueError, match="Strategy.params must be a tuple"):
        cerebro.run()
