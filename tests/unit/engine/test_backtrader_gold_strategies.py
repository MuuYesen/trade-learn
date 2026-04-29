from __future__ import annotations

import math

import pandas as pd
import pytest

import tradelearn.engine as bt
from .gold_strategies import STRATEGIES


def bars() -> pd.DataFrame:
    close = [
        10.0,
        10.3,
        10.1,
        10.8,
        11.4,
        10.7,
        9.9,
        9.4,
        10.2,
        11.1,
        12.0,
        11.2,
        10.4,
        9.8,
        10.5,
        11.5,
        12.6,
        11.7,
        10.8,
        10.0,
        10.9,
        12.0,
        13.1,
        12.2,
        11.1,
        10.3,
        11.3,
        12.5,
        13.8,
        12.6,
    ]
    frame = pd.DataFrame(
        {
            "close": close,
            "open": [value - 0.2 for value in close],
            "high": [value + 0.6 for value in close],
            "low": [value - 0.7 for value in close],
            "volume": [1000.0 + index * 25.0 for index in range(len(close))],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )
    return frame[["open", "high", "low", "close", "volume"]]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda strategy: strategy.__name__)
def test_backtrader_compat_gold_strategy_runs(strategy: type[bt.Strategy]) -> None:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars(), name="daily"))
    cerebro.addstrategy(strategy)

    [result] = cerebro.run()

    assert result.stats is not None
    assert len(result.stats.fills) >= 1
    assert math.isfinite(result.stats.summary["final_value"])
