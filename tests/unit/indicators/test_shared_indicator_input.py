"""The strict Indicator facade passes its internal line to source adapters."""

import numpy as np
import pandas as pd
import pytest

import tradelearn.engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.indicators import talib


@pytest.mark.parametrize("keyword", [False, True])
def test_custom_indicator_composes_with_persistent_source_without_public_batch_methods(keyword):
    assert not hasattr(bt.Indicator, "to_series")
    assert not hasattr(bt.Indicator, "wrap_indicator")

    class Twice(bt.Indicator):
        lines = ("value",)

        def next(self):
            self.lines.value[0] = 2 * self.data.close[0]

    class S(bt.Strategy):
        def __init__(self):
            self.values = []
            source = Twice(self.data)
            self.output = (
                talib.EMA(real=source, timeperiod=3) if keyword else talib.EMA(source, timeperiod=3)
            )

        def prenext(self):
            self.next()

        def next(self):
            self.values.append(self.output[0])

    runtime = StrategyRuntime(S, feeds={"A": "bar"}, capacity=4)
    runtime.start()
    prices = np.arange(1.0, 13.0)
    for i, price in enumerate(prices):
        runtime.step_market(
            "A",
            dict(
                ts=pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(i, "s"),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1,
            ),
        )
    runtime.stop()
    np.testing.assert_allclose(
        runtime.strategy.values,
        talib.EMA.compute(pd.Series(2 * prices), timeperiod=3),
        equal_nan=True,
    )
    assert runtime.strategy.data._history_materializations == 0
