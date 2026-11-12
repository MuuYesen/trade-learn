"""Small causal fixtures for examples that formerly vectorized during init."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import tradelearn.engine as bt


@pytest.mark.parametrize(
    "module,indicator,output,kind",
    [
        ("01_quickstart", "SMA", "sma", "mean"),
        ("02_sma_cross", "SMA", "sma", "mean"),
        ("05_migration", "SMA", "sma", "mean"),
        ("08_better_ma", "SMA", "sma", "mean"),
        ("06_turtle", "Highest", "highest", "max"),
        ("06_turtle", "Lowest", "lowest", "min"),
    ],
)
def test_local_indicator_advances_from_delivered_bars(module, indicator, output, kind):
    source = import_module(f"examples.engine.{module}")
    prices = pd.Series([10.0, 12.0, 9.0, 13.0, 11.0, 15.0, 8.0])
    frame = pd.DataFrame(
        {name: prices.to_numpy() for name in ("open", "high", "low", "close")},
        index=pd.date_range("2026-01-01", periods=len(prices), tz="UTC"),
    )
    frame["volume"] = 100.0

    class S(bt.Strategy):
        def __init__(self):
            self.values = []
            self.indicator = getattr(source, indicator)(self.data.close, period=3)

        def prenext(self):
            self.next()

        def next(self):
            self.values.append(getattr(self.indicator.lines, output)[0])

    engine = bt.Cerebro()
    engine.adddata(bt.feeds.PandasData(dataname=frame))
    engine.addstrategy(S)
    actual = engine.run()[0].values
    expected = getattr(prices.rolling(3), kind)()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
