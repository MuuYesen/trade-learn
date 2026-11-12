"""Real Engine/Lite integration, with the shared Lines owner implementation."""

import numpy as np
import pandas as pd
import pytest

import tradelearn.engine as bt
import tradelearn.lite as lite
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.indicators import talib, tdx, tv


def bars(n=100):
    x = np.linspace(10, 20, n) + np.sin(np.arange(n)) * 0.2
    return pd.DataFrame(
        dict(open=x - 0.1, high=x + 1, low=x - 1, close=x, volume=x * 10),
        index=pd.date_range("2026-01-01", periods=n, tz="UTC"),
    )


@pytest.mark.parametrize("base", [bt.Strategy, lite.Strategy])
@pytest.mark.parametrize(
    "indicator,options",
    [(talib.EMA, {"timeperiod": 30}), (tv.ema, {"length": 8}), (tdx.EMA, {"N": 8})],
)
def test_actual_shared_runtime_and_capacity(base, indicator, options):
    class S(base):
        def init(self):
            self.values = []
            self.output = indicator(self.data.close, **options)

        def prenext(self):
            self.next()

        def next(self):
            self.values.append(self.output[0])
            assert self.output[0] == self.output[0] or np.isnan(self.output[0])

    if base is bt.Strategy:
        S.__init__ = S.init
    frame = bars()
    runtime = StrategyRuntime(S, feeds={"A": "bar"}, capacity=8)
    runtime.start()
    for timestamp, row in frame.iterrows():
        runtime.step_market("A", dict(ts=timestamp, **row))
    runtime.stop()
    expected = indicator.compute(frame.close, **options)
    np.testing.assert_allclose(
        runtime.strategy.values, expected, equal_nan=True, rtol=1e-9, atol=1e-9
    )
    assert runtime.strategy.data._history_materializations == 0
    assert runtime.strategy.data._history is None


@pytest.mark.parametrize("base", [bt.Strategy, lite.Strategy])
def test_multioutput_has_one_persistent_transition(base):
    class S(base):
        def init(self):
            self.values = []
            self.output = talib.MACD(self.data.close, fastperiod=3, slowperiod=7, signalperiod=3)

        def prenext(self):
            self.next()

        def next(self):
            self.values.append(
                tuple(getattr(self.output, k)[0] for k in ("macd", "signal", "hist"))
            )

    if base is bt.Strategy:
        S.__init__ = S.init
    frame = bars()
    runtime = StrategyRuntime(S, feeds={"A": "bar"}, capacity=8)
    runtime.start()
    for timestamp, row in frame.iterrows():
        runtime.step_market("A", dict(ts=timestamp, **row))
    runtime.stop()
    np.testing.assert_allclose(
        runtime.strategy.values,
        talib.MACD(frame.close, fastperiod=3, slowperiod=7, signalperiod=3),
        equal_nan=True,
    )


def test_runtime_stop_releases_handles_without_discarding_results():
    import os
    import subprocess
    import sys

    code = """
import pandas as pd
import talib as native
from tradelearn.indicators import talib
from tradelearn.backtest.event_runtime import StrategyRuntime
import tradelearn.engine as bt
class S(bt.Strategy):
    def __init__(self):self.ma=talib.EMA(self.data.close,timeperiod=3)
    def next(self):pass
runtime=StrategyRuntime(S,feeds={'A':'bar'},capacity=8)
runtime.start()
for i in range(6):
    x=10.+i
    runtime.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=i),open=x,high=x+1,low=x-1,close=x,volume=100.))
last=runtime.strategy.ma[0]
runtime.stop()
assert runtime.strategy.ma[0]==last
old=native.get_unstable_period('EMA')
native.set_unstable_period('EMA',old+1)
native.set_unstable_period('EMA',old)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
