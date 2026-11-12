"""Actual default BT buffer return type, with Lite's convenience kept separate."""

from array import array

import backtrader as official
import numpy as np
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.lite import Strategy as LiteStrategy


def frame():
    return pd.DataFrame(
        {key: [10.0, 11.0, 12.0] for key in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=3),
    )


def run(api, stream=False):
    rows = []

    class Business(api.Strategy):
        def __init__(self):
            self.shifted = self.data.close + 1

        def next(self):
            lines = [self.data.close, self.data.datetime, self.shifted]
            rows.append(
                [
                    (type(v := line.get(size=1)), getattr(v, "typecode", None), list(v))
                    for line in lines
                ]
            )

    if stream:
        runtime = StrategyRuntime(Business, feeds={"A": "bar"})
        try:
            runtime.start()
            for stamp, row in frame().iterrows():
                runtime.step_market("A", dict(ts=stamp.tz_localize("UTC"), **row.to_dict()))
            runtime.stop()
        finally:
            runtime.cleanup()
    else:
        e = api.Cerebro(stdstats=False)
        e.adddata(api.feeds.PandasData(dataname=frame()))
        e.addstrategy(Business)
        e.run()
    return rows


@pytest.mark.parametrize("stream", [False, True])
def test_default_engine_data_date_indicator_get_returns_actual_bt_array(stream):
    expected = run(official)
    actual = run(tl, stream=stream)
    assert len(actual) == len(expected)
    for values, wanted in zip(actual, expected, strict=True):
        for value, want in zip(values, wanted, strict=True):
            assert value[:2] == want[:2] == (array, "d")
            assert value[2] == pytest.approx(want[2])


def test_plain_lite_get_keeps_numpy_window():
    runtime = StrategyRuntime(LiteStrategy, feeds={"A": "bar"})
    try:
        runtime.start()
        runtime.step_market(
            "A", dict(ts="2026-01-01T00:00:00Z", open=10, high=10, low=10, close=10, volume=1)
        )
        result = runtime.strategy.data.close.get(size=1)
        assert type(result) is np.ndarray
        assert result.tolist() == [10.0]
    finally:
        runtime.cleanup()
