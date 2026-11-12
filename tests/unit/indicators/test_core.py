"""Default source migration and retained custom indicator contract."""

import numpy as np
import pandas as pd
import pytest
import talib as reference

from tradelearn import indicators
from tradelearn.indicators.base import FunctionIndicator


@pytest.mark.parametrize("name,period", [("SMA", 3), ("EMA", 4), ("RSI", 4)])
def test_default_source_matches_real_native_reference(name, period):
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0])
    actual = getattr(indicators.talib, name)(close, timeperiod=period)
    expected = getattr(reference, name)(close.to_numpy(), timeperiod=period)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_retired_duplicate_shortcuts_are_not_discovered():
    assert indicators.DEFAULT_SOURCE == "talib"
    for name in ["sma", "ema", "rsi", "macd", "atr", "bbands", "adx", "vwap", "pta"]:
        with pytest.raises(AttributeError):
            getattr(indicators, name)


def test_custom_function_retains_batch_and_explicit_incremental_protocol():
    calls = []

    def batch(close):
        calls.append(len(close))
        return close * 2

    indicator = FunctionIndicator(
        "double", batch, bar_columns=("close",), stream_func=lambda b, s, p: b["close"][-1] * 2
    )
    assert indicator(pd.Series([1.0, 2.0])).tolist() == [2.0, 4.0]
    state = indicator.new_state()
    assert indicator.on_bar({"close": 7}, state=state) == 14
    assert calls == [2]
