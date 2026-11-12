"""Real TA-Lib persistent handles against the official batch reference."""

import numpy as np
import pandas as pd
import pytest
import talib as native

from tradelearn.indicators.talib.persistent import make_indicator


@pytest.mark.parametrize("name", native.get_functions())
def test_fixed_upstream_default_prefix(name):
    indicator = make_indicator(name)
    rng = np.random.default_rng(87)
    n = 420
    close = 50 + np.cumsum(rng.normal(0, 0.2, n))
    frame = pd.DataFrame(
        dict(
            open=close - 0.1,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=rng.uniform(100, 300, n),
            periods=np.full(n, 5.0),
            real=close,
            real0=close,
            real1=close + 1,
        )
    )
    args = [frame[col] for col in indicator._bar_columns]
    expected = indicator.compute(*args)
    actual = [
        indicator.on_bar(row, state=state)
        for state in [indicator.new_state()]
        for row in frame.to_dict("records")
    ]
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=2e-9, atol=2e-9, equal_nan=True)


def test_nan_interior_is_rejected_and_leading_nan_bounded():
    indicator = make_indicator("EMA", timeperiod=3)
    state = indicator.new_state()
    for _ in range(10000):
        assert np.isnan(indicator.on_bar({"close": np.nan}, state=state))
    assert not state.state["backend"].history[0]
    for value in [1, 2, 3, 4]:
        indicator.on_bar({"close": value}, state=state)
    with pytest.raises(ValueError, match="finite"):
        indicator.on_bar({"close": np.nan}, state=state)


def test_nondefault_period_prefix_and_capacity():
    indicator = make_indicator("EMA", timeperiod=5)
    state = indicator.new_state()
    values = np.linspace(1, 100, 10000)
    actual = [indicator.on_bar({"close": x}, state=state) for x in values]
    np.testing.assert_allclose(actual, native.EMA(values, timeperiod=5), equal_nan=True)
    assert not any(state.state["backend"].history)


@pytest.mark.parametrize("name", native.get_functions())
@pytest.mark.parametrize("kind", ["constant", "leading_nan", "short_prefix"])
def test_upstream_edge_profiles(name, kind):
    indicator = make_indicator(name)
    n = 220
    x = np.ones(n) * 20 if kind == "constant" else np.linspace(20, 25, n) + np.sin(np.arange(n))
    frame = pd.DataFrame(
        dict(
            open=x - 0.1,
            high=x + 0.5,
            low=x - 0.5,
            close=x,
            volume=x * 100,
            periods=np.full(n, 5.0),
        )
    )
    if kind == "leading_nan":
        frame.iloc[:4] = np.nan
    args = [frame[c] for c in indicator._bar_columns]
    state = indicator.new_state()
    actual = [indicator.on_bar(row, state=state) for row in frame.to_dict("records")]
    if kind == "short_prefix":
        for end in [1, 2, 4, 10, 30, 70, 220]:
            reference = indicator.compute(*(s.iloc[:end] for s in args))
            np.testing.assert_allclose(
                actual[end - 1], np.asarray(reference)[-1], equal_nan=True, rtol=2e-8, atol=2e-8
            )
    else:
        np.testing.assert_allclose(
            actual, np.asarray(indicator.compute(*args)), equal_nan=True, rtol=2e-8, atol=2e-8
        )


@pytest.mark.parametrize("name", native.get_functions())
def test_nondefault_native_parameters(name):
    from talib import abstract

    defaults = abstract.Function(name).parameters
    alternatives = {
        "timeperiod": 7,
        "fastperiod": 3,
        "slowperiod": 9,
        "signalperiod": 4,
        "matype": 1,
        "fastmatype": 1,
        "slowmatype": 1,
        "signalmatype": 1,
        "nbdevup": 1.5,
        "nbdevdn": 1.5,
        "minperiod": 3,
        "maxperiod": 12,
    }
    params = {key: alternatives[key] for key in defaults if key in alternatives}
    if name == "FRAMA":
        params["timeperiod"] = 8  # official FRAMA requires an even period
    indicator = make_indicator(name, **params)
    x = np.linspace(10, 30, 240) + np.sin(np.arange(240) / 4)
    frame = pd.DataFrame(
        dict(
            open=x - 0.2,
            high=x + 0.6,
            low=x - 0.7,
            close=x,
            volume=x * 20,
            periods=np.full(len(x), 5.0),
        )
    )
    args = [frame[c] for c in indicator._bar_columns]
    state = indicator.new_state()
    actual = [indicator.on_bar(row, state=state) for row in frame.to_dict("records")]
    np.testing.assert_allclose(
        actual, np.asarray(indicator.compute(*args)), equal_nan=True, rtol=2e-8, atol=2e-8
    )


def test_batch_and_stream_share_interior_nonfinite_rejection():
    indicator = make_indicator("EMA", timeperiod=3)
    with pytest.raises(ValueError, match="finite"):
        indicator.compute(pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]))


def test_native_stream_lease_rejects_global_settings_mutation():
    indicator = make_indicator("EMA", timeperiod=3)
    state = indicator.new_state()
    for value in [1.0, 2.0, 3.0]:
        indicator.on_bar({"close": value}, state=state)
    old = native.get_unstable_period("EMA")
    with pytest.raises(RuntimeError, match="streams are active"):
        native.set_unstable_period("EMA", old + 1)
    assert native.get_unstable_period("EMA") == old
    indicator.reset(state=state)


def test_settings_change_during_warmup_rejects_before_native_open():
    # Global settings tests require a fresh native context, independent of
    # active stream leases owned by other test/runtime instances.
    import os
    import subprocess
    import sys

    code = """
import talib
from tradelearn.indicators.talib.persistent import make_indicator
indicator=make_indicator('EMA',timeperiod=3)
state=indicator.new_state()
indicator.on_bar({'close':1.},state=state)
old=talib.get_unstable_period('EMA')
talib.set_unstable_period('EMA',old+1)
try:
    indicator.on_bar({'close':2.},state=state)
except ValueError as exc:
    assert 'global settings' in str(exc)
else:raise AssertionError('settings mutation was not detected')
finally:talib.set_unstable_period('EMA',old)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
