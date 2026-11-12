"""PyneCore fixed-version state-vector adapter qualification."""

import numpy as np
import pandas as pd
import pytest

from tradelearn.indicators.tv.persistent import SPECS, make_indicator


@pytest.mark.parametrize("name", list(SPECS))
def test_prefix_and_interleaved_isolation(name):
    indicator = make_indicator(name)
    n = 80
    x = np.arange(n, dtype=float) + 10
    frame = pd.DataFrame(
        dict(
            source=x + np.sin(x),
            close=x + np.sin(x),
            high=x + 2,
            low=x - 2,
            open=x - 0.1,
            volume=x * 10,
            source2=x[::-1],
            condition=x % 5 == 0,
            anchor=x % 20 == 0,
        )
    )
    inputs = [frame[c] for c in indicator._bar_columns]
    expected = indicator.compute(*inputs)
    state, other = indicator.new_state(), indicator.new_state()
    rows = frame.to_dict("records")
    actual = []
    for row in rows:
        actual.append(indicator.on_bar(row, state=state))
        indicator.on_bar(
            {k: (v if k in {"condition", "anchor"} else v * 2) for k, v in row.items()}, state=other
        )
    np.testing.assert_allclose(actual, np.asarray(expected), equal_nan=True, rtol=1e-10, atol=1e-10)
    for end in [1, 5, 20, 40, 80]:
        np.testing.assert_allclose(
            np.asarray(indicator.compute(*(s.iloc[:end] for s in inputs)))[-1],
            actual[end - 1],
            equal_nan=True,
        )


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("kind", ["constant", "nan"])
def test_missing_constant_and_restart(name, kind):
    indicator = make_indicator(name)
    x = np.full(100, 10.0)
    if kind == "nan":
        x[[0, 1, 20, 50]] = np.nan
    frame = pd.DataFrame(
        dict(
            source=x,
            close=x,
            high=x + 1,
            low=x - 1,
            open=x,
            volume=x * 10,
            source2=x,
            condition=np.arange(100) % 5 == 0,
            anchor=np.arange(100) % 20 == 0,
        )
    )
    expected = indicator.compute(*(frame[c] for c in indicator._bar_columns))
    state = indicator.new_state()
    actual = [indicator.on_bar(row, state=state) for row in frame.to_dict("records")]
    np.testing.assert_allclose(actual, np.asarray(expected), equal_nan=True)
    indicator.reset(state=state)
    replay = [indicator.on_bar(row, state=state) for row in frame.to_dict("records")]
    np.testing.assert_allclose(replay, actual, equal_nan=True)


@pytest.mark.parametrize("name", list(SPECS))
def test_each_public_source_can_produce_a_real_finite_output(name):
    indicator = make_indicator(name)
    x = 20 + np.arange(300) * 0.1 + np.sin(np.arange(300) * 0.7)
    frame = pd.DataFrame(
        dict(
            source=x,
            close=x,
            open=x - 0.1,
            high=x + 1,
            low=x - 1,
            volume=np.arange(300) + 100.0,
            source2=x[::-1],
            condition=np.arange(300) % 7 == 0,
            anchor=np.arange(300) % 30 == 0,
        )
    )
    output = indicator.compute(*(frame[column] for column in indicator._bar_columns))
    assert np.isfinite(np.asarray(output)).any(), name
