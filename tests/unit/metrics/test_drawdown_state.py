"""Scalar observed-value state agrees with the existing vector metric."""

import numpy as np
import pandas as pd
import pytest

from tradelearn.metrics import risk


def test_scalar_observations_match_vector_including_severe_nonpositive_values():
    update = getattr(risk, "_drawdown_observation", None)
    assert callable(update)
    values = [100.0, 120.0, 90.0, 110.0, 0.0, -50.0, 150.0]
    expected = risk.drawdown_from_equity(pd.Series(values)).tolist()
    peak = None
    actual = []
    for value in values:
        peak, ratio, money = update(value, peak)
        actual.append(ratio)
        assert money == peak - value
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, True, np.bool_(False), np.nan, np.inf])
def test_missing_or_boolean_value_is_not_zero_drawdown(value):
    update = getattr(risk, "_drawdown_observation", None)
    assert callable(update)
    with pytest.raises(ValueError):
        update(value, 100.0)


@pytest.mark.parametrize(
    "value,peak",
    [
        (0.0, None),
        (-1.0, None),
        (100.0, 0.0),
        (100.0, -1.0),
        (100.0, np.nan),
        (100.0, np.inf),
        (100.0, True),
    ],
)
def test_initial_or_stored_peak_must_be_qualified_positive(value, peak):
    update = getattr(risk, "_drawdown_observation", None)
    assert callable(update)
    with pytest.raises(ValueError):
        update(value, peak)
