"""Independent fixed upstream reference, including actual seeds and units."""

import numpy as np
import pandas as pd
import pytest

from tradelearn.indicators.tdx import _mytt_upstream as reference
from tradelearn.indicators.tdx.upstream import SUPPORTED, make_indicator


@pytest.mark.parametrize("name", SUPPORTED)
@pytest.mark.parametrize("kind", ["normal", "constant", "nan"])
def test_fixed_source_batch_and_persistent_prefix(name, kind):
    indicator = make_indicator(name)
    x = 20 + np.sin(np.arange(220) / 4) + np.arange(220) * 0.01
    if kind == "constant":
        x[:] = 20
    if kind == "nan" and name not in {
        "IF",
        "LAST",
        "COUNT",
        "EVERY",
        "EXIST",
        "FILTER",
        "BARSLAST",
        "BARSLASTCOUNT",
        "BARSSINCEN",
        "LONGCROSS",
        "VALUEWHEN",
    }:
        x[[0, 1, 20, 70]] = np.nan
    frame = pd.DataFrame(
        dict(
            close=x,
            open=x - 0.1,
            high=x + 1,
            low=x - 1,
            volume=np.full(len(x), 1000.0),
            source=x,
            source2=x[::-1],
            condition=np.arange(len(x)) % 7 == 0,
            true_value=x,
            false_value=x + 1,
            alpha=np.full(len(x), 0.2),
            lower=x - 0.3,
            upper=x + 0.3,
        )
    )
    args = [frame[col] for col in indicator._bar_columns]
    actual = [
        indicator.on_bar(row, state=state)
        for state in [indicator.new_state()]
        for row in frame.to_dict("records")
    ]
    with np.errstate(all="ignore"):
        expected = indicator.compute(*args)
    np.testing.assert_allclose(actual, np.asarray(expected), equal_nan=True, rtol=1e-8, atol=1e-8)
    for end in [5, 40, 220]:
        with np.errstate(all="ignore"):
            prefix = indicator.compute(*(arg.iloc[:end] for arg in args))
        np.testing.assert_allclose(
            actual[end - 1], np.asarray(prefix)[-1], equal_nan=True, rtol=1e-8, atol=1e-8
        )


@pytest.mark.parametrize("name", SUPPORTED)
def test_empty_has_consistent_index_and_declared_output_shape(name):
    indicator = make_indicator(name)
    empty = pd.Series(dtype=float)
    result = indicator.compute(*(empty for _ in indicator._bar_columns))
    assert result.empty
    assert indicator.minimum_period >= 1
    if len(indicator.output_names) > 1:
        assert tuple(result.columns) == indicator.output_names


def test_fixed_source_reference_hash_and_rejected_future_helpers():
    import hashlib
    from pathlib import Path

    assert (
        hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()
        == "48f9cd456f961055b013cd72a573597a13bc99ae1a4fcd84bc3076a717e6a9a3"
    )
    for name in ["CONST", "RET", "TOPRANGE", "LOWRANGE"]:
        with pytest.raises(NotImplementedError):
            make_indicator(name)


def test_bound_parameters_do_not_repeat_signature_introspection(monkeypatch):
    from types import SimpleNamespace

    from tradelearn.indicators.tdx import upstream

    indicator = make_indicator("EMA", N=7)
    state = indicator.new_state()
    indicator.on_bar({"close": 10.0}, state=state)

    def forbidden(*args, **kwargs):
        raise AssertionError("per-bar signature introspection")

    monkeypatch.setattr(upstream, "inspect", SimpleNamespace(signature=forbidden))
    assert indicator.on_bar({"close": 14.0}, state=state) == 11.0
