"""Counterexamples for the persistent indicator adapter contract."""

import pandas as pd
import pytest

from tradelearn.indicators.base import FunctionIndicator


def test_batch_only_indicator_never_claims_streaming():
    indicator = FunctionIndicator("batch", lambda x: x, bar_columns=("close",))
    with pytest.raises(NotImplementedError, match="persistent"):
        indicator.on_bar({"close": 1})


def test_stream_input_buffer_has_declared_bound():
    indicator = FunctionIndicator(
        "latest",
        lambda x: x,
        bar_columns=("close",),
        stream_func=lambda buffers, state, params: buffers["close"][-1],
        history_capacity=2,
    )
    state = indicator.new_state()
    for value in range(10000):
        assert indicator.on_bar({"close": value}, state=state) == value
    assert len(state.buffers["close"]) == 2


def test_factory_once_isolation_reset_and_parameter_guard():
    class Accumulator:
        def __init__(self):
            self.total = 0

        def update(self, close):
            self.total += close
            return self.total

    calls = []

    def factory(params):
        calls.append(dict(params))
        return Accumulator()

    indicator = FunctionIndicator(
        "sum", lambda x: x.cumsum(), bar_columns=("close",), stream_factory=factory
    )
    left, right = indicator.new_state(), indicator.new_state()
    assert indicator.on_bar({"close": 2}, state=left) == 2
    assert indicator.on_bar({"close": 3}, state=left) == 5
    assert indicator.on_bar({"close": 100}, state=right) == 100
    assert len(calls) == 2
    assert not any(left.buffers.values())
    indicator.params["changed"] = True
    with pytest.raises(ValueError, match="parameters"):
        indicator.on_bar({"close": 3}, state=left)
    indicator.reset(state=left)
    assert indicator.on_bar({"close": 1}, state=left) == 1


def test_input_conversion_is_atomic():
    indicator = FunctionIndicator(
        "x", lambda x: x, bar_columns=("high", "low"), stream_func=lambda b, s, p: b["high"][-1]
    )
    state = indicator.new_state()
    with pytest.raises((KeyError, ValueError)):
        indicator.on_bar({"high": 3}, state=state)
    assert not any(state.buffers.values())


def test_configured_clone_and_runtime_binding_keep_parameters_local():
    indicator = FunctionIndicator(
        "scale",
        lambda close, scale=2: close * scale,
        {"scale": 2},
        ("close",),
        stream_func=lambda b, s, p: b["close"][-1] * p["scale"],
    )
    configured = indicator.configured(scale=3)
    assert configured.on_bar({"close": 4}) == 12
    assert indicator.on_bar({"close": 4}) == 8
    bound = indicator.bind_stream((object(),), {"scale": 5})
    assert bound.update(4) == 20
    assert bound.update(6) == 30
    bound.reset()
    assert bound.update(2) == 10


def test_configured_batch_uses_same_parameters_as_stream_and_positional_wins():
    indicator = FunctionIndicator(
        "scale",
        lambda close, scale=2: close * scale,
        {"scale": 2},
        ("close",),
        stream_func=lambda b, s, p: b["close"][-1] * p["scale"],
    )
    configured = indicator.configured(scale=3)
    assert configured.compute(pd.Series([4])).iloc[-1] == configured.on_bar({"close": 4})
    assert configured.compute(pd.Series([4]), 5).iloc[-1] == 20


def test_warmup_follows_isolated_configuration():
    indicator = FunctionIndicator(
        "x",
        lambda close, period=2: close,
        {"period": 2},
        ("close",),
        stream_func=lambda b, s, p: b["close"][-1],
        warmup=lambda p: p["period"],
    )
    assert indicator.minimum_period == 2
    assert indicator.configured(period=5).minimum_period == 5
    assert indicator.bind_stream((object(),), {"period": 8}).indicator.minimum_period == 8


def test_reset_closes_owned_backend_exactly_once():
    closed = []

    class Backend:
        def update(self, value):
            return value

        def close(self):
            closed.append(True)

    indicator = FunctionIndicator(
        "x", lambda x: x, bar_columns=("close",), stream_factory=lambda p: Backend()
    )
    state = indicator.new_state()
    indicator.on_bar({"close": 1}, state=state)
    indicator.reset(state=state)
    indicator.reset(state=state)
    assert closed == [True]


def test_array_parameters_have_stable_equality_and_mutation_guard():
    import numpy as np

    indicator = FunctionIndicator(
        "weighted",
        lambda close, weights: close,
        {"weights": np.array([1.0, 2.0])},
        ("close",),
        stream_func=lambda b, s, p: b["close"][-1] * p["weights"].sum(),
    )
    state = indicator.new_state()
    assert indicator.on_bar({"close": 2.0}, state=state) == 6.0
    assert indicator.on_bar({"close": 3.0}, state=state) == 9.0
    indicator.params["weights"][0] = 10.0
    with pytest.raises(ValueError, match="parameters"):
        indicator.on_bar({"close": 3.0}, state=state)


def test_configured_parameter_arrays_are_owned_by_each_adapter():
    import numpy as np

    original = FunctionIndicator(
        "x", lambda close, weights: close, {"weights": np.array([1.0, 2.0])}
    )
    left = original.configured()
    right = original.configured()
    left.params["weights"][0] = 99.0
    assert right.params["weights"][0] == 1.0
    assert original.params["weights"][0] == 1.0
