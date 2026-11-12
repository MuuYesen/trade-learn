"""Event identity, dependency scheduling and restart counterexamples."""

import pytest

from tradelearn.indicators.streaming import StreamGraph


class Counter:
    warmup = 2

    def __init__(self):
        self.total = 0

    def update(self, inputs, dependencies):
        self.total += inputs["close"]
        return self.total + sum(dependencies.values())

    def reset(self):
        self.total = 0

    def snapshot(self):
        return self.total

    def restore(self, value):
        self.total = value


def graph():
    result = StreamGraph(identity=("strategy", "feed", "A", "1d"))
    result.add("a", Counter())
    result.add("b", Counter(), dependencies=("a",))
    return result


def test_dependency_once_duplicate_and_same_time_distinct_events():
    value = graph()
    assert value.update(1, "x", {"close": 2}) == {"a": 2, "b": 4}
    assert value.update(1, "x", {"close": 2}) == {"a": 2, "b": 4}
    assert not value.ready("b")
    assert value.update(2, "y", {"close": 3}) == {"a": 5, "b": 10}
    assert value.ready("b")
    with pytest.raises(ValueError, match="revision"):
        value.update(2, "y", {"close": 4})
    with pytest.raises(ValueError, match="order"):
        value.update(1, "old", {"close": 100})
    assert value.values == {"a": 5, "b": 10}


def test_restart_compatibility_and_isolation():
    a, b = graph(), graph()
    a.update(1, "x", {"close": 2})
    b.restore(a.snapshot())
    assert a.update(2, "y", {"close": 3}) == b.update(2, "y", {"close": 3})
    bad = graph()
    bad.identity = ("other",)
    with pytest.raises(ValueError, match="compatible"):
        bad.restore(a.snapshot())
    a.reset()
    assert a.update(1, "x", {"close": 1}) == {"a": 1, "b": 2}
    assert b.values == {"a": 5, "b": 10}


def test_dependency_registration_and_return_ownership():
    value = graph()
    with pytest.raises(ValueError, match="dependency"):
        value.add("c", Counter(), dependencies=("missing",))
    out = value.update(1, "x", {"close": 1})
    out["a"] = 900
    assert value.values["a"] == 1


def test_duplicate_missing_payload_and_output_availability():
    class Missing(Counter):
        warmup = 1

        def update(self, inputs, dependencies):
            return float("nan")

    value = StreamGraph(identity=("x",))
    value.add("missing", Missing())
    value.update(1, "x", {"close": float("nan")})
    value.update(1, "x", {"close": float("nan")})
    assert not value.ready("missing")


def test_corrupt_snapshot_rejects_before_backend_mutation():
    value = graph()
    value.update(1, "x", {"close": 2})
    snapshot = value.snapshot()
    snapshot["nodes"].pop("b")
    with pytest.raises(ValueError, match="snapshot"):
        value.restore(snapshot)
    assert value.update(2, "y", {"close": 3}) == {"a": 5, "b": 10}
