"""Long-lived representative working-state capacity, including native objects."""

import sys
import types
from collections import deque
from functools import partial

import numpy as np
import pytest


def owned_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, (types.FunctionType, types.ModuleType, type)):
        return 0
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size
    if isinstance(value, dict):
        return size + sum(owned_size(k, seen) + owned_size(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, deque)):
        return size + sum(owned_size(x, seen) for x in value)
    if isinstance(value, partial):
        return size + owned_size(value.args, seen) + owned_size(value.keywords, seen)
    if hasattr(value, "__dict__"):
        size += owned_size(value.__dict__, seen)
    for slot in getattr(type(value), "__slots__", ()):
        if hasattr(value, slot):
            size += owned_size(getattr(value, slot), seen)
    return size


@pytest.mark.parametrize(
    "source,name",
    [
        ("talib", "EMA"),
        ("talib", "MACD"),
        ("talib", "MAXINDEX"),
        ("tv", "ema"),
        ("tv", "valuewhen"),
        ("tv", "pivothigh"),
        ("tdx", "EMA"),
        ("tdx", "BBI"),
        ("tdx", "XSII"),
    ],
)
def test_fifty_thousand_updates_have_stable_working_state(source, name):
    import importlib

    indicator = getattr(
        importlib.import_module(f"tradelearn.indicators.{source}"), name
    ).configured()

    def forbidden(*args, **kwargs):
        raise AssertionError("streaming called the batch function")

    indicator._func = forbidden
    state = indicator.new_state()
    baseline = None
    for i in range(50000):
        x = 10 + np.sin(i / 10)
        row = dict(
            source=x,
            close=x,
            open=x - 0.1,
            high=x + 1,
            low=x - 1,
            volume=100.0,
            condition=i % 5 == 0,
            anchor=i % 100 == 0,
            source2=x,
        )
        indicator.on_bar(row, state=state)
        if i == 9999:
            baseline = owned_size(state)
    assert owned_size(state) <= baseline + 8192
    assert sum(len(x) for x in state.buffers.values()) == 0


def all_public_indicators():
    import json
    import sys
    from pathlib import Path

    from tradelearn.indicators import talib, tdx

    rows = [("talib", name) for name in talib.QUALIFIED]
    if sys.version_info < (3, 11):
        # Keep each unavailable TV capability visible, while executing all
        # supported TA/TDX capacity cases. Missing PyneCore on 3.11+ is an error.
        manifest = Path(__file__).parents[3] / "tradelearn/indicators/capabilities.json"
        capabilities = json.loads(manifest.read_text())["entries"]["tv"]
        rows.extend(
            pytest.param(
                "tv",
                name,
                marks=pytest.mark.skip(
                    reason="Fixed PyneCore requires Python 3.11+; TV unsupported on 3.10"
                ),
            )
            for name, item in capabilities.items()
            if item["public"]
        )
    else:
        from tradelearn.indicators import tv

        rows.extend(("tv", name) for name in tv.QUALIFIED)
    rows.extend(("tdx", name) for name in tdx.QUALIFIED)
    return rows


@pytest.mark.parametrize("source,name", all_public_indicators())
def test_every_public_default_has_bounded_state_after_twelve_thousand_updates(source, name):
    import importlib

    indicator = getattr(
        importlib.import_module(f"tradelearn.indicators.{source}"), name
    ).configured()

    def forbidden(*args, **kwargs):
        raise AssertionError("public stream called batch")

    indicator._func = forbidden
    state = indicator.new_state()
    row = dict(
        close=20.0,
        source=20.0,
        open=19.9,
        high=21.0,
        low=19.0,
        volume=1000.0,
        periods=5.0,
        source2=20.2,
        condition=True,
        anchor=False,
        true_value=20.0,
        false_value=21.0,
        alpha=0.2,
        lower=19.0,
        upper=21.0,
    )
    baseline = None
    try:
        for i in range(12000):
            row["condition"] = i % 7 == 0
            row["anchor"] = i % 100 == 0
            indicator.on_bar(row, state=state)
            if i == 5999:
                baseline = owned_size(state)
        assert owned_size(state) <= baseline + 8192
        assert not any(state.buffers.values())
    finally:
        indicator.reset(state=state)
