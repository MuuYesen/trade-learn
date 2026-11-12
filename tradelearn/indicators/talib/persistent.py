"""Persistent TA-Lib 0.8.3 handles; no per-update array conversion after seeding.

Interior nonfinite input is outside TA-Lib C's defined batch/stream contract.
Reject it rather than manufacturing a different missing-value formula.
"""

from __future__ import annotations

import inspect
from functools import partial

import numpy as np
import pandas as pd
import talib
from talib import abstract, stream

from tradelearn.indicators.base import FunctionIndicator


def _unstable_names():
    names = []
    for name in talib.get_functions():
        if name in {"ADXR", "MFI", "STOCHRSI"}:
            continue  # upstream-deprecated, no effect
        try:
            talib.get_unstable_period(name)
        except KeyError:
            continue
        names.append(name)
    return tuple(names)


_UNSTABLE_NAMES = _unstable_names()


def _settings():
    return talib.get_compatibility(), tuple(
        talib.get_unstable_period(name) for name in _UNSTABLE_NAMES
    )


def require_backend():
    """Reject the older array-last stream API."""
    if talib.__version__ != "0.8.3" or not hasattr(stream.SMA, "update"):
        raise ImportError("persistent indicators require TA-Lib==0.8.3")


class NativeState:
    """A bounded seed followed by one native persistent handle."""

    def __init__(self, name, params, columns, outputs):
        require_backend()
        self.name, self.params = name, params
        self.outputs = outputs
        zero = name.startswith("CDL") or name in {
            "HT_TRENDMODE",
            "MAXINDEX",
            "MININDEX",
            "MINMAXINDEX",
            "FRACTAL",
        }
        self.missing = [0.0] * outputs if zero else [float("nan")] * outputs
        if name in {"SQZMOM", "SUPERTREND"}:
            self.missing[1] = 0.0
        if name == "ZIGZAG":
            self.missing[1:] = [0.0, 0.0]
        function = abstract.Function(name)
        function.set_parameters(params)
        self.minimum = function.lookback + 1
        # Explicit limit: unusual upstream seeding beyond this is unsupported,
        # never an unbounded history fallback.
        self.capacity = max(64, 2 * self.minimum + 1)
        self.history = [[] for _ in columns]
        self.handle = None
        self.started = False
        self.leading = 0
        self.environment = _settings()

    def update(self, *values):
        """Append a finite closed bar, or retain unavailable leading history."""
        # Once opened the native stream holds a read lease and upstream itself
        # refuses every settings mutation. During seeding TL checks the snapshot.
        if self.handle is None and _settings() != self.environment:
            raise ValueError("TA-Lib global settings changed during seed; reset required")
        if not all(np.isfinite(value) for value in values):
            if not self.started and all(not np.isinf(value) for value in values):
                self.leading += 1
                return self._missing()
            raise ValueError("TA-Lib input must be finite after the first valid bar")
        self.started = True
        if self.handle is not None:
            return self._offset(self.handle.update(*values))
        for history, value in zip(self.history, values, strict=True):
            history.append(value)
        if len(self.history[0]) < self.minimum:
            return self._missing()
        try:
            self.handle = getattr(stream, self.name)(
                *(np.asarray(history, dtype=np.float64) for history in self.history), **self.params
            )
        except talib.InsufficientHistory as exc:
            if len(self.history[0]) >= self.capacity:
                raise ValueError(f"{self.name} exceeds qualified seed capacity") from exc
            return self._missing()
        self.history = [[] for _ in self.history]
        return self._offset(self.handle.value)

    def close(self):
        """Release the native stream/read lease immediately at reset."""
        self.handle = None
        self.history = [[] for _ in self.history]

    def _offset(self, value):
        if self.name in {"MAXINDEX", "MININDEX"}:
            return value + self.leading
        if self.name == "MINMAXINDEX":
            return tuple(x + self.leading for x in value)
        if self.name == "ZIGZAG":
            return (*value[:2], value[2] + self.leading)
        return value

    def _missing(self):
        return self.missing[0] if self.outputs == 1 else tuple(self.missing)


def _warmup(name, params):
    function = abstract.Function(name)
    function.set_parameters(params)
    return int(function.lookback) + 1


def make_indicator(name, **overrides):
    """Build an internal adapter. Public export requires separate qualification."""
    require_backend()
    function = abstract.Function(name)
    params = dict(function.parameters)
    unknown = overrides.keys() - params.keys()
    if unknown:
        raise TypeError(f"unknown {name} parameters: {sorted(unknown)}")
    params.update(overrides)
    function.set_parameters(params)
    columns = tuple(
        col
        for value in function.input_names.values()
        for col in (value if isinstance(value, list) else [value])
    )
    outputs = tuple(function.output_names)
    if name == "MACD":
        outputs = ("macd", "signal", "hist")
    signature = inspect.signature(getattr(talib, name))

    def batch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for key, value in params.items():
            if key not in bound.arguments:
                bound.arguments[key] = value
        bound.apply_defaults()
        inputs = [bound.arguments[key] for key in list(signature.parameters)[: len(columns)]]
        index = inputs[0].index if isinstance(inputs[0], pd.Series) else None
        if index is not None and any(
            isinstance(x, pd.Series) and not x.index.equals(index) for x in inputs
        ):
            raise ValueError("TA-Lib input indexes must exactly match")
        arrays = [np.asarray(x, dtype=float) for x in inputs]
        if arrays:
            if any(np.isinf(x).any() for x in arrays):
                raise ValueError("TA-Lib input must be finite")
            valid = np.all(np.vstack([np.isfinite(x) for x in arrays]), axis=0)
            positions = np.flatnonzero(valid)
            if len(positions) and not valid[positions[0] :].all():
                raise ValueError("TA-Lib input must be finite after the first valid bar")
        values = getattr(talib, name)(*arrays, **{k: bound.arguments[k] for k in params})
        if isinstance(values, tuple):
            return pd.DataFrame(dict(zip(outputs, values, strict=True)), index=index)
        return pd.Series(values, index=index, name=name)

    batch.__signature__ = signature
    result = FunctionIndicator(
        f"talib.{name}",
        batch,
        params,
        columns,
        stream_factory=partial(NativeState, name, columns=columns, outputs=len(outputs)),
        warmup=partial(_warmup, name),
    )
    result.output_names = outputs
    return result
