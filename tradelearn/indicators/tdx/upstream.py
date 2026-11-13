"""Persistent operators for the fixed, explicitly enumerated MyTT source.

Composite formulas reuse the frozen source's Python code with only the
whitelisted basic operators rebound. This is not an arbitrary user compiler.
No operator keeps full input history; retrospective/unbounded helpers are
excluded. Source: mpquant/MyTT 7cd36ae13cae56657284badc8d5a6b7b8ed62a37.
"""

from __future__ import annotations

import inspect
from collections import deque
from functools import partial
from types import FunctionType

import numpy as np
import pandas as pd

from tradelearn.indicators.base import FunctionIndicator

from . import _mytt_upstream as reference
from ._state import EWM

EXCLUDED = {
    "RET": "scalar history accessor, not a sequence-preserving indicator",
    "CONST": "fills history from its final future value",
    "TOPRANGE": "unbounded historical search/state not qualified",
    "LOWRANGE": "unbounded historical search/state not qualified",
}
SUPPORTED = tuple(
    name
    for name, value in vars(reference).items()
    if name.isupper() and callable(value) and name not in EXCLUDED
)
BASIC = set(SUPPORTED[: SUPPORTED.index("MACD")])
ELEMENT = {
    "RD": lambda x, D=3: np.round(x, D),
    "ABS": np.abs,
    "LN": np.log,
    "POW": lambda x, N: np.power(x, N),
    "SQRT": np.sqrt,
    "SIN": np.sin,
    "COS": np.cos,
    "TAN": np.tan,
    "MAX": np.maximum,
    "MIN": np.minimum,
    "IF": lambda condition, a, b: np.float64(a if condition else b),
    "BETWEEN": lambda x, a, b: np.float64((a < x < b) or (a > x > b)),
}
_DEFAULT = {"N": 5, "A": 5, "B": 1}
_INPUT = {
    "S": "close",
    "S1": "close",
    "S2": "source2",
    "X": "close",
    "CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "VOL": "volume",
}
_CONDITIONS = {
    "IF",
    "LAST",
    "COUNT",
    "EVERY",
    "EXIST",
    "FILTER",
    "BARSLAST",
    "BARSLASTCOUNT",
    "BARSSINCEN",
    "VALUEWHEN",
}


class _Operator:
    def __init__(self, name):
        self.name = name
        self.values = None
        self.last = np.nan
        self.count = 0
        self.total = 0.0
        self.cooldown = 0
        self.ewm = None
        self.weight = 1.0

    def window(self, x, n):
        if type(n) not in (int, np.int64, np.int32) or n < 1:
            raise ValueError("period must be a positive integer")
        if self.values is None:
            self.values = deque(maxlen=int(n))
        if self.values.maxlen != n:
            raise ValueError("dynamic periods are not qualified")
        self.values.append(x)
        return np.asarray(self.values, dtype=float)

    def update(self, name, *args):
        x = args[0]
        if name in {"EMA", "SMA"}:
            n = args[1]
            alpha = 2 / (n + 1) if name == "EMA" else (args[2] if len(args) > 2 else 1) / n
            if not 0 < alpha <= 1:
                raise ValueError("invalid smoothing parameter")
            if self.ewm is None:
                self.ewm = EWM(1)
                self.ewm.alpha = alpha
            return np.float64(self.ewm.add(x))
        if name == "DMA":
            # A is the streamed series profile. Missing A becomes one exactly
            # as the fixed source's vector-A branch specifies.
            alpha = args[1] if not np.isnan(args[1]) else 1.0
            self.last = x if self.count == 0 else alpha * x + (1 - alpha) * self.last
            self.count += 1
            return self.last
        if name in {"REF", "DIFF"}:
            n = args[1] if len(args) > 1 else 1
            if int(n) != n or n < 0:
                raise ValueError("negative/future references are not qualified")
            values = self.window(x, int(n) + 1)
            old = values[0] if len(values) > n else np.nan
            return old if name == "REF" else x - old
        if name in {"SUM", "COUNT"} and args[1] == 0:
            if np.isnan(x):
                return np.float64(np.nan)
            self.total += x
            return np.float64(self.total)
        if name in {
            "MA",
            "STD",
            "SUM",
            "COUNT",
            "HHV",
            "LLV",
            "HHVBARS",
            "LLVBARS",
            "WMA",
            "AVEDEV",
            "SLOPE",
            "FORCAST",
            "EVERY",
            "EXIST",
            "BARSSINCEN",
        }:
            n = args[1]
            values = self.window(x, n)
            valid = len(values) == n and np.isfinite(values).all()
            if not valid:
                return np.float64(False) if name in {"EVERY", "EXIST"} else np.float64(np.nan)
            if name == "MA":
                return np.mean(values)
            if name == "STD":
                return np.std(values, ddof=0)
            if name in {"SUM", "COUNT"}:
                return np.sum(values)
            if name == "HHV":
                return np.max(values)
            if name == "LLV":
                return np.min(values)
            if name == "HHVBARS":
                return np.float64(np.argmax(values[::-1]))
            if name == "LLVBARS":
                return np.float64(np.argmin(values[::-1]))
            if name == "WMA":
                return values[::-1].cumsum().sum() * 2 / n / (n + 1)
            if name == "AVEDEV":
                return np.abs(values - values.mean()).mean()
            if name == "SLOPE":
                return np.polyfit(range(n), values, deg=1)[0]
            if name == "FORCAST":
                return np.polyval(np.polyfit(range(n), values, deg=1), n - 1)
            if name == "EVERY":
                return np.float64(np.sum(values) == n)
            if name == "EXIST":
                return np.float64(np.sum(values) > 0)
            index = np.argmax(values)
            return np.float64(n - 1 - index) if index or values[0] else np.float64(np.nan)
        if name == "LAST":
            a, b = args[1:]
            if not (a > b >= 0):
                raise ValueError("LAST requires A>B>=0")
            values = self.window(x, a + 1)
            return np.float64(
                True
                if len(values) < a + 1 or not np.isfinite(values).all()
                else np.all(values[::-1][b:])
            )
        if name == "FILTER":
            if self.cooldown:
                self.cooldown -= 1
                return np.float64(0)
            if x:
                self.cooldown = int(args[1])
            return x
        if name == "BARSLAST":
            self.count = 0 if x else self.count + 1
            return np.float64(self.count)
        if name == "BARSLASTCOUNT":
            self.count = self.count + 1 if x else 0
            return np.float64(self.count)
        if name == "CROSS":
            current = x > args[1]
            output = current and self.count > 0 and not self.last
            self.last = current
            self.count += 1
            return np.float64(output)
        if name == "LONGCROSS":
            y, n = args[1:]
            values = self.window(float(x < y), n + 1)
            prior = True if len(values) < n + 1 else np.all(values[:-1])
            return np.float64(prior and x > y)
        if name == "VALUEWHEN":
            if x and not np.isnan(args[1]):
                self.last = args[1]
            return self.last
        raise NotImplementedError(name)


class SourceState:
    """A fixed composite's finite call sites retain isolated operator states."""

    def __init__(self, name, params, columns):
        self.name = name
        self.params = {key.upper(): value for key, value in params.items()}
        self.columns = columns
        self.nodes = {}
        self.position = 0
        namespace = {"np": np}
        for key in BASIC:
            namespace[key] = ELEMENT[key] if key in ELEMENT else partial(self.op, key)
        for key in SUPPORTED:
            if key not in BASIC:
                original = getattr(reference, key)
                namespace[key] = FunctionType(
                    original.__code__, namespace, key, original.__defaults__
                )
        self.fn = namespace[name]
        # Fixed scalar parameters are bound once; no inspect/binding work in
        # the per-bar primitive path (measured hot-path overhead).
        self.positional_params = tuple(
            self.params[key]
            for key in list(inspect.signature(getattr(reference, name)).parameters)[len(columns) :]
        )

    def op(self, name, *args, **kwargs):
        if kwargs:
            signature = inspect.signature(getattr(reference, name))
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            args = tuple(bound.arguments.values())
        key = self.position
        self.position += 1
        if key not in self.nodes:
            self.nodes[key] = _Operator(name)
        elif self.nodes[key].name != name:
            raise ValueError("dynamic formula topology is not qualified")
        return self.nodes[key].update(name, *args)

    def _checkpoint(self):
        from copy import deepcopy
        return deepcopy(self.nodes),self.position

    def _restore_checkpoint(self, checkpoint):
        from copy import deepcopy
        self.nodes,self.position=deepcopy(checkpoint[0]),checkpoint[1]

    def update(self, *values):
        self.position = 0
        args = tuple(np.float64(x) for x in values)
        if any(np.isinf(x) for x in args):
            raise ValueError("infinite MyTT input")
        if self.name in _CONDITIONS and args[0] not in (0, 1):
            raise ValueError("condition requires nonmissing booleans")
        with np.errstate(all="ignore"):
            result = self.fn(*args, *self.positional_params)
        if isinstance(result, tuple):
            return tuple(float(x) for x in result)
        return float(result)

    def close(self):
        self.fn = None
        self.nodes.clear()


class _Depth:
    def __init__(self, period=1):
        self.period = period

    def combine(self, other):
        return _Depth(max(self.period, getattr(other, "period", 1)))

    __add__ = __sub__ = __mul__ = __truediv__ = __pow__ = __and__ = __or__ = combine
    __radd__ = __rsub__ = __rmul__ = __rtruediv__ = __rpow__ = combine
    __lt__ = __gt__ = __le__ = __ge__ = combine

    def __neg__(self):
        return self

    def __abs__(self):
        return self


def _validate(name, params):
    params = {key.upper(): value for key, value in params.items()}
    for key, value in params.items():
        if name == "POW" and key == "N":
            continue
        if name == "BOLL" and key == "P":
            continue
        if name == "SMA" and key == "M":
            continue
        if key == "D":
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("rounding D must be an integer")
            continue
        allow_zero = (
            (name in {"SUM", "COUNT", "FILTER", "REF", "DIFF"} and key == "N")
            or (name == "LAST" and key == "B")
            or (name == "DPO" and key == "M2")
        )
        if (
            isinstance(value, bool)
            or not np.isfinite(value)
            or int(value) != value
            or value < (0 if allow_zero else 1)
        ):
            raise ValueError(f"{name}.{key} requires a qualified integer period")
    if name == "SMA" and not 0 < params["M"] <= params["N"]:
        raise ValueError("SMA requires 0<M<=N")
    if name == "LAST" and not params["A"] > params["B"]:
        raise ValueError("LAST requires A>B>=0")
    if name in {"SLOPE", "FORCAST"} and params["N"] < 2:
        raise ValueError("regression period must be >=2")


def _warmup(name, columns, params):
    _validate(name, params)
    namespace = {}

    def depth(kind, *args, **kwargs):
        if kwargs:
            bound = inspect.signature(getattr(reference, kind)).bind(*args, **kwargs)
            bound.apply_defaults()
            args = tuple(bound.arguments.values())
        period = max([1] + [x.period for x in args if isinstance(x, _Depth)])
        if kind in {"REF", "DIFF"}:
            period += int(args[1] if len(args) > 1 else 1)
        elif kind in {
            "MA",
            "STD",
            "SUM",
            "COUNT",
            "HHV",
            "LLV",
            "HHVBARS",
            "LLVBARS",
            "WMA",
            "AVEDEV",
            "SLOPE",
            "FORCAST",
            "EVERY",
            "EXIST",
            "BARSSINCEN",
        }:
            period += max(0, int(args[1]) - 1)
        elif kind == "LAST":
            period += int(args[1])
        elif kind == "LONGCROSS":
            period += int(args[2])
        elif kind == "DMA":
            period = getattr(args[0], "period", 1)
        return _Depth(period)

    for key in BASIC:
        namespace[key] = partial(depth, key)
    for key in SUPPORTED:
        if key not in BASIC:
            original = getattr(reference, key)
            namespace[key] = FunctionType(original.__code__, namespace, key, original.__defaults__)
    result = namespace[name](
        *(_Depth() for _ in columns), **{k.upper(): v for k, v in params.items()}
    )
    return max(x.period for x in result) if isinstance(result, tuple) else result.period


_OUTPUTS = {
    "MACD": ("DIF", "DEA", "MACD"),
    "KDJ": ("K", "D", "J"),
    "WR": ("WR", "WR1"),
    "BIAS": ("BIAS1", "BIAS2", "BIAS3"),
    "BOLL": ("UPPER", "MID", "LOWER"),
    "PSY": ("PSY", "PSYMA"),
    "DMI": ("PDI", "MDI", "ADX", "ADXR"),
    "TRIX": ("TRIX", "TRMA"),
    "MTM": ("MTM", "MTMMA"),
    "ROC": ("ROC", "MAROC"),
    "TAQ": ("UP", "MID", "DOWN"),
    "KTN": ("UPPER", "MID", "LOWER"),
    "EMV": ("EMV", "MAEMV"),
    "DPO": ("DPO", "MADPO"),
    "BRAR": ("AR", "BR"),
    "DFMA": ("DIF", "DIFMA"),
    "MASS": ("MASS", "MA_MASS"),
    "EXPMA": ("EXPMA1", "EXPMA2"),
    "ASI": ("ASI", "ASIT"),
    "XSII": ("TD1", "TD2", "TD3", "TD4"),
}


def make_indicator(name, **overrides):
    name = name.upper()
    if name not in SUPPORTED:
        raise NotImplementedError(f"{name} is not qualified")
    fn = getattr(reference, name)
    signature = inspect.signature(fn)
    inputs = dict(_INPUT)
    if name in _CONDITIONS:
        inputs["S"] = "condition"
    if name == "RD":
        inputs["N"] = "close"
    if name == "IF":
        inputs.update(A="true_value", B="false_value")
    if name == "DMA":
        inputs["A"] = "alpha"
    if name == "BETWEEN":
        inputs.update(A="lower", B="upper")
    columns = []
    params = {}
    for key, param in signature.parameters.items():
        if key in inputs:
            columns.append(inputs[key])
        else:
            params[key] = (
                param.default if param.default is not inspect.Parameter.empty else _DEFAULT[key]
            )
    if {k.upper() for k in overrides} - params.keys():
        raise TypeError("unknown MyTT parameters")
    params.update({k.upper(): v for k, v in overrides.items()})
    _validate(name, params)
    columns = tuple(columns)
    outputs = _OUTPUTS.get(name, (name,))
    public_signature = inspect.Signature(
        [
            *(inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD) for key in columns),
            *(
                inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=value)
                for key, value in params.items()
            ),
            inspect.Parameter("aliases", inspect.Parameter.VAR_KEYWORD),
        ]
    )

    def batch(*args, **kwargs):
        bound = public_signature.bind(*args, **kwargs)
        options = {
            **params,
            **{
                key: value
                for key, value in bound.arguments.items()
                if key not in columns and key != "aliases"
            },
        }
        options.update(
            {key.upper(): value for key, value in bound.arguments.get("aliases", {}).items()}
        )
        if options.keys() != params.keys():
            raise TypeError("unknown MyTT parameters")
        _validate(name, options)
        series = [pd.Series(bound.arguments[key]) for key in columns]
        if any(not value.index.equals(series[0].index) for value in series):
            raise ValueError("MyTT input indexes must exactly match")
        if name in _CONDITIONS and not series[0].isin([0, 1]).all():
            raise ValueError("condition requires nonmissing booleans")
        if not len(series[0]):
            return (
                pd.DataFrame(index=series[0].index, columns=outputs, dtype=float)
                if len(outputs) > 1
                else pd.Series(index=series[0].index, name=name, dtype=float)
            )
        result = fn(*(value.to_numpy(dtype=float, copy=True) for value in series), **options)
        if isinstance(result, tuple):
            return pd.DataFrame(dict(zip(outputs, result, strict=True)), index=series[0].index)
        label = name
        if name in {"MA", "EMA", "WMA", "RSI", "CCI", "ATR", "CR", "MFI"}:
            label = f"{name}_{options['N']}"
        if name == "SMA":
            label = f"SMA_{options['N']}_{options['M']}"
        if name == "VR":
            label = f"VR_{options['M1']}"
        return pd.Series(result, index=series[0].index, name=label)

    batch.__signature__ = public_signature
    result = FunctionIndicator(
        f"tdx.{name}",
        batch,
        params,
        columns,
        stream_factory=partial(SourceState, name, columns=columns),
        warmup=partial(_warmup, name, columns),
    )
    result.output_names = outputs
    return result
