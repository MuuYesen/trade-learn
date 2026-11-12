"""Pinned PyneCore state-vector adapters (6.13.3).

Only explicit input/parameter graphs are adapted. No arbitrary Python
function is compiled. Global OHLCV fields are restored after each update.
"""

from __future__ import annotations

import inspect
from functools import partial
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import RLock

import pandas as pd

from tradelearn.indicators.base import FunctionIndicator

try:
    if version("pynesys-pynecore") != "6.13.3":
        raise ImportError("TV indicators require pynesys-pynecore==6.13.3")
except PackageNotFoundError as exc:
    raise ImportError("TV indicators require pynesys-pynecore==6.13.3") from exc
lib = import_module("pynecore.lib")
ta = import_module("pynecore.lib.ta")
_bind_target = import_module("pynecore.core.instance_state")._bind_target
NA = import_module("pynecore.types.na").NA

_LOCK = RLock()
# Explicit fixed upstream scope; private helpers are never exported.
EXCLUDED = {"rci": "PyneCore 6.13.3 NaN profile fails to complete; not qualified"}
SPECS = {name: {} for name in ta.__all__ if name not in EXCLUDED}
DEFAULTS = dict(
    length=14,
    mult=2.0,
    diLength=14,
    adxSmoothing=14,
    fastlen=12,
    slowlen=26,
    siglen=9,
    offset=0,
    percentage=50,
    leftbars=3,
    rightbars=3,
    type="Traditional",
    occurrence=0,
    factor=3.0,
    atrPeriod=10,
    short_length=13,
    long_length=25,
)
_FIELDS = (
    "bar_index",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "hl2",
    "hlc3",
    "ohlc4",
    "_last_close",
)
_INPUTS = {
    "source": "close",
    "series": "close",
    "source1": "close",
    "source2": "source2",
    "condition": "condition",
    "anchor": "anchor",
    "high": "high",
    "low": "low",
}
_OHLCV = {
    "accdist",
    "atr",
    "dmi",
    "iii",
    "kc",
    "kcw",
    "mfi",
    "nvi",
    "obv",
    "pivot_point_levels",
    "pvi",
    "pvt",
    "sar",
    "stoch",
    "supertrend",
    "tr",
    "vwap",
    "vwma",
    "wad",
    "wpr",
    "wvad",
}

_SCHEMAS = {
    "accdist": ("high", "low", "close", "volume"),
    "atr": ("high", "low", "close"),
    "dmi": ("high", "low", "close"),
    "iii": ("high", "low", "close", "volume"),
    "kc": ("high", "low", "close"),
    "kcw": ("high", "low", "close"),
    "mfi": ("high", "low", "close", "volume"),
    "nvi": ("close", "volume"),
    "obv": ("close", "volume"),
    "pivot_point_levels": ("open", "high", "low", "close", "anchor"),
    "pvi": ("close", "volume"),
    "pvt": ("close", "volume"),
    "sar": ("high", "low", "close"),
    "stoch": ("high", "low", "close"),
    "supertrend": ("high", "low", "close"),
    "tr": ("high", "low", "close"),
    "vwap": ("source", "volume", "anchor"),
    "vwma": ("close", "volume"),
    "wad": ("high", "low", "close"),
    "wpr": ("high", "low", "close"),
    "wvad": ("open", "high", "low", "close", "volume"),
}
_LENGTHS = dict(
    sma=20,
    ema=20,
    wma=20,
    hma=20,
    alma=9,
    stdev=20,
    variance=20,
    roc=10,
    mom=10,
    bb=20,
    bbw=20,
    vwma=20,
    correlation=20,
    kc=20,
    kcw=20,
)


def _scalar(value):
    """Convert upstream NA/scalar/tuple outputs to Python floating-point values."""
    if isinstance(value, (tuple, list)):
        return tuple(_scalar(x) for x in value)
    return float("nan") if isinstance(value, NA) else float(value)


class PyneState:
    """One persistent native/Python state-vector tree, without a global root."""

    def __init__(self, name, params, columns, arguments):
        if version("pynesys-pynecore") != "6.13.3":
            raise ImportError("TV persistent adapters require pynesys-pynecore==6.13.3")
        self.name = name
        self.fn = _bind_target(getattr(ta, name))
        self.params, self.columns, self.arguments = params, columns, arguments
        self.index = 0
        self.previous = float("nan")

    def update(self, *values):
        """Advance this isolated Pyne state once while restoring shared library fields
        afterwards.
        """
        inputs = dict(zip(self.columns, values, strict=True))
        with _LOCK:
            previous = {key: getattr(lib, key) for key in _FIELDS}
            try:
                close = inputs.get("close", float("nan"))
                high, low = inputs.get("high", close), inputs.get("low", close)
                op = inputs.get("open", close)
                lib.bar_index = self.index
                lib.open, lib.high, lib.low, lib.close = op, high, low, close
                lib.volume = inputs.get("volume", float("nan"))
                lib.hl2 = (high + low) / 2
                lib.hlc3 = (high + low + close) / 3
                lib.ohlc4 = (op + high + low + close) / 4
                lib._last_close = self.previous
                kwargs = dict(self.params)
                for argument, column in self.arguments.items():
                    value = inputs[column]
                    if argument in {"condition", "anchor"}:
                        if value not in (0, 1):
                            raise ValueError(f"{argument} requires nonmissing booleans")
                        kwargs[argument] = bool(value)
                    else:
                        kwargs[argument] = value
                if self.name in {"cci", "mfi"}:
                    kwargs["source" if self.name == "cci" else "series"] = (high + low + close) / 3
                result = _scalar(self.fn(**kwargs))
                self.previous = close
                self.index += 1
                return result
            finally:
                for key, value in previous.items():
                    setattr(lib, key, value)


def _warmup(name, params):
    """Compute the pinned function's declared observation requirement from its parameters."""
    length = int(params.get("length", 1))
    if name == "macd":
        return int(max(params["fastlen"], params["slowlen"]) + params["siglen"] - 1)
    if name in {"rsi", "cmo", "change", "mom", "roc", "percentrank"}:
        return length + 1
    if name == "hma":
        return length + int(length**0.5) - 1
    if name == "dmi":
        return int(params["diLength"] + params["adxSmoothing"])
    if name in {"kc", "kcw"}:
        return length + int(params.get("useTrueRange", True))
    if name == "tsi":
        return int(params["short_length"] + params["long_length"])
    if name == "swma":
        return 4
    if name in {"obv", "pvt", "sar", "pivot_point_levels"}:
        return 2
    if name == "tr":
        return 1 if params.get("handle_na", False) else 2
    if name in {"pivothigh", "pivotlow"}:
        return int(params["leftbars"] + params["rightbars"] + 1)
    if "length" in params and name not in {"rising", "falling"}:
        return length
    return 1


def make_indicator(name, **overrides):
    """Build an internal fixed-source adapter for qualification."""
    if name not in SPECS:
        raise NotImplementedError(f"{name}: TV persistent capability not qualified")
    signature = inspect.signature(getattr(ta, name))
    arguments = {key: _INPUTS[key] for key in signature.parameters if key in _INPUTS}
    params = {}
    for key, parameter in signature.parameters.items():
        if key.startswith("_") or key in arguments:
            continue
        if key in overrides:
            params[key] = overrides[key]
        elif parameter.default is not inspect.Parameter.empty:
            params[key] = parameter.default
        elif key in DEFAULTS:
            params[key] = _LENGTHS.get(name, DEFAULTS[key]) if key == "length" else DEFAULTS[key]
        else:
            raise ValueError(f"{name}: no qualified default for {key}")
    unknown = overrides.keys() - params.keys()
    if unknown:
        raise TypeError(f"unknown {name} parameters: {sorted(unknown)}")
    columns = list(dict.fromkeys(arguments.values()))
    if name in _SCHEMAS:
        columns = list(_SCHEMAS[name])
    if name == "vwap":
        arguments["source"] = "source"
    columns = tuple(columns)
    factory = partial(PyneState, name, columns=columns, arguments=arguments)

    def batch(*inputs, **kwargs):
        """Replay input rows through one persistent state with exact matching indexes."""
        configuration = {**params, **kwargs}
        if kwargs.keys() - params.keys():
            raise TypeError("unknown TV parameters")
        if len(inputs) != len(columns):
            raise TypeError(f"{name} expects {columns}")
        series = [pd.Series(x) for x in inputs]
        if any(not x.index.equals(series[0].index) for x in series):
            raise ValueError("TV input indexes must exactly match")
        state = factory(configuration)
        rows = [state.update(*row) for row in zip(*series, strict=True)]
        if rows and isinstance(rows[0], tuple):
            return pd.DataFrame(rows, index=series[0].index)
        return pd.Series(rows, index=series[0].index, name=name, dtype=float)

    output_names = {
        "bb": ("basis", "upper", "lower"),
        "kc": ("basis", "upper", "lower"),
        "macd": ("macd", "signal", "hist"),
        "dmi": ("plus_di", "minus_di", "adx"),
        "supertrend": ("supertrend", "direction"),
        "pivot_point_levels": tuple(f"level_{i}" for i in range(11)),
    }.get(name, (name,))
    batch.__signature__ = inspect.Signature(
        [
            *(
                inspect.Parameter(column, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for column in columns
            ),
            *(
                inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=value)
                for key, value in params.items()
            ),
        ]
    )
    original_batch = batch

    def configuration_outputs(options):
        """Merge explicit options with defaults for output-schema selection."""
        return {**params, **options}

    def named_batch(*args, **kwargs):
        """Bind the public signature and label multi-output columns for the selected
        configuration.
        """
        bound = original_batch.__signature__.bind(*args, **kwargs)
        inputs = tuple(bound.arguments[key] for key in columns)
        options = {key: value for key, value in bound.arguments.items() if key not in columns}
        result = original_batch(*inputs, **options)
        names = (
            ("vwap", "upper", "lower")
            if name == "vwap" and configuration_outputs(options).get("stdev_mult") is not None
            else output_names
        )
        if isinstance(result, pd.DataFrame):
            result.columns = names[: result.shape[1]]
        elif result.empty and len(names) > 1:
            result = pd.DataFrame(index=result.index, columns=names, dtype=float)
        return result

    named_batch.__signature__ = batch.__signature__
    result = FunctionIndicator(
        f"tv.{name}",
        named_batch,
        params,
        columns,
        stream_factory=factory,
        warmup=partial(_warmup, name),
    )
    result.output_names = output_names
    if name == "vwap":
        result.output_resolver = lambda p: (
            ("vwap", "upper", "lower") if p.get("stdev_mult") is not None else ("vwap",)
        )
        result.output_names = result.output_resolver(params)
    return result
