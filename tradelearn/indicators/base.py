"""Indicator protocol and function wrapper."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from tradelearn.indicators.streaming import _same


class Indicator(Protocol):
    """Protocol implemented by function-style indicators."""

    name: str
    params: dict[str, Any]

    def compute(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute an indicator over a batch of data."""
        ...

    def on_bar(self, bar: Any, *, state: IndicatorState | None = None) -> float | tuple[Any, ...]:
        """Update an indicator from a single streaming bar."""
        ...


@dataclass
class IndicatorState:
    """Per-strategy streaming buffers for a function indicator."""

    buffers: dict[str, list[float]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    owner: Any = None
    params: dict[str, Any] | None = None
    failed: bool = False

    def clear(self) -> None:
        """Clear all streaming buffers."""
        backend = self.state.pop("backend", None)
        try:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        finally:
            self.buffers.clear()
            self.state.clear()
            self.owner = None
            self.params = None
            self.failed = False


class FunctionIndicator:
    """Callable indicator wrapper with batch and streaming entry points.

    Streaming requires an explicit persistent implementation. Batch-only
    functions never silently recompute history. Use one new_state() per
    strategy/feed/symbol/timeframe; reset is required after parameter changes
    or a backend exception. Input buffers have a declared finite capacity.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., pd.Series | pd.DataFrame],
        params: dict[str, Any] | None = None,
        bar_columns: tuple[str, ...] | None = None,
        stream_func: Callable[
            [dict[str, list[float]], dict[str, Any], dict[str, Any]], float | tuple[Any, ...]
        ]
        | None = None,
        *,
        stream_factory: Callable[[dict[str, Any]], Any] | None = None,
        history_capacity: int = 1,
        warmup: int | Callable[[dict[str, Any]], int] = 1,
    ) -> None:
        if not isinstance(history_capacity, int) or history_capacity < 1:
            raise ValueError("history_capacity must be a positive integer")
        self._warmup = warmup
        self._history_capacity = history_capacity
        self._stream_factory = stream_factory
        self._isolated_default = IndicatorState()
        self.name = name
        self.params = deepcopy(params or {})
        self._func = func
        self._bar_columns = bar_columns
        self._stream_func = stream_func
        self._bar_buffers: dict[str, list[float]] | None = None
        self._default_state: dict[str, Any] = {}
        self.__name__ = name
        self.__doc__ = func.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute the indicator over a batch of data."""
        args = tuple(getattr(arg, "_indicator_line", arg) for arg in args)
        kwargs = {key: getattr(value, "_indicator_line", value) for key, value in kwargs.items()}
        owner = _line_owner((*args, *kwargs.values()))
        if owner is None:
            return self.compute(*args, **kwargs)
        runtime_function = getattr(owner, "_runtime_function", None)
        if callable(runtime_function):
            return runtime_function(self, args, kwargs)
        resolved_args = tuple(_to_indicator_arg(arg) for arg in args)
        result = self.compute(
            *resolved_args, **{k: _to_indicator_arg(v) for k, v in kwargs.items()}
        )
        return owner.wrap_indicator(result, name=self.name)

    def compute(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute the indicator over a batch of data."""
        bound = inspect.signature(self._func).bind_partial(*args, **kwargs)
        explicit = set(bound.arguments) | set(kwargs)
        if self.name.startswith("tdx."):
            explicit = {key.lower() for key in explicit}
        defaults = {
            key: value
            for key, value in self.params.items()
            if (key.lower() if self.name.startswith("tdx.") else key) not in explicit
        }
        return self._func(*args, **{**defaults, **kwargs})

    @property
    def minimum_period(self) -> int:
        """Minimum seed count under this exact parameter configuration."""
        value = self._warmup(self.params) if callable(self._warmup) else self._warmup
        if not isinstance(value, int) or value < 1:
            raise ValueError("warmup must be a positive integer")
        return value

    @property
    def supports_streaming(self) -> bool:
        """Whether this object has an explicit persistent implementation."""
        return self._bar_columns is not None and (
            self._stream_func is not None or self._stream_factory is not None
        )

    def configured(self, **params: Any) -> FunctionIndicator:
        """Return an independently configured adapter without mutating a singleton."""
        if self.name.startswith("tdx."):
            aliases = {key.lower(): key for key in self.params}
            normalized = {}
            for key, value in params.items():
                target = aliases.get(key.lower(), key)
                if target in normalized and normalized[target] != value:
                    raise ValueError("conflicting TDX parameter aliases")
                normalized[target] = value
            params = normalized
        unknown = params.keys() - self.params.keys()
        if unknown:
            raise TypeError(f"unknown {self.name} parameters: {sorted(unknown)}")
        result = FunctionIndicator(
            self.name,
            self._func,
            {**self.params, **params},
            self._bar_columns,
            self._stream_func,
            stream_factory=self._stream_factory,
            history_capacity=self._history_capacity,
            warmup=self._warmup,
        )
        if hasattr(self, "output_names"):
            result.output_names = self.output_names
        if hasattr(self, "output_resolver"):
            result.output_resolver = self.output_resolver
            result.output_names = self.output_resolver(result.params)
        return result

    def bind_stream(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> StreamBinding:
        """Bind original call inputs/parameters once for the shared Lines owner.

        The caller retains the returned binding and advances update(*scalars)
        once per accepted input event. Accessing its value does not advance.
        """
        if not self.supports_streaming:
            raise NotImplementedError(f"{self.name} has no persistent streaming implementation")
        signature = inspect.signature(self._func)
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()):
            names = self._bar_columns
            inputs = tuple(args[: len(names)])
            if len(inputs) < len(names):
                inputs += tuple(kwargs[key] for key in names[len(inputs) :])
            params = {key: value for key, value in kwargs.items() if key not in names}
            if len(args) > len(names):
                raise TypeError("stream parameters must be keyword arguments")
        else:
            bound = signature.bind(*args, **kwargs)
            items = list(signature.parameters)[: len(self._bar_columns)]
            inputs = tuple(bound.arguments[key] for key in items)
            params = {}
            for key, value in bound.arguments.items():
                if key in items:
                    continue
                if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                    params.update(value)
                else:
                    params[key] = value
        return StreamBinding(self.configured(**params), inputs)

    def new_state(self) -> IndicatorState:
        """Create an isolated streaming state for this indicator."""
        if self._bar_columns is None:
            return IndicatorState()
        return IndicatorState({col: [] for col in self._bar_columns})

    def on_bar(
        self,
        bar: Any,
        *,
        state: IndicatorState | None = None,
    ) -> float | tuple[Any, ...]:
        """Advance an explicit persistent implementation exactly once."""
        if self._bar_columns is None or (
            self._stream_func is None and self._stream_factory is None
        ):
            raise NotImplementedError(f"{self.name} has no persistent streaming implementation")
        current = state if state is not None else self._isolated_default
        if current.failed:
            raise RuntimeError("indicator state failed; reset and replay required")
        if current.owner is not None and current.owner() is not self:
            raise ValueError("indicator state belongs to a different indicator")
        if current.params is not None and not _same(current.params, self.params):
            raise ValueError("indicator parameters changed; reset required")
        # Resolve every column before mutating any state.
        values = [
            float(getattr(bar, col) if hasattr(bar, col) else bar[col]) for col in self._bar_columns
        ]
        if current.owner is None:
            current.owner = weakref.ref(self)
            current.params = deepcopy(self.params)
        try:
            if self._stream_factory is not None:
                if "backend" not in current.state:
                    current.state["backend"] = self._stream_factory(deepcopy(self.params))
                return current.state["backend"].update(*values)
            buffers = self._buffers_for(current)
            for col, value in zip(self._bar_columns, values, strict=True):
                buffers[col].append(value)
                del buffers[col][: -self._history_capacity]
            return self._stream_func(buffers, current.state, self.params)
        except Exception:
            current.failed = True
            raise

    def reset(self, *, state: IndicatorState | None = None) -> None:
        """Clear streaming buffers.

        Passing a state clears only that isolated state.  Without a state, this
        clears the backwards-compatible default buffer used by direct
        ``on_bar`` calls.
        """
        if state is not None:
            state.clear()
            if self._bar_columns is not None:
                state.buffers.update({col: [] for col in self._bar_columns})
            return
        self._bar_buffers = None
        self._default_state = {}
        self._isolated_default.clear()

    def _buffers_for(self, state: IndicatorState | None) -> dict[str, list[float]]:
        if state is not None:
            if self._bar_columns is None:
                return state.buffers
            for col in self._bar_columns:
                state.buffers.setdefault(col, [])
            return state.buffers
        if self._bar_buffers is None:
            self._bar_buffers = {col: [] for col in self._bar_columns or ()}
        return self._bar_buffers


class StreamBinding:
    """Per-call input binding and isolated state shared by Engine and Lite."""

    def __init__(self, indicator: FunctionIndicator, inputs: tuple[Any, ...]):
        self.indicator, self.inputs = indicator, inputs
        self.state = indicator.new_state()
        self.value = None

    def update(self, *values: float) -> Any:
        if len(values) != len(self.indicator._bar_columns):
            raise ValueError("stream input arity mismatch")
        self.value = self.indicator.on_bar(
            dict(zip(self.indicator._bar_columns, values, strict=True)), state=self.state
        )
        return self.value

    def reset(self) -> None:
        self.indicator.reset(state=self.state)
        self.value = None


def _line_owner(args: tuple[Any, ...]) -> Any | None:
    for arg in args:
        if hasattr(arg, "to_series") and hasattr(arg, "wrap_indicator"):
            return arg
    return None


def _to_indicator_arg(arg: Any) -> Any:
    arg = getattr(arg, "_indicator_line", arg)
    if hasattr(arg, "to_series"):
        return arg.to_series()
    return arg
