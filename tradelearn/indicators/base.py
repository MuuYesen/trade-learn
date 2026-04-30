"""Indicator protocol and function wrapper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


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

    def clear(self) -> None:
        """Clear all streaming buffers."""
        self.buffers.clear()
        self.state.clear()


class FunctionIndicator:
    """Callable indicator wrapper with batch and streaming entry points.

    For streaming use (on_bar), pass ``bar_columns`` to declare which OHLCV
    fields the function needs.  Each call to ``on_bar`` appends to a streaming
    state buffer and recomputes over the full history — correct but O(n).

    Module-level instances such as ``ta.sma`` are shared.  Use ``new_state()``
    and pass that state to ``on_bar`` when a streaming indicator is used by a
    long-lived strategy or event loop.
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
    ) -> None:
        self.name = name
        self.params = params or {}
        self._func = func
        self._bar_columns = bar_columns
        self._stream_func = stream_func
        self._bar_buffers: dict[str, list[float]] | None = None
        self._default_state: dict[str, Any] = {}
        self.__name__ = name
        self.__doc__ = func.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute the indicator over a batch of data."""
        owner = _line_owner(args)
        if owner is None:
            return self.compute(*args, **kwargs)
        resolved_args = tuple(_to_indicator_arg(arg) for arg in args)
        result = self.compute(*resolved_args, **kwargs)
        return owner.wrap_indicator(result, name=self.name)

    def compute(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute the indicator over a batch of data."""
        return self._func(*args, **kwargs)

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
        """Update from a single bar and return the current indicator value.

        ``bar`` may be a ``StreamBar`` dataclass or any mapping with the
        required column keys.  Recomputes over the full buffer each call.
        """
        if self._bar_columns is None:
            raise NotImplementedError(f"{self.name} streaming on_bar is not implemented")
        buffers = self._buffers_for(state)
        for col in self._bar_columns:
            if hasattr(bar, col):
                val = float(getattr(bar, col))
            else:
                val = float(bar[col])
            buffers[col].append(val)
        if self._stream_func is not None:
            stream_state = state.state if state is not None else self._default_state
            return self._stream_func(buffers, stream_state, self.params)
        series_args = [pd.Series(buffers[col]) for col in self._bar_columns]
        result = self.compute(*series_args, **self.params)
        if result is None:
            return float("nan")
        if isinstance(result, pd.DataFrame):
            last = result.iloc[-1]
            return tuple(float(v) if v == v else float("nan") for v in last)
        val = result.iloc[-1]
        return float(val) if val == val else float("nan")

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


def _line_owner(args: tuple[Any, ...]) -> Any | None:
    for arg in args:
        if hasattr(arg, "to_series") and hasattr(arg, "wrap_indicator"):
            return arg
    return None


def _to_indicator_arg(arg: Any) -> Any:
    if hasattr(arg, "to_series"):
        return arg.to_series()
    return arg
