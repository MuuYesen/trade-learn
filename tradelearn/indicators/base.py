"""Indicator protocol and function wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd


class Indicator(Protocol):
    """Protocol implemented by function-style indicators."""

    name: str
    params: dict[str, Any]

    def compute(self, *args: Any, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Compute an indicator over a batch of data."""
        ...

    def on_bar(self, bar: Any) -> float | tuple[Any, ...]:
        """Update an indicator from a single streaming bar."""
        ...


class FunctionIndicator:
    """Callable indicator wrapper with batch and streaming entry points.

    For streaming use (on_bar), pass ``bar_columns`` to declare which OHLCV
    fields the function needs.  Each call to ``on_bar`` appends to an internal
    buffer and recomputes over the full history — correct but O(n).

    Note: ``FunctionIndicator`` instances at module level (e.g. ``ta.sma``) are
    singletons.  If two independent strategies share the same instance and both
    call ``on_bar``, they will share state.  Create a fresh instance per strategy
    when that matters.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., pd.Series | pd.DataFrame],
        params: dict[str, Any] | None = None,
        bar_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.params = params or {}
        self._func = func
        self._bar_columns = bar_columns
        self._bar_buffers: dict[str, list[float]] | None = None
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

    def on_bar(self, bar: Any) -> float | tuple[Any, ...]:
        """Update from a single bar and return the current indicator value.

        ``bar`` may be a ``StreamBar`` dataclass or any mapping with the
        required column keys.  Recomputes over the full buffer each call.
        """
        if self._bar_columns is None:
            raise NotImplementedError(f"{self.name} streaming on_bar is not implemented")
        if self._bar_buffers is None:
            self._bar_buffers = {col: [] for col in self._bar_columns}
        for col in self._bar_columns:
            if hasattr(bar, col):
                val = float(getattr(bar, col))
            else:
                val = float(bar[col])
            self._bar_buffers[col].append(val)
        series_args = [pd.Series(self._bar_buffers[col]) for col in self._bar_columns]
        result = self.compute(*series_args, **self.params)
        if result is None:
            return float("nan")
        if isinstance(result, pd.DataFrame):
            last = result.iloc[-1]
            return tuple(float(v) if v == v else float("nan") for v in last)
        val = result.iloc[-1]
        return float(val) if val == val else float("nan")

    def reset(self) -> None:
        """Clear the streaming buffer (useful between strategy runs)."""
        self._bar_buffers = None


def _line_owner(args: tuple[Any, ...]) -> Any | None:
    for arg in args:
        if hasattr(arg, "to_series") and hasattr(arg, "wrap_indicator"):
            return arg
    return None


def _to_indicator_arg(arg: Any) -> Any:
    if hasattr(arg, "to_series"):
        return arg.to_series()
    return arg
