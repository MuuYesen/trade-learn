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
    """Callable indicator wrapper with batch and streaming entry points."""

    def __init__(
        self,
        name: str,
        func: Callable[..., pd.Series | pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Create a function-backed indicator."""
        self.name = name
        self.params = params or {}
        self._func = func
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
        """Raise until streaming mode is implemented."""
        raise NotImplementedError(f"{self.name} streaming on_bar is not implemented")


def _line_owner(args: tuple[Any, ...]) -> Any | None:
    for arg in args:
        if hasattr(arg, "to_series") and hasattr(arg, "wrap_indicator"):
            return arg
    return None


def _to_indicator_arg(arg: Any) -> Any:
    if hasattr(arg, "to_series"):
        return arg.to_series()
    return arg
