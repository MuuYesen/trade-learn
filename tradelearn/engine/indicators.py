from __future__ import annotations

import inspect
from typing import Any

from tradelearn.engine.base import Params, collect_param_defaults, split_param_kwargs


class MetaSimple(type):
    """Metaclass for Backtrader-style custom Indicator classes."""

    def __call__(cls, *args: Any, **kwargs: Any):
        _param_names, p_defaults = collect_param_defaults(cls)
        p_kwargs, other_kwargs = split_param_kwargs(cls, kwargs)

        instance = cls.__new__(cls, *args, **other_kwargs)
        instance.p = instance.params = Params(p_defaults, **p_kwargs)

        from .base import _G

        data = None
        for arg in args:
            if hasattr(arg, "lines") or hasattr(arg, "_values"):
                data = arg
                break

        instance.data = data if data is not None else other_kwargs.get("data", _G.current_data)
        instance.datas = [arg for arg in args if hasattr(arg, "lines") or hasattr(arg, "_values")]
        for idx, item in enumerate(instance.datas):
            setattr(instance, f"data{idx}", item)
        instance.l = instance.lines = Params({line: None for line in instance.lines_def})

        sig = inspect.signature(cls.__init__)
        real_params = list(sig.parameters.values())[1:]
        takes_var_args = any(param.kind == param.VAR_POSITIONAL for param in real_params)
        fixed_param_count = len(
            [
                param
                for param in real_params
                if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY)
            ]
        )
        if not takes_var_args and fixed_param_count == 0 and args:
            instance.__init__(**other_kwargs)
        else:
            instance.__init__(*args, **other_kwargs)

        if _G.current_strategy is not None:
            _G.current_strategy._register_indicator(instance)

        return instance


class Indicator(metaclass=MetaSimple):
    """Base class for Engine custom indicators.

    Built-in indicators are intentionally not exposed from ``tradelearn.engine``.
    Use ``bt.talib`` / ``bt.tdx`` / ``bt.tv`` for built-in vector indicators, or
    subclass this class for strategy-specific custom indicators.
    """

    lines = ()
    params = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def min_period(self) -> int:
        period = 0
        for name in self.lines_def:
            line = getattr(self.lines, name)
            if hasattr(line, "min_period"):
                period = max(period, line.min_period)
        return period

    def _advance(self, i: int) -> None:
        for line_name in self.lines_def:
            line = getattr(self.lines, line_name)
            if hasattr(line, "_advance"):
                line._advance(i)

    @property
    def lines_def(self) -> tuple[str, ...]:
        cls_lines = getattr(type(self), "lines", ())
        return cls_lines if isinstance(cls_lines, tuple) else tuple(cls_lines)

    def __getattr__(self, name: str) -> Any:
        if name in getattr(self, "lines_def", ()):
            return getattr(self.lines, name)
        if name == "l":
            return self.lines
        if name == "p":
            return self.params
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __bool__(self) -> bool:
        return bool(self.lines[0])

    def to_series(self):
        return self.lines[0].to_series()

    def wrap_indicator(self, values: Any, name: str | None = None):
        return self.lines[0].wrap_indicator(values, name=name)

    def __getitem__(self, index: int):
        return self.lines[0][index]

    def __call__(self, ago: int = 0):
        return self.lines[0](ago)

    def __len__(self) -> int:
        return len(self.lines[0])

    def __mul__(self, other: Any):
        return self.lines[0] * other

    def __truediv__(self, other: Any):
        return self.lines[0] / other

    def __add__(self, other: Any):
        return self.lines[0] + other

    def __sub__(self, other: Any):
        return self.lines[0] - other

    def __lt__(self, other: Any):
        return self.lines[0] < other

    def __gt__(self, other: Any):
        return self.lines[0] > other


__all__ = ["Indicator"]
