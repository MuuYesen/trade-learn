from __future__ import annotations

import inspect
from typing import Any

from tradelearn.backtest.lines import DelayedLine, IndicatorLine, Lines, LineSeries
from tradelearn.backtest.models import (
    BaseAnalyzer,
    BaseBroker,
    BaseSizer,
    Order,
    Params,
)
from tradelearn.backtest.strategy import Strategy as CoreStrategy

__all__ = [
    "BaseAnalyzer",
    "BaseBroker",
    "BaseSizer",
    "DelayedLine",
    "IndicatorLine",
    "LineRoot",
    "LineSeries",
    "Lines",
    "MetaParams",
    "Params",
    "_G",
    "_notify_order",
    "collect_param_defaults",
    "set_current_data",
    "set_current_datas",
    "set_current_strategy",
    "split_param_kwargs",
]


# Centralized context to avoid import-time shadowing
class _GlobalContext:
    def __init__(self):
        self.current_data = None
        self.current_datas = []
        self.current_strategy = None

_G = _GlobalContext()

def set_current_data(data: Any) -> None:
    _G.current_data = data

def set_current_datas(datas: list[Any]) -> None:
    _G.current_datas = datas

def set_current_strategy(strategy: Any) -> None:
    _G.current_strategy = strategy


def collect_param_defaults(cls: type) -> tuple[list[str], list[tuple[str, Any]]]:
    names: list[str] = []
    defaults: list[tuple[str, Any]] = []
    for base_cls in reversed(cls.mro()):
        param_defaults = getattr(base_cls, "params", [])
        if isinstance(param_defaults, dict):
            names.extend(param_defaults.keys())
            defaults.extend(param_defaults.items())
        elif isinstance(param_defaults, (list, tuple)):
            for item in param_defaults:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    names.append(item[0])
                    defaults.append((item[0], item[1]))
    return names, defaults


def split_param_kwargs(cls: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    param_names, _defaults = collect_param_defaults(cls)
    param_name_set = set(param_names)
    param_kwargs = {}
    other_kwargs = {}
    for key, value in kwargs.items():
        if key in param_name_set:
            param_kwargs[key] = value
        else:
            other_kwargs[key] = value
    return param_kwargs, other_kwargs


class MetaParams(type):
    """Metaclass to handle Backtrader-style parameter stripping and lifecycle hooks."""
    def __call__(cls, *args, **kwargs):
        # 1. Collect all params from MRO
        p_kwargs, other_kwargs = split_param_kwargs(cls, kwargs)

        # 2. Instantiate
        instance = cls.__new__(cls)

        # 3. Base Init (Params and Lines)
        if hasattr(instance, '_base_init'):
            instance._base_init(**p_kwargs)

        # 4. Data Assignment
        datas = [arg for arg in args if hasattr(arg, 'lines') or hasattr(arg, '_values')]
        if not datas and _G.current_data is not None:
            datas = [_G.current_data]
            
        if datas:
            instance.datas = datas
            instance.data = datas[0]
            for i, d in enumerate(datas):
                setattr(instance, f'data{i}', d)
        else:
            print(f"DEBUG: {cls.__name__} has NO DATA. _G.current_data={_G.current_data}")

        # 5. Setup Hook (Data and aliasing)
        if hasattr(instance, '_setup'):
            instance._setup()
            
        # 5.5 Register Indicator to Strategy
        # 5.5 Register Indicator to Strategy
        if _G.current_strategy is not None and not isinstance(instance, CoreStrategy):
             if hasattr(_G.current_strategy, '_register_indicator'):
                 _G.current_strategy._register_indicator(instance)

        # 6. User Init with context management
        prev_strat = _G.current_strategy
        prev_data = _G.current_data
        
        try:
            if not prev_strat and hasattr(instance, 'next'): # Likely a strategy
                set_current_strategy(instance)
                if hasattr(instance, 'data'):
                    set_current_data(instance.data)

            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:] # Skip 'self'
            
            has_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
            if has_var_args:
                instance.__init__(*args, **other_kwargs)
            else:
                pos_params = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                instance.__init__(*args[:len(pos_params)], **other_kwargs)
        finally:
            set_current_strategy(prev_strat)
            set_current_data(prev_data)
            
        if prev_strat is not None and not isinstance(instance, prev_strat.__class__):
            if hasattr(instance, 'lines') and hasattr(prev_strat, '_register_indicator'):
                prev_strat._register_indicator(instance)

        return instance

class LineRoot(metaclass=MetaParams):
    """Base class for anything that has lines (DataFeeds, Indicators)."""
    def _base_init(self, **kwargs):
        _param_names, all_cls_params = collect_param_defaults(self.__class__)
            
        self.params = self.p = Params(all_cls_params, **kwargs)

        if not hasattr(self, 'lines') or not isinstance(self.lines, Lines):
            self.lines = Lines(self)
        self.l = self.lines
        
        # Initialize line placeholders if class defines 'lines'
        line_names = getattr(self.__class__, 'lines', [])
        if isinstance(line_names, (list, tuple)):
            for name in line_names:
                if not hasattr(self.lines, name):
                    setattr(self.lines, name, None)

        # Initialize core strategy state if this is a strategy
        if hasattr(self, 'next') and not hasattr(self, '_positions'):
            self._sizers = {}
            self._sizer = None
            self._positions = {}
            self._pending_size = {}
            self._indicators = []
            self._manual_min_period = 0

    def _advance(self, i: int) -> None:
        if hasattr(self, 'lines'):
            for line in self.lines:
                if line: line._advance(i)

    def __getattr__(self, name: str) -> Any:
        if name != 'lines' and hasattr(self, 'lines'):
            line = getattr(self.lines, name, None)
            if line is not None: return line
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def _notify_order(strategy: Any, order: Order) -> None:
    strategy.notify_order(order)
