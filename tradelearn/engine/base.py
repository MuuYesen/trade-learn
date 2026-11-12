from __future__ import annotations

import inspect
from typing import Any
from tradelearn.backtest.parameters import Params, collect_param_defaults, split_param_kwargs

from tradelearn.backtest.lines import DelayedLine, IndicatorLine, Lines, LineSeries, line_names
from tradelearn.backtest.models import _notify_order
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.core.strategy_context import (
    _G,
    engine_context,
    set_current_data,
    set_current_datas,
    set_current_strategy,
)

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
    "TimeFrame",
    "_G",
    "_notify_order",
    "collect_param_defaults",
    "engine_context",
    "set_current_data",
    "set_current_datas",
    "set_current_strategy",
    "split_param_kwargs",
]


class TimeFrame:
    (Ticks, MicroSeconds, Seconds, Minutes, Days, Weeks, Months, Years, NoTimeFrame) = range(1, 10)
    Names = ["", "Ticks", "MicroSeconds", "Seconds", "Minutes", "Days", "Weeks", "Months", "Years", "NoTimeFrame"]
    names = Names

    @classmethod
    def getname(cls, tframe, compression=None):
        name = cls.Names[tframe]
        return name if compression > 1 or tframe == cls.NoTimeFrame else name[:-1]

    @classmethod
    def TFrame(cls, name):
        return getattr(cls, name)

    @classmethod
    def TName(cls, tframe):
        return cls.Names[tframe]


class BaseBroker:
    def __init__(self, **kwargs):
        pass

    def setcash(self, cash: float):
        pass

    def setcommission(self, commission: float):
        pass

    def getcash(self) -> float:
        return 0.0

    def getvalue(self) -> float:
        return 0.0

    def get_cash(self) -> float:
        return self.getcash()

    def get_value(self) -> float:
        return self.getvalue()


class BaseSizer:
    pass


class BaseAnalyzer:
    def __init__(self, **kwargs):
        self.strategy = None

    def _on_order(self, order: Any):
        pass

    def _on_bar(self, bar: Any) -> None:
        pass

    def _on_start(self) -> None:
        pass

    def _on_fill(self, fill: Any) -> None:
        pass

    def _on_trade(self, trade: Any) -> None:
        pass

    def stop(self):
        pass

    def _on_end(self, stats: Any) -> None:
        pass

    def get_analysis(self) -> dict[str, Any]:
        return {}



class BaseObserver:
    """Marker for components whose owner is bound before user construction."""


def _bind_component_data_aliases(component):
    for index, data in enumerate(component.datas):
        aliases = data.lines.getlinealiases()
        if not aliases:
            aliases = ("close", "low", "high", "open", "volume", "openinterest", "datetime")
        for number, name in enumerate(aliases):
            line = getattr(data.lines, name, None)
            if line is None:
                continue
            setattr(component, f"data{index}_{name}", line)
            setattr(component, f"data{index}_{number}", line)
            if index == 0:
                setattr(component, f"data_{name}", line)
                setattr(component, f"data_{number}", line)


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
        datas = ([] if isinstance(instance, BaseAnalyzer) else
                 [arg for arg in args if hasattr(arg, 'lines') or hasattr(arg, '_values')])
        if not datas and _G.current_datas:
            datas = list(_G.current_datas)
        elif not datas and _G.current_data is not None:
            datas = [_G.current_data]
            
        if datas:
            instance.datas = datas
            instance.data = datas[0]
            for i, d in enumerate(datas):
                setattr(instance, f'data{i}', d)
        else:
            instance.datas = []
            instance.data = None

        runtime = getattr(datas[0], "runtime", None) if datas else None
        if runtime is not None and hasattr(runtime, "broker"):
            instance.broker = runtime.broker

        component_context = (getattr(instance, '_initialize_component_context', None)
                             if isinstance(instance, (BaseAnalyzer, BaseObserver)) else None)
        if callable(component_context):
            component_context(_G.current_strategy)

        # 5. Setup Hook (Data and aliasing)
        if not isinstance(instance, CoreStrategy) and hasattr(instance, '_setup'):
            instance._setup()
            
        # 5.5 Register Indicator to Strategy
        # 5.5 Register Indicator to Strategy
        if (_G.current_strategy is not None and not isinstance(instance, CoreStrategy)
                and not callable(component_context)):
             if hasattr(_G.current_strategy, '_register_indicator'):
                 _G.current_strategy._register_indicator(instance)

        # 6. User Init with context management
        prev_strat = _G.current_strategy
        
        context_kwargs = {}
        if not prev_strat and hasattr(instance, 'next'): # Likely a strategy
            context_kwargs["strategy"] = instance
            if hasattr(instance, 'data'):
                context_kwargs["data"] = instance.data
            if getattr(instance, "datas", None) is not None:
                context_kwargs["datas"] = instance.datas

        with engine_context(**context_kwargs):
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:] # Skip 'self'
            
            has_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
            if has_var_args:
                instance.__init__(*args, **other_kwargs)
            else:
                pos_params = [
                    p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                instance.__init__(*args[:len(pos_params)], **other_kwargs)
            
        if (prev_strat is not None and not isinstance(instance, prev_strat.__class__)
                and not callable(component_context)):
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
        declared_lines = line_names(type(self))
        if isinstance(declared_lines, (list, tuple)):
            for name in declared_lines:
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
                if line:
                    line._advance(i)

    def __getattr__(self, name: str) -> Any:
        if name != 'lines' and hasattr(self, 'lines'):
            line = getattr(self.lines, name, None)
            if line is not None:
                return line
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
