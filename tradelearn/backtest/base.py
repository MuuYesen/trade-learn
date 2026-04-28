"""Base classes and shared state for the TradeLearn backtesting engine."""

from __future__ import annotations
from typing import Any, List, Optional, Type, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tradelearn.backtest.engine import Order, Position, Strategy

# Global context for Backtrader-style implicit data resolution
_CURRENT_DATA = None
_CURRENT_DATAS = []
_CURRENT_STRATEGY = None

def set_current_data(data: Any) -> None:
    global _CURRENT_DATA
    _CURRENT_DATA = data

def set_current_datas(datas: List[Any]) -> None:
    global _CURRENT_DATAS
    _CURRENT_DATAS = datas

def set_current_strategy(strategy: Any) -> None:
    global _CURRENT_STRATEGY
    _CURRENT_STRATEGY = strategy

import inspect

class MetaParams(type):
    """Metaclass to handle Backtrader-style parameter stripping and lifecycle hooks."""
    def __call__(cls, *args, **kwargs):
        # 1. Separate params from other kwargs
        p_defaults = getattr(cls, 'params', [])
        p_names = []
        if isinstance(p_defaults, dict):
            p_names = list(p_defaults.keys())
        elif isinstance(p_defaults, (list, tuple)):
            p_names = [x[0] for x in p_defaults if isinstance(x, (list, tuple))]

        p_kwargs = {}
        other_kwargs = {}
        for k, v in kwargs.items():
            if k in p_names:
                p_kwargs[k] = v
            else:
                other_kwargs[k] = v

        # 2. Instantiate
        instance = cls.__new__(cls)

        # 3. Base Init (Setup lines, params)
        if hasattr(instance, '_base_init'):
            instance._base_init(**p_kwargs)

        # 4. Separate datas from args for implicit assignment
        datas = [arg for arg in args if hasattr(arg, 'lines') or hasattr(arg, '_values')]
        if datas:
            if not hasattr(instance, 'datas'): instance.datas = datas
            if not hasattr(instance, 'data'): instance.data = datas[0]
            for i, d in enumerate(datas):
                setattr(instance, f'data{i}', d)

        # 5. Strategy Setup (Specific to Strategy)
        if hasattr(instance, '_setup'):
            instance._setup()

        # 6. User Init - call with appropriate number of positional args
        # Check signature of cls.__init__
        try:
            # Ensure _CURRENT_STRATEGY is set during user __init__
            from tradelearn.backtest import base
            prev_strat = base._CURRENT_STRATEGY
            if not prev_strat and hasattr(instance, 'next'): # Likely a strategy
                base.set_current_strategy(instance)

            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())
            # Skip 'self'
            params = params[1:]
            
            has_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
            if has_var_args:
                instance.__init__(*args, **other_kwargs)
            else:
                pos_params = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                # Pass as many args as there are positional parameters
                instance.__init__(*args[:len(pos_params)], **other_kwargs)
        except (ValueError, TypeError):
            # Fallback if signature can't be determined
            instance.__init__(*args, **other_kwargs)
        finally:
            if 'prev_strat' in locals() and prev_strat:
                set_current_strategy(prev_strat)
            
        # 7. Register with current strategy if this is an indicator
        if prev_strat and not isinstance(instance, prev_strat.__class__):
            if cls.__name__ not in ('Strategy', 'DataFeed', 'Cerebro'):
                prev_strat._register_indicator(instance)

        return instance

class LineRoot(metaclass=MetaParams):
    """Base class for anything that has lines (DataFeeds, Indicators)."""
    def __init__(self, *args, **kwargs) -> None:
        # Metaclass already called _base_init, but we keep this for direct instantiation if any
        pass

    def _base_init(self, **kwargs):
        # Initialize params: merge class defaults with instance kwargs
        cls_params = getattr(self.__class__, 'params', [])
        self.params = self.p = Params(cls_params, **kwargs)

        # Always create instance-level Lines container to shadow class-level tuple
        if not hasattr(self, 'lines') or not isinstance(self.lines, Lines):
            self.lines = Lines(self)
        self.l = self.lines
        
        # Initialize line placeholders if class defines 'lines'
        line_names = getattr(self.__class__, 'lines', [])
        if isinstance(line_names, (list, tuple)):
            for name in line_names:
                if not hasattr(self.lines, name):
                    setattr(self.lines, name, None)

    def _advance(self, i: int) -> None:
        if hasattr(self, 'lines'):
            for line in self.lines:
                if line: line._advance(i)

    def __getattr__(self, name: str) -> Any:
        if name != 'lines' and hasattr(self, 'lines'):
            line = getattr(self.lines, name, None)
            if line is not None: return line
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class Lines:
    """Container for line series with ordered access."""
    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._lines = []
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'): 
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value
            # Ensure _lines reflects the order defined in the class attribute 'lines'
            cls_lines = getattr(self._owner.__class__, 'lines', [])
            if isinstance(cls_lines, (list, tuple)) and name in cls_lines:
                idx = list(cls_lines).index(name)
                while len(self._lines) <= idx:
                    self._lines.append(None)
                self._lines[idx] = value
            elif value not in self._lines:
                self._lines.append(value)
    
    def __getitem__(self, i: int) -> Any: return self._lines[i]
    def __iter__(self): return iter(self._lines)
    def __len__(self): return len(self._lines)

class LineSeries:
    """A single line of data."""
    def __init__(self, values: Any) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._cursor = 0
        self._is_datetime = False
        self.min_period = 0

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def __getitem__(self, ago: int) -> Any:
        try:
            return self._values[self._cursor + ago]
        except IndexError:
            return np.nan

    def __call__(self, ago: int = 0) -> LineSeries:
        return DelayedLine(self, ago)

    def datetime(self, ago: int = 0) -> Any:
        import pandas as pd
        val = self[ago]
        if pd.isna(val):
            return None
        return pd.to_datetime(val, unit='s' if val < 1e11 else 'ms')

    def date(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.date() if dt else None

    def time(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.time() if dt else None

    def __len__(self) -> int: return len(self._values)

    # Comparisons
    def __lt__(self, other): return self[0] < (other[0] if hasattr(other, "__getitem__") else other)
    def __gt__(self, other): return self[0] > (other[0] if hasattr(other, "__getitem__") else other)
    def __le__(self, other): return self[0] <= (other[0] if hasattr(other, "__getitem__") else other)
    def __ge__(self, other): return self[0] >= (other[0] if hasattr(other, "__getitem__") else other)
    def __eq__(self, other): return self[0] == (other[0] if hasattr(other, "__getitem__") else other)

    # Math
    def __add__(self, other): return self._math_op(other, np.add)
    def __sub__(self, other): return self._math_op(other, np.subtract)
    def __mul__(self, other): return self._math_op(other, np.multiply)
    def __truediv__(self, other): return self._math_op(other, np.divide)
    
    def _math_op(self, other, op):
        v1 = self._values
        target = other.lines[0] if hasattr(other, "lines") else other
        res = None
        if hasattr(target, "_values"):
            v2 = target._values
            if v1.shape != v2.shape:
                max_len = max(len(v1), len(v2))
                v1_ext = np.full(max_len, np.nan)
                v1_ext[:len(v1)] = v1
                v2_ext = np.full(max_len, np.nan)
                v2_ext[:len(v2)] = v2
                res = LineSeries(op(v1_ext, v2_ext))
            else:
                res = LineSeries(op(v1, v2))
            res.min_period = max(getattr(self, "min_period", 0), getattr(target, "min_period", 0))
        else:
            res = LineSeries(op(v1, target))
            res.min_period = getattr(self, "min_period", 0)
        return res

class DelayedLine(LineSeries):
    """A line series shifted by 'ago' periods."""
    def __init__(self, source: LineSeries, ago: int) -> None:
        self._source = source
        self._ago = ago
        shifted = pd.Series(source._values).shift(-ago).values
        super().__init__(shifted)
        self._is_datetime = source._is_datetime
        self.min_period = source.min_period + abs(ago)

class IndicatorLine(LineSeries):
    """Compatibility stub."""
    def __init__(self, source: LineSeries, shift: int):
        super().__init__(pd.Series(source._values).shift(-shift).values)

class Params:
    def __init__(self, defaults: Any, **kwargs):
        if isinstance(defaults, dict):
            for name, val in defaults.items(): setattr(self, name, val)
        elif isinstance(defaults, (list, tuple)):
            for name, val in defaults: setattr(self, name, val)
        for name, val in kwargs.items(): setattr(self, name, val)

class BaseBroker:
    def __init__(self, **kwargs): pass
    def setcash(self, cash: float): pass
    def setcommission(self, commission: float): pass
    def getcash(self) -> float: return 0.0
    def getvalue(self) -> float: return 0.0
    
    # Aliases for Backtrader compatibility
    def get_cash(self) -> float: return self.getcash()
    def get_value(self) -> float: return self.getvalue()

class BaseSizer: pass
class BaseAnalyzer:
    def __init__(self, **kwargs): self.strategy = None
    def on_order(self, order: Order): pass
    def on_trade(self, trade: Any): pass
    def stop(self): pass

def _notify_order(strategy: Strategy, order: Order) -> None:
    strategy.notify_order(order)
