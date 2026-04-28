from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.core.models import Order, Params
from tradelearn.backtest.core.strategy import Strategy as CoreStrategy


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

class MetaParams(type):
    """Metaclass to handle Backtrader-style parameter stripping and lifecycle hooks."""
    def __call__(cls, *args, **kwargs):
        # 1. Collect all params from MRO
        p_names = []
        all_p_defaults = []
        for base_cls in cls.mro():
            p_defaults = getattr(base_cls, 'params', [])
            if isinstance(p_defaults, dict):
                p_names.extend(p_defaults.keys())
                all_p_defaults.extend(p_defaults.items())
            elif isinstance(p_defaults, (list, tuple)):
                p_names.extend([x[0] for x in p_defaults if isinstance(x, (list, tuple))])
                all_p_defaults.extend(p_defaults)

        p_kwargs = {}
        other_kwargs = {}
        for k, v in kwargs.items():
            if k in p_names: p_kwargs[k] = v
            else: other_kwargs[k] = v

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
        all_cls_params = []
        for base_cls in self.__class__.mro():
            p = getattr(base_cls, 'params', [])
            if isinstance(p, (list, tuple)): all_cls_params.extend(p)
            elif isinstance(p, dict): all_cls_params.extend(p.items())
            
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

class Lines:
    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._lines = []
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'): 
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value
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
    def __init__(
        self,
        values: Any,
        is_datetime: bool = False,
        buffer: Any | None = None,
        buffer_name: str | None = None,
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._cursor = 0
        self._is_datetime = is_datetime
        self._buffer = buffer
        self._buffer_name = buffer_name
        self._series_cache = None
        self.min_period = 0

    def datetime(self, ago: int = 0) -> Any:
        val = self[ago]
        if val is None: return None
        # Tradelearn stores as numeric timestamp (seconds)
        unit = 's' if abs(val) < 1e11 else 'ms'
        import pandas as pd
        return pd.to_datetime(val, unit=unit).to_pydatetime()
    
    def date(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.date() if dt else None

    def time(self, ago: int = 0) -> Any:
        dt = self.datetime(ago)
        return dt.time() if dt else None

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def __len__(self) -> int:
        buffer = self._buffer
        if buffer is not None:
            return buffer.cursor + 1
        return self._cursor + 1

    def __getitem__(self, ago: Any) -> Any:
        values = self._values
        buffer = self._buffer
        if ago == 0:
            if not self._is_datetime:
                if buffer is not None and self._buffer_name is not None:
                    return buffer.value(self._buffer_name, ago=0)
                cursor = self._cursor
                if cursor < 0 or cursor >= len(values):
                    return np.nan
                return values[cursor]
            return self._format_value(self._current_value(0))
        if ago == -1:
            if not self._is_datetime:
                if buffer is not None and self._buffer_name is not None:
                    return buffer.value(self._buffer_name, ago=1)
                idx = self._cursor - 1
                if idx < 0 or idx >= len(values):
                    return np.nan
                return values[idx]
            return self._format_value(self._current_value(1))
        if not isinstance(ago, (int, slice, np.integer)):
            # Support indexing by data object (common in multi-data strategies)
            return self
        if isinstance(ago, slice):
            return values[ago]
        cursor = buffer.cursor if buffer is not None else self._cursor
        idx = cursor + int(ago)
        if idx < 0 or idx >= len(values):
            return np.nan
        value = values[idx]
        if not self._is_datetime:
            return value
        return self._format_value(value)

    def _current_value(self, ago: int) -> Any:
        if self._buffer is not None and self._buffer_name is not None:
            return self._buffer.value(self._buffer_name, ago=ago)
        idx = self._cursor - ago
        if idx < 0 or idx >= len(self._values):
            return np.nan
        return self._values[idx]

    def _format_value(self, value: Any) -> Any:
        if not self._is_datetime:
            return value
        if pd.isna(value):
            return None
        return pd.to_datetime(value, unit='s' if abs(value) < 1e11 else 'ms', utc=True)

    def __call__(self, ago: int = 0) -> LineSeries:
        return DelayedLine(self, ago)

    def datetime(self, ago: int = 0) -> Any:
        import pandas as pd
        val = self[ago]
        if pd.isna(val): return None
        if self._is_datetime:
            return pd.to_datetime(val)
        return pd.to_datetime(val, unit='s' if val < 1e11 else 'ms')

    def __bool__(self) -> bool:
        val = self[0]
        return bool(val) and not np.isnan(val)

    def __lt__(self, other): return self[0] < (other[0] if hasattr(other, "__getitem__") else other)
    def __gt__(self, other): return self[0] > (other[0] if hasattr(other, "__getitem__") else other)
    def __le__(self, other): return self[0] <= (other[0] if hasattr(other, "__getitem__") else other)
    def __ge__(self, other): return self[0] >= (other[0] if hasattr(other, "__getitem__") else other)
    def __eq__(self, other): return self[0] == (other[0] if hasattr(other, "__getitem__") else other)

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
    def __init__(self, source: LineSeries, ago: int) -> None:
        self._source = source
        self._ago = ago
        import pandas as pd
        shifted = pd.Series(source._values).shift(-ago).values
        super().__init__(shifted)
        self._is_datetime = source._is_datetime
        self.min_period = source.min_period + abs(ago)

class IndicatorLine(LineSeries):
    def __init__(self, source: LineSeries, shift: int):
        import pandas as pd
        super().__init__(pd.Series(source._values).shift(-shift).values)

class BaseBroker:
    def __init__(self, **kwargs): pass
    def setcash(self, cash: float): pass
    def setcommission(self, commission: float = 0.0, margin: float = 0.0, mult: float = 1.0): pass
    def getcash(self) -> float: return 0.0
    def getvalue(self) -> float: return 0.0
    def get_cash(self) -> float: return self.getcash()
    def get_value(self) -> float: return self.getvalue()

class BaseSizer: pass
class BaseAnalyzer:
    def __init__(self, **kwargs): self.strategy = None
    def on_order(self, order: Order): pass
    def stop(self): pass

def _notify_order(strategy: Any, order: Order) -> None:
    strategy.notify_order(order)
