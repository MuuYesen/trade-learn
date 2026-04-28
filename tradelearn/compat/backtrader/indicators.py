from __future__ import annotations
import pandas as pd
import numpy as np
import inspect
from typing import Any, Optional

class Params:
    def __init__(self, **kwargs):
        self._keys = []
        for k, v in kwargs.items():
            if k not in self._keys: self._keys.append(k)
            setattr(self, k, v)
    
    def __setattr__(self, name, value):
        if not name.startswith('_') and name not in getattr(self, '_keys', []):
            if hasattr(self, '_keys'):
                self._keys.append(name)
        super().__setattr__(name, value)

    def __getitem__(self, i): 
        return getattr(self, self._keys[i])

def bt_ema(series, period):
    if len(series) < period: return pd.Series([np.nan]*len(series))
    res = np.zeros(len(series))
    res[:] = np.nan
    s = series.to_numpy()
    first_idx = pd.Series(s).first_valid_index()
    if first_idx is None: return pd.Series([np.nan]*len(series))
    start = first_idx + period
    if start > len(s): return pd.Series([np.nan]*len(series))
    
    sma = np.mean(s[first_idx:start])
    res[start-1] = sma
    alpha = 2.0 / (period + 1.0)
    for i in range(start, len(series)):
        if not np.isnan(s[i]):
            res[i] = res[i-1] + alpha * (s[i] - res[i-1])
    return pd.Series(res, index=series.index)

def bt_wilder(series, period):
    if len(series) < period: return pd.Series([np.nan]*len(series))
    res = np.zeros(len(series))
    res[:] = np.nan
    s = series.to_numpy()
    first_idx = pd.Series(s).first_valid_index()
    if first_idx is None: return pd.Series([np.nan]*len(series))
    start = first_idx + period
    if start > len(s): return pd.Series([np.nan]*len(series))
    
    res[start-1] = np.mean(s[first_idx:start])
    for i in range(start, len(series)):
        if not np.isnan(s[i]):
            res[i] = (res[i-1] * (period - 1) + s[i]) / period
    return pd.Series(res, index=series.index)

class MetaSimple(type):
    def __call__(cls, *args, **kwargs):
        p_defs = getattr(cls, 'params', ())
        p_names = []
        p_defaults = {}
        if isinstance(p_defs, (list, tuple)):
            for p in p_defs:
                if isinstance(p, (list, tuple)):
                    p_names.append(p[0])
                    p_defaults[p[0]] = p[1]
        
        p_kwargs = {}
        other_kwargs = {}
        for k, v in kwargs.items():
            if k in p_names: p_kwargs[k] = v
            else: other_kwargs[k] = v
            
        final_p = p_defaults.copy()
        final_p.update(p_kwargs)
        
        instance = cls.__new__(cls, *args, **other_kwargs)
        instance.p = instance.params = Params(**final_p)
        
        from tradelearn.backtest.base import _CURRENT_DATA
        data = None
        new_args = []
        for a in args:
            if hasattr(a, 'lines') or hasattr(a, '_values'):
                if data is None: data = a
                new_args.append(a)
            else:
                new_args.append(a)
        
        instance.data = data if data is not None else other_kwargs.get('data', _CURRENT_DATA)
        instance.l = instance.lines = Params(**{line: None for line in instance.lines_def})
        
        sig = inspect.signature(cls.__init__)
        init_params = list(sig.parameters.values())
        real_params = init_params[1:] 
        takes_var_args = any(p.kind == p.VAR_POSITIONAL for p in real_params)
        fixed_param_count = len([p for p in real_params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])
        
        if not takes_var_args and fixed_param_count == 0 and len(new_args) > 0:
            instance.__init__(**other_kwargs)
        else:
            instance.__init__(*args, **other_kwargs)
        
        from tradelearn.backtest.base import _CURRENT_STRATEGY
        if _CURRENT_STRATEGY:
            _CURRENT_STRATEGY._register_indicator(instance)
            
        return instance

class Indicator(metaclass=MetaSimple):
    lines = ()
    params = ()
    
    def __init__(self, *args, **kwargs):
        pass

    @property
    def min_period(self):
        m = 0
        for name in self.lines_def:
            line = getattr(self.lines, name)
            if hasattr(line, 'min_period'):
                m = max(m, line.min_period)
        return m

    def _advance(self, i):
        for line_name in self.lines_def:
            line = getattr(self.lines, line_name)
            if hasattr(line, '_advance'):
                line._advance(i)

    @property
    def lines_def(self):
        cls_lines = getattr(type(self), 'lines', ())
        return cls_lines if isinstance(cls_lines, (list, tuple)) else (cls_lines,)

    def __getattr__(self, name):
        if name in getattr(self, 'lines_def', ()):
            return getattr(self.lines, name)
        if name == 'l': return self.lines
        if name == 'p': return self.params
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __bool__(self):
        return bool(self.lines[0])

    def __getitem__(self, i): return self.lines[0][i]
    def __call__(self, ago=0): return self.lines[0](ago)
    def __len__(self): return len(self.lines[0])
    def __mul__(self, other): return self.lines[0] * other
    def __truediv__(self, other): return self.lines[0] / other
    def __add__(self, other): return self.lines[0] + other
    def __sub__(self, other): return self.lines[0] - other
    def __lt__(self, other): return self.lines[0] < other
    def __gt__(self, other): return self.lines[0] > other

def _series(data, target_len=None):
    if data is None: return pd.Series()
    if hasattr(data, 'close'): # Backtrader default
        res = pd.Series(data.close._values)
    elif hasattr(data, 'lines'):
        res = pd.Series(data.lines[0]._values)
    elif hasattr(data, '_values'):
        res = pd.Series(data._values)
    else:
        res = pd.Series(data)
    if target_len and len(res) < target_len:
        res = pd.concat([pd.Series([np.nan]*(target_len-len(res))), res]).reset_index(drop=True)
    return res

def _wrap(data, values, min_period=0):
    from tradelearn.compat.backtrader.strategy import LineSeries
    obj = LineSeries(values.to_numpy())
    obj.min_period = min_period
    return obj

def _base_p(data):
    if hasattr(data, 'min_period'): return data.min_period
    if hasattr(data, 'lines') and hasattr(data.lines[0], 'min_period'):
        return data.lines[0].min_period
    return 0

class SMA(Indicator):
    lines = ('sma',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        res = s.rolling(self.p.period).mean()
        self.lines.sma = _wrap(self.data, res, min_period=_base_p(self.data) + self.p.period - 1)

MovingAverageSimple = SMA
SimpleMovingAverage = SMA

class EMA(Indicator):
    lines = ('ema',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        res = bt_ema(s, self.p.period)
        self.lines.ema = _wrap(self.data, res, min_period=_base_p(self.data) + self.p.period - 1)

ExponentialMovingAverage = EMA
MovingAverageExponential = EMA

class MACD(Indicator):
    lines = ('macd', 'signal', 'histo')
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9))
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        me1 = bt_ema(s, self.p.period_me1)
        me2 = bt_ema(s, self.p.period_me2)
        macd_val = me1 - me2
        signal = bt_ema(macd_val, self.p.period_signal)
        base = _base_p(self.data)
        m_period = base + max(self.p.period_me1, self.p.period_me2) + self.p.period_signal - 2
        self.lines.macd = _wrap(self.data, macd_val, min_period=base + max(self.p.period_me1, self.p.period_me2))
        self.lines.signal = _wrap(self.data, signal, min_period=m_period)
        self.lines.histo = _wrap(self.data, macd_val - signal, min_period=m_period)

class RSI(Indicator):
    lines = ('rsi',)
    params = (('period', 14), ('movav', None))
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        g_smooth = bt_wilder(gain, self.p.period)
        l_smooth = bt_wilder(loss, self.p.period)
        rs = g_smooth / l_smooth.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        base = _base_p(self.data)
        rsi.iloc[:base + self.p.period] = np.nan
        self.lines.rsi = _wrap(self.data, rsi.fillna(100.0), min_period=base + self.p.period)

class RSI_SMA(RSI):
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        g_smooth = gain.rolling(self.p.period).mean()
        l_smooth = loss.rolling(self.p.period).mean()
        rs = g_smooth / l_smooth.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        base = _base_p(self.data)
        rsi.iloc[:base + self.p.period + 1] = np.nan
        self.lines.rsi = _wrap(self.data, rsi.fillna(100.0), min_period=base + self.p.period + 1)

class ATR(Indicator):
    lines = ('atr',)
    params = (('period', 14),)
    def __init__(self, *args, **kwargs):
        h = _series(getattr(self.data, 'high', self.data))
        l = _series(getattr(self.data, 'low', self.data))
        c = _series(getattr(self.data, 'close', self.data))
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        tr.iloc[0] = np.nan
        atr = bt_wilder(tr, self.p.period)
        base = _base_p(self.data)
        self.lines.atr = _wrap(self.data, atr, min_period=base + self.p.period + 1)
class TrueRange(Indicator):
    lines = ('tr',)
    def __init__(self, *args, **kwargs):
        h = _series(getattr(self.data, 'high', self.data))
        l = _series(getattr(self.data, 'low', self.data))
        c = _series(getattr(self.data, 'close', self.data))
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        tr.iloc[0] = np.nan
        self.lines.tr = _wrap(self.data, tr, min_period=_base_p(self.data) + 1)

TR = TrueRange

class CrossOver(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        res = np.zeros(len(s0))
        diff = s0 - s1
        signs = np.sign(diff)
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).to_numpy()
        # Only trigger if both current and previous signs are not NaN and not zero
        mask = (~np.isnan(signs)) & (~np.isnan(filled_signs))
        res[mask & (signs > 0) & (filled_signs <= 0)] = 1.0
        res[mask & (signs < 0) & (filled_signs >= 0)] = -1.0
        m0 = _base_p(d0)
        m1 = _base_p(d1)
        self.lines.crossover = _wrap(d0, pd.Series(res), min_period=max(m0, m1) + 1)

class CrossUp(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        diff = s0 - s1
        signs = np.sign(diff)
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).to_numpy()
        mask = (~np.isnan(signs)) & (~np.isnan(filled_signs))
        res = mask & (signs > 0) & (filled_signs <= 0)
        m0 = _base_p(d0)
        m1 = _base_p(d1)
        self.lines.crossover = _wrap(d0, pd.Series(res.astype(float)), min_period=max(m0, m1) + 1)

class CrossDown(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        diff = s0 - s1
        signs = np.sign(diff)
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).to_numpy()
        mask = (~np.isnan(signs)) & (~np.isnan(filled_signs))
        res = mask & (signs < 0) & (filled_signs >= 0)
        m0 = _base_p(d0)
        m1 = _base_p(d1)
        self.lines.crossover = _wrap(d0, pd.Series(res.astype(float)), min_period=max(m0, m1) + 1)

class Lowest(Indicator):
    lines = ('lowest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        res = s.rolling(self.p.period).min()
        base = _base_p(self.data)
        self.lines.lowest = _wrap(self.data, res, min_period=base + self.p.period)

class Highest(Indicator):
    lines = ('highest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        s = _series(self.data)
        res = s.rolling(self.p.period).max()
        base = _base_p(self.data)
        self.lines.highest = _wrap(self.data, res, min_period=base + self.p.period)

class DonchianChannels(Indicator):
    lines = ('upper', 'lower', 'middle')
    params = (('period', 20),)
    def __init__(self, *args, **kwargs):
        hi = _series(getattr(self.data, 'high', self.data))
        lo = _series(getattr(self.data, 'low', self.data))
        upper = hi.rolling(self.p.period).max()
        lower = lo.rolling(self.p.period).min()
        base = _base_p(self.data)
        self.lines.upper = _wrap(self.data, upper, min_period=base + self.p.period + 2)
        self.lines.lower = _wrap(self.data, lower, min_period=base + self.p.period + 2)
        self.lines.middle = _wrap(self.data, (upper + lower) / 2.0, min_period=base + self.p.period + 2)
