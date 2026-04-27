
import numpy as np
import pandas as pd
from tradelearn.backtest import base
from tradelearn.backtest.base import (
    LineRoot, LineSeries,
    set_current_data
)

def _series(line, target_len=None):
    """Universal extractor for Pandas Series, handles scalars and forced alignment."""
    if hasattr(line, 'lines'):
        if hasattr(line.lines, 'close'):
            line = line.lines.close
        else:
            line = line.lines[0]
    
    if hasattr(line, '_values'):
        res = pd.Series(line._values)
    elif isinstance(line, (pd.Series, np.ndarray)):
        res = pd.Series(line)
    elif isinstance(line, (int, float)):
        res = pd.Series([float(line)])
    else:
        res = pd.Series(line)
        
    res = res.reset_index(drop=True)
    
    # Forced Alignment
    if target_len is not None and len(res) != target_len:
        if len(res) == 1:
            res = pd.Series(np.full(target_len, res.iloc[0]))
        else:
            new_res = np.full(target_len, np.nan)
            limit = min(len(res), target_len)
            new_res[:limit] = res.values[:limit]
            res = pd.Series(new_res)
            
    return res

def _wrap(source, values):
    """Wraps result in a LineSeries and updates strategy warmup."""
    vals = values.values if hasattr(values, 'values') else np.asarray(values)
    line = LineSeries(vals)
    non_nan = np.where(~np.isnan(vals))[0]
    line.min_period = int(non_nan[0]) if len(non_nan) > 0 else 0
    if base._CURRENT_STRATEGY:
        base._CURRENT_STRATEGY.addminperiod(line.min_period)
    return line

class Indicator(LineRoot):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'data') or self.data is None:
            if args and hasattr(args[0], 'lines'):
                self.data = args[0]
            else:
                self.data = base._CURRENT_DATA
        super().__init__(*args, **kwargs)
            
    def __getitem__(self, ago): return self.lines[0][ago]
    def __call__(self, ago): return self.lines[0](ago)
    
    def __add__(self, other): return self.lines[0] + other
    def __sub__(self, other): return self.lines[0] - other
    def __mul__(self, other): return self.lines[0] * other
    def __truediv__(self, other): return self.lines[0] / other
    def __lt__(self, other): return self.lines[0] < other
    def __gt__(self, other): return self.lines[0] > other

class SMA(Indicator):
    lines = ('sma',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        res = s.rolling(self.p.period).mean()
        self.lines.sma = _wrap(self.data, res)

MovingAverageSimple = SMA

class EMA(Indicator):
    lines = ('ema',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        res = s.ewm(span=self.p.period, adjust=False).mean()
        self.lines.ema = _wrap(self.data, res)

class MACD(Indicator):
    lines = ('macd', 'signal', 'histo')
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        me1 = s.ewm(span=self.p.period_me1, adjust=False).mean()
        me2 = s.ewm(span=self.p.period_me2, adjust=False).mean()
        macd = me1 - me2
        signal = macd.ewm(span=self.p.period_signal, adjust=False).mean()
        self.lines.macd = _wrap(self.data, macd)
        self.lines.signal = _wrap(self.data, signal)
        self.lines.histo = _wrap(self.data, macd - signal)

class RSI(Indicator):
    lines = ('rsi',)
    params = (('period', 14),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        
        # Wilder's Smoothing: first value is SMA, then EWM
        # For simplicity and performance, we'll use a close approximation 
        # or implement the exact recursive formula if needed.
        # Backtrader uses:
        # gain = gain.ewm(alpha=1/period, adjust=False).mean()
        # but the first (period) bars are SMA.
        
        alpha = 1.0 / self.p.period
        
        def wilder_smooth(series, period):
            res = np.full(len(series), np.nan)
            if len(series) < period: return res
            # Initial SMA (from index 0 to period-1)
            first_sma = series.iloc[:period].mean()
            res[period-1] = first_sma
            # Recursive (from index period onwards)
            for i in range(period, len(series)):
                res[i] = (res[i-1] * (period - 1) + series.iloc[i]) / period
            return pd.Series(res)

        g_smooth = wilder_smooth(gain, self.p.period)
        l_smooth = wilder_smooth(loss, self.p.period)
        
        rs = g_smooth / l_smooth.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        self.lines.rsi = _wrap(self.data, rsi.fillna(100.0))

class RSI_SMA(RSI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DonchianChannels(Indicator):
    lines = ('upper', 'lower', 'middle')
    params = (('period', 20),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s_h = _series(self.data.high)
        s_l = _series(self.data.low)
        upper = s_h.rolling(self.p.period).max()
        lower = s_l.rolling(self.p.period).min()
        self.lines.upper = _wrap(self.data, upper)
        self.lines.lower = _wrap(self.data, lower)
        self.lines.middle = _wrap(self.data, (upper + lower) / 2.0)

def _shift(arr, n=1):
    res = np.full(arr.shape, np.nan)
    if n > 0: res[n:] = arr[:-n]
    elif n < 0: res[:n] = arr[-n:]
    return res

class CrossOver(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        super().__init__()
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        
        s0_1, s1_1 = _shift(s0, 1), _shift(s1, 1)
        
        cond_up = (s0 > s1) & (s0_1 <= s1_1)
        cond_down = (s0 < s1) & (s0_1 >= s1_1)
        res = np.zeros(len(s0))
        res[cond_up] = 1.0
        res[cond_down] = -1.0
        self.lines.crossover = _wrap(d0, res)

class CrossUp(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        super().__init__()
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        s0_1, s1_1 = _shift(s0, 1), _shift(s1, 1)
        res = (s0 > s1) & (s0_1 <= s1_1)
        self.lines.crossover = _wrap(d0, res.astype(float))

class CrossDown(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        super().__init__()
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        s0_1, s1_1 = _shift(s0, 1), _shift(s1, 1)
        res = (s0 < s1) & (s0_1 >= s1_1)
        self.lines.crossover = _wrap(d0, res.astype(float))

class Highest(Indicator):
    lines = ('highest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        res = _series(self.data).rolling(self.p.period).max()
        self.lines.highest = _wrap(self.data, res)

class Lowest(Indicator):
    lines = ('lowest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        res = _series(self.data).rolling(self.p.period).min()
        self.lines.lowest = _wrap(self.data, res)

class ATR(Indicator):
    lines = ('atr',)
    params = (('period', 14),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data is None:
            raise ValueError("ATR requires data feed")
            
        h = _series(self.data.high)
        l = _series(self.data.low)
        c = _series(self.data.close).shift(1)
        tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
        
        def wilder_smooth(series, period):
            res = np.full(len(series), np.nan)
            if len(series) < period: return res
            first_sma = series.iloc[:period].mean()
            res[period-1] = first_sma
            for i in range(period, len(series)):
                res[i] = (res[i-1] * (period - 1) + series.iloc[i]) / period
            return pd.Series(res)

        atr = wilder_smooth(tr, self.p.period)
        self.lines.atr = _wrap(self.data, atr)
