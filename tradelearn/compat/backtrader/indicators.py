
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

def _wrap(source, values, min_period=None):
    """Wraps result in a LineSeries and updates strategy warmup."""
    vals = values.values if hasattr(values, 'values') else np.asarray(values)
    line = LineSeries(vals)
    if min_period is not None:
        line.min_period = min_period
    else:
        non_nan = np.where(~np.isnan(vals))[0]
        line.min_period = int(non_nan[0]) if len(non_nan) > 0 else 0
        
    if base._CURRENT_STRATEGY:
        base._CURRENT_STRATEGY.addminperiod(line.min_period)
    return line


def bt_ema(series, period):
    """Backtrader-style EMA initialization: first value is SMA(period)."""
    res = np.full(len(series), np.nan)
    if len(series) < period: return pd.Series(res)
    # Initial SMA
    first_sma = series.iloc[:period].mean()
    res[period-1] = first_sma
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(series)):
        val = series.iloc[i]
        if not np.isnan(val):
            res[i] = (val - res[i-1]) * alpha + res[i-1]
        else:
            res[i] = res[i-1]
    return pd.Series(res)

def bt_wilder(series, period):
    """Backtrader-style Wilder's Smoothing (SmoothedMovingAverage)."""
    res = np.full(len(series), np.nan)
    if len(series) < period: return pd.Series(res)
    # Initial SMA
    first_sma = series.iloc[:period].mean()
    res[period-1] = first_sma
    for i in range(period, len(series)):
        val = series.iloc[i]
        if not np.isnan(val):
            res[i] = (res[i-1] * (period - 1) + val) / period
        else:
            res[i] = res[i-1]
    return pd.Series(res)

class Indicator(LineRoot):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'data') or self.data is None:
            if args and (hasattr(args[0], 'lines') or hasattr(args[0], '_values')):
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
        self.lines.sma = _wrap(self.data, res, min_period=self.p.period - 1)

MovingAverageSimple = SMA

class EMA(Indicator):
    lines = ('ema',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        res = bt_ema(s, self.p.period)
        self.lines.ema = _wrap(self.data, res, min_period=self.p.period - 1)

class MACD(Indicator):
    lines = ('macd', 'signal', 'histo')
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        me1 = bt_ema(s, self.p.period_me1)
        me2 = bt_ema(s, self.p.period_me2)
        macd = me1 - me2
        signal = bt_ema(macd, self.p.period_signal)
        # MACD min_period is max of me1, me2 plus signal period
        m_period = max(self.p.period_me1, self.p.period_me2) + self.p.period_signal - 1
        self.lines.macd = _wrap(self.data, macd, min_period=max(self.p.period_me1, self.p.period_me2) - 1)
        self.lines.signal = _wrap(self.data, signal, min_period=m_period)
        self.lines.histo = _wrap(self.data, macd - signal, min_period=m_period)

class RSI(Indicator):
    lines = ('rsi',)
    params = (
        ('period', 14),
        ('movav', None), # Will default to SmoothedMovingAverage if None
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = _series(self.data)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        
        movav = self.p.movav
        if movav is None:
            g_smooth = bt_wilder(gain, self.p.period)
            l_smooth = bt_wilder(loss, self.p.period)
            m_period = self.p.period # RSI Wilder needs period+1 bars to start
        else:
            # Handle class-based movav like bt.ind.SMA
            g_ind = movav(gain, period=self.p.period)
            l_ind = movav(loss, period=self.p.period)
            g_smooth = g_ind.lines[0]._values
            l_smooth = l_ind.lines[0]._values
            m_period = g_ind.lines[0].min_period
        
        # Avoid division by zero
        rs = pd.Series(g_smooth) / pd.Series(l_smooth).replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        self.lines.rsi = _wrap(self.data, rsi.fillna(100.0), min_period=m_period)

class RSI_SMA(RSI):
    def __init__(self, *args, **kwargs):
        kwargs['movav'] = SMA
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
        m_period = self.p.period - 1
        self.lines.upper = _wrap(self.data, upper, min_period=m_period)
        self.lines.lower = _wrap(self.data, lower, min_period=m_period)
        self.lines.middle = _wrap(self.data, (upper + lower) / 2.0, min_period=m_period)

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
        
        res = np.zeros(len(s0))
        # Backtrader's CrossOver uses non-zero difference memory
        # We simulate this by finding the last non-zero difference at each point
        diff = s0 - s1
        
        # Vectorized non-zero diff: 
        # 1. Get signs of diffs
        # 2. Forward fill non-zero signs
        signs = np.sign(diff)
        # We need to handle NaNs and Zeros
        # Replace 0 with NaN for forward filling, then ffill
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        # Use pandas to ffill
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).fillna(0).to_numpy()
        
        # Trigger when current sign is positive and previous non-zero sign was <= 0
        cond_up = (signs > 0) & (filled_signs <= 0)
        # Trigger when current sign is negative and previous non-zero sign was >= 0
        cond_down = (signs < 0) & (filled_signs >= 0)
        
        res[cond_up] = 1.0
        res[cond_down] = -1.0
        
        m0 = getattr(d0, "min_period", 0) if hasattr(d0, "min_period") else 0
        m1 = getattr(d1, "min_period", 0) if hasattr(d1, "min_period") else 0
        m_period = max(m0, m1, 1)
        self.lines.crossover = _wrap(d0, res, min_period=m_period)

class CrossUp(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        super().__init__()
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        
        diff = s0 - s1
        signs = np.sign(diff)
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).fillna(0).to_numpy()
        
        res = (signs > 0) & (filled_signs <= 0)
        
        m0 = getattr(d0, "min_period", 0) if hasattr(d0, "min_period") else 0
        m1 = getattr(d1, "min_period", 0) if hasattr(d1, "min_period") else 0
        m_period = max(m0, m1, 1)
        self.lines.crossover = _wrap(d0, res.astype(float), min_period=m_period)

class CrossDown(Indicator):
    lines = ('crossover',)
    def __init__(self, d0, d1):
        super().__init__()
        v0 = _series(d0)
        v1 = _series(d1, target_len=len(v0))
        s0, s1 = v0.to_numpy(), v1.to_numpy()
        
        diff = s0 - s1
        signs = np.sign(diff)
        signs_to_fill = np.where(signs == 0, np.nan, signs)
        filled_signs = pd.Series(signs_to_fill).ffill().shift(1).fillna(0).to_numpy()
        
        res = (signs < 0) & (filled_signs >= 0)
        
        m0 = getattr(d0, "min_period", 0) if hasattr(d0, "min_period") else 0
        m1 = getattr(d1, "min_period", 0) if hasattr(d1, "min_period") else 0
        m_period = max(m0, m1, 1)
        self.lines.crossover = _wrap(d0, res.astype(float), min_period=m_period)

class Highest(Indicator):
    lines = ('highest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        res = _series(self.data).rolling(self.p.period).max()
        self.lines.highest = _wrap(self.data, res, min_period=self.p.period - 1)

class Lowest(Indicator):
    lines = ('lowest',)
    params = (('period', 30),)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        res = _series(self.data).rolling(self.p.period).min()
        self.lines.lowest = _wrap(self.data, res, min_period=self.p.period - 1)

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
        
        # Backtrader's first TR is High - Low
        tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
        tr.iloc[0] = h.iloc[0] - l.iloc[0]
        
        atr = bt_wilder(tr, self.p.period)
        self.lines.atr = _wrap(self.data, atr)
