"""Shared causal numerical primitives. No centered windows or backward filling."""

import numpy as np
import pandas as pd

from tradelearn.indicators import tv


def sma(s, n):
    return tv.sma(s, n)


def ema(s, n):
    return tv.ema(s, n)


def rma(s, n):
    return tv.rma(s, n)


def wma(s, n):
    return tv.wma(s, n)


def div(a, b):
    return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b


def source(f, key):
    return {
        "hl2": (f.high + f.low) / 2,
        "hlc3": (f.high + f.low + f.close) / 3,
        "ohlc4": (f.open + f.high + f.low + f.close) / 4,
    }.get(key, f[key] if key in f else f.close)


def tr(f):
    return pd.concat(
        [f.high - f.low, (f.high - f.close.shift()).abs(), (f.low - f.close.shift()).abs()], axis=1
    ).max(axis=1)


def regression(s, n):
    x = np.arange(n)
    z = x - x.mean()
    den = (z * z).sum()
    slope = s.rolling(n).apply(lambda y: np.dot(z, y) / den, raw=True) if n > 1 else s * 0
    fit = s.rolling(n).mean() + slope * (n - 1) / 2
    err = (
        s.rolling(n).apply(
            lambda y: np.sqrt(
                np.sum((y - (y.mean() + np.dot(z, y) / den * z)) ** 2) / max(n - 2, 1)
            ),
            raw=True,
        )
        if n > 1
        else s * 0
    )
    return fit, slope, err


def recursive(s, n, kind):
    out = np.full(len(s), np.nan)
    previous = np.nan
    seed = ema(s, n)
    change = s.diff(n).abs()
    noise = s.diff().abs().rolling(n).sum()
    er = div(change, noise)
    for i, v in enumerate(s.to_numpy()):
        if not np.isfinite(v):
            continue
        if not np.isfinite(previous):
            previous = seed.iloc[i]
            if not np.isfinite(previous):
                continue
        elif kind == "kama":
            if not np.isfinite(er.iloc[i]):
                continue
            previous += ((er.iloc[i] * (2 / 3 - 2 / 31) + 2 / 31) ** 2) * (v - previous)
        else:
            previous += (v - previous) / (n * max(abs(v / previous), 1e-6) ** 4) if previous else v
        out[i] = previous
    return pd.Series(out, index=s.index)
