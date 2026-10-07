"""Causal event studies and explicit OHLCV profile allocation."""

import numpy as np
import pandas as pd


def profile(f, p, visible_range):
    if visible_range:
        if isinstance(visible_range, dict):
            a, b = visible_range.get("from"), visible_range.get("to")
        else:
            a, b = visible_range

        def ts(v):
            return (
                pd.to_datetime(v, unit="s", utc=True)
                if isinstance(v, (int, float))
                else pd.to_datetime(v, utc=True)
            )

        if a is not None:
            f = f.loc[f.index >= ts(a)]
        if b is not None:
            f = f.loc[f.index <= ts(b)]
    if "bars" in p:
        f = f.tail(p["bars"])
    if f.empty:
        return []
    low = float(f.low.min())
    high = float(f.high.max())
    if high == low:
        high = low + max(abs(low) * 1e-8, 1e-8)
    edges = np.linspace(low, high, p["bins"] + 1)
    volumes = np.zeros(p["bins"])
    for bar in f.itertuples():
        if bar.high == bar.low:
            volumes[
                min(max(np.searchsorted(edges, bar.close, side="right") - 1, 0), len(volumes) - 1)
            ] += bar.volume
        else:
            volumes += (
                np.maximum(0, np.minimum(edges[1:], bar.high) - np.maximum(edges[:-1], bar.low))
                / (bar.high - bar.low)
                * bar.volume
            )
    return [
        dict(low=float(a), high=float(b), volume=float(v))
        for a, b, v in zip(edges[:-1], edges[1:], volumes, strict=True)
    ]


def zigzag(f, deviation):
    # Emit only a newly confirmed reversal at its confirmation time, never backdate.
    result = pd.Series(np.nan, index=f.index)
    direction = 0
    extreme = float(f.close.iloc[0]) if len(f) else 0
    for i, price in enumerate(f.close):
        if direction >= 0:
            extreme = max(extreme, price)
            if price <= extreme * (1 - deviation / 100):
                result.iloc[i] = extreme
                direction = -1
                extreme = price
        else:
            extreme = min(extreme, price)
            if price >= extreme * (1 + deviation / 100):
                result.iloc[i] = extreme
                direction = 1
                extreme = price
    return result


def pivots(f, anchor):
    idx = f.index
    if anchor == "day":
        groups = idx.strftime("%Y-%m-%d")
    elif anchor == "week":
        groups = idx.strftime("%G-%V")
    else:
        groups = idx.strftime("%Y-%m")
    periods = (
        f.assign(group=groups)
        .groupby("group", sort=False)
        .agg({"high": "max", "low": "min", "close": "last"})
        .shift()
    )
    h = pd.Series(groups, index=idx).map(periods.high)
    low = pd.Series(groups, index=idx).map(periods.low)
    c = pd.Series(groups, index=idx).map(periods.close)
    p = (h + low + c) / 3
    return dict(
        pivot=p,
        r1=2 * p - low,
        s1=2 * p - h,
        r2=p + h - low,
        s2=p - h + low,
        r3=h + 2 * (p - low),
        s3=low - 2 * (h - p),
    )


def volatility_stop(f, n, mult, method):
    from .core import ema, rma, tr

    ranges = tr(f)
    average = (ema if method == "Exponential" else rma)(ranges, n)
    out = pd.Series(np.nan, index=f.index)
    long = True
    extreme = float(f.close.iloc[:n].max())
    next_stop = np.nan
    for i in range(n - 1, len(f)):
        close = f.close.iloc[i]
        distance = average.iloc[i] * mult
        if np.isfinite(next_stop):
            out.iloc[i] = next_stop
            if long and close < next_stop:
                long = False
                extreme = close
            elif not long and close > next_stop:
                long = True
                extreme = close
            else:
                extreme = max(extreme, close) if long else min(extreme, close)
        next_stop = extreme - distance if long else extreme + distance
    return out


def fractals(f, k):
    # Report at confirmation time. Newer bars must be strictly beyond the pivot;
    # allow up to four equal-price older bars before the strict k-bar flank.
    def events(values, is_high):
        a = values.to_numpy()
        out = np.full(len(a), np.nan)
        for i in range(2 * k, len(a)):
            pivot = i - k
            price = a[pivot]
            newer = a[pivot + 1 : i + 1]
            if not np.all(newer < price if is_high else newer > price):
                continue
            for plateau in range(5):
                end = pivot - plateau
                start = end - k
                if start < 0:
                    break
                equal = a[end:pivot]
                if not np.all(equal == price):
                    continue
                flank = a[start:end]
                if np.all(flank < price if is_high else flank > price):
                    out[i] = price
                    break
        return pd.Series(out, index=f.index)

    return dict(high=events(f.high, True), low=events(f.low, False))
