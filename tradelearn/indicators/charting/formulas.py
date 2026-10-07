"""Indicator compositions; primitive moving averages use the native TV adapter."""

import numpy as np
import pandas as pd

from tradelearn.indicators import tv

from .core import div, ema, recursive, regression, rma, sma, source, tr, wma
from .special import pivots, zigzag


def compute(f, name, p, secondary=None):
    o, h, low, c, v = f.open, f.high, f.low, f.close, f.volume
    s = source(f, p.get("source", "close"))
    n = p.get("length", 14)
    mult = p.get("mult", 2.0)
    hl = (h + low) / 2
    typical = (h + low + c) / 3
    delta = c.diff()

    def atr():
        return tv.atr(h, low, c, n)

    def val(x):
        return {"value": x}

    def band(mid, width):
        return dict(mid=mid, upper=mid + width, lower=mid - width)

    direct = {
        "Moving Average": "sma",
        "Moving Average Exponential": "ema",
        "Moving Average Weighted": "wma",
        "Smoothed Moving Average": "rma",
        "Relative Strength Index": "rsi",
        "Rate Of Change": "roc",
        "Momentum": "mom",
        "Chande Momentum Oscillator": "cmo",
        "Standard Deviation": "stdev",
    }
    if name in direct:
        return val(getattr(tv, direct[name])(s, n))
    if name == "Hull Moving Average":
        return val(wma(2 * wma(s, max(n // 2, 1)) - wma(s, n), max(int(np.sqrt(n) + 0.5), 1)))
    if name in ["Moving Average Double", "Moving Average Triple", "Moving Average Multiple"]:
        average = {"Simple": sma, "Exponential": ema, "Weighted": wma}[p["method"]]
        return {
            "ma" + k.split("_")[1]: average(c, v) for k, v in p.items() if k.startswith("length_")
        }
    if name == "Arnaud Legoux Moving Average":
        return val(tv.alma(s, n, p["offset"], p["sigma"]))
    if name in ["Double EMA", "Triple EMA", "TRIX"]:
        a = ema(np.log(s) if name == "TRIX" else s, n)
        b = ema(a, n)
        d = ema(b, n)
        return val(
            d.diff() * 10000
            if name == "TRIX"
            else 2 * a - b
            if name == "Double EMA"
            else 3 * a - 3 * b + d
        )
    if name == "Moving Average Adaptive":
        logs = np.log(s / s.shift())
        factor = 0.1 * (np.log(s / s.shift(n)) / (logs.rolling(n).std(ddof=0) * np.sqrt(n))).abs()
        out = []
        previous = np.nan
        for price, weight in zip(s, factor, strict=True):
            value = previous + (price - previous) * weight
            out.append(value)
            previous = price if not np.isfinite(value) else value
        return val(pd.Series(out, index=f.index))
    if name == "McGinley Dynamic":
        return val(recursive(s, n, "mcginley"))
    if name == "Moving Average Hamming":
        weights = np.sin((1 + np.arange(1, n + 1)) / n * np.pi / 2)[::-1]
        return val(s.rolling(n).apply(lambda a: np.dot(a, weights) / weights.sum(), raw=True))
    if name == "Guppy Multiple Moving Average":
        return {"ma" + str(k): ema(s, k) for k in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]}
    if name in ["Average Price", "Typical Price", "Median Price"]:
        return val(
            {"Average Price": (o + h + low + c) / 4, "Typical Price": typical, "Median Price": hl}[
                name
            ]
        )
    if name == "Average True Range":
        return val(atr())
    if name == "Average Directional Index":
        return val(tv.dmi(h, low, c, n, p["adx_smoothing"])["adx"])
    if name == "Directional Movement":
        d = tv.dmi(h, low, c, n, p["adx_smoothing"])
        plus, minus, adx = [d.iloc[:, i] for i in range(3)]
        return dict(
            plus=plus,
            minus=minus,
            adx=adx,
            dx=100 * (plus - minus).abs() / (plus + minus).replace(0, 1),
            adxr=(adx + adx.shift(n - 1)) / 2,
        )
    if name == "Commodity Channel Index":
        return val(tv.cci(h, low, c, n))
    if name == "Money Flow Index":
        return val(tv.mfi(h, low, c, v, n))
    if name == "Parabolic SAR":
        return val(tv.sar(h, low, c, p["start"], p["increment"], p["maximum"]))
    if name == "SuperTrend":
        d = tv.supertrend(h, low, c, n, mult)
        return dict(value=d.iloc[:, 0], direction=d.iloc[:, 1])
    if name == "Ichimoku Cloud":
        conversion = (h.rolling(9).max() + low.rolling(9).min()) / 2
        base = (h.rolling(26).max() + low.rolling(26).min()) / 2
        return dict(
            conversion=conversion,
            base=base,
            lagging=c,
            span_a=((conversion + base) / 2).shift(25),
            span_b=((h.rolling(52).max() + low.rolling(52).min()) / 2).shift(25),
        )
    if name == "Bollinger Bands":
        return band(sma(s, n), s.rolling(n).std(ddof=0) * mult)
    if name in ["Bollinger Bands %B", "Bollinger Bands Width"]:
        mid = sma(s, n)
        width = s.rolling(n).std(ddof=0) * mult
        return val(div(s - mid + width, 2 * width) if name.endswith("%B") else div(2 * width, mid))
    if name in ["Donchian Channels", "Price Channel"]:
        upper = h.rolling(n).max()
        lower = low.rolling(n).min()
        return dict(mid=(upper + lower) / 2, upper=upper, lower=lower)
    if name == "Moving Average Channel":
        return dict(upper=sma(h, p["high_length"]), lower=sma(low, p["low_length"]))
    if name == "Envelopes":
        return band(sma(s, n), sma(s, n) * p["percent"] / 100)
    if name == "Keltner Channels":
        return band(ema(c, n), ema(tr(f), n) * mult)
    if name in [
        "Least Squares Moving Average",
        "Linear Regression Curve",
        "Linear Regression Slope",
        "Standard Error",
        "Standard Error Bands",
    ]:
        fit, slope, err = regression(s, n)
        return (
            {
                k: {"Simple": sma, "Exponential": ema, "Weighted": wma}[p["method"]](
                    v, p["average_length"]
                )
                for k, v in band(fit, err * mult).items()
            }
            if name == "Standard Error Bands"
            else val(slope if name.endswith("Slope") else err if name == "Standard Error" else fit)
        )
    if name == "VWMA":
        return val(div((s * v).rolling(n).sum(), v.rolling(n).sum()))
    if name == "VWAP":
        a = p["anchor"]
        groups = f.index.strftime(
            "%Y-%m-%d"
            if a == "day"
            else "%G-%V"
            if a == "week"
            else "%Y-%m"
            if a == "month"
            else "all"
        )
        total = v.groupby(groups).cumsum()
        mid = div((s * v).groupby(groups).cumsum(), total)
        variance = (div((s * s * v).groupby(groups).cumsum(), total) - mid * mid).clip(lower=0)
        return band(mid, np.sqrt(variance) * mult)
    if name in ["EMA Cross", "MA Cross", "MA with EMA Cross"]:
        fast = p["fast"]
        slow = p["slow"]
        return dict(
            fast=(ema if name == "EMA Cross" else sma)(s, fast),
            slow=(sma if name == "MA Cross" else ema)(s, slow),
        )
    if name in ["MACD", "Price Oscillator", "Volume Oscillator"]:
        x = v if name == "Volume Oscillator" else s
        avg = sma if name == "Price Oscillator" else ema
        fast = avg(x, p["fast"])
        slow = avg(x, p["slow"])
        diff = fast - slow
        if name != "MACD":
            return val(div(diff, slow) * 100)
        signal = ema(diff, p["signal"])
        return dict(macd=diff, signal=signal, histogram=diff - signal)
    if name == "Awesome Oscillator":
        return val(sma(hl, 5) - sma(hl, 34))
    if name == "Balance of Power":
        return val(div(c - o, h - low))
    if name == "Aroon":
        return dict(
            up=h.rolling(n + 1).apply(
                lambda a: np.flatnonzero(a == a.max())[-1] / n * 100, raw=True
            ),
            down=low.rolling(n + 1).apply(
                lambda a: np.flatnonzero(a == a.min())[-1] / n * 100, raw=True
            ),
        )
    ad = (div(2 * c - h - low, h - low).fillna(0) * v).cumsum()
    if name == "Accumulation/Distribution":
        return val(ad)
    if name == "Chaikin Money Flow":
        return val(
            div((div(2 * c - h - low, h - low).fillna(0) * v).rolling(n).sum(), v.rolling(n).sum())
        )
    if name == "Chaikin Oscillator":
        return val(ema(ad, p["fast"]) - ema(ad, p["slow"]))
    if name == "Chaikin Volatility":
        return val(ema(h - low, n).pct_change(n, fill_method=None) * 100)
    if name == "Chande Kroll Stop":
        long = h.rolling(n).max() - atr() * mult
        short = low.rolling(n).min() + atr() * mult
        return dict(
            long=long.rolling(p["stop_length"]).max(), short=short.rolling(p["stop_length"]).min()
        )
    if name == "Choppiness Index":
        return val(
            100
            * np.log10(div(tr(f).rolling(n).sum(), h.rolling(n).max() - low.rolling(n).min()))
            / np.log10(n)
        )
    if name == "Chop Zone":
        slope = div(ema(c, 34).shift() - ema(c, 34), typical) * div(
            25 * h.rolling(30).min(), h.rolling(30).max() - h.rolling(30).min()
        )
        angle = np.sign(-slope) * np.floor(np.degrees(np.arctan(slope.abs())) + 0.5)
        zones = np.select(
            [
                angle >= 5,
                angle >= 3.57,
                angle >= 2.14,
                angle >= 0.71,
                angle <= -5,
                angle <= -3.57,
                angle <= -2.14,
                angle <= -0.71,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7],
            default=8,
        )
        zone = pd.Series(zones, index=f.index).where(angle.notna())
        return dict(value=pd.Series(1.0, index=f.index).where(zone.notna()), color=zone)
    if name == "Connors RSI":
        streak = []
        last = 0
        for d in delta:
            last = max(last, 0) + 1 if d > 0 else min(last, 0) - 1 if d < 0 else 0
            streak.append(last)
        rank = (
            c.pct_change(fill_method=None)
            .rolling(p["rank_length"])
            .apply(lambda a: (a[:-1] <= a[-1]).sum() / len(a) * 100, raw=True)
        )
        return val(
            (
                tv.rsi(s, p["rsi_length"])
                + tv.rsi(pd.Series(streak, index=f.index, dtype=float), p["streak_length"])
                + rank
            )
            / 3
        )
    if name == "Coppock Curve":
        return val(wma(tv.roc(c, p["long"]) + tv.roc(c, p["short"]), n))
    if name in ["Correlation Coefficient", "Correlation - Log", "Spread", "Ratio"]:
        second = secondary.close.reindex(f.index)
        if name == "Spread":
            return val(c - second)
        if name == "Ratio":
            return val(div(c, second))
        return val(
            np.log(c / c.shift()).rolling(n).corr(np.log(second / second.shift())).round(3)
            if name == "Correlation - Log"
            else c.rolling(n).corr(second)
        )
    if name == "Detrended Price Oscillator":
        return val(s - sma(s, n).shift(n // 2 + 1))
    if name == "Ease Of Movement":
        return val(sma(div(hl.diff() * (h - low), v) * 10000, n))
    if name == "Elder's Force Index":
        return val(ema(delta * v, n))
    if name == "Fisher Transform":
        x = (hl - hl.rolling(n).min()) / (hl.rolling(n).max() - hl.rolling(n).min()).clip(
            lower=0.001
        )
        a = 0.0
        b = 0.0
        out = []
        for q in x:
            if not np.isfinite(q):
                out.append(np.nan)
                continue
            a = 0.66 * (q - 0.5) + 0.67 * a
            a = 0.999 if a > 0.99 else -0.999 if a < -0.99 else a
            b = 0.5 * np.log((1 + a) / max(1 - a, 0.001)) + 0.5 * b
            out.append(b)
        result = pd.Series(out, index=f.index)
        return dict(value=result, signal=result.shift())
    if name == "Klinger Oscillator":
        force = v.where(typical.diff() >= 0, -v)
        k = ema(force, p["fast"]) - ema(force, p["slow"])
        return dict(value=k, signal=ema(k, p["signal"]))
    if name == "Know Sure Thing":
        k = sum(
            weight * sma(tv.roc(c, roc), smooth)
            for weight, roc, smooth in [
                (i, p["roc_" + str(i)], p["smooth_" + str(i)]) for i in range(1, 5)
            ]
        )
        return dict(value=k, signal=sma(k, p["signal"]))
    if name == "Mass Index":
        return val(div(ema(h - low, 9), ema(ema(h - low, 9), 9)).rolling(n).sum())
    if name == "Majority Rule":
        return val((s > s.shift()).astype(float).rolling(n).mean() * 100)
    if name == "Net Volume":
        return val(np.sign(delta).fillna(0) * v)
    if name == "On Balance Volume":
        return val(tv.obv(c, v))
    if name == "Price Volume Trend":
        return val((c.pct_change(fill_method=None).fillna(0) * v).cumsum())
    if name == "Relative Vigor Index":

        def smooth(x):
            return (x + 2 * x.shift() + 2 * x.shift(2) + x.shift(3)) / 6

        r = div(sma(smooth(c - o), n), sma(smooth(h - low), n))
        return dict(value=r, signal=smooth(r))
    if name == "Relative Volatility Index":
        sd = c.rolling(n).std(ddof=0)
        up = ema(sd.where(delta > 0, 0), 14)
        down = ema(sd.where(delta <= 0, 0), 14)
        return val(100 * div(up, up + down))
    if name in ["True Strength Index", "SMI Ergodic Indicator/Oscillator"]:
        t = tv.tsi(c, p["short"], p["long"])
        if name == "True Strength Index":
            return dict(value=t, signal=ema(t, p["signal"]))
        signal = ema(t, p["signal"])
        return dict(value=t, signal=signal, histogram=t - signal)
    if name == "Trend Strength Index":
        return val(c.rolling(n).corr(pd.Series(np.arange(len(f)), index=f.index)))
    if name in ["Stochastic", "Stochastic RSI"]:
        x = tv.rsi(c, n) if name.endswith("RSI") else c
        hi = x if name.endswith("RSI") else h
        lo = x if name.endswith("RSI") else low
        window = p.get("stoch_length", n)
        k = sma(
            100
            * div(
                x - lo.rolling(window).min(), hi.rolling(window).max() - lo.rolling(window).min()
            ),
            p["smooth_k"],
        )
        return dict(k=k, d=sma(k, p["smooth_d"]))
    if name == "Ultimate Oscillator":
        bp = c - pd.concat([low, c.shift()], axis=1).min(axis=1)
        ranges = tr(f)
        return val(
            100
            * sum(
                w * div(bp.rolling(k).sum(), ranges.rolling(k).sum())
                for w, k in [(4, p["short"]), (2, p["medium"]), (1, p["long"])]
            )
            / 7
        )
    if name == "Williams %R":
        return val(-100 * div(h.rolling(n).max() - c, h.rolling(n).max() - low.rolling(n).min()))
    if name == "Vortex Indicator":
        return dict(
            plus=div((h - low.shift()).abs().rolling(n).sum(), tr(f).rolling(n).sum()),
            minus=div((low - h.shift()).abs().rolling(n).sum(), tr(f).rolling(n).sum()),
        )
    if name == "Williams Alligator":
        return dict(jaw=rma(hl, 21).shift(8), teeth=rma(hl, 13).shift(5), lips=rma(hl, 8).shift(3))
    if name == "Williams Fractal":
        from .special import fractals

        return fractals(f, p["period"])
    if name == "Volume":
        return val(v)
    if name == "Zig Zag":
        return val(zigzag(f, p["deviation"]))
    if name == "Pivot Points Standard":
        return pivots(f, p["anchor"])
    if name == "Advance/Decline":
        up = (c > o).rolling(n).sum()
        down = (c < o).rolling(n).sum()
        return val(div(up, down))
    if name == "Accumulative Swing Index":
        a = (h - c.shift()).abs()
        b = (low - c.shift()).abs()
        d = (c.shift() - o.shift()).abs()
        e = h - low
        r = pd.Series(
            np.select(
                [(a >= b) & (a >= e), (b > a) & (b >= e)],
                [a - b / 2 + d / 4, b - a / 2 + d / 4],
                default=e + d / 4,
            ),
            index=f.index,
        )
        swing = (
            50
            * div(c - c.shift() + (c - o) / 2 + (c.shift() - o.shift()) / 4, r)
            * pd.concat([a, b], axis=1).max(axis=1)
            / p["limit_move"]
        )
        return val(swing.fillna(0).cumsum())
    if name in [
        "Historical Volatility",
        "Volatility Close-to-Close",
        "Volatility Zero Trend Close-to-Close",
        "Volatility O-H-L-C",
        "Volatility Index",
    ]:
        returns = np.log(div(c, c.shift()))
        ann = p.get("annualization", 252)
        if name == "Volatility O-H-L-C":
            raw = 0.5 * np.log(h / low) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
            closed = p["market_closed"]
            if closed:
                raw = 0.12 * np.log(o / c.shift()) ** 2 / closed + 0.88 * raw / (1 - closed)
            days = f.index.to_series().diff().dt.total_seconds() / 86400
            variance = (raw / days).rolling(n).mean()
        elif name == "Volatility Zero Trend Close-to-Close":
            variance = (
                (returns**2 / (f.index.to_series().diff().dt.total_seconds() / 86400))
                .rolling(n)
                .mean()
            )
        elif name == "Volatility Index":
            from .special import volatility_stop

            return val(volatility_stop(f, n, mult, p["method"]))
        else:
            variance = returns.rolling(n).var(ddof=0 if name == "Historical Volatility" else 1)
        return val(np.sqrt(variance.clip(lower=0) * ann) * 100)
    raise ValueError("No implementation for " + name)
