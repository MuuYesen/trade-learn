"""Small MyTT-compatible formula surface used by ``ta.tdx`` tests.

The v2 indicator facade keeps the historical ``tradelearn.query.tec.MyTT``
import path as a compatibility shim.  Implementations are vectorized with
pandas/numpy and are intentionally deterministic for batch backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _s(values) -> pd.Series:
    return pd.Series(_arr(values))


def _nan_to_zero(values) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def MA(values, N=5):
    return _s(values).rolling(int(N), min_periods=1).mean().to_numpy()


def EMA(values, N=5):
    return _s(values).ewm(span=int(N), adjust=False).mean().to_numpy()


def SMA(values, N=5, M=1):
    values = _arr(values)
    n = int(N)
    m = float(M)
    out = np.empty_like(values, dtype=float)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (m * values[i] + (n - m) * out[i - 1]) / n
    return out


def WMA(values, N=5):
    weights = np.arange(1, int(N) + 1, dtype=float)
    return _s(values).rolling(int(N), min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True,
    ).to_numpy()


def MACD(values, SHORT=12, LONG=26, M=9):
    dif = EMA(values, SHORT) - EMA(values, LONG)
    dea = EMA(dif, M)
    return dif, dea, (dif - dea) * 2


def KDJ(close, high, low, N=9, M1=3, M2=3):
    close_s = _s(close)
    high_n = _s(high).rolling(int(N), min_periods=1).max()
    low_n = _s(low).rolling(int(N), min_periods=1).min()
    rsv = ((close_s - low_n) / (high_n - low_n).replace(0, np.nan) * 100).fillna(0).to_numpy()
    k = SMA(rsv, M1, 1)
    d = SMA(k, M2, 1)
    return k, d, 3 * k - 2 * d


def RSI(close, N=24):
    diff = _s(close).diff().fillna(0)
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    total = _s(SMA(up + down, N, 1)).replace(0, np.nan)
    return (_s(SMA(up, N, 1)) / total * 100).fillna(0).to_numpy()


def WR(close, high, low, N=10, N1=6):
    def _wr(n):
        high_n = _s(high).rolling(int(n), min_periods=1).max()
        low_n = _s(low).rolling(int(n), min_periods=1).min()
        width = (high_n - low_n).replace(0, np.nan)
        return ((high_n - _s(close)) / width * 100).fillna(0).to_numpy()
    return _wr(N), _wr(N1)


def BIAS(close, L1=6, L2=12, L3=24):
    close_s = _s(close)
    def _bias(n):
        ma = _s(MA(close, n)).replace(0, np.nan)
        return ((close_s - ma) / ma * 100).fillna(0).to_numpy()
    return _bias(L1), _bias(L2), _bias(L3)


def BOLL(close, N=20, P=2):
    mid = MA(close, N)
    std = _s(close).rolling(int(N), min_periods=1).std(ddof=0).fillna(0).to_numpy()
    return mid + float(P) * std, mid, mid - float(P) * std


def PSY(close, N=12, M=6):
    up = (_s(close).diff() > 0).astype(float)
    psy = up.rolling(int(N), min_periods=1).mean().to_numpy() * 100
    return psy, MA(psy, M)


def CCI(close, high, low, N=14):
    tp = (_s(close) + _s(high) + _s(low)) / 3
    ma = tp.rolling(int(N), min_periods=1).mean()
    md = (tp - ma).abs().rolling(int(N), min_periods=1).mean().replace(0, np.nan)
    return ((tp - ma) / (0.015 * md)).fillna(0).to_numpy()


def ATR(close, high, low, N=20):
    close_s = _s(close)
    tr = pd.concat([
        (_s(high) - _s(low)).abs(),
        (_s(high) - close_s.shift(1)).abs(),
        (_s(low) - close_s.shift(1)).abs(),
    ], axis=1).max(axis=1).fillna(0)
    return MA(tr, N)


def BBI(close, M1=3, M2=6, M3=12, M4=20):
    return (MA(close, M1) + MA(close, M2) + MA(close, M3) + MA(close, M4)) / 4


def DMI(close, high, low, M1=14, M2=6):
    high_s, low_s = _s(high), _s(low)
    tr = ATR(close, high, low, 1)
    hd = high_s.diff().fillna(0)
    ld = -low_s.diff().fillna(0)
    p_dm = np.where((hd > 0) & (hd > ld), hd, 0.0)
    m_dm = np.where((ld > 0) & (ld > hd), ld, 0.0)
    tr_ma = _s(MA(tr, M1)).replace(0, np.nan)
    pdi = (_s(MA(p_dm, M1)) / tr_ma * 100).fillna(0).to_numpy()
    mdi = (_s(MA(m_dm, M1)) / tr_ma * 100).fillna(0).to_numpy()
    dx = (np.abs(pdi - mdi) / pd.Series(pdi + mdi).replace(0, np.nan) * 100).fillna(0).to_numpy()
    adx = MA(dx, M2)
    return pdi, mdi, adx, MA(adx, M2)


def TRIX(close, M1=12, M2=20):
    tr = _s(EMA(EMA(EMA(close, M1), M1), M1)).pct_change().fillna(0).to_numpy() * 100
    return tr, MA(tr, M2)


def VR(close, volume, M1=26):
    close_s = _s(close)
    vol = _s(volume)
    up = vol.where(close_s > close_s.shift(1), 0).rolling(int(M1), min_periods=1).sum()
    down = vol.where(close_s <= close_s.shift(1), 0)
    down = down.rolling(int(M1), min_periods=1).sum().replace(0, np.nan)
    return (up / down * 100).fillna(0).to_numpy()


def MTM(close, N=12, M=6):
    mtm = (_s(close) - _s(close).shift(int(N))).fillna(0).to_numpy()
    return mtm, MA(mtm, M)


def ROC(close, N=12, M=6):
    previous = _s(close).shift(int(N)).replace(0, np.nan)
    roc = ((_s(close) - _s(close).shift(int(N))) / previous * 100).fillna(0).to_numpy()
    return roc, MA(roc, M)


def TAQ(high, low, N=20):
    up = _s(high).rolling(int(N), min_periods=1).max().to_numpy()
    down = _s(low).rolling(int(N), min_periods=1).min().to_numpy()
    return up, (up + down) / 2, down


def KTN(close, high, low, N=20, M=10):
    mid = EMA(close, N)
    atr = ATR(close, high, low, M)
    return mid + atr, mid, mid - atr


def CR(close, high, low, N=20):
    mid_prev = ((_s(high) + _s(low)) / 2).shift(1)
    up = (_s(high) - mid_prev).clip(lower=0).rolling(int(N), min_periods=1).sum()
    down = (mid_prev - _s(low)).clip(lower=0)
    down = down.rolling(int(N), min_periods=1).sum().replace(0, np.nan)
    return (up / down * 100).fillna(0).to_numpy()


def EMV(high, low, volume, N=14, M=9):
    mid = (_s(high) + _s(low)) / 2
    distance = mid.diff().fillna(0)
    box_ratio = (_s(volume) / 10000) / (_s(high) - _s(low)).replace(0, np.nan)
    emv = (distance / box_ratio).replace([np.inf, -np.inf], 0).fillna(0).to_numpy()
    emv = MA(emv, N)
    return emv, MA(emv, M)


def DPO(close, M1=20, M2=10, M3=6):
    dpo = (_s(close) - _s(MA(close, M1)).shift(int(M2))).fillna(0).to_numpy()
    return dpo, MA(dpo, M3)


def BRAR(open_, close, high, low, M1=26):
    ar_up = (_s(high) - _s(open_)).rolling(int(M1), min_periods=1).sum()
    ar_down = (_s(open_) - _s(low)).rolling(int(M1), min_periods=1).sum()
    ar = (ar_up / ar_down.replace(0, np.nan) * 100).fillna(0).to_numpy()
    br_up = (_s(high) - _s(close).shift(1)).clip(lower=0)
    br_up = br_up.rolling(int(M1), min_periods=1).sum()
    br_down = (_s(close).shift(1) - _s(low)).clip(lower=0)
    br_down = br_down.rolling(int(M1), min_periods=1).sum().replace(0, np.nan)
    br = (br_up / br_down * 100).fillna(0).to_numpy()
    return ar, br


def DFMA(close, N1=10, N2=50, M=10):
    dif = MA(close, N1) - MA(close, N2)
    return dif, MA(dif, M)


def MASS(high, low, N1=9, N2=25, M=6):
    spread = _s(high) - _s(low)
    mass = (_s(EMA(spread, N1)) / _s(EMA(EMA(spread, N1), N1)).replace(0, np.nan)).fillna(0)
    mass = mass.rolling(int(N2), min_periods=1).sum().to_numpy()
    return mass, MA(mass, M)


def EXPMA(close, N1=12, N2=50):
    return EMA(close, N1), EMA(close, N2)


def OBV(close, volume):
    direction = np.sign(_s(close).diff().fillna(0))
    return (direction * _s(volume)).cumsum().to_numpy()


def MFI(close, high, low, volume, N=14):
    typical = (_s(close) + _s(high) + _s(low)) / 3
    money = typical * _s(volume)
    pos = money.where(typical > typical.shift(1), 0).rolling(int(N), min_periods=1).sum()
    neg = money.where(typical <= typical.shift(1), 0)
    neg = neg.rolling(int(N), min_periods=1).sum().replace(0, np.nan)
    return (100 - 100 / (1 + pos / neg)).fillna(0).to_numpy()


def ASI(open_, close, high, low, M1=26, M2=10):
    si = (_s(close) - _s(close).shift(1)).fillna(0) + (_s(close) - _s(open_)) / 2
    asi = si.cumsum().to_numpy()
    return asi, MA(asi, M2)


def XSII(close, high, low, N=102, M=7):
    mid = MA(close, M)
    td1 = _s(high).rolling(int(N), min_periods=1).max().to_numpy()
    td4 = _s(low).rolling(int(N), min_periods=1).min().to_numpy()
    return td1, mid + (td1 - td4) / 3, mid - (td1 - td4) / 3, td4
