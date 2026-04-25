"""Tongdaxin-compatible indicator namespace.

The functions in this module wrap the vendored MyTT formulas and return
pandas objects with stable names while preserving the input index.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from tradelearn.indicators.base import FunctionIndicator
from tradelearn.query.tec import MyTT


def _index(values: pd.Series | Sequence[float]) -> pd.Index | None:
    """Return the index for pandas inputs."""
    if isinstance(values, pd.Series):
        return values.index
    return None


def _array(values: pd.Series | Sequence[float]) -> Sequence[float]:
    """Return raw values for MyTT functions."""
    if isinstance(values, pd.Series):
        return values.to_numpy()
    return values


def _series(
    values: Sequence[float],
    *,
    index: pd.Index | None,
    name: str,
) -> pd.Series:
    """Create a named series with an optional source index."""
    return pd.Series(values, index=index, name=name)


def _ma(close: pd.Series | Sequence[float], n: int = 5, **kwargs: int) -> pd.Series:
    """Simple moving average using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    return _series(MyTT.MA(_array(close), n), index=_index(close), name=f"MA_{n}")


def _ema(close: pd.Series | Sequence[float], n: int = 5, **kwargs: int) -> pd.Series:
    """Exponential moving average using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    return _series(MyTT.EMA(_array(close), n), index=_index(close), name=f"EMA_{n}")


def _sma(
    close: pd.Series | Sequence[float],
    n: int = 5,
    m: int = 1,
    **kwargs: int,
) -> pd.Series:
    """Chinese-style SMA using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    return _series(MyTT.SMA(_array(close), n, m), index=_index(close), name=f"SMA_{n}_{m}")


def _macd(
    close: pd.Series | Sequence[float],
    short: int = 12,
    long: int = 26,
    m: int = 9,
    **kwargs: int,
) -> pd.DataFrame:
    """MACD using Tongdaxin semantics."""
    short = kwargs.pop("SHORT", short)
    long = kwargs.pop("LONG", long)
    m = kwargs.pop("M", m)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    dif, dea, macd_value = MyTT.MACD(_array(close), SHORT=short, LONG=long, M=m)
    return pd.DataFrame(
        {"DIF": dif, "DEA": dea, "MACD": macd_value},
        index=_index(close),
    )


def _kdj(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    **kwargs: int,
) -> pd.DataFrame:
    """KDJ using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    k, d, j = MyTT.KDJ(
        _array(close),
        _array(high),
        _array(low),
        N=n,
        M1=m1,
        M2=m2,
    )
    return pd.DataFrame({"K": k, "D": d, "J": j}, index=_index(close))


def _rsi(close: pd.Series | Sequence[float], n: int = 24, **kwargs: int) -> pd.Series:
    """RSI using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")
    return _series(MyTT.RSI(_array(close), N=n), index=_index(close), name=f"RSI_{n}")


ma = FunctionIndicator("tdx.ma", _ma, {"n": 5})
MA = FunctionIndicator("tdx.MA", _ma, {"N": 5})
ema = FunctionIndicator("tdx.ema", _ema, {"n": 5})
EMA = FunctionIndicator("tdx.EMA", _ema, {"N": 5})
sma = FunctionIndicator("tdx.sma", _sma, {"n": 5, "m": 1})
SMA = FunctionIndicator("tdx.SMA", _sma, {"N": 5, "M": 1})
macd = FunctionIndicator("tdx.macd", _macd, {"short": 12, "long": 26, "m": 9})
MACD = FunctionIndicator("tdx.MACD", _macd, {"SHORT": 12, "LONG": 26, "M": 9})
kdj = FunctionIndicator("tdx.kdj", _kdj, {"n": 9, "m1": 3, "m2": 3})
KDJ = FunctionIndicator("tdx.KDJ", _kdj, {"N": 9, "M1": 3, "M2": 3})
rsi = FunctionIndicator("tdx.rsi", _rsi, {"n": 24})
RSI = FunctionIndicator("tdx.RSI", _rsi, {"N": 24})

__all__ = [
    "EMA",
    "KDJ",
    "MA",
    "MACD",
    "RSI",
    "SMA",
    "ema",
    "kdj",
    "ma",
    "macd",
    "rsi",
    "sma",
]
