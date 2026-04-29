"""TradingView-style indicator namespace.

This namespace is reserved for TradingView/Pine semantics.  The batch functions
below keep the same public shape as ``ta.*`` while importing pyneCore as the
declared backend dependency; a later Pine runtime adapter can swap the internals
without changing strategy code.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as pta
import pynecore.lib.ta as pyne_ta

from tradelearn.indicators.base import FunctionIndicator

BACKEND = "pynecore"
_PYNE_TA = pyne_ta


def _sma(close: pd.Series, length: int = 20) -> pd.Series:
    return pta.sma(close, length=length)


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    result = pta.bbands(close, length=length, std=std).iloc[:, :3].copy()
    result.columns = ["lower", "mid", "upper"]
    return result


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    return pta.rsi(close, length=length)


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    result = pta.macd(close, fast=fast, slow=slow, signal=signal).copy()
    result.columns = ["macd", "hist", "signal"]
    return result[["macd", "signal", "hist"]]


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    return pta.atr(high, low, close, length=length)


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    result = pta.adx(high, low, close, length=length).copy()
    result.columns = ["adx", "dmp", "dmn"]
    return result


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    return pta.vwap(high, low, close, volume)


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    result = pta.supertrend(high, low, close, length=length, multiplier=multiplier).copy()
    result.columns = ["supertrend", "direction", "long", "short"]
    return result


def _ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
) -> pd.DataFrame:
    visible, _forward = pta.ichimoku(high, low, close, tenkan=tenkan, kijun=kijun, senkou=senkou)
    result = visible.copy()
    result.columns = ["span_a", "span_b", "tenkan", "kijun", "chikou"]
    return result


sma = FunctionIndicator("tv.sma", _sma, {"length": 20})
bbands = FunctionIndicator("tv.bbands", _bbands, {"length": 20, "std": 2.0})
rsi = FunctionIndicator("tv.rsi", _rsi, {"length": 14})
macd = FunctionIndicator("tv.macd", _macd, {"fast": 12, "slow": 26, "signal": 9})
atr = FunctionIndicator("tv.atr", _atr, {"length": 14})
adx = FunctionIndicator("tv.adx", _adx, {"length": 14})
vwap = FunctionIndicator("tv.vwap", _vwap, {})
supertrend = FunctionIndicator("tv.supertrend", _supertrend, {"length": 10, "multiplier": 3.0})
ichimoku = FunctionIndicator("tv.ichimoku", _ichimoku, {"tenkan": 9, "kijun": 26, "senkou": 52})

SMA = sma
BBANDS = bbands
RSI = rsi
MACD = macd
ATR = atr
ADX = adx
VWAP = vwap
SUPERTREND = supertrend
ICHIMOKU = ichimoku

__all__ = [
    "ADX",
    "ATR",
    "BACKEND",
    "BBANDS",
    "ICHIMOKU",
    "MACD",
    "RSI",
    "SMA",
    "SUPERTREND",
    "VWAP",
    "adx",
    "atr",
    "bbands",
    "ichimoku",
    "macd",
    "rsi",
    "sma",
    "supertrend",
    "vwap",
]
