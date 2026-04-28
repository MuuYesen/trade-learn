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


sma = FunctionIndicator("tv.sma", _sma, {"length": 20})
rsi = FunctionIndicator("tv.rsi", _rsi, {"length": 14})
macd = FunctionIndicator("tv.macd", _macd, {"fast": 12, "slow": 26, "signal": 9})

__all__ = ["BACKEND", "macd", "rsi", "sma"]
