"""pandas-ta-classic backed TA-Lib-style indicator adapters."""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as pta


def _sma(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    return pta.sma(close, length=timeperiod)


def _ema(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    return pta.ema(close, length=timeperiod)


def _rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    return pta.rsi(close, length=timeperiod)


def _macd(
    close: pd.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> pd.DataFrame:
    result = pta.macd(close, fast=fastperiod, slow=slowperiod, signal=signalperiod).copy()
    result.columns = ["macd", "hist", "signal"]
    return result[["macd", "signal", "hist"]]


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    return pta.atr(high, low, close, length=timeperiod)


__all__ = ["_atr", "_ema", "_macd", "_rsi", "_sma"]
