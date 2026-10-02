"""pandas-ta-classic backed indicator adapters."""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as pta


def _sma(close: pd.Series, length: int = 30) -> pd.Series:
    """Return pandas-ta-classic simple moving averages for the requested close-price window."""
    return pta.sma(close, length=length)


def _ema(close: pd.Series, length: int = 30) -> pd.Series:
    """Return pandas-ta-classic exponential moving averages for the requested window."""
    return pta.ema(close, length=length)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute close-price relative strength using pandas-ta-classic smoothing."""
    return pta.rsi(close, length=length)


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Normalize pandas-ta-classic MACD output to macd, signal, and hist columns."""
    result = pta.macd(close, fast=fast, slow=slow, signal=signal).copy()
    result.columns = ["macd", "hist", "signal"]
    return result[["macd", "signal", "hist"]]


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Compute pandas-ta-classic average true range from high, low, and close prices."""
    return pta.atr(high, low, close, length=length)


__all__ = ["_atr", "_ema", "_macd", "_rsi", "_sma"]
