"""Trend indicators backed by pandas-ta-classic."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn.indicators.base import FunctionIndicator


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    NOTE: Values may differ from future ta.tdx.macd / ta.tv.macd variants.
    Choose the namespace that matches your market convention.
    """
    result = pta.macd(close, fast=fast, slow=slow, signal=signal).copy()
    result.columns = ["macd", "hist", "signal"]
    return result


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    """Average Directional Index.

    NOTE: Values may differ from ta.tdx.dmi / ta.tv.adx variants.
    Choose the namespace that matches your market convention.
    """
    result = pta.adx(high, low, close, length=length).copy()
    result.columns = ["adx", "dmp", "dmn"]
    return result


macd = FunctionIndicator("macd", _macd, {"fast": 12, "slow": 26, "signal": 9})
adx = FunctionIndicator("adx", _adx, {"length": 14})
