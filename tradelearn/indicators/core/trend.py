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


macd = FunctionIndicator("macd", _macd, {"fast": 12, "slow": 26, "signal": 9})
