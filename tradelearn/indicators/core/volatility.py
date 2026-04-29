"""Volatility indicators backed by pandas-ta-classic."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn.indicators.base import FunctionIndicator


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Average True Range.

    NOTE: Values may differ from ta.tdx.atr / ta.tv.atr variants.
    Choose the namespace that matches your market convention.
    """
    return pta.atr(high, low, close, length=length)


atr = FunctionIndicator("atr", _atr, {"length": 14})
