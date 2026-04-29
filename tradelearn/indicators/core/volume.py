"""Volume indicators backed by pandas-ta-classic."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn.indicators.base import FunctionIndicator


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price.

    NOTE: Values may differ from future ta.tdx.vwap / ta.tv.vwap variants.
    Choose the namespace that matches your market convention.
    """
    return pta.vwap(high, low, close, volume)


vwap = FunctionIndicator("vwap", _vwap, {})
