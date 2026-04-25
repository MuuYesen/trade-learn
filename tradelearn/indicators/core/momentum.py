"""Momentum indicators backed by pandas-ta-classic."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn.indicators.base import FunctionIndicator


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index.

    NOTE: Values may differ from future ta.tdx.rsi / ta.tv.rsi variants.
    Choose the namespace that matches your market convention.
    """
    return pta.rsi(close, length=length)


rsi = FunctionIndicator("rsi", _rsi, {"length": 14})
