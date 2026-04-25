"""Overlap indicators backed by pandas-ta-classic."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn.indicators.base import FunctionIndicator


def _sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average.

    NOTE: Values may differ from future ta.tdx.ma / ta.tv.sma variants.
    Choose the namespace that matches your market convention.
    """
    return pta.sma(close, length=period)


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.

    NOTE: Values may differ from future ta.tdx.boll / ta.tv.bbands variants.
    Choose the namespace that matches your market convention.
    """
    result = pta.bbands(close, length=length, std=std).iloc[:, :3].copy()
    result.columns = ["lower", "mid", "upper"]
    return result


sma = FunctionIndicator("sma", _sma, {"period": 20})
bbands = FunctionIndicator("bbands", _bbands, {"length": 20, "std": 2.0})
