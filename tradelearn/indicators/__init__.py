"""Generic technical indicator namespace.

The default ``ta.*`` functions are thin wrappers over pandas-ta-classic.
Same-name indicators may differ from future ``ta.tdx`` and ``ta.tv`` namespaces.
Choose the namespace for the market convention you need.
"""

from tradelearn.indicators import tdx
from tradelearn.indicators.base import FunctionIndicator, Indicator
from tradelearn.indicators.core.momentum import rsi
from tradelearn.indicators.core.overlap import bbands, sma
from tradelearn.indicators.core.trend import macd

tdx30 = tdx

__all__ = [
    "FunctionIndicator",
    "Indicator",
    "bbands",
    "macd",
    "rsi",
    "sma",
    "tdx",
    "tdx30",
]
