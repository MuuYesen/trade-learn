"""trade-learn public package namespace."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.2.0"

__all__ = ["__version__", "ta", "pta", "talib", "tdx", "tv"]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        return importlib.import_module("tradelearn.indicators")
    if name == "pta":
        return importlib.import_module("tradelearn.indicators.pta")
    if name == "talib":
        return importlib.import_module("tradelearn.indicators.talib")
    if name == "tdx":
        return importlib.import_module("tradelearn.indicators.tdx")
    if name == "tv":
        return importlib.import_module("tradelearn.indicators.tv")
    if name == "lab":
        from tradelearn import lab
        return lab
    if name == "cli":
        from tradelearn import cli
        return cli
    if name == "feeds":
        return importlib.import_module("tradelearn.engine.feeds")
    if name == "analyzers":
        from tradelearn.engine import analyzers
        return analyzers
    if name == "observers":
        from tradelearn.engine import observers
        return observers
    if name == "sizers":
        from tradelearn.engine import sizers
        return sizers
    if name == "Cerebro":
        from tradelearn.engine import Cerebro
        return Cerebro
    if name == "OptReturn":
        from tradelearn.engine import OptReturn
        return OptReturn
    if name == "Strategy":
        from tradelearn.engine import Strategy
        return Strategy
    if name == "DataFeed":
        from tradelearn.engine import DataFeed
        return DataFeed
    if name == "Sizer":
        from tradelearn.engine import Sizer
        return Sizer
    if name == "FixedSize":
        from tradelearn.engine import FixedSize
        return FixedSize
    if name == "PercentSizer":
        from tradelearn.engine import PercentSizer
        return PercentSizer
    if name == "AllInSizer":
        from tradelearn.engine import AllInSizer
        return AllInSizer
    if name == "TimeFrame":
        from tradelearn.engine import TimeFrame
        return TimeFrame
    if name == "Order":
        from tradelearn.engine import Order
        return Order
    if name == "CommInfoBase":
        from tradelearn.engine import CommInfoBase
        return CommInfoBase
    if name == "Analyzer":
        from tradelearn.engine import Analyzer
        return Analyzer
    if name in {"num2date", "date2num"}:
        return getattr(importlib.import_module("tradelearn.engine"), name)
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
