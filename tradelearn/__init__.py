"""trade-learn public package namespace."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.2.0"

__all__ = ["__version__", "ta"]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        return importlib.import_module("tradelearn.indicators")
    if name == "lab":
        from tradelearn import lab
        return lab
    if name == "cli":
        from tradelearn import cli
        return cli
    if name in ["ind", "indicators"]:
        return importlib.import_module("tradelearn.compat.backtrader.indicators")
    if name == "feeds":
        return importlib.import_module("tradelearn.compat.backtrader.feeds")
    if name == "analyzers":
        from tradelearn.compat.backtrader import analyzers
        return analyzers
    if name == "observers":
        from tradelearn.compat.backtrader import observers
        return observers
    if name == "sizers":
        from tradelearn.compat.backtrader import sizers
        return sizers
    if name == "Cerebro":
        from tradelearn.compat.backtrader import Cerebro
        return Cerebro
    if name == "Strategy":
        from tradelearn.compat.backtrader import Strategy
        return Strategy
    if name == "DataFeed":
        from tradelearn.compat.backtrader import DataFeed
        return DataFeed
    if name == "Sizer":
        from tradelearn.compat.backtrader import Sizer
        return Sizer
    if name == "FixedSize":
        from tradelearn.compat.backtrader import FixedSize
        return FixedSize
    if name == "PercentSizer":
        from tradelearn.compat.backtrader import PercentSizer
        return PercentSizer
    if name == "AllInSizer":
        from tradelearn.compat.backtrader import AllInSizer
        return AllInSizer
    if name == "TimeFrame":
        from tradelearn.backtest.models import TimeFrame
        return TimeFrame
    if name == "Order":
        from tradelearn.compat.backtrader import Order
        return Order
    if name == "CommInfoBase":
        from tradelearn.compat.backtrader import CommInfoBase
        return CommInfoBase
    if name == "Analyzer":
        from tradelearn.compat.backtrader import Analyzer
        return Analyzer
    if name in {"num2date", "date2num"}:
        return getattr(importlib.import_module("tradelearn.compat.backtrader"), name)
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
