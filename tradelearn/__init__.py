"""trade-learn public package namespace."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.2.0"

__all__ = ["__version__", "ta"]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        from tradelearn import indicators
        return indicators
    if name == "lab":
        from tradelearn import lab
        return lab
    if name == "cli":
        from tradelearn import cli
        return cli
    if name in ["ind", "indicators"]:
        from tradelearn.compat.backtrader import indicators
        return indicators
    if name == "analyzers":
        from tradelearn.backtest import analyzers
        return analyzers
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
        from tradelearn.backtest.core.models import TimeFrame
        return TimeFrame
    if name == "Analyzer":
        from tradelearn.backtest import Analyzer
        return Analyzer
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
