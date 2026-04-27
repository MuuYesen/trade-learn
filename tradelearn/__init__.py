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
        from tradelearn.backtest import engine
        return engine
    if name == "Cerebro":
        from tradelearn.backtest.engine import Cerebro
        return Cerebro
    if name == "Strategy":
        from tradelearn.backtest.engine import Strategy
        return Strategy
    if name == "Analyzer":
        from tradelearn.backtest.engine import Analyzer
        return Analyzer
    if name == "Sizer":
        from tradelearn.backtest.engine import Sizer
        return Sizer
    if name == "TimeFrame":
        from tradelearn.backtest.engine import TimeFrame
        return TimeFrame
    if name == "FixedSize":
        from tradelearn.backtest.engine import FixedSize
        return FixedSize
    if name == "PercentSizer":
        from tradelearn.backtest.engine import PercentSizer
        return PercentSizer
    if name == "AllInSizer":
        from tradelearn.backtest.engine import AllInSizer
        return AllInSizer
    if name == "DataFeed":
        from tradelearn.backtest.engine import DataFeed
        return DataFeed
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
