from tradelearn.backtest.data import RollingBarBuffer, SharedBarBuffer
from tradelearn.backtest.engine import run_backtest
from tradelearn.backtest.event_runner import (
    EventRunner,
    EventSnapshot,
    HistoricalDriver,
    LiveDriver,
    PaperDriver,
)
from tradelearn.backtest.indicator_cache import (
    BatchIndicatorCache,
    IndicatorCache,
    RollingIndicatorCache,
)
from tradelearn.backtest.lines import DelayedLine, IndicatorLine, Lines, LineSeries
from tradelearn.backtest.models import (
    BarRangeSlippage,
    BarSnapshot,
    BaseAnalyzer,
    BaseBroker,
    BaseSizer,
    CNAStockCommission,
    CommissionModel,
    ExecutedInfo,
    FixedCommission,
    FixedSlippage,
    Order,
    PercentCommission,
    PercentSlippage,
    Position,
    SlippageModel,
    Stats,
    TieredCommission,
    TimeFrame,
    Trade,
    _notify_order,
)
from tradelearn.backtest.strategy import Strategy as CoreStrategy


def __getattr__(name):
    if name == "SimBroker":
        from tradelearn.backtest.broker import RustBroker

        return RustBroker
    if name == "Analyzer":
        from tradelearn.compat.backtrader.analyzer import Analyzer

        return Analyzer
    if name == "Cerebro":
        from tradelearn.compat.backtrader.cerebro import Cerebro

        return Cerebro
    if name == "Strategy":
        from tradelearn.compat.backtrader.strategy import Strategy

        return Strategy
    if name == "DataFeed":
        from tradelearn.compat.backtrader.datafeed import DataFeed

        return DataFeed
    if name == "FixedSize":
        from tradelearn.compat.backtrader.sizer import FixedSize

        return FixedSize
    if name == "PercentSizer":
        from tradelearn.compat.backtrader.sizer import PercentSizer

        return PercentSizer
    if name == "AllInSizer":
        from tradelearn.compat.backtrader.sizer import AllInSizer

        return AllInSizer
    if name == "analyzers":
        from tradelearn.compat.backtrader import analyzers

        return analyzers
    if name == "observers":
        from tradelearn.compat.backtrader import observers

        return observers
    if name == "grid_search":
        from tradelearn.compat.backtrader.grid import grid_search

        return grid_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Analyzer",
    "BarRangeSlippage",
    "BarSnapshot",
    "BaseAnalyzer",
    "BaseBroker",
    "BaseSizer",
    "CNAStockCommission",
    "Cerebro",
    "CommissionModel",
    "CoreStrategy",
    "DataFeed",
    "ExecutedInfo",
    "EventRunner",
    "EventSnapshot",
    "HistoricalDriver",
    "BatchIndicatorCache",
    "FixedCommission",
    "FixedSlippage",
    "FixedSize",
    "grid_search",
    "IndicatorCache",
    "DelayedLine",
    "IndicatorLine",
    "Lines",
    "LineSeries",
    "Order",
    "PercentCommission",
    "PercentSlippage",
    "PercentSizer",
    "Position",
    "LiveDriver",
    "PaperDriver",
    "RollingBarBuffer",
    "RollingIndicatorCache",
    "SharedBarBuffer",
    "AllInSizer",
    "analyzers",
    "observers",
    "SimBroker",
    "SlippageModel",
    "Stats",
    "Strategy",
    "TieredCommission",
    "TimeFrame",
    "Trade",
    "_notify_order",
    "run_backtest",
]
