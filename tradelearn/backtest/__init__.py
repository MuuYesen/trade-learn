from tradelearn.backtest.core.data import RollingBarBuffer, SharedBarBuffer
from tradelearn.backtest.core.engine import run_backtest
from tradelearn.backtest.core.event_runner import (
    EventRunner,
    EventSnapshot,
    HistoricalDriver,
    LiveDriver,
    PaperDriver,
)
from tradelearn.backtest.core.indicator_cache import (
    BatchIndicatorCache,
    IndicatorCache,
    RollingIndicatorCache,
)
from tradelearn.backtest.core.models import (
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
from tradelearn.backtest.core.strategy import Strategy as CoreStrategy


def __getattr__(name):
    if name == "SimBroker":
        from tradelearn.backtest.core.brokers.rust import RustBroker

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
    "ExecutedInfo",
    "EventRunner",
    "EventSnapshot",
    "HistoricalDriver",
    "BatchIndicatorCache",
    "FixedCommission",
    "FixedSlippage",
    "IndicatorCache",
    "Order",
    "PercentCommission",
    "PercentSlippage",
    "Position",
    "LiveDriver",
    "PaperDriver",
    "RollingBarBuffer",
    "RollingIndicatorCache",
    "SharedBarBuffer",
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
