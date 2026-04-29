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
    Trade,
    _notify_order,
)
from tradelearn.backtest.strategy import Strategy as CoreStrategy


def __getattr__(name):
    if name == "SimBroker":
        from tradelearn.backtest.broker import RustBroker

        return RustBroker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BarRangeSlippage",
    "BarSnapshot",
    "CNAStockCommission",
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
    "DelayedLine",
    "IndicatorLine",
    "Lines",
    "LineSeries",
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
    "TieredCommission",
    "Trade",
    "_notify_order",
    "run_backtest",
]
