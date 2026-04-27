from __future__ import annotations
from typing import Any, List, Dict, Type

from tradelearn.backtest.strategy import Strategy
from tradelearn.backtest.datafeed import DataFeed
from tradelearn.backtest.sizer import FixedSize
from tradelearn.backtest.analyzer import Analyzer

class Cerebro:
    """Main orchestrator for backtesting."""
    def __init__(self) -> None:
        self.datas: List[DataFeed] = []
        self.strats: List[Tuple[Type[Strategy], tuple, dict]] = []
        from tradelearn.backtest.brokers.rust_broker import RustBroker
        self.broker = RustBroker()
        self._sizer_spec = (FixedSize, {})
        self.analyzers: Dict[str, Tuple[Type[Analyzer], dict]] = {}

    def adddata(self, data: Any, name: str | None = None) -> None:
        if hasattr(data, 'columns') and hasattr(data, 'index'):
            # Auto-wrap pandas DataFrame
            data = DataFeed(data, name=name)
        elif name: 
            data._name = name
        self.datas.append(data)

    def addstrategy(self, strategy: Type[Strategy], *args: Any, **kwargs: Any) -> None:
        self.strats.append((strategy, args, kwargs))

    def addanalyzer(self, analyzer: Type[Analyzer], *args, _name=None, **kwargs) -> None:
        name = _name or analyzer.__name__.lower()
        self.analyzers[name] = (analyzer, kwargs)

    def run(self) -> List[Strategy]:
        # Main execution logic orchestrated here
        # This will call the engine's core loops
        from tradelearn.backtest.engine import run_backtest
        return run_backtest(self)
