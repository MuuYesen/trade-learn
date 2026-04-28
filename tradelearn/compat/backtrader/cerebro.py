from __future__ import annotations
from typing import Any, List, Dict, Type, Tuple

from tradelearn.compat.backtrader.strategy import Strategy
from tradelearn.compat.backtrader.datafeed import DataFeed
from tradelearn.compat.backtrader.sizer import FixedSize
from .analyzer import Analyzer

class Cerebro:
    """Main orchestrator for backtesting (Backtrader facade)."""
    def __init__(self, match_mode: str = 'exact') -> None:
        self.datas: List[DataFeed] = []
        self.strats: List[Tuple[Type[Strategy], tuple, dict]] = []
        self.match_mode = match_mode
        from tradelearn.backtest.brokers.rust_broker import RustBroker
        self.broker = RustBroker(match_mode=match_mode)
        self._sizer_spec = (FixedSize, {})
        self.analyzers: Dict[str, Tuple[Type[Analyzer], dict]] = {}

    def setcash(self, cash: float) -> None:
        self.broker._cash = self.broker._active_cash = cash

    def setcommission(self, commission: float = 0.0, margin: float = 0.0, mult: float = 1.0, 
                      comminfo: Any = None, name: str | None = None) -> None:
        """Set commission parameters or a custom CommInfo object."""
        if comminfo:
            self.broker.set_comminfo(comminfo)
        else:
            self.broker.commission_ratio = commission
            self.broker._mult = mult

    def adddata(self, data: Any, name: str | None = None) -> None:
        if hasattr(data, 'columns') and hasattr(data, 'index'):
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
        from tradelearn.backtest.core.engine import run_backtest
        return run_backtest(self)
