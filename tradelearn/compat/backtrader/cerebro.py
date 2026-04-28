from __future__ import annotations
from typing import Any, List, Dict, Type, Tuple

from tradelearn.compat.backtrader.strategy import Strategy
from tradelearn.compat.backtrader.datafeed import DataFeed
from tradelearn.compat.backtrader.sizer import FixedSize
from tradelearn.backtest.core.models import FixedCommission, FixedSlippage
from .analyzer import Analyzer

class Cerebro:
    """Main orchestrator for backtesting (Backtrader facade)."""
    def __init__(
        self,
        match_mode: str = 'exact',
        callback_batch: int = 1,
        trade_on_close: bool = False,
        exactbars: bool = False,
        stdstats: bool = True,
        slippage: Any | None = None,
        commission: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.datas: List[DataFeed] = []
        self.strats: List[Tuple[Type[Strategy], tuple, dict]] = []
        self.match_mode = match_mode
        self.callback_batch = int(callback_batch)
        self.trade_on_close = bool(trade_on_close)
        self.exactbars = bool(exactbars)
        self.stdstats = bool(stdstats)
        self.kwargs = kwargs
        from tradelearn.backtest.core.brokers.rust import RustBroker
        self.broker = RustBroker(match_mode=match_mode)
        self.broker._slippage_model = slippage or FixedSlippage()
        self.broker._commission_model = commission or FixedCommission()
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
