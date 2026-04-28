from __future__ import annotations

from typing import Any

from tradelearn.backtest.core.models import FixedCommission, FixedSlippage
from tradelearn.compat.backtrader.datafeed import DataFeed
from tradelearn.compat.backtrader.sizer import FixedSize
from tradelearn.compat.backtrader.strategy import Strategy

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
        mode: str = "backtest",
        **kwargs: Any,
    ) -> None:
        self.datas: list[DataFeed] = []
        self.strats: list[tuple[type[Strategy], tuple, dict]] = []
        self.match_mode = match_mode
        self.callback_batch = int(callback_batch)
        self.trade_on_close = bool(trade_on_close)
        self.exactbars = bool(exactbars)
        self.stdstats = bool(stdstats)
        if mode not in {"backtest", "paper", "live"}:
            raise ValueError("mode must be one of 'backtest', 'paper', or 'live'")
        self.mode = mode
        self.kwargs = kwargs
        from tradelearn.backtest.core.brokers.rust import RustBroker
        self.broker = RustBroker(match_mode=match_mode)
        self.broker._slippage_model = slippage or FixedSlippage()
        self.broker._commission_model = commission or FixedCommission()
        self._sizer_spec = (FixedSize, {})
        self.analyzers: dict[str, tuple[type[Analyzer], dict]] = {}

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

    def addstrategy(self, strategy: type[Strategy], *args: Any, **kwargs: Any) -> None:
        self.strats.append((strategy, args, kwargs))

    def addanalyzer(self, analyzer: type[Analyzer], *args, _name=None, **kwargs) -> None:
        name = _name or analyzer.__name__.lower()
        self.analyzers[name] = (analyzer, kwargs)

    def run(self) -> list[Strategy]:
        if self.mode != "backtest":
            return self._run_event_mode()
        # Main execution logic orchestrated here
        from tradelearn.backtest.core.engine import run_backtest
        return run_backtest(self)

    def _run_event_mode(self) -> list[Strategy]:
        from tradelearn.backtest.core.event_runner import EventRunner, LiveDriver, PaperDriver
        from tradelearn.core import BrokerEventPump

        strategy_cls, args, kwargs = self.strats[0]
        strategy = strategy_cls(*args, **kwargs)
        strategy.broker = self.broker
        sizer_cls, sizer_kwargs = self._sizer_spec
        strategy.setsizer(sizer_cls(**sizer_kwargs))
        strategy.start()

        pump = self.kwargs.get("broker_event_pump")
        if pump is None:
            poller = self.kwargs.get("broker_event_poller", lambda: ())
            pump = BrokerEventPump(poller)
        runner = EventRunner(
            strategy=strategy,
            broker_event_pump=pump,
            buffer_capacity=int(self.kwargs.get("buffer_capacity", 512)),
        )
        if self.mode == "paper":
            bars = self.kwargs.get("event_bars", ())
            PaperDriver(runner, bars).run_once()
        else:
            poller = self.kwargs.get("live_poller")
            if poller is None:
                raise ValueError("live mode requires live_poller")
            LiveDriver(runner, poller).poll_once()
        strategy.stop()
        return [strategy]
