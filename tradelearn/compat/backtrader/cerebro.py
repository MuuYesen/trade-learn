from __future__ import annotations

from typing import Any

from tradelearn.backtest.models import FixedCommission, FixedSlippage
from tradelearn.compat.backtrader.base import TimeFrame
from tradelearn.compat.backtrader.datafeed import DataFeed
from tradelearn.compat.backtrader.observers import ObserverCollection, Value
from tradelearn.compat.backtrader.sizers import FixedSize
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
        from tradelearn.backtest.broker import RustBroker
        self.broker = RustBroker(match_mode=match_mode)
        self.broker.configure_matching(
            trade_on_close=self.trade_on_close,
            slippage=slippage or FixedSlippage(),
            commission=commission or FixedCommission(),
        )
        self._sizer_spec = (FixedSize, {})
        self.analyzers: dict[str, tuple[type[Analyzer], dict]] = {}
        self.observers: dict[str, tuple[type[Any], dict]] = {}
        self.writers: list[tuple[type[Any], tuple, dict]] = []
        self.timers: list[tuple[tuple, dict]] = []
        self.calendars: list[Any] = []
        self._runstop = False

    def setcash(self, cash: float) -> None:
        self.broker.setcash(cash)

    def setcommission(self, commission: float = 0.0, margin: float = 0.0, mult: float = 1.0, 
                      comminfo: Any = None, name: str | None = None) -> None:
        """Set commission parameters or a custom CommInfo object."""
        if comminfo:
            self.broker.set_comminfo(comminfo)
        else:
            self.broker.setcommission(commission=commission, margin=margin, mult=mult)

    def set_coc(self, coc: bool = True) -> None:
        self.trade_on_close = bool(coc)
        self.broker.configure_matching(trade_on_close=self.trade_on_close)

    def adddata(self, data: Any, name: str | None = None) -> Any:
        if hasattr(data, 'columns') and hasattr(data, 'index'):
            data = DataFeed(data, name=name)
        elif name: 
            data._name = name
        self.datas.append(data)
        return data

    def resampledata(
        self,
        data: Any,
        timeframe: int,
        compression: int = 1,
        name: str | None = None,
        **kwargs: Any,
    ) -> DataFeed:
        from tradelearn.data.resampler import resample_frame

        source_frame = data if hasattr(data, "columns") and hasattr(data, "index") else data._frame
        resampled = resample_frame(source_frame, timeframe=timeframe, compression=compression)
        source_name = name or getattr(data, "_name", None) or "data"
        tf_name = TimeFrame.getname(timeframe, compression)
        return self.adddata(DataFeed(resampled, name=f"{source_name}_{tf_name}"), **kwargs)

    def replaydata(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        if args or {"timeframe", "compression"} & set(kwargs):
            return self.resampledata(data, *args, **kwargs)
        return self.adddata(data, name=kwargs.get("name"))

    def addstrategy(self, strategy: type[Strategy], *args: Any, **kwargs: Any) -> None:
        self.strats.append((strategy, args, kwargs))

    def optstrategy(self, strategy: type[Strategy], *args: Any, **kwargs: Any) -> None:
        from itertools import product

        grid_keys = [key for key, value in kwargs.items() if isinstance(value, (list, tuple))]
        if not grid_keys:
            self.addstrategy(strategy, *args, **kwargs)
            return
        fixed_kwargs = {key: value for key, value in kwargs.items() if key not in grid_keys}
        for values in product(*(kwargs[key] for key in grid_keys)):
            combo = dict(fixed_kwargs)
            combo.update(dict(zip(grid_keys, values, strict=True)))
            self.addstrategy(strategy, *args, **combo)

    def addanalyzer(self, analyzer: type[Analyzer], *args, _name=None, **kwargs) -> None:
        name = _name or kwargs.pop("name", None) or analyzer.__name__.lower()
        self.analyzers[name] = (analyzer, kwargs)

    def addobserver(self, observer: type[Any], *args: Any, _name=None, **kwargs: Any) -> None:
        name = _name or kwargs.pop("name", None) or observer.__name__.lower()
        self.observers[name] = (observer, kwargs)

    def addwriter(self, writer: type[Any], *args: Any, **kwargs: Any) -> None:
        self.writers.append((writer, args, kwargs))

    def addtimer(self, *args: Any, **kwargs: Any) -> None:
        self.timers.append((args, kwargs))

    def addcalendar(self, calendar: Any) -> None:
        self.calendars.append(calendar)

    def addsizer(self, sizer: type[Any], **kwargs: Any) -> None:
        self._sizer_spec = (sizer, kwargs)

    def setsizer(self, sizer: type[Any], **kwargs: Any) -> None:
        self.addsizer(sizer, **kwargs)

    def runstop(self) -> None:
        self._runstop = True

    def plot(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def run(self) -> list[Strategy]:
        if self.mode != "backtest":
            return self._run_event_mode()
        # Main execution logic orchestrated here
        from tradelearn.backtest.engine import run_backtest
        results: list[Strategy] = []
        specs = self.strats or [(Strategy, (), {})]
        original_strats = self.strats
        for spec in specs:
            self.strats = [spec]
            self._prepare_strategy_context()
            try:
                results.extend(run_backtest(self))
            finally:
                self._reset_strategy_context()
        self.strats = original_strats
        return results

    def _attach_observers(self, strategy: Any) -> None:
        strategy.observers = ObserverCollection()
        observer_specs = dict(self.observers)
        if self.stdstats:
            observer_specs.setdefault("value", (Value, {}))
        for name, (observer_cls, observer_kwargs) in observer_specs.items():
            observer = observer_cls(**observer_kwargs)
            observer._set(strategy)
            strategy.observers[name] = observer

    def _prepare_strategy_context(self) -> None:
        from tradelearn.compat.backtrader.base import (
            set_current_data,
            set_current_datas,
            set_current_strategy,
        )

        if self.datas:
            set_current_data(self.datas[0])
            set_current_datas(self.datas)
        else:
            set_current_data(None)
            set_current_datas([])
        set_current_strategy(None)

    def _bind_strategy_context(self, strategy: Any) -> None:
        from tradelearn.compat.backtrader.base import set_current_strategy

        set_current_strategy(strategy)

    def _reset_strategy_context(self) -> None:
        from tradelearn.compat.backtrader.base import (
            set_current_data,
            set_current_datas,
            set_current_strategy,
        )

        set_current_strategy(None)
        set_current_data(None)
        set_current_datas([])

    def _run_event_mode(self) -> list[Strategy]:
        from tradelearn.backtest.event_runner import EventRunner, LiveDriver, PaperDriver
        from tradelearn.core import BrokerEventPump

        self._prepare_strategy_context()
        strategy_cls, args, kwargs = self.strats[0]
        try:
            strategy = strategy_cls(*args, **kwargs)
            self._bind_strategy_context(strategy)
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
        finally:
            self._reset_strategy_context()
