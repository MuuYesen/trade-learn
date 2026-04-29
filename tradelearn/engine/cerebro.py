from __future__ import annotations

from collections import OrderedDict
from typing import Any

from tradelearn.backtest.models import FixedCommission, FixedSlippage
from tradelearn.engine.base import TimeFrame
from tradelearn.engine.datafeed import DataFeed
from tradelearn.engine.observers import ObserverCollection, Value
from tradelearn.engine.sizers import FixedSize
from tradelearn.engine.strategy import Strategy

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
        self.datasbyname: OrderedDict[str, DataFeed] = OrderedDict()
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
        self.stores: list[Any] = []
        self.timers: list[tuple[tuple, dict]] = []
        self.calendars: list[Any] = []
        self._runstop = False
        self.signals: list[tuple[int, type, tuple, dict]] = []
        self._signal_strat: tuple[type, tuple, dict] | None = None
        self._signal_concurrent: bool = False
        self._signal_accumulate: bool = False

    def setcash(self, cash: float) -> None:
        self.broker.setcash(cash)

    def getbroker(self) -> Any:
        return self.broker

    def setbroker(self, broker: Any) -> None:
        self.broker = broker

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
        data_name = getattr(data, "_name", None)
        if data_name is not None:
            self.datasbyname[str(data_name)] = data
        return data

    def chaindata(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            raise ValueError("chaindata requires at least one data feed")
        first = None
        for data in args:
            added = self.adddata(data, name=kwargs.get("name") if len(args) == 1 else None)
            if first is None:
                first = added
        return first

    def rolloverdata(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            raise ValueError("rolloverdata requires at least one data feed")
        first = None
        for data in args:
            added = self.adddata(data, name=kwargs.get("name") if len(args) == 1 else None)
            if first is None:
                first = added
        return first

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

    def _writer_results(self, method: str) -> list[Any]:
        results: list[Any] = []
        for writer_cls, args, kwargs in self.writers:
            writer = writer_cls(*args, **kwargs)
            callback = getattr(writer, method, None)
            if not callable(callback):
                continue
            value = callback()
            if isinstance(value, list):
                results.extend(value)
            elif value is not None:
                results.append(value)
        return results

    def getwriterheaders(self) -> list[Any]:
        return self._writer_results("getheaders")

    def getwriterinfo(self) -> list[Any]:
        return self._writer_results("getinfo")

    def getwritervalues(self) -> list[Any]:
        return self._writer_results("getvalues")

    def addstore(self, store: Any) -> None:
        self.stores.append(store)

    def addtimer(self, *args: Any, **kwargs: Any) -> None:
        self.timers.append((args, kwargs))

    def addcalendar(self, calendar: Any) -> None:
        self.calendars.append(calendar)

    def addsizer(self, sizer: type[Any], **kwargs: Any) -> None:
        self._sizer_spec = (sizer, kwargs)

    def setsizer(self, sizer: type[Any], **kwargs: Any) -> None:
        self.addsizer(sizer, **kwargs)

    def add_signal(self, sigtype: int, sigcls: type, *sigargs: Any, **sigkwargs: Any) -> None:
        self.signals.append((sigtype, sigcls, sigargs, sigkwargs))

    def signal_strategy(self, stratcls: type, *args: Any, **kwargs: Any) -> None:
        self._signal_strat = (stratcls, args, kwargs)

    def signal_concurrent(self, onoff: bool) -> None:
        self._signal_concurrent = onoff

    def signal_accumulate(self, onoff: bool) -> None:
        self._signal_accumulate = onoff

    def runstop(self) -> None:
        self._runstop = True

    def plot(self, path: str | None = None, *args: Any, **kwargs: Any) -> Any:
        """Return or write a Lightweight Charts market replay HTML plot."""
        reporter = self._last_reporter()
        title = kwargs.pop("title", "Tradelearn Engine Market Replay")
        if path is not None:
            return reporter.write_market_replay_html(path, title=title)
        return [reporter.market_replay_html(title=title)]

    def report(self, path: str = "report.html", benchmark: Any | None = None) -> Any:
        """Write a Tradelearn HTML report for the most recent run."""
        return self._last_reporter().html(path, benchmark=benchmark)

    def run(self) -> list[Strategy]:
        if self.mode != "backtest":
            return self._run_event_mode()
        specs = list(self.strats)
        if self.signals:
            from tradelearn.engine.signal import SignalStrategy
            signal_args: tuple[Any, ...] = ()
            signal_kwargs: dict[str, Any] = {}
            strat_cls: type[Any] | None = None
            if self._signal_strat is not None:
                strat_cls, signal_args, signal_kwargs = self._signal_strat
            elif specs:
                first_cls, first_args, first_kwargs = specs[0]
                if issubclass(first_cls, SignalStrategy):
                    strat_cls = first_cls
                    signal_args = first_args
                    signal_kwargs = dict(first_kwargs)
                    specs = specs[1:]
            if strat_cls is None:
                strat_cls = SignalStrategy
            signal_kwargs = {
                **signal_kwargs,
                "signals": list(self.signals),
                "_accumulate": self._signal_accumulate,
                "_concurrent": self._signal_concurrent,
            }
            specs.insert(0, (
                strat_cls,
                signal_args,
                signal_kwargs,
            ))
        from tradelearn.backtest.engine import run_backtest
        results: list[Strategy] = []
        specs = specs or [(Strategy, (), {})]
        original_strats = self.strats
        for spec in specs:
            self._runstop = False
            self.strats = [spec]
            self._prepare_strategy_context()
            try:
                results.extend(run_backtest(self))
            finally:
                self._reset_strategy_context()
        self.strats = original_strats
        self._last_results = results
        return results

    def _last_reporter(self):
        results = getattr(self, "_last_results", None)
        if not results:
            raise RuntimeError("run() must be called before plot() or report()")
        stats = getattr(results[0], "stats", None)
        if stats is None:
            raise RuntimeError("last run did not produce stats")
        from tradelearn.report import Reporter

        return Reporter(stats, market_data=self._report_market_data())

    def _report_market_data(self):
        if not self.datas:
            return None
        return getattr(self.datas[0], "_frame", None)

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
        from tradelearn.engine.base import (
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
        from tradelearn.engine.base import set_current_strategy

        set_current_strategy(strategy)

    def _reset_strategy_context(self) -> None:
        from tradelearn.engine.base import (
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
