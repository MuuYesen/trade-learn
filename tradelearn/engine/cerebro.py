from __future__ import annotations

from collections import OrderedDict
from typing import Any

from tradelearn.backtest.feed import is_panel_ohlcv_frame, panel_symbol_level
from tradelearn.backtest.reporting import reporter_from_results
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.core.costs import FixedCommission, FixedSlippage
from tradelearn.core.logging import get_logger
from tradelearn.core.strategy_events import invoke_component
from tradelearn.engine.analyzer import AnalyzerCollection
from tradelearn.engine.base import TimeFrame, Params, engine_context
from tradelearn.engine.datafeed import DataFeed
from tradelearn.engine.observers import ObserverCollection, Value
from tradelearn.engine.sizers import FixedSize
from tradelearn.engine.strategy import Strategy

from .analyzer import Analyzer

LOGGER = get_logger("engine.cerebro")


class OptReturn:
    """Lightweight optimization result returned by ``Cerebro.run(optreturn=True)``."""

    def __init__(self, params: Any, **kwargs: Any) -> None:
        self.p = self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)


def _strategy_name(strategy_cls: type[Any]) -> str:
    return getattr(strategy_cls, "__name__", str(strategy_cls))


def _feed_rows(datas: list[Any]) -> int:
    return int(sum(len(getattr(data, "_frame", ())) for data in datas))


class Cerebro:
    """Main orchestrator for backtesting (Backtrader facade)."""
    _bt_defaults = dict(preload=True, runonce=True, maxcpus=None, stdstats=True,
        oldbuysell=False, oldtrades=False, lookahead=0, exactbars=False,
        optdatas=True, optreturn=True, objcache=False, live=False, writer=False,
        tradehistory=False, oldsync=False, tz=None, cheat_on_open=False,
        broker_coo=True, quicknotify=False)

    @classmethod
    def _validate_options(cls, options):
        unknown = options.keys() - cls._bt_defaults.keys()
        if unknown:
            raise TypeError(f"unknown Cerebro parameters: {sorted(unknown)}")
        supported = {"stdstats", "optreturn", "runonce", "maxcpus"}
        for key, value in options.items():
            if key not in supported and value != cls._bt_defaults[key]:
                raise NotImplementedError(f"Cerebro {key}={value!r} is not implemented")
        if options.get("maxcpus") not in (None, 1):
            raise NotImplementedError("parallel optimization is not implemented; use maxcpus=1")

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError("Cerebro accepts keyword parameters")
        self._validate_options(kwargs)
        options = {**self._bt_defaults, **kwargs}
        self._initialize(stdstats=options["stdstats"], optreturn=options["optreturn"])
        self.p = self.params = Params(options)

    def _initialize(
        self,
        match_mode: str = 'exact',
        callback_batch: int = 1,
        trade_on_close: bool = False,
        exactbars: bool = False,
        stdstats: bool = True,
        slippage: Any | None = None,
        commission: Any | None = None,
        mode: str = "backtest",
        stats_mode: str = "full",
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
        if stats_mode not in {"full", "lazy"}:
            raise ValueError("stats_mode must be one of 'full' or 'lazy'")
        self.stats_mode = stats_mode
        self.optreturn = bool(kwargs.pop("optreturn", True))
        self.kwargs = kwargs
        from tradelearn.backtest.broker import RustBroker
        self.setbroker(RustBroker(match_mode=match_mode))
        self.broker.configure_matching(
            trade_on_close=self.trade_on_close,
            slippage=slippage or FixedSlippage(),
            commission=commission or FixedCommission(),
        )
        self.runtime_config = BacktestRuntimeConfig.from_owner(self)
        self._sizer_spec = (FixedSize, {})
        self.analyzers: dict[str, tuple[type[Analyzer], tuple[Any, ...], dict[str, Any]]] = {}
        self._analyzer_name_counts: dict[str, int] = {}
        self.observers: dict[str, tuple[type[Any], tuple[Any, ...], dict]] = {}
        self.writers: list[tuple[type[Any], tuple, dict]] = []
        self.stores: list[Any] = []
        self.timers: list[tuple[tuple, dict]] = []
        self.calendars: list[Any] = []
        self._runstop = False
        self.signals: list[tuple[int, type, tuple, dict]] = []
        self._signal_strat: tuple[type, tuple, dict] | None = None
        self._signal_concurrent: bool = False
        self._signal_accumulate: bool = False
        self._dooptimize = False
        self.optcbs: list[Any] = []

    def _setcash(self, cash: float) -> None:
        self.broker.setcash(cash)

    def getbroker(self) -> Any:
        return self.broker

    def setbroker(self, broker: Any) -> Any:
        from tradelearn.backtest.broker import RustBroker
        if isinstance(broker, RustBroker):
            from tradelearn.engine.comminfo import CommInfoBase
            if broker._mult != 1:
                raise NotImplementedError("commission economics with a contract multiplier are not qualified")
            broker._bt_notifications = True
            broker._comminfo_factory = CommInfoBase
            if broker._comminfo is None:
                broker._set_comminfo(CommInfoBase(commission=broker.commission_ratio,percabs=True))
        self.broker = broker
        broker.cerebro = self
        return broker

    def _setcommission(self, commission: float = 0.0, margin: float = 0.0, mult: float = 1.0,
                      comminfo: Any = None, name: str | None = None) -> None:
        """Set commission parameters or a custom CommInfo object."""
        if comminfo:
            self.broker._set_comminfo(comminfo)
        else:
            self.broker.setcommission(commission=commission, margin=margin, mult=mult)

    def _set_coc(self, coc: bool = True) -> None:
        self.trade_on_close = bool(coc)
        self.broker.configure_matching(trade_on_close=self.trade_on_close)

    def adddata(self, data: Any, name: Any | None = None) -> Any:
        if hasattr(data,"columns") and hasattr(data,"index"):
            raise TypeError("Cerebro.adddata requires a feed; use feeds.PandasData(dataname=frame)")
        if not hasattr(data,"lines") and not getattr(data,"is_live",False):
            raise TypeError("Cerebro.adddata requires a data feed")
        if name is not None:
            data._name=name
        data._id=len(self.datas)+1
        self.datas.append(data)
        data_name = getattr(data, "_name", None)
        if data_name is not None:
            self.datasbyname[data_name] = data
        return data

    def chaindata(self, *args, **kwargs):
        raise NotImplementedError("chaindata requires native feed lifecycle semantics; no multi-feed substitution")

    def rolloverdata(self, *args, **kwargs):
        raise NotImplementedError("rolloverdata requires native feed lifecycle semantics; no multi-feed substitution")

    def resampledata(self, dataname, name=None, **kwargs):
        raise NotImplementedError("resampledata incremental feed semantics are not implemented; prepare explicit source feeds")

    def replaydata(self, data, **kwargs):
        raise NotImplementedError("replaydata incremental replay is not implemented")

    def addstrategy(self, strategy: type[Strategy], *args: Any, **kwargs: Any) -> int:
        self.strats.append((strategy, args, kwargs))
        return len(self.strats)-1

    def optstrategy(self, strategy: type[Strategy], *args: Any, **kwargs: Any) -> None:
        from itertools import product

        self._dooptimize = True
        if not kwargs:
            self.addstrategy(strategy, *args)
            return
        opt_items = {
            key: value if isinstance(value, (list, tuple, range)) else (value,)
            for key, value in kwargs.items()
        }
        for values in product(*opt_items.values()):
            combo = dict(zip(opt_items.keys(), values, strict=True))
            self.addstrategy(strategy, *args, **combo)

    def optcallback(self, cb: Any) -> None:
        self.optcbs.append(cb)

    def addanalyzer(self, ancls: type[Analyzer], *args: Any, **kwargs: Any) -> None:
        base_name = kwargs.pop("_name", None) or ancls.__name__.lower()
        suffix = self._analyzer_name_counts.get(base_name, 0)
        self._analyzer_name_counts[base_name] = suffix + 1
        name = base_name + (str(suffix) if suffix else "")
        self.analyzers[name] = (ancls, args, kwargs)

    def addobserver(self, obscls: type[Any], *args: Any, **kwargs: Any) -> None:
        # Only obsname selects the BT collection; other keywords belong to the observer.
        name = kwargs.pop("obsname", None) or obscls.__name__.lower()
        self.observers[name] = (obscls, args, kwargs)

    def addwriter(self, writer: type[Any], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("writer lifecycle is not implemented; use an independent Reporter")

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

    def _getwriterheaders(self) -> list[Any]:
        return self._writer_results("getheaders")

    def _getwriterinfo(self) -> list[Any]:
        return self._writer_results("getinfo")

    def _getwritervalues(self) -> list[Any]:
        return self._writer_results("getvalues")

    def addstore(self, store: Any) -> None:
        self.stores.append(store)

    def add_timer(self, when, offset=__import__("datetime").timedelta(0), repeat=__import__("datetime").timedelta(0), weekdays=[], weekcarry=False, monthdays=[], monthcarry=True, allow=None, tzdata=None, cheat=False, *args, **kwargs):
        raise NotImplementedError("timer scheduling is not implemented")

    def addcalendar(self, calendar: Any) -> None:
        raise NotImplementedError("calendar scheduling is not implemented")

    def addsizer(self, sizer: type[Any], **kwargs: Any) -> None:
        self._sizer_spec = (sizer, kwargs)

    def _setsizer(self, sizer: type[Any], **kwargs: Any) -> None:
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
        driver = getattr(self, '_active_store_driver', None)
        if driver is not None:
            driver.stop()

    def plot(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Reject the unqualified BT plotting backend explicitly."""
        raise NotImplementedError("BT plot is not qualified; use tradelearn.report.Reporter explicitly")

    def _report(
        self,
        path: str = "report.html",
        benchmark: Any | None = None,
        sections: Any | None = None,
    ) -> Any:
        """Write a Tradelearn report for the most recent run."""
        return self._last_reporter().report(path, benchmark=benchmark, sections=sections)

    def run(self, **kwargs: Any) -> list[Strategy]:
        self._validate_options(kwargs)
        from tradelearn_runtime.assembly import select_driver
        driver = select_driver(self)
        if kwargs.get('monitor') is not None and driver is None:
            raise ValueError('remote monitor requires a Store-owned native driver')
        if driver is not None:
            return driver._run_cerebro(self, kwargs)
        self._apply_run_kwargs(kwargs)
        self.trade_on_close = bool(getattr(self.broker, "_trade_on_close", self.trade_on_close))
        has_live_data = any(bool(getattr(data, "is_live", False)) for data in self.datas)
        if self.mode == "backtest" and has_live_data:
            self.mode = "live"
            if "live_poller" not in self.kwargs:
                self.kwargs["live_poller"] = self._live_data_poller
            broker_connect = getattr(self.broker, "connect", None)
            if callable(broker_connect) and not self.broker.is_connected():
                broker_connect()
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
        LOGGER.info(
            "Cerebro run started mode=%s strategies=%s feeds=%s rows=%s cash=%s commission=%s",
            self.mode,
            ",".join(_strategy_name(spec[0]) for spec in specs),
            len(self.datas),
            _feed_rows(self.datas),
            self.broker.getcash(),
            getattr(self.broker, "_commission", "n/a"),
        )
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
        formatted = self._format_run_results(results)
        if results:
            summary = getattr(getattr(results[-1], "stats", None), "summary", {}) or {}
            LOGGER.info(
                "Cerebro run finished strategies=%s final_value=%s return_pct=%s "
                "total_trades=%s total_fills=%s",
                len(results),
                summary.get("final_value", "n/a"),
                summary.get("return_pct", "n/a"),
                summary.get("total_trades", "n/a"),
                summary.get("total_fills", "n/a"),
            )
        return formatted

    def _apply_run_kwargs(self, kwargs):
        self._validate_options(kwargs)
        if "optreturn" in kwargs:
            self.optreturn = bool(kwargs["optreturn"])
        if "stdstats" in kwargs:
            self.stdstats = bool(kwargs["stdstats"])
        for key, value in kwargs.items():
            setattr(self.p, key, value)

    def _live_data_poller(self):
        bars = []
        for data in self.datas:
            poll = getattr(data, "poll", None)
            if callable(poll):
                bars.extend(list(poll()))
        return bars

    def _format_run_results(self, results):
        if not self._dooptimize:
            return results
        formatted = [[OptReturn(strategy.params,
                      analyzers=getattr(strategy,"analyzers",None), strategycls=type(strategy))
                      if self.optreturn else strategy] for strategy in results]
        for combination in formatted:
            for callback in self.optcbs:
                callback(combination)
        return formatted

    def _last_reporter(self):
        results = getattr(self, "_last_results", None)
        return reporter_from_results(results, self.datas)

    def _attach_observers(self, strategy: Any) -> None:
        strategy.observers = ObserverCollection()
        observer_specs = dict(self.observers)
        if self.stdstats:
            observer_specs.setdefault("value", (Value, {}))
        for name, spec in observer_specs.items():
            if len(spec) == 3:
                observer_cls, observer_args, observer_kwargs = spec
            else:
                observer_cls, observer_kwargs = spec
                observer_args = ()
            with engine_context(strategy=strategy, data=strategy.data, datas=strategy.datas):
                observer = observer_cls(*observer_args, **observer_kwargs)
            observer._set(strategy)
            strategy.observers[name] = observer

    def _attach_analyzers(self, strategy: Any) -> None:
        strategy.analyzers = AnalyzerCollection()
        for name, ana_spec in self.analyzers.items():
            if len(ana_spec) == 3:
                ana_cls, ana_args, ana_kwargs = ana_spec
            else:
                ana_cls, ana_kwargs = ana_spec
                ana_args = ()
            with engine_context(strategy=strategy, data=strategy.data, datas=strategy.datas):
                analyzer = ana_cls(*ana_args, **ana_kwargs)
            if getattr(analyzer, "strategy", None) is not strategy:
                analyzer.strategy = strategy
            strategy.analyzers[name] = analyzer

    def _start_analyzers(self, strategy: Any) -> None:
        for analyzer in getattr(strategy, "analyzers", {}).values():
            on_start = getattr(analyzer, "on_start", None)
            if callable(on_start):
                invoke_component(strategy, on_start)
            start = getattr(analyzer, "start", None)
            if callable(start):
                invoke_component(strategy, start)

    def _finish_event_analyzers(self, strategy: Any, snapshots: list[Any]) -> None:
        stats = {
            "bars": len(snapshots),
            "broker_events": sum(snapshot.dispatched_events for snapshot in snapshots),
            "mode": self.mode,
        }
        for analyzer in getattr(strategy, "analyzers", {}).values():
            on_end = getattr(analyzer, "on_end", None)
            if callable(on_end):
                invoke_component(strategy, on_end, stats)
        analyzer_results = {
            name: invoke_component(strategy, analyzer.get_analysis)
            for name, analyzer in getattr(strategy, "analyzers", {}).items()
        }
        strategy.analyzer_results = analyzer_results
        self.analyzer_results = analyzer_results
        for analyzer in getattr(strategy, "analyzers", {}).values():
            stop = getattr(analyzer, "stop", None)
            if callable(stop):
                invoke_component(strategy, stop)

    def _prepare_strategy_context(self) -> None:
        from tradelearn.engine.base import engine_context

        if self.datas:
            data = self.datas[0]
            datas = self.datas
        else:
            data = None
            datas = []
        self._strategy_context = engine_context(data=data, datas=datas, strategy=None)
        self._strategy_context.__enter__()

    def _bind_strategy_context(self, strategy: Any) -> None:
        from tradelearn.engine.base import set_current_strategy

        set_current_strategy(strategy)

    def _reset_strategy_context(self) -> None:
        context = getattr(self, "_strategy_context", None)
        if context is not None:
            context.__exit__(None, None, None)
            self._strategy_context = None
            return
        from tradelearn.engine.base import set_current_data, set_current_datas, set_current_strategy

        set_current_strategy(None)
        set_current_data(None)
        set_current_datas([])

    def _run_event_mode(self) -> list[Strategy]:
        from itertools import chain

        from tradelearn.backtest.event_runner import (
            EventRunner,
            LiveDriver,
            PaperDriver,
            _coerce_stream_bar,
        )
        from tradelearn.core.broker_events import BrokerEventPump
        from tradelearn.backtest.event_runtime import StrategyRuntime

        if self.mode == "paper":
            def source():
                return self.kwargs.get("event_bars", ())
        else:
            source = self.kwargs.get("live_poller")
            if source is None:
                raise ValueError("live mode requires live_poller")
        feeds = {
            str(getattr(data, "_name", None) or f"data{index}"): "bar"
            for index, data in enumerate(self.datas)
        }
        prefetched = None
        if not feeds:
            bars = iter(source())
            first = next(bars, None)
            feeds = {_coerce_stream_bar(first).symbol if first is not None else "data0": "bar"}
            prefetched = chain((), bars) if first is None else chain((first,), bars)
        strategy_cls, args, kwargs = (self.strats or [(Strategy, (), {})])[0]
        runtime = StrategyRuntime(
            strategy_cls,
            strategy_args=args,
            parameters=kwargs,
            feeds=feeds,
            broker=self.broker,
            capacity=int(self.kwargs.get("buffer_capacity", 512)),
        )
        strategy = runtime.strategy
        sizer_cls, sizer_kwargs = self._sizer_spec
        strategy.setsizer(sizer_cls(**sizer_kwargs))
        self._attach_analyzers(strategy)
        try:
            self._start_analyzers(strategy)
            pump = self.kwargs.get("broker_event_pump")
            if pump is None:
                pump = BrokerEventPump(self.kwargs.get("broker_event_poller", lambda: ()))
            for analyzer in strategy.analyzers.values():
                on_broker_event = getattr(analyzer, "on_broker_event", None)
                if callable(on_broker_event):
                    pump.on_event(lambda event, callback=on_broker_event: runtime.invoke_component(callback, event))
            runner = EventRunner(runtime=runtime, broker_event_pump=pump)
            runtime.start()
            remaining = prefetched if prefetched is not None else source()
            if self.mode == "paper":
                snapshots = PaperDriver(runner, remaining).run_once()
            else:
                snapshots = LiveDriver(runner, lambda: remaining).poll_once()
        except BaseException as error:
            runtime.cleanup(error)
            raise
        else:
            runner.stop()
        self._finish_event_analyzers(strategy, snapshots)
        self._last_results = [strategy]
        return [strategy]
