from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.models import BarSnapshot
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.backtest.strategy import Strategy as CoreStrategy
from tradelearn.utils.console import smart_tqdm as tqdm


TRADE_STATISTICS_REVISION = "flat-to-flat-fees-v1"


def _build_observer_nexts(strategy: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(observer, "_next_step", observer.next)
        for observer in getattr(strategy, "observers", {}).values()
        if callable(getattr(observer, "next", None))
    )


def _observer_step(runtime: Any, observer_nexts: tuple[Any, ...]) -> None:
    for observer_next in observer_nexts:
        runtime.invoke_component(observer_next)


def _current_bar(data: Any) -> BarSnapshot:
    idx = max(0, int(getattr(data, "_cursor", 0)))
    runtime = getattr(data, "runtime", None)
    source_ns = getattr(runtime, "_last_ts", {}).get(getattr(data, "_name", None))
    if source_ns is not None:
        timestamp = pd.Timestamp(source_ns, unit="ns", tz="UTC")
    elif hasattr(data, "_datetime"):
        if getattr(data, '_clock_unit', 's') == 'ns':
            timestamp = pd.Timestamp(int(data._clock_timestamps[idx]), unit='ns', tz='UTC')
        else:
            timestamp = pd.to_datetime(data._datetime[idx], unit="s", utc=True)
    elif getattr(data, "_buffer", None) is not None:
        raw_timestamp = data._buffer.value("datetime", ago=0)
        timestamp = (pd.to_datetime(raw_timestamp, unit="s", utc=True)
                     if pd.notna(raw_timestamp) else None)
    else:
        timestamp = None
    return BarSnapshot(
        datetime=timestamp,
        open=float(data.open[0]),
        high=float(data.high[0]),
        low=float(data.low[0]),
        close=float(data.close[0]),
        volume=float(data.volume[0]),
        data=data,
    )


def _build_analyzer_bar_callbacks(strategy: Any) -> tuple[Any, ...]:
    callbacks = []
    for analyzer in getattr(strategy, "analyzers", {}).values():
        on_bar = getattr(analyzer, "on_bar", None)
        if not callable(on_bar):
            continue
        if not getattr(analyzer, "is_streaming", False) and "on_bar" not in type(analyzer).__dict__:
            continue
        callbacks.append(on_bar)
    return tuple(callbacks)


def _analyzer_bar_step(strategy: Any, analyzer_bar_callbacks: tuple[Any, ...]) -> None:
    if not analyzer_bar_callbacks or strategy.data is None:
        return
    bar = _current_bar(strategy.data)
    for on_bar in analyzer_bar_callbacks:
        strategy._runtime.invoke_component(on_bar, bar)



from tradelearn.backtest._stats import (
    _build_equity_from_fills,
    _build_equity_returns,
    _build_stats,
    _canonical_drawdown,
    _fills_frame,
    _final_unrealized_pnl,
    _orders_frame,
    _positions_frame,
    _resolve_close_by_data,
    _resolve_close_series,
    _return_pct,
    _stats_config,
    _summary_annual_return,
    _summary_calmar,
    _summary_drawdown_stats,
    _summary_exposure_pct,
    _summary_pnl,
    _summary_sharpe,
    _summary_sortino,
    _summary_trade_metrics,
    _summary_trade_stats,
    _summary_volatility,
    _trade_summary,
    _trades_frame,
    _win_rate_pct,
)


def _build_bar_advancers(
    strategy: Any,
    datas: list[Any],
    indicators: list[Any],
    *,
    include_data: bool = True,
) -> tuple[Any, ...]:
    """Build a stable per-bar advance plan for data feeds and indicators."""
    bar_advancers = []
    seen_advancer_ids = set()
    for data in datas:
        if include_data:
            bar_advancers.append(data._advance)
        seen_advancer_ids.add(id(data))
    for indicator in indicators:
        advance = getattr(indicator, "_advance", None)
        if callable(advance) and id(indicator) not in seen_advancer_ids:
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(indicator))
    strategy_advance = getattr(strategy, "_advance", None)
    strategy_lines = getattr(strategy, "lines", None)
    strategy_has_lines = strategy_lines is None or len(strategy_lines) > 0
    if strategy_advance is not None and strategy_has_lines:
        bar_advancers.append(strategy_advance)
    for attr, val in strategy.__dict__.items():
        if attr.startswith("_"):
            continue
        if id(getattr(val, "_feed", val)) in seen_advancer_ids:
            continue
        advance = getattr(val, "_advance", None)
        if callable(advance):
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(val))
    return tuple(bar_advancers)


def _build_data_advance_plan(datas: list[Any], *, unit="s") -> Any | None:
    """Build a Rust primary-clock cursor runner for multi-data runs."""
    if len(datas) <= 1:
        return None
    from .clock import native_timestamps
    primary_timestamps = native_timestamps(datas[0], unit)
    if all(
        len(data._datetime) == len(primary_timestamps)
        and np.array_equal(native_timestamps(data, unit), primary_timestamps)
        for data in datas[1:]
    ):
        return None
    try:
        from tradelearn._rust import RustBarRunner
    except (ImportError, AttributeError):
        try:
            from tradelearn._rust import RustPrimaryClockPlan as RustBarRunner
        except (ImportError, AttributeError):
            return None
    return RustBarRunner(
        primary_timestamps,
        [native_timestamps(data, unit) for data in datas[1:]],
    )


def _build_clocked_multi_data_runner(datas: list[Any], *, unit="s") -> Any | None:
    """Build the Rust clocked multi-data runner when the feed schema is supported."""
    if len(datas) <= 1:
        return None
    try:
        from tradelearn._rust import RustClockedMultiDataRunner
    except (ImportError, AttributeError):
        return None

    from .clock import native_timestamps
    symbols: list[str] = []
    timestamps: list[np.ndarray] = []
    opens: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    volumes: list[np.ndarray] = []
    from .data import native_data_symbols
    execution_symbols = native_data_symbols(datas)
    for index, data in enumerate(datas):
        required = ("_datetime", "_open", "_high", "_low", "_close", "_volume")
        if not all(hasattr(data, attr) for attr in required):
            return None
        symbols.append(execution_symbols[index])
        timestamps.append(native_timestamps(data, unit))
        opens.append(np.asarray(data._open, dtype=np.float64))
        highs.append(np.asarray(data._high, dtype=np.float64))
        lows.append(np.asarray(data._low, dtype=np.float64))
        closes.append(np.asarray(data._close, dtype=np.float64))
        volumes.append(np.asarray(data._volume, dtype=np.float64))
    return RustClockedMultiDataRunner(symbols, timestamps, opens, highs, lows, closes, volumes)


def _load_rust_backtest_engine() -> Any:
    try:
        from tradelearn._rust import RustBacktestEngine
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "tradelearn._rust is not available, so RustBroker cannot run order matching. "
            "If you installed from PyPI, make sure you run outside a source checkout that "
            "contains a local tradelearn/ package and install with `pip install trade-learn`. "
            "If you are developing from source, build the extension first with "
            "`python -m maturin develop --release`."
        ) from exc
    return RustBacktestEngine


def run_backtest(cerebro: Any) -> list[Any]:
    """Unified backtest engine that runs any strategy inheriting from core.Strategy."""
    runtime_config = BacktestRuntimeConfig.from_owner(cerebro)
    cerebro.runtime_config = runtime_config
    strategy_cls, args, kwargs = cerebro.strats[0]
    bind_strategy_context = getattr(cerebro, "_bind_strategy_context", None)
    from .event_runtime import StrategyRuntime
    runtime = StrategyRuntime(strategy_cls, strategy_args=args, parameters=kwargs,
                              profile="batch", data_feeds=cerebro.datas, broker=cerebro.broker)
    # Preserve bound historical buffers while honoring BT's event-mode callback
    # policy for imperative Indicators on held secondary source clocks.
    runtime._bt_event_indicators = runtime._bt_line_views and not bool(
        getattr(getattr(cerebro, "p", None), "runonce", True)
    )
    try:
        return _run_backtest_runtime(cerebro, runtime, runtime_config, bind_strategy_context)
    except BaseException as error:
        runtime.cleanup(error)
        raise


def _run_backtest_runtime(cerebro, runtime, runtime_config, bind_strategy_context):
    strategy = runtime.strategy
    strategy.cerebro = cerebro
    if callable(bind_strategy_context):
        bind_strategy_context(strategy)

    # Core attributes
    strategy.datas = cerebro.datas
    if cerebro.datas:
        strategy.data = cerebro.datas[0]
    strategy.broker = cerebro.broker

    # ... (Rust Engine Initialization omitted for brevity but preserved in real file) ...
    # ---------------------------------------------------------
    # Rust Engine Initialization
    # ---------------------------------------------------------
    from .broker import RustBroker

    if isinstance(cerebro.broker, RustBroker):
        data = cerebro.datas[0]
        cerebro.broker._clock_unit = runtime._clock.unit
        from .clock import native_timestamps
        # Robustly handle both native containers and facade data feeds.
        if hasattr(data, "_datetime"):
            timestamps = native_timestamps(data, runtime._clock.unit)
            opens = data._open
            highs = data._high
            lows = data._low
            closes = data._close
            volumes = data._volume
        else:  # LineSeries fallback
            timestamps = np.array(data.datetime._values, dtype=np.int64)
            opens = np.array(data.open._values, dtype=np.float64)
            highs = np.array(data.high._values, dtype=np.float64)
            lows = np.array(data.low._values, dtype=np.float64)
            closes = np.array(data.close._values, dtype=np.float64)
            volumes = np.array(data.volume._values, dtype=np.float64)

        RustBacktestEngine = _load_rust_backtest_engine()
        commission_model = runtime_config.commission_model
        cn_a_stock_commission = type(commission_model).__name__ == "CNAStockCommission"
        rust_engine = RustBacktestEngine(
            timestamps,
            opens,
            highs,
            lows,
            closes,
            volumes,
            runtime_config.cash,
            runtime_config.commission,
            runtime_config.trade_on_close,
            False,
            False,
            0.0,
            0.0,
            False,
            False,
            False,
            float(cerebro.broker._mult),
            1.0,
            runtime_config.match_mode == "smart",
            cn_a_stock_commission,
            float(getattr(commission_model, "commission_rate", 0.00025)),
            float(getattr(commission_model, "min_commission", 5.0)),
            float(getattr(commission_model, "stamp_tax_rate", 0.001)),
            float(getattr(commission_model, "transfer_fee_rate", 0.00002)),
        )
        cerebro.broker.bind_engine(rust_engine)
        if hasattr(cerebro.broker, "bind_datas"):
            cerebro.broker.bind_datas(cerebro.datas)
        cerebro.broker._open_prices = opens
        cerebro.broker._high_prices = highs
        cerebro.broker._low_prices = lows
        cerebro.broker._close_prices = closes

    # 2. Initialize Sizer & Analyzers
    sizer_cls, sizer_kwargs = cerebro._sizer_spec
    strategy.setsizer(sizer_cls(**sizer_kwargs))

    # Support indicators from both facades
    indicators = getattr(strategy, "_indicators", [])
    indicators_bt = getattr(strategy, "_indicators_bt", [])

    from tradelearn.backtest.components import AnalyzerCollection
    strategy.analyzers = AnalyzerCollection()
    for name, ana_spec in cerebro.analyzers.items():
        if len(ana_spec) == 3:
            ana_cls, ana_args, ana_kwargs = ana_spec
        else:
            ana_cls, ana_kwargs = ana_spec
            ana_args = ()
        ana_inst = ana_cls(*ana_args, **ana_kwargs)
        ana_inst.strategy = strategy
        strategy.analyzers[name] = ana_inst

    attach_observers = getattr(cerebro, "_attach_observers", None)
    if callable(attach_observers):
        attach_observers(strategy)

    # 3. Component start is coordinated before the business start callback.
    def start_components():
        for ana in strategy.analyzers.values():
            on_start = getattr(ana, "on_start", None)
            if callable(on_start):
                runtime.invoke_component(on_start)
            if hasattr(ana, "start"):
                runtime.invoke_component(ana.start)
        for observer in getattr(strategy, "observers", {}).values():
            runtime.invoke_component(observer.start)
    runtime._component_start_callback = start_components
    install_timers=getattr(cerebro,"_install_timers",None)
    if callable(install_timers):runtime._component_after_start_callback=lambda:install_timers(runtime)
    runtime.start()
    analyzer_bar_callbacks = _build_analyzer_bar_callbacks(strategy)
    observer_nexts = _build_observer_nexts(strategy)
    has_analyzer_bar_callbacks = bool(analyzer_bar_callbacks)
    has_observer_nexts = bool(observer_nexts)

    limit = len(runtime._clock)
    broker = cerebro.broker
    has_cheat_timers=any(timer.p.cheat for timer in getattr(cerebro,'_timers',()))
    use_clocked_multi_data_runner = (
        not has_cheat_timers
        and isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and len(cerebro.datas) > 1
        and not runtime._clock.requires_union
        and not any(hasattr(data, '_execution_quote_values') for data in cerebro.datas)
    )
    clocked_multi_data_runner = (
        (_build_clocked_multi_data_runner(cerebro.datas) if runtime._clock.unit == 's' else
         _build_clocked_multi_data_runner(cerebro.datas, unit=runtime._clock.unit)) if use_clocked_multi_data_runner else None
    )
    data_advance_plan = (
        runtime._clock if runtime._clock.requires_union else
        None if clocked_multi_data_runner is not None else (_build_data_advance_plan(cerebro.datas) if runtime._clock.unit == 's' else
              _build_data_advance_plan(cerebro.datas, unit=runtime._clock.unit))
    )
    data_advance_methods = tuple(data._advance for data in cerebro.datas)
    bar_advancers = _build_bar_advancers(
        strategy,
        cerebro.datas,
        indicators + indicators_bt,
        include_data=data_advance_plan is None and clocked_multi_data_runner is None,
    )

    if data_advance_plan is None:

        def advance_datas(i: int) -> None:
            return None
    else:

        def advance_datas(i: int) -> None:
            cursors = data_advance_plan.cursors_at(i)
            for advance_data, cursor in zip(data_advance_methods, cursors, strict=False):
                if cursor >= 0:
                    advance_data(cursor)
                else:
                    advance_data(-1)
            return cursors

    runtime.configure_bound(bar_advancers, advance_datas if data_advance_plan is not None else None)
    strategy_pre_next = runtime.advance_bound

    notify_cashvalue = None
    if type(strategy).notify_cashvalue is not CoreStrategy.notify_cashvalue:
        notify_cashvalue = lambda *args: runtime.invoke_business("notify_cashvalue", *args)
    broker_step = broker.step if broker else None
    broker_process_fills = broker.process_fills if broker else None
    broker_getcash = broker.getcash if broker else None
    broker_getvalue = broker.getvalue if broker else None
    begin_order_buffering = getattr(broker, "begin_order_buffering", None) if broker else None
    flush_order_buffer = getattr(broker, "flush_order_buffer", None) if broker else None
    drain_order_buffer = getattr(broker, "drain_order_buffer", None) if broker else None
    begin_terminal_order_suppression = (
        getattr(broker, "begin_terminal_order_suppression", None) if broker else None
    )
    end_terminal_order_suppression = (
        getattr(broker, "end_terminal_order_suppression", None) if broker else None
    )
    strategy_next = lambda name: runtime.invoke_business(name)

    use_rust_bar_loop = (
        not has_cheat_timers
        and isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and hasattr(broker._engine, "run_bar_loop")
        and len(cerebro.datas) <= 1
        and not runtime._clock.requires_union
        and not any(hasattr(data, '_execution_quote_values') for data in cerebro.datas)
        and not bool(getattr(cerebro, "trade_on_close", False))
    )

    class _RunStopped(Exception):
        """Internal sentinel used to break out of Rust callback-driven loops."""

    def raise_if_runstopped() -> None:
        if bool(getattr(cerebro, "_runstop", False)):
            raise _RunStopped

    def terminal_bar(i: int) -> bool:
        return i >= limit - 1

    def run_strategy_next_python(i: int, callback_name: str) -> None:
        if terminal_bar(i) and begin_terminal_order_suppression is not None:
            begin_terminal_order_suppression()
            try:
                strategy_next(callback_name)
            finally:
                if end_terminal_order_suppression is not None:
                    end_terminal_order_suppression()
            if flush_order_buffer is not None:
                flush_order_buffer()
            return
        if begin_order_buffering is not None:
            begin_order_buffering()
            strategy_next(callback_name)
            flush_order_buffer()
            if getattr(broker, "_trade_on_close", False):
                broker_process_fills(strategy, i)
        else:
            strategy_next(callback_name)

    def run_strategy_next_drained(i: int, callback_name: str) -> list[Any] | None:
        if terminal_bar(i) and begin_terminal_order_suppression is not None:
            begin_terminal_order_suppression()
            try:
                strategy_next(callback_name)
            finally:
                if end_terminal_order_suppression is not None:
                    end_terminal_order_suppression()
            orders = drain_order_buffer()
            return orders if orders else None
        begin_order_buffering()
        strategy_next(callback_name)
        orders = drain_order_buffer()
        return orders if orders else None

    def on_bar(i: int) -> list[Any]:
        strategy_pre_next(i)

        # Existing Python coordinator supplies the pre-match timer phase;
        # the same Rust kernel still owns every order match and account fact.
        cheat_callback=getattr(runtime,'_timer_cheat_callback',None)
        if callable(cheat_callback):
            if broker:broker._curr_idx=i
            cheat_callback()
        # Broker Match
        if broker:
            broker_step(i)
            if begin_order_buffering is not None:
                begin_order_buffering()
            broker_process_fills(strategy, i)
            if notify_cashvalue is not None:
                notify_cashvalue(broker_getcash(), broker_getvalue())

        # Strategy Next
        ready, _ = runtime.step_bound(i, run_strategy_next_python)
        if not ready and has_observer_nexts and getattr(runtime, "_bt_line_views", False):
            _observer_step(runtime, observer_nexts)
        if ready:
            if bool(getattr(cerebro, "_runstop", False)):
                return []
            if has_analyzer_bar_callbacks:
                _analyzer_bar_step(strategy, analyzer_bar_callbacks)
            if has_observer_nexts:
                _observer_step(runtime, observer_nexts)
        return []

    use_multi_data_rust_runner = clocked_multi_data_runner is not None
    _show_pbar = bool(getattr(strategy, "verbose", True) and not bool(os.environ.get("TQDM_DISABLE")))
    pbar = tqdm(total=limit, desc="Backtest.run", unit="bar", leave=True, delay=0, disable=not _show_pbar)
    try:
        def _make_rust_bar_callback(*, multi: bool, trade_on_close: bool):
            dispatch = run_strategy_next_python if trade_on_close else run_strategy_next_drained

            def _finish_bar(i: int, fills: list[Any], cash: float, size: float, price: float):
                broker._curr_idx = i
                begin_order_buffering()
                broker._step_fills_from_collect = fills if fills is not None else []
                broker._rust_state_cache = (i, cash, size, price)
                broker_process_fills(strategy, i)
                if notify_cashvalue is not None:
                    notify_cashvalue(broker_getcash(), broker_getvalue())
                ready, orders = runtime.step_bound(i, dispatch)
                raise_if_runstopped()
                if not ready and has_observer_nexts and getattr(runtime, "_bt_line_views", False):
                    _observer_step(runtime, observer_nexts)
                if ready:
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(runtime, observer_nexts)
                    return None if trade_on_close else orders
                if trade_on_close:
                    return None
                orders = orders or drain_order_buffer()
                return orders if orders else None

            if multi:
                def on_rust_bar_multi(
                    i: int,
                    data_cursors: list[int],
                    fills: list[Any],
                    cash: float,
                    size: float,
                    price: float,
                ) -> list[Any] | None:
                    pbar.update(1)
                    runtime.advance_bound(i, data_cursors)
                    return _finish_bar(i, fills, cash, size, price)

                return on_rust_bar_multi

            def on_rust_bar(
                i: int, fills: list[Any], cash: float, size: float, price: float
            ) -> list[Any] | None:
                pbar.update(1)
                strategy_pre_next(i)
                return _finish_bar(i, fills, cash, size, price)

            return on_rust_bar

        if use_multi_data_rust_runner:
            on_rust_bar_multi = _make_rust_bar_callback(
                multi=True, trade_on_close=bool(getattr(broker, "_trade_on_close", False))
            )
            try:
                clocked_multi_data_runner.run(broker._engine, broker, on_rust_bar_multi, 0, limit)
            except _RunStopped:
                pass
        elif use_rust_bar_loop:
            on_rust_bar = _make_rust_bar_callback(multi=False, trade_on_close=False)
            try:
                broker._engine.run_bar_loop(broker, on_rust_bar, 0, limit)
            except _RunStopped:
                pass
        else:
            for i in range(limit):
                on_bar(i)
                pbar.update(1)
                if bool(getattr(cerebro, "_runstop", False)):
                    break

    finally:
        pbar.close()

    # 4. Lifecycle Stop
    runtime.stop()
    metrics_engine = getattr(strategy, "metrics_engine", None)
    lazy_stats = (
        getattr(cerebro, "stats_mode", "full") == "lazy"
        and metrics_engine is None
        and not strategy.analyzers
    )
    stats = _build_stats(cerebro, strategy, lazy_artifacts=lazy_stats)
    if metrics_engine is not None:
        metrics_engine.compute(stats)
    for ana in strategy.analyzers.values():
        on_end = getattr(ana, "on_end", None)
        if callable(on_end):
            runtime.invoke_component(on_end, stats)
    for ana in strategy.analyzers.values():
        if hasattr(ana,"stop"):
            runtime.invoke_component(ana.stop)
    analyzer_results = {name: ana.get_analysis() for name, ana in strategy.analyzers.items()}
    stats.analyzers = analyzer_results
    strategy.stats = stats
    strategy.analyzer_results = analyzer_results
    cerebro.stats = stats
    cerebro.analyzer_results = analyzer_results
    for observer in getattr(strategy, "observers", {}).values():
        callback = getattr(observer,"stop",getattr(observer,"_stop",None))
        if callable(callback):
            runtime.invoke_component(callback)

    return [strategy]
