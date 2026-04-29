from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.models import BarSnapshot, Stats
from tradelearn.backtest.strategy import Strategy as CoreStrategy


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def getbyname(self, name: str) -> Any:
        return self[name]


def _build_observer_nexts(strategy: Any) -> tuple[Any, ...]:
    return tuple(
        observer.next
        for observer in getattr(strategy, "observers", {}).values()
        if callable(getattr(observer, "next", None))
    )


def _observer_step(observer_nexts: tuple[Any, ...]) -> None:
    for observer_next in observer_nexts:
        observer_next()


def _current_bar(data: Any) -> BarSnapshot:
    idx = max(0, int(getattr(data, "_cursor", 0)))
    timestamp = data.datetime[0] if hasattr(data, "datetime") else idx
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
        on_bar(bar)


def _orders_frame(broker: Any) -> pd.DataFrame:
    rows = []
    for order in getattr(broker, "_orders", []):
        data = getattr(order, "data", None)
        rows.append(
            {
                "ref": order.ref,
                "datetime": broker._fill_datetime(data) if data is not None else None,
                "data": getattr(data, "_name", None),
                "side": "buy" if order.isbuy() else "sell",
                "exectype": order.exectype,
                "status": order.getstatusname(),
                "size": order.size,
                "executed_size": order.executed.size,
                "executed_price": order.executed.price,
            }
        )
    return pd.DataFrame(rows)


def _fills_frame(broker: Any) -> pd.DataFrame:
    fills = broker.fills_frame() if hasattr(broker, "fills_frame") else pd.DataFrame()
    if fills.empty:
        return pd.DataFrame(columns=["order_ref", "data", "size", "price"])
    rows = fills.rename(columns={"ref": "order_ref"}).copy()
    if "data" not in rows.columns:
        rows["data"] = None
    return rows


def _positions_frame(strategy: Any, fills: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    columns = [
        "datetime",
        "data",
        "size",
        "avg_price",
        "mark_price",
        "value",
        "unrealized_pnl",
        "realized_pnl",
        "margin_used",
    ]
    if fills.empty:
        return pd.DataFrame(columns=columns)

    def is_missing_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

    position_size = 0.0
    avg_price = 0.0
    realized_pnl = 0.0
    rows = []
    data_name = getattr(getattr(strategy, "data", None), "_name", None)
    sizes = fills["size"].to_numpy(dtype=float, copy=False)
    prices = fills["price"].to_numpy(dtype=float, copy=False)
    datetimes = fills["datetime"].to_numpy(copy=False) if "datetime" in fills else None
    data_values = fills["data"].to_numpy(copy=False) if "data" in fills else None
    fallback_datetime = index[-1] if len(index) else None

    for row_idx, (signed_size, price) in enumerate(zip(sizes, prices, strict=True)):
        previous_size = position_size
        new_size = position_size + signed_size
        if previous_size == 0 or previous_size * signed_size > 0:
            total_abs = abs(previous_size) + abs(signed_size)
            avg_price = (
                (abs(previous_size) * avg_price + abs(signed_size) * price) / total_abs
                if total_abs
                else 0.0
            )
        elif previous_size * new_size <= 0:
            realized_pnl += (price - avg_price) * previous_size
            avg_price = price if new_size else 0.0
        position_size = 0.0 if abs(new_size) < 1e-9 else new_size
        mark_price = price
        value = position_size * mark_price
        fill_datetime = datetimes[row_idx] if datetimes is not None else None
        if is_missing_scalar(fill_datetime):
            fill_datetime = fallback_datetime
        fill_data = data_values[row_idx] if data_values is not None else None
        if is_missing_scalar(fill_data):
            fill_data = data_name
        rows.append(
            {
                "datetime": fill_datetime,
                "data": fill_data,
                "size": position_size,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "value": value,
                "unrealized_pnl": (mark_price - avg_price) * position_size
                if position_size
                else 0.0,
                "realized_pnl": realized_pnl,
                "margin_used": abs(value),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_equity_returns(strategy: Any, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    values = []
    observers_get = getattr(
        getattr(strategy, "observers", {}),
        "get",
        lambda key, default=None: None,
    )
    observer_value = observers_get("value", None)
    if observer_value is not None:
        values = list(getattr(observer_value, "values", []))
    if not values:
        values = [float(strategy.broker.getvalue())] * len(index)
    if len(values) < len(index):
        fill_value = values[-1] if values else float(strategy.broker.getvalue())
        values.extend([fill_value] * (len(index) - len(values)))
    equity = pd.Series(values[: len(index)], index=index, name="equity", dtype=float)
    returns = equity.pct_change().fillna(0.0)
    returns.name = "returns"
    return equity, returns


def _stats_config(cerebro: Any) -> dict[str, Any]:
    return {
        "callback_batch": getattr(cerebro, "callback_batch", 1),
        "trade_on_close": bool(getattr(cerebro, "trade_on_close", False)),
        "exactbars": bool(getattr(cerebro, "exactbars", False)),
        "stdstats": bool(getattr(cerebro, "stdstats", True)),
        "broker": {
            "cash": getattr(cerebro.broker, "_cash", 0.0),
            "commission": getattr(cerebro.broker, "commission_ratio", 0.0),
        },
    }


def _build_stats(cerebro: Any, strategy: Any, *, lazy_artifacts: bool = False) -> Stats:
    data = strategy.data
    frame = getattr(data, "_frame", None)
    if not isinstance(frame, pd.DataFrame):
        primary = cerebro.datas[0] if getattr(cerebro, "datas", None) else None
        frame = getattr(primary, "_frame", pd.DataFrame())
    index = frame.index

    if lazy_artifacts:
        final_cash = float(strategy.broker.getcash())
        final_value = float(strategy.broker.getvalue())
        total_orders = float(len(getattr(strategy.broker, "_orders", ())))
        total_fills = float(len(getattr(strategy.broker, "_fills", ())))
        trade_summary = getattr(strategy.broker, "trade_summary", None)
        total_trades = float(trade_summary()[0]) if callable(trade_summary) else 0.0

        def equity_factory() -> pd.Series:
            return _build_equity_returns(strategy, index)[0]

        def returns_factory() -> pd.Series:
            return _build_equity_returns(strategy, index)[1]

        def fills_factory() -> pd.DataFrame:
            return _fills_frame(strategy.broker)

        def positions_factory() -> pd.DataFrame:
            return _positions_frame(strategy, fills_factory(), index)

        return Stats(
            returns=returns_factory,
            equity=equity_factory,
            trades=lambda: pd.DataFrame(),
            positions=positions_factory,
            orders=lambda: _orders_frame(strategy.broker),
            summary={
                "bars": float(len(index)),
                "final_cash": final_cash,
                "final_value": final_value,
                "final_realized_pnl": 0.0,
                "final_unrealized_pnl": 0.0,
                "final_margin_used": 0.0,
                "max_drawdown": 0.0,
                "sharpe": float("nan"),
                "total_trades": total_trades,
                "total_orders": total_orders,
                "total_fills": total_fills,
            },
            analyzers={},
            config=_stats_config(cerebro),
            fills=fills_factory,
        )

    equity, returns = _build_equity_returns(strategy, index)
    drawdowns = (equity.cummax() - equity) / equity.cummax().replace(0, np.nan)
    fills = _fills_frame(strategy.broker)
    orders = _orders_frame(strategy.broker)
    positions = _positions_frame(strategy, fills, index)
    summary = {
        "bars": float(len(index)),
        "final_cash": float(strategy.broker.getcash()),
        "final_value": float(strategy.broker.getvalue()),
        "final_realized_pnl": 0.0,
        "final_unrealized_pnl": 0.0,
        "final_margin_used": 0.0,
        "max_drawdown": float(drawdowns.fillna(0.0).max()) if not drawdowns.empty else 0.0,
        "sharpe": float("nan"),
        "total_trades": 0.0,
        "total_orders": float(len(orders)),
        "total_fills": float(len(fills)),
    }
    return Stats(
        returns=returns,
        equity=equity,
        trades=pd.DataFrame(),
        positions=positions,
        orders=orders,
        summary=summary,
        analyzers={},
        config=_stats_config(cerebro),
        fills=fills,
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
        if id(val) in seen_advancer_ids:
            continue
        advance = getattr(val, "_advance", None)
        if callable(advance):
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(val))
    return tuple(bar_advancers)


def _build_data_advance_plan(datas: list[Any]) -> Any | None:
    """Build a Rust primary-clock cursor runner for multi-data runs."""
    if len(datas) <= 1:
        return None
    try:
        from tradelearn._rust import RustBarRunner
    except (ImportError, AttributeError):
        try:
            from tradelearn._rust import RustPrimaryClockPlan as RustBarRunner
        except (ImportError, AttributeError):
            return None
    return RustBarRunner(
        np.asarray(datas[0]._datetime, dtype=np.int64),
        [np.asarray(data._datetime, dtype=np.int64) for data in datas[1:]],
    )


def run_backtest(cerebro: Any) -> list[Any]:
    """Unified backtest engine that runs any strategy inheriting from core.Strategy."""
    strategy_cls, args, kwargs = cerebro.strats[0]
    bind_strategy_context = getattr(cerebro, "_bind_strategy_context", None)

    strategy = strategy_cls(*args, **kwargs)
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
        from tradelearn._rust import RustBacktestEngine

        data = cerebro.datas[0]
        # Robustly handle both native containers and facade data feeds.
        if hasattr(data, "_datetime"):
            timestamps = data._datetime
            opens = data._open
            highs = data._high
            lows = data._low
            closes = data._close
            volumes = data._volume
        else:  # BT Legacy / LineSeries fallback
            timestamps = np.array(data.datetime._values, dtype=np.int64)
            opens = np.array(data.open._values, dtype=np.float64)
            highs = np.array(data.high._values, dtype=np.float64)
            lows = np.array(data.low._values, dtype=np.float64)
            closes = np.array(data.close._values, dtype=np.float64)
            volumes = np.array(data.volume._values, dtype=np.float64)

        rust_engine = RustBacktestEngine(
            timestamps,
            opens,
            highs,
            lows,
            closes,
            volumes,
            float(cerebro.broker._cash),
            float(cerebro.broker.commission_ratio),
            bool(getattr(cerebro, "trade_on_close", False)),
            False,
            False,
            0.0,
            0.0,
            False,
            False,
            False,
            float(cerebro.broker._mult),
            1.0,
            cerebro.broker.match_mode == "smart",
        )
        cerebro.broker.bind_engine(rust_engine)
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

    strategy.analyzers = _AttrDict()
    for name, (ana_cls, ana_kwargs) in cerebro.analyzers.items():
        ana_inst = ana_cls(**ana_kwargs)
        ana_inst.strategy = strategy
        strategy.analyzers[name] = ana_inst

    attach_observers = getattr(cerebro, "_attach_observers", None)
    if callable(attach_observers):
        attach_observers(strategy)

    # 3. Lifecycle Start
    if hasattr(strategy, "_setup"):
        strategy._setup()
    strategy.init()
    strategy.start()
    for ana in strategy.analyzers.values():
        on_start = getattr(ana, "on_start", None)
        if callable(on_start):
            on_start()
        if hasattr(ana, "start"):
            ana.start()
    for observer in getattr(strategy, "observers", {}).values():
        observer.start()
    analyzer_bar_callbacks = _build_analyzer_bar_callbacks(strategy)
    observer_nexts = _build_observer_nexts(strategy)
    has_analyzer_bar_callbacks = bool(analyzer_bar_callbacks)
    has_observer_nexts = bool(observer_nexts)

    limit = cerebro.datas[0].buflen()
    # Calculate min_period from all indicators (mostly for BT facade)
    min_period = int(getattr(strategy, "_manual_min_period", 0))
    for ind in indicators + indicators_bt:
        if hasattr(ind, "min_period"):
            m = ind.min_period
            if callable(m):
                m = m()
            min_period = max(min_period, int(m))
    if min_period == 0:
        min_period = 1

    data_advance_plan = _build_data_advance_plan(cerebro.datas)
    bar_advancers = _build_bar_advancers(
        strategy,
        cerebro.datas,
        indicators + indicators_bt,
        include_data=data_advance_plan is None,
    )

    if data_advance_plan is None:

        def advance_datas(i: int) -> None:
            return None
    else:

        def advance_datas(i: int) -> None:
            for data, cursor in zip(cerebro.datas, data_advance_plan.cursors_at(i), strict=False):
                if cursor >= 0:
                    data._advance(cursor)
                else:
                    data._advance(-1)

    if hasattr(strategy, "_set_bar_advancers"):
        if data_advance_plan is None:
            strategy._set_bar_advancers(bar_advancers)
        else:

            def advance_bar(i: int) -> None:
                advance_datas(i)
                for advance in bar_advancers:
                    advance(i)

            strategy._set_bar_advancers((advance_bar,))
        strategy_pre_next = strategy._pre_next
    else:

        def strategy_pre_next(cursor: int) -> None:
            advance_datas(cursor)
            for advance in bar_advancers:
                advance(cursor)

    notify_cashvalue = None
    if type(strategy).notify_cashvalue is not CoreStrategy.notify_cashvalue:
        notify_cashvalue = strategy.notify_cashvalue
    broker = cerebro.broker
    broker_step = broker.step if broker else None
    broker_process_fills = broker.process_fills if broker else None
    broker_getcash = broker.getcash if broker else None
    broker_getvalue = broker.getvalue if broker else None
    begin_order_buffering = getattr(broker, "begin_order_buffering", None) if broker else None
    flush_order_buffer = getattr(broker, "flush_order_buffer", None) if broker else None
    drain_order_buffer = getattr(broker, "drain_order_buffer", None) if broker else None
    strategy_next = strategy.next

    use_rust_bar_loop = (
        isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and hasattr(broker._engine, "run_bar_loop")
        and not bool(getattr(cerebro, "trade_on_close", False))
    )
    min_start = min_period - 1

    class _RunStopped(Exception):
        """Internal sentinel used to break out of Rust callback-driven loops."""

    def raise_if_runstopped() -> None:
        if bool(getattr(cerebro, "_runstop", False)):
            raise _RunStopped

    def on_bar(i: int) -> list[Any]:
        strategy_pre_next(i)

        # Broker Match
        if broker:
            broker_step(i)
            broker_process_fills(strategy, i)
            if notify_cashvalue is not None:
                notify_cashvalue(broker_getcash(), broker_getvalue())

        # Strategy Next
        if i >= min_start:
            if begin_order_buffering is not None:
                begin_order_buffering()
                strategy_next()
                flush_order_buffer()
                if getattr(broker, "_trade_on_close", False):
                    broker_process_fills(strategy, i)
            else:
                strategy_next()
            if bool(getattr(cerebro, "_runstop", False)):
                return []
            if has_analyzer_bar_callbacks:
                _analyzer_bar_step(strategy, analyzer_bar_callbacks)
            if has_observer_nexts:
                _observer_step(observer_nexts)
        return []

    use_multi_data_rust_runner = (
        use_rust_bar_loop and data_advance_plan is not None and hasattr(data_advance_plan, "run")
    )

    if use_multi_data_rust_runner:
        if notify_cashvalue is None:

            def on_rust_bar_multi(
                i: int,
                data_cursors: list[int],
                fills: list[Any],
                cash: float,
                size: float,
                price: float,
            ) -> list[Any] | None:
                for data, cursor in zip(cerebro.datas, data_cursors, strict=False):
                    data._advance(cursor)
                for advance in bar_advancers:
                    advance(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                if i >= min_start:
                    begin_order_buffering()
                    strategy_next()
                    orders = drain_order_buffer()
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders if orders else None
                return None
        else:

            def on_rust_bar_multi(
                i: int,
                data_cursors: list[int],
                fills: list[Any],
                cash: float,
                size: float,
                price: float,
            ) -> list[Any] | None:
                for data, cursor in zip(cerebro.datas, data_cursors, strict=False):
                    data._advance(cursor)
                for advance in bar_advancers:
                    advance(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                notify_cashvalue(broker_getcash(), broker_getvalue())
                if i >= min_start:
                    begin_order_buffering()
                    strategy_next()
                    orders = drain_order_buffer()
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders if orders else None
                return None

        try:
            data_advance_plan.run(broker._engine, broker, on_rust_bar_multi, 0, limit)
        except _RunStopped:
            pass
    elif use_rust_bar_loop:
        if notify_cashvalue is None:

            def on_rust_bar(
                i: int, fills: list[Any], cash: float, size: float, price: float
            ) -> list[Any] | None:
                strategy_pre_next(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                if i >= min_start:
                    begin_order_buffering()
                    strategy_next()
                    orders = drain_order_buffer()
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders if orders else None
                return None
        else:

            def on_rust_bar(
                i: int, fills: list[Any], cash: float, size: float, price: float
            ) -> list[Any] | None:
                strategy_pre_next(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                notify_cashvalue(broker_getcash(), broker_getvalue())
                if i >= min_start:
                    begin_order_buffering()
                    strategy_next()
                    orders = drain_order_buffer()
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders if orders else None
                return None

        try:
            broker._engine.run_bar_loop(broker, on_rust_bar, 0, limit)
        except _RunStopped:
            pass
    else:
        for i in range(limit):
            on_bar(i)
            if bool(getattr(cerebro, "_runstop", False)):
                break

    # 4. Lifecycle Stop
    strategy.stop()
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
            on_end(stats)
    analyzer_results = {name: ana.get_analysis() for name, ana in strategy.analyzers.items()}
    stats.analyzers = analyzer_results
    strategy.stats = stats
    strategy.analyzer_results = analyzer_results
    cerebro.stats = stats
    cerebro.analyzer_results = analyzer_results
    for ana in strategy.analyzers.values():
        if hasattr(ana, "stop"):
            ana.stop()
    for observer in getattr(strategy, "observers", {}).values():
        observer.stop()

    return [strategy]
