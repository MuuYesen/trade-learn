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


def _observer_step(strategy: Any) -> None:
    for observer in getattr(strategy, "observers", {}).values():
        observer.next()


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


def _analyzer_bar_step(strategy: Any) -> None:
    analyzers = getattr(strategy, "analyzers", {})
    if not analyzers or strategy.data is None:
        return
    bar = _current_bar(strategy.data)
    for analyzer in analyzers.values():
        on_bar = getattr(analyzer, "on_bar", None)
        if callable(on_bar):
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


def _build_stats(cerebro: Any, strategy: Any) -> Stats:
    data = strategy.data
    frame = getattr(data, "_frame", None)
    if not isinstance(frame, pd.DataFrame):
        primary = cerebro.datas[0] if getattr(cerebro, "datas", None) else None
        frame = getattr(primary, "_frame", pd.DataFrame())
    index = frame.index
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
    drawdowns = (equity.cummax() - equity) / equity.cummax().replace(0, np.nan)
    fills = _fills_frame(strategy.broker)
    orders = _orders_frame(strategy.broker)
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
        positions=pd.DataFrame(),
        orders=orders,
        summary=summary,
        analyzers={},
        config={
            "callback_batch": getattr(cerebro, "callback_batch", 1),
            "trade_on_close": bool(getattr(cerebro, "trade_on_close", False)),
            "exactbars": bool(getattr(cerebro, "exactbars", False)),
            "stdstats": bool(getattr(cerebro, "stdstats", True)),
            "broker": {
                "cash": getattr(cerebro.broker, "_cash", 0.0),
                "commission": getattr(cerebro.broker, "commission_ratio", 0.0),
            },
        },
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
        [int(ts) for ts in datas[0]._datetime],
        [[int(ts) for ts in data._datetime] for data in datas[1:]],
    )


def run_backtest(cerebro: Any) -> list[Any]:
    """Unified backtest engine that runs any strategy inheriting from core.Strategy."""
    strategy_cls, args, kwargs = cerebro.strats[0]
    bind_strategy_context = getattr(cerebro, "_bind_strategy_context", None)

    strategy = strategy_cls(*args, **kwargs)
    if callable(bind_strategy_context):
        bind_strategy_context(strategy)

    # Core attributes
    strategy.datas = cerebro.datas
    if cerebro.datas:
        strategy.data = cerebro.datas[0]
    strategy.broker = cerebro.broker

    if hasattr(strategy, "_setup"):
        strategy._setup()

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
        cerebro.broker._engine = rust_engine
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
            _analyzer_bar_step(strategy)
            _observer_step(strategy)
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
                    _analyzer_bar_step(strategy)
                    _observer_step(strategy)
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
                    _analyzer_bar_step(strategy)
                    _observer_step(strategy)
                    return orders if orders else None
                return None

        data_advance_plan.run(broker._engine, broker, on_rust_bar_multi, 0, limit)
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
                    _analyzer_bar_step(strategy)
                    _observer_step(strategy)
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
                    _analyzer_bar_step(strategy)
                    _observer_step(strategy)
                    return orders if orders else None
                return None

        broker._engine.run_bar_loop(broker, on_rust_bar, 0, limit)
    else:
        for i in range(limit):
            on_bar(i)

    # 4. Lifecycle Stop
    strategy.stop()
    stats = _build_stats(cerebro, strategy)
    metrics_engine = getattr(strategy, "metrics_engine", None)
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
