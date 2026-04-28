from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, List
from tradelearn.backtest.core.strategy import Strategy as CoreStrategy

# These are still needed for Backtrader compatibility
from tradelearn.compat.backtrader.base import (
    set_current_data, set_current_datas, set_current_strategy
)

def run_backtest(cerebro: Any) -> List[Any]:
    """Unified backtest engine that runs any strategy inheriting from core.Strategy."""
    strategy_cls, args, kwargs = cerebro.strats[0]
    
    # 1. Initialize Context for Compatibility Facades
    if cerebro.datas:
        set_current_data(cerebro.datas[0])
        set_current_datas(cerebro.datas)
    
    set_current_strategy(None)
    strategy = strategy_cls(*args, **kwargs)
    set_current_strategy(strategy)
    
    # Core attributes
    strategy.datas = cerebro.datas
    if cerebro.datas: strategy.data = cerebro.datas[0]
    strategy.broker = cerebro.broker
    
    if hasattr(strategy, '_setup'):
        strategy._setup()

    # Reset context after init
    set_current_data(None)
    set_current_datas([])
    
    # ... (Rust Engine Initialization omitted for brevity but preserved in real file) ...
    # ---------------------------------------------------------
    # Rust Engine Initialization
    # ---------------------------------------------------------
    from .brokers.rust import RustBroker
    if isinstance(cerebro.broker, RustBroker):
        from tradelearn._rust import RustBacktestEngine
        data = cerebro.datas[0]
        # Robustly handle both core.DataContainer and compat.backtrader.DataFeed
        if hasattr(data, '_datetime'):
            timestamps = data._datetime
            opens = data._open
            highs = data._high
            lows = data._low
            closes = data._close
            volumes = data._volume
        else: # BT Legacy / LineSeries fallback
            timestamps = np.array(data.datetime._values, dtype=np.int64)
            opens = np.array(data.open._values, dtype=np.float64)
            highs = np.array(data.high._values, dtype=np.float64)
            lows = np.array(data.low._values, dtype=np.float64)
            closes = np.array(data.close._values, dtype=np.float64)
            volumes = np.array(data.volume._values, dtype=np.float64)
        
        rust_engine = RustBacktestEngine(
            timestamps, opens, highs, lows, closes, volumes,
            float(cerebro.broker._cash),
            float(cerebro.broker.commission_ratio),
            False, False, False, 
            0.0, 0.0, False, False, False, 
            float(cerebro.broker._mult), 1.0,
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
    indicators = getattr(strategy, '_indicators', [])
    indicators_bt = getattr(strategy, '_indicators_bt', [])
    
    strategy.analyzers = {}
    for name, (ana_cls, ana_kwargs) in cerebro.analyzers.items():
        ana_inst = ana_cls(**ana_kwargs)
        ana_inst.strategy = strategy
        strategy.analyzers[name] = ana_inst
    
    # 3. Lifecycle Start
    if hasattr(strategy, '_setup'):
        strategy._setup()
    strategy.init()
    strategy.start()
    for ana in strategy.analyzers.values():
        if hasattr(ana, 'start'): ana.start()

    limit = cerebro.datas[0].buflen()
    # Calculate min_period from all indicators (mostly for BT facade)
    min_period = int(getattr(strategy, '_manual_min_period', 0))
    for ind in indicators + indicators_bt:
        if hasattr(ind, 'min_period'):
            m = ind.min_period
            if callable(m): m = m()
            min_period = max(min_period, int(m))
    if min_period == 0: min_period = 1

    bar_advancers = []
    seen_advancer_ids = set()
    for d in cerebro.datas:
        bar_advancers.append(d._advance)
        seen_advancer_ids.add(id(d))
    for ind in indicators + indicators_bt:
        if hasattr(ind, '_advance') and id(ind) not in seen_advancer_ids:
            bar_advancers.append(ind._advance)
            seen_advancer_ids.add(id(ind))
    strategy_advance = getattr(strategy, '_advance', None)
    if strategy_advance is not None:
        bar_advancers.append(strategy_advance)
    for attr, val in strategy.__dict__.items():
        if attr.startswith('_'):
            continue
        if id(val) in seen_advancer_ids:
            continue
        advance = getattr(val, '_advance', None)
        if callable(advance):
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(val))
    bar_advancers = tuple(bar_advancers)
    notify_cashvalue = None
    if type(strategy).notify_cashvalue is not CoreStrategy.notify_cashvalue:
        notify_cashvalue = strategy.notify_cashvalue
    broker = cerebro.broker
    broker_step = broker.step if broker else None
    broker_process_fills = broker.process_fills if broker else None
    broker_getcash = broker.getcash if broker else None
    broker_getvalue = broker.getvalue if broker else None
    begin_order_buffering = getattr(broker, 'begin_order_buffering', None) if broker else None
    flush_order_buffer = getattr(broker, 'flush_order_buffer', None) if broker else None
    drain_order_buffer = getattr(broker, 'drain_order_buffer', None) if broker else None
    strategy_next = strategy.next
    
    use_rust_bar_loop = (
        isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and hasattr(broker._engine, "run_bar_loop")
    )
    min_start = min_period - 1

    def on_bar(i: int) -> list[Any]:
        for advance in bar_advancers:
            advance(i)
        
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
            else:
                strategy_next()
        return []

    if use_rust_bar_loop:
        if notify_cashvalue is None:
            def on_rust_bar(i: int, fills: list[Any], cash: float, size: float, price: float) -> list[Any]:
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
                    return drain_order_buffer()
                return []
        else:
            def on_rust_bar(i: int, fills: list[Any], cash: float, size: float, price: float) -> list[Any]:
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
                    return drain_order_buffer()
                return []

        broker._engine.run_bar_loop(broker, on_rust_bar, 0, limit)
    else:
        for i in range(limit):
            on_bar(i)
    
    # 4. Lifecycle Stop
    strategy.stop()
    for ana in strategy.analyzers.values():
        if hasattr(ana, 'stop'): ana.stop()

    set_current_strategy(None)
    set_current_data(None)
    set_current_datas([])

    return [strategy]
