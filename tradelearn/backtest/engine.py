"""Core execution engine and component hub for TradeLearn."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, List

from tradelearn.backtest.base import (
    LineSeries, _notify_order, set_current_data, 
    set_current_datas, set_current_strategy
)
from tradelearn.backtest.strategy import Strategy
from tradelearn.backtest.datafeed import DataFeed
from tradelearn.backtest.sizer import Sizer, FixedSize, PercentSizer, AllInSizer
from tradelearn.backtest.cerebro import Cerebro
from tradelearn.backtest.models import Order, Position, ExecutedInfo
from tradelearn.backtest.analyzer import Analyzer

def run_backtest(cerebro: Cerebro) -> List[Strategy]:
    strategy_cls, args, kwargs = cerebro.strats[0]
    
    # 1. Initialize Context BEFORE Strategy __init__
    if cerebro.datas:
        set_current_data(cerebro.datas[0])
        set_current_datas(cerebro.datas)
    
    set_current_strategy(None)
    # The strategy __init__ will now find datas/data via global context
    strategy = strategy_cls(*args, **kwargs)
    set_current_strategy(strategy)
    
    # Ensure they are also set as attributes
    strategy.datas = cerebro.datas
    if cerebro.datas: strategy.data = cerebro.datas[0]
    
    # Reset context after init
    set_current_data(None)
    set_current_datas([])
    
    # ---------------------------------------------------------
    # Rust Engine Initialization
    # ---------------------------------------------------------
    from tradelearn.backtest.brokers.rust_broker import RustBroker
    if isinstance(cerebro.broker, RustBroker):
        from tradelearn._rust import RustBacktestEngine
        data = cerebro.datas[0]
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
            float(cerebro.broker._mult), 1.0
        )
        cerebro.broker._engine = rust_engine
    
    strategy.broker = cerebro.broker
    
    # 2. Initialize Sizer
    sizer_cls, sizer_kwargs = cerebro._sizer_spec
    strategy.setsizer(sizer_cls(**sizer_kwargs))
    # Indicators are now self-registered via LineRoot._base_init
    indicators = strategy._indicators
    
    limit = cerebro.datas[0].buflen()
    min_period = strategy._min_period
    
    for i in range(limit):
        # 1. Advance all lines
        for d in cerebro.datas: d._advance(i)
        for ind in indicators: ind._advance(i)
        
        # 2. Broker Match
        if cerebro.broker:
            cerebro.broker.step(i)
            cerebro.broker.process_fills(strategy, i)
            strategy.notify_cashvalue(cerebro.broker.getcash(), cerebro.broker.getvalue())

        # 3. Strategy Next
        if i >= min_period:
            strategy.next()
    
    return [strategy]
