from __future__ import annotations
from typing import Any, Callable, List, Dict
import numpy as np
import pandas as pd
from tradelearn.backtest.core.strategy import Strategy as CoreStrategy
from tradelearn.backtest.core.data import DataContainer

class BacktestingDataProxy:
    """Proxy for DataContainer to support backtesting.py style capitalized attributes."""
    def __init__(self, container: DataContainer):
        self._container = container

    def __getattr__(self, name: str) -> Any:
        # Map capitalized OHLCV to core lower case
        mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
        core_name = mapping.get(name, name.lower())
        arr = self._container.get_array(core_name)
        # In backtesting.py, self.data.Close is a full series up to current bar
        # For performance, we might just return the whole array and let user use [-1]
        return arr[:self._container._cursor + 1]

    def __len__(self) -> int:
        return len(self._container)

class Strategy(CoreStrategy):
    """Facade for backtesting.py style strategies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backtesting_data = None
        self._indicators_bt: List[Any] = []

    def _setup(self):
        # Called by engine before start/init
        self._backtesting_data = BacktestingDataProxy(self.data)
        # backtesting.py uses 'init' instead of 'start'
        if hasattr(self, 'init'):
            self.init()

    @property
    def data(self) -> Any:
        # Override to return the proxy if setup
        return self._backtesting_data or super().data

    @data.setter
    def data(self, value: Any):
        # Allow engine to set the core data
        # We'll wrap it in _setup
        self.__dict__['data'] = value

    def I(self, func: Callable, *args, **kwargs) -> Any:
        """Indicator wrapper."""
        # Execute func with args
        res = func(*args, **kwargs)
        # If it returns a series/array, wrap it so it advances
        self._indicators_bt.append(res)
        return res

    def next(self):
        # Default next does nothing
        pass

    def _advance(self, i: int):
        # This facade might need to do something on each step
        pass
