from __future__ import annotations
from typing import Any, Type, List, Dict
import pandas as pd
from tradelearn.backtest.core.engine import run_backtest
from tradelearn.backtest.core.brokers.rust import RustBroker
from .strategy import Strategy

from tradelearn.compat.backtrader.datafeed import DataFeed

class Backtest:
    """Facade for backtesting.py Backtest class."""
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Type[Strategy],
        cash: float = 10_000,
        commission: float = .0,
        margin: float = 1.,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = False
    ):
        self._data = data
        self._strategy_cls = strategy
        self._cash = cash
        self._commission = commission
        
        # Internal state to match run_backtest expectations
        self.datas = [DataFeed(data)]
        self.strats = [(strategy, (), {})]
        self.match_mode = 'smart' # Use smart mode for higher performance
        from tradelearn.compat.backtrader.sizer import FixedSize
        self._sizer_spec = (FixedSize, {})
        self.broker = RustBroker(cash=cash, commission=commission)
        self.analyzers = {}

    def run(self, **kwargs) -> pd.Series:
        """Run the backtest and return statistics."""
        # Update strategy params from kwargs if any
        if kwargs:
            self.strats = [(self._strategy_cls, (), kwargs)]
            
        results = run_backtest(self)
        strategy_instance = results[0]
        
        # Generate statistics
        final_value = self.broker.getvalue()
        
        trades = getattr(strategy_instance, '_trades', [])
        wins = [t for t in trades if getattr(t, 'pnl', 0) > 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        
        return pd.Series({
            "Equity Final [$]": final_value,
            "Return [%]": (final_value / self._cash - 1) * 100,
            "# Trades": len(trades),
            "Win Rate [%]": win_rate,
            "Sharpe Ratio": 0.0, # Placeholder
            "Max. Drawdown [%]": 0.0, # Placeholder
            "Return (Ann.) [%]": 0.0, # Placeholder
            "Avg. Trade [%]": 0.0, # Placeholder
        })

    def plot(self, *args, **kwargs):
        """Mock plot method."""
        print("Plotting is not yet implemented in tradelearn.compat.backtesting")
