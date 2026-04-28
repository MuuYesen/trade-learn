from __future__ import annotations

import pandas as pd

from tradelearn.backtest.core.broker import RustBroker
from tradelearn.backtest.core.engine import run_backtest
from tradelearn.compat.backtrader.datafeed import DataFeed

from .strategy import Strategy


class Backtest:
    """Facade for backtesting.py Backtest class."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: type[Strategy],
        cash: float = 10_000,
        commission: float = .0,
        margin: float = 1.,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = False,
        match_mode: str = 'exact' # Default to exact for alignment
    ):
        self._data = data
        self._strategy_cls = strategy
        self._cash = cash
        self._commission = commission

        # Internal state to match run_backtest expectations
        self.datas = [DataFeed(data)]
        self.strats = [(strategy, (), {})]
        self.match_mode = match_mode
        from tradelearn.compat.backtrader.sizer import FixedSize
        self._sizer_spec = (FixedSize, {})
        self.broker = RustBroker(cash=cash, commission=commission, match_mode=match_mode)
        self.analyzers = {}

    def run(self, **kwargs) -> pd.Series:
        """Run the backtest and return statistics."""
        # Reset broker for fresh run
        self.broker = RustBroker(
            cash=self._cash,
            commission=self._commission,
            match_mode=self.match_mode,
        )

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
            "Win Rate [%]": win_rate
        })

    def optimize(self, **kwargs) -> pd.Series:
        """Grid search optimization."""
        import itertools
        from concurrent.futures import ProcessPoolExecutor

        keys = list(kwargs.keys())
        values = list(kwargs.values())
        grid = list(itertools.product(*values))

        print(f"Starting Grid Search: {len(grid)} combinations...")

        with ProcessPoolExecutor() as executor:
            # Pass data and class separately to avoid pickling the whole 'self' if not needed
            # But here it's easier to just pass what we need
            results = list(executor.map(_optimize_worker,
                                        itertools.repeat(self._data),
                                        itertools.repeat(self._strategy_cls),
                                        itertools.repeat(self._cash),
                                        itertools.repeat(self._commission),
                                        itertools.repeat(self.match_mode),
                                        itertools.repeat(keys),
                                        grid))

        # Find best result based on Return [%]
        best_res, best_params = max(results, key=lambda x: x[0]["Return [%]"])
        print(f"Best Params: {best_params}")
        return best_res

    def plot(self, *args, **kwargs):
        """Mock plot method."""
        print("Plotting is not yet implemented in tradelearn.compat.backtesting")

def _optimize_worker(data, strategy_cls, cash, commission, match_mode, keys, params_values):
    params = dict(zip(keys, params_values, strict=False))
    bt = Backtest(data, strategy_cls, cash=cash, commission=commission, match_mode=match_mode)
    return bt.run(**params), params
