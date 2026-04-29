from __future__ import annotations

import pandas as pd

from tradelearn.backtest.broker import RustBroker
from tradelearn.backtest.engine import run_backtest
from tradelearn.engine.datafeed import DataFeed

from .strategy import Strategy


class Backtest:
    """Tradelearn Lite quick backtest facade."""

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
        holding: dict[str, float] | None = None,
        trade_start_date: str | pd.Timestamp | None = None,
        lot_size: int = 1,
        fail_fast: bool = True,
        storage: dict | None = None,
        match_mode: str = 'exact' # Default to exact for alignment
    ):
        self._data = data
        self._strategy_cls = strategy
        self._cash = cash
        self._commission = commission
        self._holding = holding or {}
        self._margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        self.trade_start_date = trade_start_date
        self.lot_size = lot_size
        self.fail_fast = fail_fast
        self._storage = storage if storage is not None else {}

        # Internal state to match the shared backtest runtime expectations.
        self.datas = [DataFeed(self._normalize_data(data), name="Asset")]
        self.strats = [(strategy, (), {})]
        self.match_mode = match_mode
        self.stats_mode = "lazy"
        from tradelearn.engine.sizers import FixedSize
        self._sizer_spec = (FixedSize, {})
        self.broker = RustBroker(cash=cash, commission=commission, match_mode=match_mode)
        self.broker._storage = self._storage
        self.analyzers = {}

    @staticmethod
    def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            if len(frame.columns.levels[0]) != 1:
                raise NotImplementedError(
                    "tradelearn.lite currently supports one ticker per Backtest run"
                )
            frame = frame.xs(frame.columns.levels[0][0], axis=1, level=0)
        lower = {column: str(column).lower() for column in frame.columns}
        frame = frame.rename(columns=lower)
        required = {"open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"data must contain OHLC columns; missing {sorted(missing)}")
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        if not isinstance(frame.index, pd.DatetimeIndex):
            numeric_index = pd.to_numeric(frame.index, errors="coerce")
            if not pd.isna(numeric_index).any() and numeric_index.max() > 10_000_000:
                frame.index = pd.to_datetime(numeric_index, unit="s", utc=True)
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
        return frame

    def run(self, **kwargs) -> pd.Series:
        """Run the backtest and return statistics."""
        # Reset broker for fresh run
        self.broker = RustBroker(
            cash=self._cash,
            commission=self._commission,
            match_mode=self.match_mode,
        )
        self.broker._storage = self._storage

        # Tradelearn 1.x passes strategy params through run(**kwargs).
        if kwargs:
            self.strats = [(self._strategy_cls, (), kwargs)]

        results = run_backtest(self)
        strategy_instance = results[0]

        # Generate statistics
        final_value = self.broker.getvalue()
        trade_summary = getattr(self.broker, "trade_summary", None)
        if callable(trade_summary):
            trade_count, wins = trade_summary()
        else:
            trades = getattr(strategy_instance, "_trades", [])
            trade_count = len(trades)
            wins = sum(1 for trade in trades if getattr(trade, "pnl", 0) > 0)
        win_rate = (wins / trade_count * 100) if trade_count else 0.0

        records = {
            key: value.copy()
            for key, value in getattr(strategy_instance, "_records", {}).items()
        }

        return pd.Series({
            "Equity Final [$]": final_value,
            "Return [%]": (final_value / self._cash - 1) * 100,
            "# Trades": trade_count,
            "Win Rate [%]": win_rate,
            "_strategy": strategy_instance,
            "_records": records,
        })

    def optimize(self, **kwargs) -> pd.Series:
        """Simple grid search optimization."""
        import itertools
        from concurrent.futures import ProcessPoolExecutor

        keys = list(kwargs.keys())
        values = list(kwargs.values())
        grid = list(itertools.product(*values))

        print(f"Starting Grid Search: {len(grid)} combinations...")

        with ProcessPoolExecutor() as executor:
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
        print("Plotting is not yet implemented in tradelearn.lite")

def _optimize_worker(data, strategy_cls, cash, commission, match_mode, keys, params_values):
    params = dict(zip(keys, params_values, strict=False))
    bt = Backtest(data, strategy_cls, cash=cash, commission=commission, match_mode=match_mode)
    return bt.run(**params), params
