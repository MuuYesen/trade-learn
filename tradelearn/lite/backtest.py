from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pandas as pd

from tradelearn.backtest.broker import RustBroker
from tradelearn.backtest.engine import run_backtest
from tradelearn.backtest.feed import (
    build_data_feeds,
    is_normalized_ohlcv_frame,
    normalize_ohlcv_frame,
)
from tradelearn.backtest.models import Stats
from tradelearn.backtest.optimize import expand_grid
from tradelearn.backtest.reporting import reporter_from_stats

from .strategy import Strategy


class LiteStats(Mapping[str, Any]):
    """Lite run result.

    Summary values are available through mapping access, while detailed
    artifacts delegate to the shared backtest Stats object.
    """

    def __init__(self, stats: Stats, strategy: Strategy, records: dict[str, pd.Series]) -> None:
        self.stats = stats
        self.strategy = strategy
        self.records = records
        self.summary = dict(stats.summary)

    def __getitem__(self, key: str) -> Any:
        return self.summary[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.summary)

    def __len__(self) -> int:
        return len(self.summary)

    def get(self, key: str, default: Any = None) -> Any:
        return self.summary.get(key, default)

    @property
    def equity(self) -> pd.Series:
        return self.stats.equity

    @property
    def returns(self) -> pd.Series:
        return self.stats.returns

    @property
    def fills(self) -> pd.DataFrame:
        return self.stats.fills

    @property
    def trades(self) -> pd.DataFrame:
        return self.stats.trades

    @property
    def positions(self) -> pd.DataFrame:
        return self.stats.positions

    @property
    def orders(self) -> pd.DataFrame:
        return self.stats.orders

    @property
    def config(self) -> dict[str, Any]:
        return self.stats.config


class Backtest:
    """Tradelearn Lite quick backtest facade."""

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
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
        self.datas = build_data_feeds(
            data,
            assume_normalized=_can_skip_normalize_data(data),
        )
        self.strats = [(strategy, (), {})]
        self.match_mode = match_mode
        self.stats_mode = "full"
        from tradelearn.backtest.sizer import FixedSize
        self._sizer_spec = (FixedSize, {})
        self.broker = RustBroker(cash=cash, commission=commission, match_mode=match_mode)
        self.broker.configure_matching(trade_on_close=self.trade_on_close)
        self.broker.set_storage(self._storage)
        self.analyzers = {}

    @staticmethod
    def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
        return normalize_ohlcv_frame(data)

    def run(self, **kwargs) -> LiteStats:
        """Run the backtest and return statistics."""
        # Reset broker for fresh run
        self.broker = RustBroker(
            cash=self._cash,
            commission=self._commission,
            match_mode=self.match_mode,
        )
        self.broker.configure_matching(trade_on_close=self.trade_on_close)
        self.broker.set_storage(self._storage)

        # Lite passes strategy params through run(**kwargs).
        if kwargs:
            self.strats = [(self._strategy_cls, (), kwargs)]

        results = run_backtest(self)
        strategy_instance = results[0]
        self._last_results = results
        self._last_strategy = strategy_instance
        self._last_stats = getattr(strategy_instance, "stats", None)
        if self._last_stats is None:
            raise RuntimeError("backtest runtime did not produce stats")

        records = {
            key: value.copy()
            for key, value in getattr(strategy_instance, "_records", {}).items()
        }
        return LiteStats(self._last_stats, strategy_instance, records)

    def optimize(self, **kwargs) -> LiteStats:
        """Simple grid search optimization."""
        from concurrent.futures import ProcessPoolExecutor
        from itertools import repeat

        grid = expand_grid(kwargs)

        print(f"Starting Grid Search: {len(grid)} combinations...")

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(_optimize_worker,
                                        repeat(self._data),
                                        repeat(self._strategy_cls),
                                        repeat(self._cash),
                                        repeat(self._commission),
                                        repeat(self.match_mode),
                                        grid))

        best_res, best_params = max(results, key=lambda x: x[0]["return_pct"])
        print(f"Best Params: {best_params}")
        return best_res

    def plot(self, *args, **kwargs):
        """Return market replay charts for the most recent Lite run."""
        reporter = self._last_reporter()
        chart = reporter.market_replay_chart()
        return [] if chart is None else [chart]

    def report(self, path: str = "report.html", benchmark=None):
        """Write a Tradelearn report for the most recent Lite run."""
        return self._last_reporter().report(path, benchmark=benchmark)

    def log_mlflow(
        self,
        experiment_name: str = "tradelearn-lite",
        run_name: str | None = None,
        *,
        uri: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
        artifact_file: str = "stats.json",
        artifact_path: str = "tradelearn",
        artifact_bundle: bool = True,
        log_report: bool = True,
        log_plot: bool = True,
        mlflow_module: Any | None = None,
    ) -> str:
        """Log the most recent Lite run to MLflow."""
        from .mlflow import log_lite_run

        return log_lite_run(
            self,
            experiment_name=experiment_name,
            run_name=run_name,
            uri=uri,
            params=params,
            tags=tags,
            nested=nested,
            artifact_file=artifact_file,
            artifact_path=artifact_path,
            artifact_bundle=artifact_bundle,
            log_report=log_report,
            log_plot=log_plot,
            mlflow_module=mlflow_module,
        )

    def _last_reporter(self):
        stats = getattr(self, "_last_stats", None)
        if stats is None:
            raise RuntimeError("run() must be called before plot() or report()")
        return reporter_from_stats(stats, getattr(self, "datas", None))

def _optimize_worker(data, strategy_cls, cash, commission, match_mode, params):
    bt = Backtest(data, strategy_cls, cash=cash, commission=commission, match_mode=match_mode)
    return bt.run(**params), params


def _can_skip_normalize_data(data: pd.DataFrame | dict[str, pd.DataFrame]) -> bool:
    if isinstance(data, dict):
        return bool(data) and all(is_normalized_ohlcv_frame(frame) for frame in data.values())
    return is_normalized_ohlcv_frame(data)
