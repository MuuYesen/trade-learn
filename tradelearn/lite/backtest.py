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
from tradelearn.backtest.models import Stats, SummaryDict
from tradelearn.backtest._optimize import expand_grid
from tradelearn.backtest.reporting import reporter_from_stats
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.core import get_logger
from tradelearn.utils.console import smart_tqdm as tqdm, smart_print

from .strategy import Strategy

LOGGER = get_logger("lite.backtest")


class LiteStats(Mapping[str, Any]):
    """Lite run result.

    Summary values are available through mapping access, while detailed
    artifacts delegate to the shared backtest Stats object.
    """

    def __init__(self, stats: Stats, strategy: Strategy, records: dict[str, pd.Series]) -> None:
        self.stats = stats
        self.strategy = strategy
        self.records = records
        self.summary = stats.summary if isinstance(stats.summary, SummaryDict) else SummaryDict(stats.summary)

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
        stats_mode: str = "full",
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
        if stats_mode not in {"full", "lazy"}:
            raise ValueError("stats_mode must be one of 'full' or 'lazy'")
        self.stats_mode = stats_mode
        from tradelearn.backtest.sizer import FixedSize
        self._sizer_spec = (FixedSize, {})
        self.broker = RustBroker(cash=cash, commission=commission, match_mode=match_mode)
        self.broker.configure_matching(trade_on_close=self.trade_on_close)
        self.broker.set_storage(self._storage)
        self.analyzers = {}
        self.runtime_config = BacktestRuntimeConfig.from_owner(self)

    @staticmethod
    def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
        return normalize_ohlcv_frame(data)

    def run(self, **kwargs) -> LiteStats:
        """Run the backtest and return statistics."""
        feeds = len(self.datas)
        rows = _feed_rows(self.datas)
        label = f"Backtest ({self._strategy_cls.__name__}, {feeds} feed{'s' if feeds != 1 else ''}, {rows:,} bars)"
        LOGGER.info(
            "Backtest started strategy=%s feeds=%s rows=%s cash=%s commission=%s",
            self._strategy_cls.__name__,
            feeds,
            rows,
            self._cash,
            self._commission,
        )
        # Reset broker for fresh run
        self.broker = RustBroker(
            cash=self._cash,
            commission=self._commission,
            match_mode=self.match_mode,
        )
        self.broker.configure_matching(trade_on_close=self.trade_on_close)
        self.broker.set_storage(self._storage)
        self.runtime_config = BacktestRuntimeConfig.from_owner(self)

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
        result = LiteStats(self._last_stats, strategy_instance, records)
        LOGGER.info(
            "Backtest finished strategy=%s final_value=%s return_pct=%s total_trades=%s total_fills=%s",
            self._strategy_cls.__name__,
            _summary_value(result.summary, "final_value"),
            _summary_value(result.summary, "return_pct"),
            _summary_value(result.summary, "total_trades"),
            _summary_value(result.summary, "total_fills"),
        )
        return result

    def optimize(self, **kwargs) -> LiteStats:
        """Simple grid search optimization."""
        from concurrent.futures import ProcessPoolExecutor
        from itertools import repeat

        grid = expand_grid(kwargs)

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_optimize_worker,
                                        repeat(self._data),
                                        repeat(self._strategy_cls),
                                        repeat(self._cash),
                                        repeat(self._commission),
                                        repeat(self.match_mode),
                                        grid), total=len(grid), desc="Backtest.optimize"))

        best_res, best_params = max(results, key=lambda x: x[0]["return_pct"])
        smart_print(f"Best Params: {best_params}")
        return best_res

    def plot(self, *args, **kwargs):
        """Return market replay charts for the most recent Lite run."""
        reporter = self._last_reporter()
        chart = reporter.market_replay_chart()
        return [] if chart is None else [chart]

    def report(self, path: str = "report.html", benchmark=None, sections=None):
        """Write a Tradelearn report for the most recent Lite run."""
        with tqdm(total=1, desc="Backtest.report", leave=True) as pbar:
            result = self._last_reporter().report(path, benchmark=benchmark, sections=sections)
            pbar.update(1)
        return result

    def log_mlflow(
        self,
        experiment_name: str = "tradelearn-lite",
        run_name: str | None = None,
        *,
        uri: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
        log_mlflow: bool = True,
        upload_artifacts: bool = True,
        log_artifacts: bool | None = None,
        artifact_path: str | None = None,
        log_report: bool = True,
        log_plot: bool = False,
        mlflow_module: Any | None = None,
    ) -> str:
        """Log the most recent Lite run to MLflow."""
        if not log_mlflow:
            return "skipped"
        from .mlflow import log_lite_run

        return log_lite_run(
            self,
            experiment_name=experiment_name,
            run_name=run_name,
            uri=uri,
            params=params,
            tags=tags,
            nested=nested,
            upload_artifacts=upload_artifacts,
            log_artifacts=log_artifacts,
            artifact_path=artifact_path,
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


def _feed_rows(datas: list[Any]) -> int:
    return int(sum(len(getattr(data, "_frame", ())) for data in datas))


def _summary_value(summary: Mapping[str, Any], key: str) -> Any:
    return summary.get(key, "n/a")
