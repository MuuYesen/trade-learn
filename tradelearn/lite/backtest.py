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
from tradelearn.core.logging import get_logger
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
    def metric_metadata(self) -> dict[str, Any]:
        return self.stats.metric_metadata

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
        unsupported = {
            "margin": margin != 1.,
            "hedging": hedging,
            "exclusive_orders": exclusive_orders,
            "holding": bool(holding),
            "trade_start_date": trade_start_date is not None,
            "lot_size": lot_size != 1,
            "fail_fast": not fail_fast,
        }
        for name, requested in unsupported.items():
            if requested:
                raise NotImplementedError(f"Lite Backtest does not implement non-default {name}")
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
        # Preserve the effective economic configuration while resetting run state.
        from tradelearn.backtest._optimize import fresh_broker
        self.broker = fresh_broker(self.broker)
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
        """Run a finite grid sequentially with isolated, identically configured runs."""
        if self._storage:
            raise NotImplementedError("Lite optimize requires empty storage for isolated trials")
        grid = expand_grid(kwargs)
        if not grid:
            raise ValueError("Lite optimize requires a nonempty parameter grid")
        from copy import deepcopy
        from tradelearn.backtest._optimize import (
            historical_template, effective_config, execute_trial, cancel_remaining, strategy_config, materialize_spec,
        )
        template = historical_template(self)
        config = effective_config(template)
        self.trial_records = []
        best_res, best_params = None, None
        for index, params in enumerate(tqdm(grid, desc="Backtest.optimize")):
            try:
                owner = deepcopy(template)
                trial_params = materialize_spec((self._strategy_cls, (), params))[2]
                owner.strats = [(self._strategy_cls, (), trial_params)]
                trial = execute_trial(self.trial_records, params,
                    {**config, "strategies": [strategy_config(self._strategy_cls, (), params)]},
                    lambda: owner.run(**trial_params), lambda result: result["return_pct"])
            except BaseException:
                cancel_remaining(self.trial_records, [(item,
                    {**config, "strategies": [strategy_config(self._strategy_cls, (), item)]})
                    for item in grid[index + 1:]])
                raise
            if best_res is None or trial["return_pct"] > best_res["return_pct"]:
                best_res, best_params = trial, params
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
        experiment_name: str | None = None,
        run_name: str | None = None,
        *,
        research_result: Any = None,
        uri: str | None = None,
        log_report: bool = False,
        attachments: tuple[str, ...] = (),
    ) -> Any:
        """Record the last result using the independent post-run integration."""
        from .mlflow import log_lite_run

        return log_lite_run(
            self, experiment_name=experiment_name, run_name=run_name,
            research_result=research_result, uri=uri,
            log_report=log_report, attachments=attachments,
        )

    def _last_reporter(self):
        stats = getattr(self, "_last_stats", None)
        if stats is None:
            raise RuntimeError("run() must be called before plot() or report()")
        return reporter_from_stats(stats, getattr(self, "datas", None))

def _can_skip_normalize_data(data: pd.DataFrame | dict[str, pd.DataFrame]) -> bool:
    if isinstance(data, dict):
        return bool(data) and all(is_normalized_ohlcv_frame(frame) for frame in data.values())
    return is_normalized_ohlcv_frame(data)


def _feed_rows(datas: list[Any]) -> int:
    return int(sum(len(getattr(data, "_frame", ())) for data in datas))


def _summary_value(summary: Mapping[str, Any], key: str) -> Any:
    return summary.get(key, "n/a")
