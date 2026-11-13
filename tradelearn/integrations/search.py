"""Parameter grid execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradelearn.backtest._optimize import expand_grid
from tradelearn.engine.cerebro import Cerebro
from tradelearn.engine.feeds import PandasData
from tradelearn.engine.strategy import Strategy


@dataclass(frozen=True)
class GridSearchResult:
    """One grid-search run result."""

    params: dict[str, Any]
    strategy: Strategy
    analyzer_results: dict[str, Any]
    trial_record: dict[str, Any] = field(default_factory=dict)


def grid_search(
    strategy: type[Strategy],
    data: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    *,
    mlflow: dict[str, Any] | None = None,
    cerebro_factory: Any = None,
) -> list[GridSearchResult]:
    """Run a parameter grid; optionally log each existing result after execution."""

    from copy import deepcopy

    from tradelearn.backtest._optimize import (
        cancel_remaining,
        effective_config,
        execute_trial,
        historical_template,
        strategy_config,
        strategy_defaults,
    )
    grid = expand_grid(param_grid)
    if not grid:
        return []
    owner = Cerebro() if cerebro_factory is None else cerebro_factory()
    if not isinstance(owner, Cerebro):
        raise TypeError('cerebro_factory must return a Cerebro')
    owner = historical_template(owner)
    owner.adddata(PandasData(dataname=data))
    template = historical_template(owner)
    config = effective_config(template)
    records = []
    results: list[GridSearchResult] = []
    for index, params in enumerate(grid):
        try:
            cerebro = deepcopy(template)
            cerebro.strats = []
            cerebro._dooptimize = False
            cerebro.addstrategy(strategy, **deepcopy({**strategy_defaults(strategy), **params}))
            [instance] = execute_trial(records, params,
                {**config, "strategies": [strategy_config(strategy, (), params)]}, cerebro.run,
                lambda value: dict(value[0].stats.summary))
            results.append(GridSearchResult(params=params, strategy=instance,
                analyzer_results=dict(instance.analyzer_results), trial_record=records[-1]))
        except BaseException as exc:
            cancel_remaining(records, [(item,
                {**config, "strategies": [strategy_config(strategy, (), item)]})
                for item in grid[index + 1:]])
            try:
                exc.trial_records = records
            except (AttributeError, TypeError):
                pass
            if mlflow is not None:
                try:
                    _log_trials(results, records, mlflow)
                except Exception as tracking_error:
                    # A secondary tracking error must not replace the trial error.
                    exc.tracking_error_type = type(tracking_error).__name__
            raise
    if mlflow is not None:
        _log_trials(results, records, mlflow)
    return results


def _log_trials(results, records, mlflow):
    from tradelearn.integrations.mlflow import log
    from tradelearn.research.run import ResearchResult
    options = dict(mlflow)
    if "experiment" in options:
        options["experiment_name"] = options.pop("experiment")
    options.pop("nested", None)
    research = ResearchResult(name='grid_search', artifacts={'trials': records})
    primary = [result.strategy for result in results] if results else research
    logged = log(primary, research_result=research, **options)
    for result in results:
        result.strategy.mlflow_result = logged
