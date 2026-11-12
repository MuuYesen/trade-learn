"""Parameter grid execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
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


def grid_search(
    strategy: type[Strategy],
    data: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    *,
    mlflow: dict[str, Any] | None = None,
) -> list[GridSearchResult]:
    """Run a parameter grid; optionally log each existing result after execution."""

    results: list[GridSearchResult] = []
    for params in expand_grid(param_grid):
        cerebro = Cerebro()
        cerebro.adddata(PandasData(dataname=data))
        cerebro.addstrategy(strategy, **params)
        [strategy_instance] = cerebro.run()
        results.append(
            GridSearchResult(
                params=params,
                strategy=strategy_instance,
                analyzer_results=dict(strategy_instance.analyzer_results),
            )
        )
    if mlflow is not None and results:
        from tradelearn.integrations.mlflow import log

        options = dict(mlflow)
        if "experiment" in options:
            options["experiment_name"] = options.pop("experiment")
        options.pop("nested", None)
        logged = log([result.strategy for result in results], **options)
        for result in results:
            result.strategy.mlflow_result = logged
    return results
