"""Parameter grid execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from tradelearn.backtest.analyzers import MLflowAnalyzer
from tradelearn.backtest.engine import Cerebro, Strategy


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
    """Run a simple strategy parameter grid with nested MLflow runs."""

    results: list[GridSearchResult] = []
    for index, params in enumerate(_expand_grid(param_grid)):
        cerebro = Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(strategy, **params)
        if mlflow is not None:
            analyzer_params = dict(mlflow)
            analyzer_params.setdefault("nested", True)
            analyzer_params.setdefault("run_name", f"{strategy.__name__}[{index}]")
            cerebro.addanalyzer(MLflowAnalyzer, name="mlflow", **analyzer_params)
        [strategy_instance] = cerebro.run()
        results.append(
            GridSearchResult(
                params=params,
                strategy=strategy_instance,
                analyzer_results=dict(strategy_instance.analyzer_results),
            )
        )
    return results


def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, item, strict=True)) for item in product(*values)]
