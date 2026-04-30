"""Legacy query facade backed by v2 factor implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd


class Query:
    """Compatibility entry point for legacy factor helpers.

    The v2 data/query path lives under :mod:`tradelearn.data` and
    :mod:`tradelearn.factor`.  This class keeps historical imports available
    without reintroducing retired provider dependencies.
    """

    @staticmethod
    def alphas101(data: pd.DataFrame, names: list[str] | tuple[str, ...] | None = None):
        """Return Alpha101 factors using the v2 factor facade."""
        from tradelearn.factor.alpha import alpha101

        return alpha101(data, names=names)

    @staticmethod
    def alphas191(
        stock_data: pd.DataFrame,
        bench_data: pd.DataFrame | None = None,
        names: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ):
        """Return Alpha191 factors using the v2 factor facade."""
        from tradelearn.factor.alpha import alpha191

        return alpha191(stock_data, bench_data, names=names, **kwargs)


__all__ = ["Query"]
