"""Backtrader-compatible Cerebro facade."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradelearn.backtest import Cerebro as BacktestCerebro
from tradelearn.backtest import DataFeed


class Cerebro(BacktestCerebro):
    """Cerebro facade accepting backtrader-style PandasData feeds."""

    def adddata(self, data: pd.DataFrame | DataFeed, name: str | None = None) -> DataFeed:
        feed_name = name if name is not None else getattr(data, "_name", None)
        return super().adddata(data, name=feed_name)

    def addstrategy(self, strategy: type[Any], *args: Any, **params: Any) -> None:
        if args:
            raise TypeError("compat.backtrader.Cerebro.addstrategy only supports keyword params")
        super().addstrategy(strategy, **params)
