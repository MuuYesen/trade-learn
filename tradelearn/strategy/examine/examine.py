"""Legacy Examine facade backed by the v2 factor layer."""

from __future__ import annotations

import pandas as pd

from tradelearn.factor import FactorAnalyzer
from tradelearn.report import Reporter


class Examine:
    """Compatibility entry point for historical factor examination helpers."""

    @staticmethod
    def single_factor(data: pd.DataFrame, col: str, filename: str = "./examine.html") -> None:
        """Write a factor tear-sheet style HTML report for one factor column."""
        analyzer = _factor_analyzer(data, col)
        returns = _portfolio_returns(data)
        Reporter(
            {
                "returns": returns,
                "trades": pd.DataFrame(),
                "analyzers": {"factor": analyzer},
                "summary": {"strategy_name": col},
                "config": {"strategy": col},
            }
        ).html(filename)

    @staticmethod
    def factor_compare(
        data: pd.DataFrame,
        ind: str | None = None,
        cir: str | None = None,
        f_col: str | None = None,
    ) -> pd.DataFrame:
        """Return summary diagnostics for alpha columns in a legacy data frame."""
        del ind, cir
        columns = (
            [f_col]
            if f_col
            else [column for column in data.columns if column.startswith("alpha")]
        )
        rows = []
        for column in columns:
            analyzer = _factor_analyzer(data, column)
            rows.append({"name": column.split(".")[0], **analyzer.summary()})
        if not rows:
            return pd.DataFrame().rename_axis("name")
        return pd.DataFrame(rows).set_index("name")


def _factor_analyzer(data: pd.DataFrame, factor_column: str) -> FactorAnalyzer:
    """Build a FactorAnalyzer from legacy date/code/close data."""
    required = {"date", "code", "close", factor_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index(["date", "code"]).sort_index()
    factor = pd.Series(frame[factor_column], index=frame.index, dtype="float64")
    prices = pd.Series(frame["close"], index=frame.index, dtype="float64")
    return FactorAnalyzer(factor=factor, prices=prices)


def _portfolio_returns(data: pd.DataFrame) -> pd.Series:
    """Return equal-weight daily returns from legacy close prices."""
    close = data.pivot(index="date", columns="code", values="close").sort_index()
    returns = close.pct_change().mean(axis=1).dropna()
    returns.index = pd.to_datetime(returns.index)
    returns.name = "returns"
    return returns
