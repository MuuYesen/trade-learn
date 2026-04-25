"""Reporter facade backed by tradelearn.metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tradelearn import metrics
from tradelearn.report.excel import write_excel_report
from tradelearn.report.explore import explore_trades


class Reporter:
    """Build report-ready summaries and series from backtest stats."""

    def __init__(self, stats: Any, periods: int = 252) -> None:
        """Create a reporter from a Stats-like object or mapping."""
        self.stats = stats
        self.periods = periods

    def summary(self, benchmark: pd.Series | None = None) -> dict[str, float | int | str]:
        """Return report summary statistics."""
        returns = self._get("returns")
        trades = self._get("trades", default=pd.DataFrame())
        summary = dict(self._get("summary", default={}) or {})
        computed: dict[str, float | int] = {
            "annual_return": metrics.annual_return(returns, periods=self.periods),
            "cumulative_return": metrics.cum_returns(returns).iloc[-1],
            "annual_volatility": metrics.volatility(returns, periods=self.periods),
            "sharpe_ratio": metrics.sharpe(returns, periods=self.periods),
            "calmar_ratio": metrics.calmar(returns, periods=self.periods),
            "sortino_ratio": metrics.sortino(returns, periods=self.periods),
            "max_drawdown": metrics.max_drawdown(returns),
            "max_dd_duration": self._max_drawdown_duration(returns),
        }
        if benchmark is not None:
            computed.update(
                {
                    "alpha": metrics.alpha(returns, benchmark, periods=self.periods),
                    "beta": metrics.beta(returns, benchmark),
                    "information_ratio": metrics.information_ratio(
                        returns,
                        benchmark,
                        self.periods,
                    ),
                }
            )
        computed.update(
            {
                "win_rate": metrics.win_rate(trades),
                "profit_factor": metrics.profit_factor(trades),
                "avg_win": metrics.avg_win(trades),
                "avg_loss": metrics.avg_loss(trades),
                "total_trades": int(len(trades)),
                "turnover": self._turnover(),
            }
        )
        return computed | summary

    def equity_curve(self) -> pd.Series:
        """Return normalized equity curve starting at 1.0."""
        return metrics.cum_returns(self._get("returns"), starting_value=1.0)

    def drawdown(self) -> pd.Series:
        """Return drawdown series."""
        return metrics.drawdown_series(self._get("returns"))

    def excel(self, path: str) -> Any:
        """Write an Excel report."""
        return write_excel_report(self, path)

    def explore(self) -> Any:
        """Open an interactive pygwalker explorer for trades."""
        return explore_trades(self._get("trades", default=pd.DataFrame()))

    def _get(self, name: str, default: Any = None) -> Any:
        """Read a field from mapping or object stats."""
        if isinstance(self.stats, Mapping):
            return self.stats.get(name, default)
        return getattr(self.stats, name, default)

    def _turnover(self) -> float:
        """Return explicit turnover summary or NaN until positions are supported."""
        summary = self._get("summary", default={}) or {}
        if "turnover" in summary:
            return float(summary["turnover"])
        return np.nan

    @staticmethod
    def _max_drawdown_duration(returns: pd.Series) -> int:
        """Return longest drawdown run length in periods."""
        drawdown = metrics.drawdown_series(returns)
        current = 0
        longest = 0
        for value in drawdown:
            if value < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest
