"""Alphalens-style factor analysis facade backed by tradelearn.metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradelearn.metrics import factor as factor_metrics


@dataclass(frozen=True)
class FactorAnalyzer:
    """Analyze cross-sectional factor values and forward returns."""

    factor: pd.Series
    forward_returns: pd.Series | None = None
    prices: pd.Series | None = None
    periods: int = 252
    quantiles: int = 5

    def ic(self) -> pd.Series:
        """Return per-date Pearson information coefficient."""
        return factor_metrics.ic(self.factor, self._forward_returns())

    def rank_ic(self) -> pd.Series:
        """Return per-date Spearman rank information coefficient."""
        return factor_metrics.rank_ic(self.factor, self._forward_returns())

    def ic_ir(self) -> float:
        """Return annualized IC information ratio."""
        return factor_metrics.ic_ir(self.ic(), periods=self.periods)

    def quantile_returns(self) -> pd.DataFrame:
        """Return mean forward returns by factor quantile."""
        return factor_metrics.quantile_returns(
            self.factor,
            self._forward_returns(),
            quantiles=self.quantiles,
        )

    def factor_returns(self) -> pd.DataFrame:
        """Return quantile returns derived from configured prices."""
        if self.prices is None:
            return self.quantile_returns()
        return factor_metrics.factor_returns(
            self.factor,
            self.prices,
            quantiles=self.quantiles,
        )

    def turnover(self) -> pd.Series:
        """Return factor rank turnover."""
        return factor_metrics.turnover(self.factor)

    def autocorrelation(self) -> pd.Series:
        """Return factor rank autocorrelation."""
        return factor_metrics.autocorrelation(self.factor)

    def summary(self) -> dict[str, float]:
        """Return scalar factor diagnostics."""
        ic_values = self.ic()
        rank_ic_values = self.rank_ic()
        turnover_values = self.turnover()
        autocorrelation_values = self.autocorrelation()
        return {
            "ic_mean": float(ic_values.mean()),
            "ic_std": float(ic_values.std(ddof=1)),
            "ic_ir": self.ic_ir(),
            "rank_ic_mean": float(rank_ic_values.mean()),
            "turnover_mean": float(turnover_values.mean()),
            "autocorrelation_mean": float(autocorrelation_values.mean()),
        }

    def _forward_returns(self) -> pd.Series:
        """Return configured or price-derived forward returns."""
        if self.forward_returns is not None:
            return self.forward_returns
        if self.prices is not None:
            forward = self.prices.groupby(level=1).pct_change().shift(-1)
            forward.name = "forward_returns"
            return forward
        raise ValueError("FactorAnalyzer requires forward_returns or prices")
